"""Trích thực thể, quan hệ và ngưỡng chẩn đoán từ các chunk bằng LLM, rồi kiểm chứng bằng code.

Ba loại tác vụ cho mỗi chunk:
- guided: tên mục khớp luật (vd. "Triệu chứng" → HAS_SYMPTOM), LLM chỉ liệt kê thực thể của mục đó;
- generic: mục/bài viết không khớp luật, LLM trích bộ ba theo ontology;
- threshold: đoạn có con số kèm đơn vị xét nghiệm, LLM trích ngưỡng chẩn đoán.
Mọi kết quả LLM được cache theo nội dung nên chạy lại hoặc chạy tiếp sau khi dừng đều không tốn thêm thời gian.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterator

from agents.llm import LLM, Usage
from config import CACHE_DIR, THINK
from kb.schema import (DISEASE_PRIORITY_RULES, DISEASE_SECTION_RULES, DRUG_SECTION_RULES, RELATIONS, TYPE_LABELS_VI, EntityList,
                       ThresholdList, TripleList)
from kb.textutil import display_name, evidence_ratio, name_supported, plain, sentence_containing, strip_accents

PROMPT_VERSION = "v1"
MIN_EVIDENCE_RATIO = 0.9
# Tên chỉ là từ chung ("biến chứng nguy hiểm", "triệu chứng ban đầu"...): không trỏ tới thực thể cụ thể nào
GENERIC_NAME = re.compile(r"^(cac |nhung |mot so )?(bien chung|trieu chung|dau hieu|benh ly|benh|van de suc khoe|nguy co"
                          r"|tinh trang|hau qua)( (nguy hiem|nghiem trong|lau dai|man tinh|cap tinh|khac|thuong gap"
                          r"|ban dau|suc khoe|nang|tiem an))*$")
THRESHOLD_HINT = re.compile(r"(hba1c|duong huyet|glucose).{0,80}\d+([.,]\d+)?\s*(%|mg/dl|mmol)", re.S)

SYSTEM_BASE = ("Bạn là chuyên gia trích xuất tri thức y khoa từ văn bản tiếng Việt của nhà thuốc Long Châu. "
               "Chỉ trích những gì đoạn văn NÓI RÕ, không suy diễn, không thêm kiến thức bên ngoài. "
               "Trường evidence phải là cụm từ hoặc câu sao chép NGUYÊN VĂN từ đoạn văn. "
               "Tên thực thể ngắn gọn (2–8 từ), viết như danh từ, không kèm giải thích.")


def section_tasks(chunk: dict) -> list[dict]:
    heading = plain(chunk["heading"])
    tasks = []
    if chunk["page_type"] == "benh":
        rules = DISEASE_SECTION_RULES
    elif chunk["page_type"] in {"thuoc", "thanh-phan"}:
        rules = DRUG_SECTION_RULES
    else:
        rules = []
    if chunk["page_type"] == "benh":
        priority = next((rule for rule in DISEASE_PRIORITY_RULES if re.search(rule[0], heading)), None)
        if priority:
            rules = [priority]
    matched_any = False
    for pattern, relation, direction, types in rules:
        if re.search(pattern, heading):
            matched_any = True
            if relation is None:  # mục liều dùng: cố ý không đưa vào KG
                return []
            tasks.append({"kind": "guided", "relation": relation, "direction": direction, "types": types})
            if chunk["page_type"] != "benh":
                break  # trang thuốc: mỗi mục một quan hệ ("chống chỉ định" không được rơi vào "chỉ định")
    if not matched_any:
        tasks.append({"kind": "generic"})
    if THRESHOLD_HINT.search(plain(chunk["text"])):
        tasks.append({"kind": "threshold"})
    return tasks


def _cache_key(llm: LLM, chunk: dict, task: dict) -> str:
    raw = json.dumps([PROMPT_VERSION, llm.model, task, chunk["main_entities"], chunk["heading"], chunk["text"]],
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


def _guided_prompt(chunk: dict, task: dict) -> tuple[str, str]:
    main = ", ".join(chunk["main_entities"])
    relation_vi = RELATIONS[task["relation"]][2]
    types_vi = " hoặc ".join(f"{t} ({TYPE_LABELS_VI[t]})" for t in task["types"])
    if task["direction"] == "out":
        goal = f"Liệt kê các thực thể loại {types_vi} mà đoạn văn nêu rõ theo quan hệ: '{main}' — {relation_vi} — [thực thể]."
    else:
        goal = f"Liệt kê các thực thể loại {types_vi} mà đoạn văn nêu rõ theo quan hệ: [thực thể] — {relation_vi} — '{main}'."
    system = f"{SYSTEM_BASE}\n\n{goal}\nNếu đoạn văn không nêu thực thể nào phù hợp, trả về danh sách rỗng."
    user = f"Trang: {chunk['title']}\nMục: {chunk['heading']}\n\nĐoạn văn:\n{chunk['text']}"
    return system, user


def _generic_prompt(chunk: dict) -> tuple[str, str]:
    lines = [f"- {name}: {'|'.join(sorted(head))} → {'|'.join(sorted(tail))} ({desc})"
             for name, (head, tail, desc) in RELATIONS.items()]
    system = (f"{SYSTEM_BASE}\n\nTrích các quan hệ (bộ ba) được nêu rõ trong đoạn văn, CHỈ dùng các quan hệ sau:\n"
              + "\n".join(lines)
              + "\nLoại thực thể: " + ", ".join(f"{k} = {v}" for k, v in TYPE_LABELS_VI.items())
              + "\nNếu không có quan hệ nào rõ ràng, trả về danh sách rỗng.")
    user = f"Trang: {chunk['title']}\nMục: {chunk['heading']}\n\nĐoạn văn:\n{chunk['text']}"
    return system, user


def _threshold_prompt(chunk: dict) -> tuple[str, str]:
    system = (f"{SYSTEM_BASE}\n\nTrích các NGƯỠNG CHẨN ĐOÁN (con số của xét nghiệm HbA1c hoặc đường huyết dùng để xác định "
              "một tình trạng như đái tháo đường, tiền đái tháo đường, hạ đường huyết, mức bình thường). "
              "Giữ đúng đơn vị trong văn bản. Bỏ qua mục tiêu điều trị và liều thuốc. Nếu không có, trả về danh sách rỗng.")
    user = f"Trang: {chunk['title']}\nMục: {chunk['heading']}\n\nĐoạn văn:\n{chunk['text']}"
    return system, user


def run_task(llm: LLM, chunk: dict, task: dict, usage: Usage | None = None) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(llm, chunk, task)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if task["kind"] == "guided":
        result = llm.structured(EntityList, *_guided_prompt(chunk, task), think=THINK["extract"], usage=usage)
    elif task["kind"] == "generic":
        result = llm.structured(TripleList, *_generic_prompt(chunk), think=THINK["extract"], usage=usage)
    else:
        result = llm.structured(ThresholdList, *_threshold_prompt(chunk), think=THINK["extract"], usage=usage)
    data = result.model_dump()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# Chữ đứng ngay trước con số trong câu trích → phép so sánh (cụm dài xét trước cụm ngắn)
OP_WORDS = [
    (r"lon hon hoac bang|cao hon hoac bang|it nhat|>=|≥", ">="),
    (r"nho hon hoac bang|thap hon hoac bang|toi da|<=|≤", "<="),
    (r"lon hon|cao hon|tren|vuot qua|>", ">"),
    (r"nho hon|thap hon|duoi|<", "<"),
]


def _number_regex(value: float) -> str:
    whole, _, frac = f"{value:g}".partition(".")
    return rf"(?<![\d.,]){whole}" + (rf"[.,]{frac}" if frac else r"(?:[.,]0+)?") + r"(?!\d)"


def refine_threshold(th: dict) -> dict:
    """Sửa ngưỡng theo đúng chữ trong câu trích. Model hay ghi "dưới 140" thành <=, và với "≥100 và <126" chỉ giữ ≥100.
    Khoảng có cận trên lấy từ "<"/"dưới" được đánh dấu max_exclusive (giá trị bằng cận trên không thuộc khoảng)."""
    th = dict(th)
    text = strip_accents(th["evidence"].lower())
    number = _number_regex(th["value"])
    if th["op"] != "range":
        for pattern, op in OP_WORDS:
            if re.search(rf"(?:{pattern})\s*{number}", text):
                th["op"] = op
                break
        if re.search(rf"\btu\s*{number}\s*\S*\s*tro len", text):     # "từ 6,5% trở lên"
            th["op"] = ">="
    if th["op"] in {">=", ">"}:
        upper = re.search(rf"{number}.{{0,30}}?\b(?:va|den)?\s*(?:<|duoi|nho hon)\s*(\d+(?:[.,]\d+)?)", text)
        if upper and float(upper.group(1).replace(",", ".")) > th["value"]:
            th.update(op="range", value_max=float(upper.group(1).replace(",", ".")), max_exclusive=True)
    return th


def to_records(chunk: dict, task: dict, data: dict) -> Iterator[dict]:
    """Chuyển kết quả LLM thành bản ghi quan hệ, kèm lý do nếu bị loại."""
    base = {"chunk_id": chunk["id"], "source_url": chunk["url"], "heading": chunk["heading"], "task": task["kind"]}
    if task["kind"] == "guided":
        head_type = "Drug" if chunk["page_type"] in {"thuoc", "thanh-phan"} else "Disease"
        for entity in data.get("entities", []):
            for main in chunk["main_entities"]:
                if task["direction"] == "out":
                    head, h_type, tail, t_type = main, head_type, entity["name"], entity["type"]
                else:
                    head, h_type, tail, t_type = entity["name"], entity["type"], main, head_type
                yield {**base, "head": head, "head_type": h_type, "relation": task["relation"], "tail": tail,
                       "tail_type": t_type, "evidence": entity["evidence"], "extracted": entity["name"],
                       "extracted_type": entity["type"], "allowed_types": task["types"], "confidence": 0.9}
    elif task["kind"] == "generic":
        for triple in data.get("triples", []):
            yield {**base, **triple, "extracted": f"{triple['head']} | {triple['tail']}", "confidence": 0.7}
    else:
        for th in map(refine_threshold, data.get("thresholds", [])):
            props = {k: th[k] for k in ("op", "value", "value_max", "unit")}
            if th.get("max_exclusive"):
                props["max_exclusive"] = True
            yield {**base, "head": th["condition"], "head_type": "Disease", "relation": "DIAGNOSED_BY",
                   "tail": th["test"], "tail_type": "Test", "evidence": th["evidence"], "extracted": th["test"],
                   "props": props, "confidence": 0.9}


def validate(record: dict, chunk_text: str) -> str | None:
    """Trả về lý do loại bỏ, hoặc None nếu bản ghi hợp lệ."""
    relation = record["relation"]
    if relation not in RELATIONS:
        return "quan hệ ngoài ontology"
    head_types, tail_types, _ = RELATIONS[relation]
    if record["head_type"] not in head_types or record["tail_type"] not in tail_types:
        return f"sai miền: {record['head_type']} -{relation}-> {record['tail_type']}"
    if "allowed_types" in record and record["extracted_type"] not in record["allowed_types"]:
        return f"loại {record['extracted_type']} không thuộc mục này"
    if evidence_ratio(record["evidence"], chunk_text) < MIN_EVIDENCE_RATIO:
        return "câu trích không có trong văn bản"
    names = record["extracted"].split(" | ")
    if not all(name_supported(name, record["evidence"]) or name_supported(name, chunk_text) for name in names):
        return "tên thực thể không xuất hiện trong văn bản"
    if len(plain(record["head"])) < 2 or len(plain(record["tail"])) < 2 or plain(record["head"]) == plain(record["tail"]):
        return "tên rỗng hoặc tự trỏ"
    if GENERIC_NAME.match(plain(record["head"])) or GENERIC_NAME.match(plain(record["tail"])):
        return "tên quá chung chung"
    if re.match(r"\d", plain(record["head"])) or re.match(r"\d", plain(record["tail"])):
        return "tên là con số (ngưỡng phải nằm trong thuộc tính của DIAGNOSED_BY)"
    if relation == "DIAGNOSED_BY" and "props" in record and record["props"]["unit"] == "%" \
            and "a1c" not in plain(record["tail"]):
        return "đơn vị % chỉ dùng cho HbA1c"
    if relation == "DIAGNOSED_BY" and "props" in record:
        numbers = {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", record["evidence"])}
        wanted = [record["props"]["value"]] + ([record["props"]["value_max"]] if record["props"].get("value_max") else [])
        if not all(v in numbers for v in wanted):
            return "con số ngưỡng không có trong câu trích"
    return None


def extract_chunk(llm: LLM, chunk: dict, usage: Usage | None = None) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for task in section_tasks(chunk):
        data = run_task(llm, chunk, task, usage)
        for record in to_records(chunk, task, data):
            record["head"], record["tail"] = display_name(record["head"]), display_name(record["tail"])
            if evidence_ratio(record["evidence"], chunk["text"]) < MIN_EVIDENCE_RATIO:
                # model hay ghép câu; nếu tên thực thể có nguyên văn trong đoạn thì lấy chính câu gốc làm bằng chứng
                names = record["extracted"].split(" | ")
                sentence = sentence_containing(names[-1], chunk["text"]) or sentence_containing(names[0], chunk["text"])
                if sentence:
                    record["evidence"], record["evidence_repaired"] = sentence, True
            reason = validate(record, chunk["text"])
            (rejected if reason else accepted).append({**record, "reason": reason} if reason else record)
    return accepted, rejected
