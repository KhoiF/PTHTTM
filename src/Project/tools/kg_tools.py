"""Công cụ GraphRAG cho agent: tra Knowledge Graph trên Neo4j bằng các truy vấn Cypher chỉ-đọc có sẵn.

Mỗi công cụ trả về `facts`: danh sách {fact, quote, source_url, title}. Knowledge Agent gom các facts này thành
danh sách bằng chứng đánh số để Advisor trích dẫn [n].
"""

from __future__ import annotations

import re
from functools import lru_cache

import config
from kb.feature_bridge import FEATURES
from kb.schema import format_threshold
from kb.load_neo4j import get_driver, hidden_labels, index_prefix, run
from kb.textutil import name_key, plain, source_title

HIDDEN_RELATIONS = {"MENTIONS", "FROM"}
RELATION_VI = {
    "HAS_SYMPTOM": "có triệu chứng", "HAS_RISK_FACTOR": "có yếu tố nguy cơ", "HAS_COMPLICATION": "có thể gây biến chứng",
    "DIAGNOSED_BY": "được chẩn đoán bằng", "TREATED_BY": "được điều trị bằng", "PREVENTED_BY": "được phòng ngừa bằng",
    "BELONGS_TO": "thuộc nhóm", "HAS_SIDE_EFFECT": "có tác dụng phụ", "CONTRAINDICATED_IN": "chống chỉ định ở",
    "MAPS_TO": "tương ứng với",
}
MGDL_PER_MMOLL = 18.0
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


@lru_cache
def _driver():
    return get_driver()


@lru_cache
def _llm():
    from agents.llm import LLM

    return LLM()


def _q(query: str, **params):
    return run(_driver(), query.replace("$LABEL", config.KG_LABEL).replace("$PREFIX", index_prefix()), **params)


def _title_map(urls) -> dict:
    rows = _q("MATCH (s:$LABEL:Source) WHERE s.url IN $urls RETURN s.url AS url, s.title AS title", urls=list(urls))
    return {r["url"]: source_title(r["title"]) for r in rows}


def _facts_from_rel(rows) -> list[dict]:
    titles = _title_map({u for r in rows for u in (r["source_urls"] or [])})
    facts = []
    for r in rows:
        urls = r["source_urls"] or []
        fact = f"{r['head']} {RELATION_VI.get(r['relation'], r['relation'])} {r['tail']}"
        if r.get("op"):
            fact += f" ({format_threshold(r['op'], r['value'], r.get('value_max'), bool(r.get('max_exclusive')))} {r['unit']})"
        facts.append({"fact": fact, "quote": (r["evidence"] or [""])[0], "source_url": urls[0] if urls else None,
                      "title": titles.get(urls[0]) if urls else None,
                      "triple": [r["head"], r["relation"], r["tail"]]})
    return facts


def link_entities(text: str, k: int = 5) -> dict:
    """Tìm các thực thể trong Knowledge Graph khớp với một cụm từ (tên bệnh, triệu chứng, thuốc, xét nghiệm...).

    Args:
        text: Cụm từ tiếng Việt cần tìm, ví dụ "biến chứng bàn chân" hoặc "metformin"
        k: Số kết quả tối đa
    """
    k = max(1, min(int(k or 5), 10))
    scores: dict[str, float] = {}
    info: dict[str, dict] = {}
    tokens = [LUCENE_SPECIAL.sub(r"\\\1", t) for t in name_key(text).split() if len(t) > 1]
    if tokens:
        query = f'name_key:"{" ".join(tokens)}"^3 OR name_key:({" OR ".join(tokens)})'
        rows = _q("""CALL db.index.fulltext.queryNodes('$PREFIX_entity_names', $q, {limit: 50}) YIELD node, score
                     WHERE node:$LABEL RETURN node.id AS id, node.name AS name, labels(node) AS labels, score""", q=query)
        top = max((r["score"] for r in rows), default=1.0) or 1.0
        for r in rows:
            scores[r["id"]] = 0.4 * r["score"] / top
            info[r["id"]] = r
    vector = _llm().embed([text])[0].tolist()
    rows = _q("""CALL db.index.vector.queryNodes('$PREFIX_entity_embedding', 50, $v) YIELD node, score
                 WHERE node:$LABEL RETURN node.id AS id, node.name AS name, labels(node) AS labels, score""", v=vector)
    for r in rows:
        scores[r["id"]] = scores.get(r["id"], 0.0) + 0.6 * r["score"]
        info.setdefault(r["id"], r)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return {"entities": [{"id": i, "name": info[i]["name"],
                          "type": next(l for l in info[i]["labels"] if l not in hidden_labels()),
                          "score": round(s, 3)} for i, s in ranked]}


def _resolve(entity: str, types: tuple[str, ...] = ()) -> dict | None:
    if ":" in entity:
        rows = _q("MATCH (n:$LABEL {id: $id}) RETURN n.id AS id, n.name AS name", id=entity)
        if rows:
            return dict(rows[0])
    for candidate in link_entities(entity, k=8)["entities"]:
        if not types or candidate["type"] in types:
            return candidate
    return None


def _resolve_feature(feature: str) -> str | None:
    if feature in FEATURES:
        return feature
    key = name_key(feature)
    for column, spec in FEATURES.items():
        if key in {name_key(column), name_key(spec["label"])} or name_key(spec["label"]) in key:
            return column
    return None


def _convert(value: float, unit_from: str | None, unit_to: str) -> float:
    if unit_from == unit_to or unit_from is None:
        return value
    if unit_from == "mg/dL" and unit_to == "mmol/L":
        return value / MGDL_PER_MMOLL
    if unit_from == "mmol/L" and unit_to == "mg/dL":
        return value * MGDL_PER_MMOLL
    return value


def _met(value: float, op: str, low: float, high: float | None, max_exclusive: bool = False) -> bool:
    in_range = high is not None and low <= value and (value < high if max_exclusive else value <= high)
    return {">=": value >= low, ">": value > low, "<=": value <= low, "<": value < low, "range": in_range}[op]


def explain_feature(feature: str, value: float | None = None) -> dict:
    """Giải thích một chỉ số của bệnh nhân bằng Knowledge Graph: chỉ số đó ứng với xét nghiệm/yếu tố nguy cơ nào,
    các ngưỡng chẩn đoán liên quan và giá trị của bệnh nhân có vượt ngưỡng không.

    Args:
        feature: Tên cột dữ liệu, ví dụ "HbA1c_level", "blood_glucose_level", "bmi", "hypertension", "smoking_history", "age"
        value: Giá trị của bệnh nhân (HbA1c theo %, đường huyết theo mg/dL), bỏ trống nếu không có
    """
    column = _resolve_feature(feature)
    if column is None:
        return {"error": f"không có đặc trưng {feature!r}", "valid": list(FEATURES)}
    spec = FEATURES[column]
    try:
        value = None if value in (None, "", "None") else float(value)
    except (TypeError, ValueError):
        value = None
    targets = _q("""MATCH (f:$LABEL:Feature {id: $fid})-[:MAPS_TO]->(t)
                    RETURN t.id AS id, t.name AS name, [l IN labels(t) WHERE NOT l IN $hidden][0] AS type""",
                 fid=f"Feature:{column}", hidden=hidden_labels())
    result = {"feature": column, "label": spec["label"], "value": value, "unit": spec.get("unit"), "note": spec.get("note"),
              "mapped_to": [dict(t) for t in targets], "thresholds": [], "facts": []}
    test_ids = [t["id"] for t in targets if t["type"] == "Test"]
    if test_ids:
        rows = _q("""MATCH (d:$LABEL)-[r:DIAGNOSED_BY]->(t:$LABEL) WHERE t.id IN $ids AND r.op IS NOT NULL
                     RETURN d.name AS head, 'DIAGNOSED_BY' AS relation, t.name AS tail, r.op AS op, r.value AS value,
                            r.value_max AS value_max, r.max_exclusive AS max_exclusive, r.unit AS unit,
                            r.evidence AS evidence, r.source_urls AS source_urls""",
                  ids=test_ids)
        skip, keep = spec.get("exclude", []), spec.get("keep", [])
        rows = [r for r in rows if any(k in plain(r["tail"]) for k in keep)
                or not any(x in plain(f"{r['tail']} {(r['evidence'] or [''])[0]}") for x in skip)]
        result["facts"] += _facts_from_rel(rows)
        for r in rows:
            item = {"condition": r["head"], "test": r["tail"], "op": r["op"], "value": r["value"], "value_max": r["value_max"],
                    "max_exclusive": bool(r["max_exclusive"]), "unit": r["unit"], "source_url": (r["source_urls"] or [None])[0]}
            if value is not None:
                item["patient_value_converted"] = round(_convert(value, spec.get("unit"), r["unit"]), 2)
                item["met"] = _met(item["patient_value_converted"], r["op"], r["value"], r["value_max"], item["max_exclusive"])
            result["thresholds"].append(item)
    other_ids = [t["id"] for t in targets if t["type"] != "Test"]
    if other_ids:
        rows = _q("""MATCH (d:$LABEL:Disease)-[r:HAS_RISK_FACTOR]->(x:$LABEL) WHERE x.id IN $ids
                     RETURN d.name AS head, 'HAS_RISK_FACTOR' AS relation, x.name AS tail, r.evidence AS evidence,
                            r.source_urls AS source_urls LIMIT 12""", ids=other_ids)
        result["facts"] += _facts_from_rel(rows)
    return result


def neighbors(entity: str, relations: list[str] | None = None, hops: int = 1, limit: int = 20) -> dict:
    """Liệt kê các quan hệ của một thực thể trong Knowledge Graph (triệu chứng, biến chứng, yếu tố nguy cơ, điều trị...).

    Args:
        entity: Tên thực thể, ví dụ "Đái tháo đường tuýp 2", "Tăng huyết áp"
        relations: Lọc theo loại quan hệ, ví dụ ["HAS_COMPLICATION"]; bỏ trống để lấy tất cả
        hops: 1 hoặc 2 bước
        limit: Số quan hệ tối đa
    """
    node = _resolve(entity)
    if node is None:
        return {"error": f"không tìm thấy thực thể {entity!r}"}
    hops = 2 if int(hops or 1) >= 2 else 1
    limit = max(1, min(int(limit or 20), 40))
    wanted = [r for r in (relations or []) if r in RELATION_VI] or None
    rows = _q(f"""MATCH p = (n:$LABEL {{id: $id}})-[*1..{hops}]-(m:$LABEL)
                  WHERE none(r IN relationships(p) WHERE type(r) IN $hidden)
                    AND ($wanted IS NULL OR type(last(relationships(p))) IN $wanted)
                  WITH last(relationships(p)) AS r
                  RETURN DISTINCT startNode(r).name AS head, type(r) AS relation, endNode(r).name AS tail,
                         r.op AS op, r.value AS value, r.value_max AS value_max, r.max_exclusive AS max_exclusive,
                         r.unit AS unit, r.evidence AS evidence, r.source_urls AS source_urls, r.confidence AS confidence
                  ORDER BY confidence DESC LIMIT $limit""",
              id=node["id"], hidden=list(HIDDEN_RELATIONS | {"MAPS_TO"}), wanted=wanted, limit=limit)
    return {"entity": node["name"], "facts": _facts_from_rel(rows)}


def drug_info(drug: str) -> dict:
    """Tra thông tin một thuốc/hoạt chất: nhóm thuốc, bệnh được dùng để điều trị, tác dụng phụ, chống chỉ định.
    Không bao gồm liều dùng.

    Args:
        drug: Tên thuốc hoặc hoạt chất, ví dụ "Metformin", "Januvia", "nhóm ức chế SGLT2"
    """
    node = _resolve(drug, ("Drug", "DrugClass"))
    if node is None:
        return {"error": f"không tìm thấy thuốc {drug!r}"}
    rows = _q("""MATCH (n:$LABEL {id: $id})
                 OPTIONAL MATCH (n)-[:BELONGS_TO]->(c:$LABEL)
                 WITH n, collect(c) AS classes
                 UNWIND [n] + classes AS x
                 MATCH (x)-[r]-(m:$LABEL)
                 WHERE type(r) IN ['BELONGS_TO', 'HAS_SIDE_EFFECT', 'CONTRAINDICATED_IN', 'TREATED_BY']
                 RETURN DISTINCT startNode(r).name AS head, type(r) AS relation, endNode(r).name AS tail,
                        r.evidence AS evidence, r.source_urls AS source_urls, r.confidence AS confidence
                 ORDER BY confidence DESC LIMIT 30""", id=node["id"])
    return {"drug": node["name"], "facts": _facts_from_rel(rows),
            "note": "Không cung cấp liều dùng; việc dùng thuốc phải theo chỉ định của bác sĩ."}


def search_chunks(query: str, k: int = 4) -> dict:
    """Tìm các đoạn văn gốc từ Long Châu liên quan nhất tới câu hỏi (tìm kiếm ngữ nghĩa).

    Args:
        query: Câu truy vấn tiếng Việt
        k: Số đoạn tối đa (1–6)
    """
    k = max(1, min(int(k or 4), 6))
    vector = _llm().embed([query])[0].tolist()
    rows = _q("""CALL db.index.vector.queryNodes('$PREFIX_chunk_embedding', $n, $v) YIELD node, score
                 WHERE node:$LABEL
                 OPTIONAL MATCH (node)-[:MENTIONS]->(e:$LABEL)
                 RETURN node.id AS id, node.heading AS heading, node.title AS title, node.url AS url, node.text AS text,
                        score, collect(DISTINCT e.name)[..8] AS mentions
                 ORDER BY score DESC LIMIT $k""", n=k * 4, v=vector, k=k)
    facts = [{"fact": f"{source_title(r['title'])} — mục \"{r['heading']}\"", "quote": r["text"][:700],
              "source_url": r["url"], "title": source_title(r["title"]), "score": round(r["score"], 3)} for r in rows]
    return {"facts": facts, "mentions": sorted({m for r in rows for m in r["mentions"]})}


KG_TOOLS = [link_entities, explain_feature, neighbors, drug_info, search_chunks]
