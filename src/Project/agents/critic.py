"""Critic Agent: kiểm số liệu, trích dẫn, liều thuốc bằng code; kiểm an toàn y khoa bằng LLM."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agents import prompts
from agents.advisor import evidence_block, lab_block, risk_block
from agents.base import Step, default_llm
from config import MAX_CRITIC_ROUNDS, THINK
from kb.textutil import evidence_ratio, plain

NUMBER_WITH_UNIT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|mg/dl|mmol/l)", re.I)
# "mg" đứng trước "/dL" là đơn vị đường huyết, không phải liều thuốc
DOSAGE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(mg|mcg|µg|ui|iu|đơn vị)\b(?!\s*/)|\b\d+\s*viên\b|\b(uống|tiêm)\s+\d|\bliều\s+\d",
                    re.I)
TOLERANCE = 0.06
PLACEHOLDER_CITE = re.compile(r"\[(n|\?|…|\.\.\.|x)\]", re.I)
DIAGNOSIS = re.compile(r"\bban\b.{0,20}\b(bi|mac)\s+(benh\s+)?(tieu duong|dai thao duong)")
# Câu nói "lúc đói" phải có từ rào đón khi người dùng không nói đường huyết đo lúc đói (dataset không ghi thời điểm đo)
FASTING_HEDGES = re.compile(r"\b(neu|gia dinh|gia su|khong ro|chua ro|trong truong hop|tuy thuoc|truong hop)\b")
# Cách gọi khác nhau của cùng một xét nghiệm: câu trả lời dùng tên nào trong nhóm cũng được
TEST_SYNONYMS = [("duong huyet", "glucose", "duong mau"), ("hba1c", "a1c")]
HEDGES = re.compile(r"nguy co|co the|kha nang|khong|chua|neu|lieu|nghi ngo|kiem tra|xac nhan|xet nghiem")


class CriticReview(BaseModel):
    """Đánh giá an toàn y khoa của bản nháp."""

    recommends_doctor: bool = Field(description="Bản nháp khuyên gặp bác sĩ / cơ sở y tế")
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Câu trích nguyên văn từ bản nháp chứa khẳng định y khoa không có trong bằng chứng")


def diagnosis_sentences(draft: str) -> list[str]:
    """Câu kết luận người dùng đã mắc bệnh mà không có từ rào đón (nguy cơ, có thể, cần xác nhận...)."""
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", draft):
        text = plain(sentence)
        if DIAGNOSIS.search(text) and not HEDGES.search(text):
            found.append(sentence.strip())
    return found


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def allowed_numbers(state: dict) -> set[float]:
    allowed: set[float] = set()
    risk = state.get("risk") or {}
    for p in [risk.get("probability"), risk.get("threshold")] + [w.get(k) for w in risk.get("what_if", [])
                                                                  for k in ("before", "after")]:
        if p is not None:
            allowed |= {round(p * 100, 1), round(p * 100), round(p * 100, 2)}
    for w in risk.get("what_if", []):
        allowed |= {abs(round(w["delta"] * 100, 1)), abs(round(w["delta"] * 100))}
        allowed |= {float(v) for v in w["changes"].values() if isinstance(v, (int, float))}
    for value in (state.get("patient") or {}).values():
        if isinstance(value, (int, float)):
            allowed.add(float(value))
    for check in state.get("lab_checks") or []:
        allowed |= {v for v in (check.get("value"), check.get("value_max"), check.get("patient_value_converted"))
                    if v is not None}
    for e in state.get("evidence") or []:
        allowed |= _numbers(e["fact"]) | _numbers(e["quote"])
    allowed |= _numbers(state.get("question") or "") | _numbers(state.get("follow_up") or "")
    return allowed


def rule_issues(state: dict, draft: str) -> list[str]:
    issues = []
    evidence = state.get("evidence") or []
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", draft)]
    invalid = sorted({n for n in cited if not 1 <= n <= len(evidence)})
    if invalid:
        issues.append(f"Trích dẫn {invalid} không tồn tại; chỉ được dùng [1]–[{len(evidence)}].")
    if evidence and not cited:
        issues.append("Chưa có trích dẫn [n] nào cho thông tin y khoa.")
    if PLACEHOLDER_CITE.search(draft):
        issues.append("Có trích dẫn chưa điền số (ví dụ \"[n]\"); thay bằng số thật hoặc bỏ đi.")
    allowed = allowed_numbers(state)
    for raw, unit in NUMBER_WITH_UNIT.findall(draft):
        value = float(raw.replace(",", "."))
        if not any(abs(value - a) <= TOLERANCE for a in allowed):
            issues.append(f"Con số {raw}{unit} không khớp kết quả mô hình hay nguồn; chỉ dùng số đã cung cấp.")
    if DOSAGE.search(draft):
        issues.append("Không được nêu liều/hàm lượng thuốc (mg, số viên, số lần dùng).")
    text = plain(draft)
    if any(c["feature"] == "blood_glucose_level" for c in state.get("lab_checks") or []) \
            and "doi" not in plain(state.get("question") or "").split():
        fasting = [s for s in re.split(r"(?<=[.!?\n])\s*", draft) if "luc doi" in plain(s)]
        if any(not FASTING_HEDGES.search(plain(s)) for s in fasting):
            issues.append("Người dùng không nói đường huyết đo lúc đói; câu nhắc 'lúc đói' phải nói rõ đó là giả định "
                          "(ví dụ: 'nếu đây là kết quả đo lúc đói…').")
    for check in state.get("lab_checks") or []:
        test = plain(check["test"])
        names = next((group for group in TEST_SYNONYMS if any(name in test for name in group)), (test.split()[0],))
        if check["met"] and check["condition"] and not any(name in text for name in names):
            issues.append(f"Phải nêu rằng {check['test']} của người dùng đạt ngưỡng '{check['condition']}' theo nguồn.")
    return list(dict.fromkeys(issues))


def supported_by_rules(claim: str, state: dict, allowed: set[float]) -> bool:
    """Câu có trích dẫn hợp lệ, hoặc mọi con số trong câu đều lấy từ mô hình/nguồn, thì không coi là thiếu nguồn."""
    n_evidence = len(state.get("evidence") or [])
    if any(1 <= int(n) <= n_evidence for n in re.findall(r"\[(\d+)\]", claim)):
        return True
    numbers = _numbers(re.sub(r"\[\d+\]", "", claim))
    return bool(numbers) and all(any(abs(x - a) <= TOLERANCE for a in allowed) for x in numbers)


def critic_node(state: dict, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("Critic")
    draft = state.get("draft") or ""
    issues = rule_issues(state, draft)
    review = llm.structured(
        CriticReview, prompts.CRITIC,
        f"KẾT QUẢ MÔ HÌNH:\n{risk_block(state)}\n\n{lab_block(state)}\n\nBẰNG CHỨNG:\n"
        f"{evidence_block(state.get('evidence') or [])}\n\nBẢN NHÁP:\n{draft}", think=THINK["critic"], usage=step.usage)
    for sentence in diagnosis_sentences(draft):
        issues.append(f"Không được khẳng định người dùng mắc bệnh: \"{sentence}\". Chỉ nói về nguy cơ và khuyên xét nghiệm.")
    if not review.recommends_doctor:
        issues.append("Cần khuyên người dùng gặp bác sĩ để được tư vấn/chẩn đoán.")
    # chỉ giữ khẳng định mà Critic trích đúng từ bản nháp (model nhỏ đôi khi tự bịa câu để bắt lỗi)
    allowed = allowed_numbers(state)
    claims = [c for c in review.unsupported_claims
              if evidence_ratio(c, draft) >= 0.8 and not supported_by_rules(c, state, allowed)]
    issues += [f"Khẳng định chưa có nguồn, hãy bỏ hoặc trích dẫn: \"{claim}\"" for claim in claims[:3]]

    counters = dict(state.get("counters") or {})
    counters["critic_rounds"] = counters.get("critic_rounds", 0) + 1
    passed = not issues
    if passed:
        decision = "đạt → Finalize"
    elif counters["critic_rounds"] < MAX_CRITIC_ROUNDS:
        decision = f"bác bỏ ({len(issues)} lỗi) → trả về Advisor"
    else:
        decision = f"còn {len(issues)} lỗi nhưng hết lượt sửa → Finalize"
    return {"critique": {"passed": passed, "issues": issues, "review": review.model_dump()}, "counters": counters,
            "trace": [step.entry(decision, issues=issues)]}
