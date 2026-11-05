"""Knowledge Agent: tìm bằng chứng trong Knowledge Graph (GraphRAG), tự mở rộng truy vấn khi chưa đủ."""

from __future__ import annotations

from agents import prompts
from agents.base import Step, default_llm, percent
from config import MAX_KNOWLEDGE_ROUNDS, THINK
from tools import kg_tools

MIN_EVIDENCE = 3
MAX_EVIDENCE = 10
PREFETCH_FACTS_PER_FEATURE = 2
LAB_FEATURES = ["HbA1c_level", "blood_glucose_level"]


def add_facts(evidence: list[dict], facts: list[dict], origin: str) -> int:
    seen = {(e["source_url"], e["quote"][:120]) for e in evidence}
    added = 0
    for fact in facts:
        key = (fact.get("source_url"), (fact.get("quote") or "")[:120])
        if not key[0] or not key[1] or key in seen or len(evidence) >= MAX_EVIDENCE:
            continue
        seen.add(key)
        evidence.append({"n": len(evidence) + 1, "fact": fact["fact"], "quote": fact["quote"],
                         "source_url": fact["source_url"], "title": fact.get("title"), "origin": origin,
                         "triple": fact.get("triple")})
        added += 1
    return added


def patient_kg_tools(patient: dict) -> list:
    """Bộ công cụ KG; explain_feature tự điền giá trị của bệnh nhân nếu model để trống."""

    def explain_feature(feature: str, value: float | None = None) -> dict:
        column = kg_tools._resolve_feature(feature)
        if value in (None, "", "None", 0) and column and patient.get(column) is not None:
            value = patient[column]
        return kg_tools.explain_feature(feature, value)

    explain_feature.__doc__ = kg_tools.explain_feature.__doc__
    return [kg_tools.link_entities, kg_tools.neighbors, kg_tools.drug_info, explain_feature, kg_tools.search_chunks]


def _prefetch(state: dict, evidence: list[dict]) -> tuple[list[dict], list[str]]:
    """Đối chiếu các chỉ số quan trọng của bệnh nhân với KG (chạy bằng code, không qua LLM)."""
    patient, risk = state.get("patient") or {}, state.get("risk") or {}
    features = [f for f in LAB_FEATURES if patient.get(f) is not None]
    features += [t["feature"] for t in risk.get("top_features", []) if t["feature"] != "gender"]
    lab_checks = []
    used = []
    for feature in list(dict.fromkeys(features))[:4]:
        result = kg_tools.explain_feature(feature, patient.get(feature))
        add_facts(evidence, result.get("facts", [])[:PREFETCH_FACTS_PER_FEATURE], f"explain_feature:{feature}")
        used.append(feature)
        for t in result.get("thresholds", []):
            if t.get("met") is not None:
                lab_checks.append({"feature": feature, "patient_value": patient.get(feature),
                                   "patient_unit": result.get("unit"), **t})
    return lab_checks, used


def _risk_line(state: dict) -> str:
    risk = state.get("risk")
    if not risk:
        return "Chưa có đánh giá nguy cơ."
    tops = ", ".join(f"{t['label']}={t['value']} ({t['direction']})" for t in risk.get("top_features", [])[:4])
    return f"Xác suất mô hình: {percent(risk['probability'])}; yếu tố chính: {tops}."


def knowledge_node(state: dict, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("Knowledge")
    options = state.get("options") or {}
    use_kg = options.get("use_kg", True)
    counters = dict(state.get("counters") or {})
    round_no = counters.get("knowledge_rounds", 0) + 1
    evidence = [dict(e) for e in state.get("evidence") or []]
    lab_checks = list(state.get("lab_checks") or [])

    prefetched = []
    if round_no == 1 and use_kg and state.get("risk"):
        checks, prefetched = _prefetch(state, evidence)
        lab_checks += checks

    tools = patient_kg_tools(state.get("patient") or {}) if use_kg else [kg_tools.search_chunks]
    question = (state.get("question") or "").strip() or "(không có câu hỏi — hãy tìm bằng chứng giải thích kết quả đánh giá nguy cơ)"
    known = "\n".join(f"- {e['fact']}" for e in evidence) or "(chưa có)"
    notes = "\n".join(state.get("knowledge_notes") or [])
    user = (f"Câu hỏi: {question}\nÝ định: {state.get('intent')}\n{_risk_line(state)}\n"
            f"Bằng chứng đã có:\n{known}\n" + (f"Ghi chú vòng trước:\n{notes}\n" if notes else ""))
    run = llm.run_tools(prompts.KNOWLEDGE if use_kg else prompts.KNOWLEDGE_VECTOR, user, tools,
                        think=THINK["knowledge"], max_steps=5, usage=step.usage, need_text=False)
    added = 0
    for call in run.tool_calls:
        if isinstance(call["result"], dict):
            added += add_facts(evidence, call["result"].get("facts", []), call["tool"])

    from_llm = sum(1 for e in evidence if not e["origin"].startswith("explain_feature:"))
    needs_question_facts = state.get("intent") in {"ask", "both"}
    enough = len(evidence) >= MIN_EVIDENCE and (from_llm >= 1 or not needs_question_facts)
    counters["knowledge_rounds"] = round_no
    queries = [f"{c['tool']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})" for c in run.tool_calls]
    if not enough and round_no < MAX_KNOWLEDGE_ROUNDS:
        note = (f"Vòng {round_no}: mới có {len(evidence)} bằng chứng ({from_llm} liên quan câu hỏi). Đã thử: "
                f"{'; '.join(queries) or 'chưa gọi công cụ'}. Hãy dùng từ khóa khác/đồng nghĩa hoặc search_chunks.")
        decision = f"vòng {round_no}: chưa đủ bằng chứng ({len(evidence)}) → tự lặp lại với truy vấn khác"
        return {"evidence": evidence, "lab_checks": lab_checks, "counters": counters, "knowledge_done": False,
                "knowledge_notes": (state.get("knowledge_notes") or []) + [note],
                "trace": [step.entry(decision, prefetch=prefetched, queries=queries, added=added)]}
    decision = f"vòng {round_no}: {len(evidence)} bằng chứng → Advisor"
    return {"evidence": evidence, "lab_checks": lab_checks, "counters": counters, "knowledge_done": True,
            "trace": [step.entry(decision, prefetch=prefetched, queries=queries, added=added,
                                 met_thresholds=[f"{c['test']} {c['op']} {c['value']}" for c in lab_checks if c["met"]])]}
