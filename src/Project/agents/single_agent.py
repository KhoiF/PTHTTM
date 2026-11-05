"""Ablation D: một agent duy nhất cầm toàn bộ công cụ (ML + KG), không phân vai, không Critic."""

from __future__ import annotations

from agents import prompts
from agents.base import Step, default_llm
from agents.graph import finalize_node
from config import THINK
from ml import predictor
from tools import kg_tools


def predict_risk_full_args(gender: str, age: float, bmi: float, hypertension: int, heart_disease: int,
                           smoking_history: str = "No Info", HbA1c_level: float | None = None,
                           blood_glucose_level: float | None = None) -> dict:
    """Dự đoán xác suất mắc tiểu đường bằng mô hình học máy từ thông tin bệnh nhân.

    Args:
        gender: "Male" hoặc "Female"
        age: Tuổi (năm)
        bmi: Chỉ số khối cơ thể
        hypertension: 1 nếu tăng huyết áp, 0 nếu không
        heart_disease: 1 nếu có bệnh tim, 0 nếu không
        smoking_history: never, former, current, ever, not current hoặc No Info
        HbA1c_level: HbA1c (%) nếu có
        blood_glucose_level: Đường huyết (mg/dL) nếu có
    """
    result = predictor.predict_risk(dict(gender=gender, age=age, bmi=bmi, hypertension=hypertension,
                                         heart_disease=heart_disease, smoking_history=smoking_history,
                                         HbA1c_level=HbA1c_level, blood_glucose_level=blood_glucose_level))
    result.pop("patient", None)
    return result


def run_single(question: str = "", form: dict | None = None, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("SingleAgent")
    form = {k: v for k, v in (form or {}).items() if v not in (None, "")}
    user = question or "Hãy đánh giá nguy cơ tiểu đường của tôi."
    if form:
        user += "\nHồ sơ: " + ", ".join(f"{k}={v}" for k, v in form.items())
    tools = [predict_risk_full_args, kg_tools.link_entities, kg_tools.neighbors, kg_tools.drug_info,
             kg_tools.explain_feature, kg_tools.search_chunks]
    run = llm.run_tools(prompts.SINGLE, user, tools, think=THINK["single"], max_steps=8, usage=step.usage)
    evidence, risk = [], None
    for call in run.tool_calls:
        result = call["result"] if isinstance(call["result"], dict) else {}
        if call["tool"] == "predict_risk_full_args" and result.get("ok"):
            risk = {**result, "top_features": [], "what_if": []}
        for fact in result.get("facts", []):
            evidence.append({"n": len(evidence) + 1, **{k: fact.get(k) for k in ("fact", "quote", "source_url", "title")},
                             "origin": call["tool"]})
    state = {"question": question, "form": form, "risk": risk, "evidence": evidence, "draft": run.text,
             "patient": form, "lab_checks": [],
             "trace": [step.entry(f"{len(run.tool_calls)} lần gọi công cụ", tools=[c["tool"] for c in run.tool_calls])]}
    state.update(finalize_node(state) | {"trace": state["trace"] + finalize_node(state)["trace"]})
    return state
