"""Risk Agent: gọi mô hình ML (dự đoán, SHAP, what-if) và quyết định có cần thêm dữ liệu không."""

from __future__ import annotations

from agents import prompts
from agents.base import Step, default_llm, percent
from config import THINK
from ml.predictor import FEATURE_LABELS
from tools.ml_tools import make_ml_tools


def _patient_text(patient: dict) -> str:
    return "\n".join(f"- {FEATURE_LABELS[k]}: {v}" for k, v in patient.items() if k in FEATURE_LABELS and v is not None)


def risk_node(state: dict, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("Risk")
    patient = state["patient"]
    previous = state.get("risk") or {}
    if previous.get("patient_snapshot") == patient:
        return {"trace": [step.entry("hồ sơ không đổi → giữ kết quả cũ, chuyển Knowledge")]}

    tools = make_ml_tools(patient)
    run = llm.run_tools(prompts.RISK, "Hồ sơ bệnh nhân:\n" + _patient_text(patient), tools, need_text=False,
                        think=THINK["risk"], max_steps=5, usage=step.usage)
    by_tool: dict[str, list] = {}
    for call in run.tool_calls:
        by_tool.setdefault(call["tool"], []).append(call["result"])
    fallbacks = []
    prediction = next((r for r in reversed(by_tool.get("predict_risk", [])) if r.get("ok")), None)
    if prediction is None:  # đảm bảo luôn có dự đoán dù model quên gọi công cụ
        prediction, _ = tools[0](), fallbacks.append("predict_risk")
    explanation = next(iter(by_tool.get("explain_risk", [])), None)
    if explanation is None:
        explanation, _ = tools[1](), fallbacks.append("explain_risk")

    risk = {**prediction, "top_features": explanation["top_features"],
            "what_if": [r for r in by_tool.get("what_if", []) if r.get("ok")],
            "patient_snapshot": patient, "fallbacks": fallbacks}
    counters = dict(state.get("counters") or {})
    request = None
    if prediction["grey_zone"] and prediction["missing_labs"] and not counters.get("labs_requested"):
        request, counters["labs_requested"] = "labs", True
        decision = f"{percent(prediction['probability'])} trong vùng xám, thiếu xét nghiệm → nhờ Intake tìm thêm"
    else:
        decision = f"{percent(prediction['probability'])} (mô hình {prediction['model_used']}) → Knowledge"
    follow_up = state.get("follow_up")
    if prediction["missing_labs"] and not follow_up:
        follow_up = ("Kết quả dựa trên mô hình sàng lọc chưa có xét nghiệm; nên làm xét nghiệm HbA1c và đường huyết "
                     "để đánh giá chính xác hơn.")
    return {"risk": risk, "risk_request": request, "counters": counters, "follow_up": follow_up,
            "trace": [step.entry(decision, tools=[c["tool"] for c in run.tool_calls], fallbacks=fallbacks)]}
