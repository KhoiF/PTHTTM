"""Advisor Agent: viết câu trả lời tiếng Việt từ kết quả mô hình + bằng chứng đánh số."""

from __future__ import annotations

import re
import time

from agents import prompts
from agents.base import Step, default_llm, percent
from config import THINK
from kb.schema import format_threshold


def risk_block(state: dict) -> str:
    risk = state.get("risk")
    if not risk:
        return ("Không có đánh giá nguy cơ cá nhân. ĐÂY LÀ CÂU HỎI KIẾN THỨC: trả lời thẳng vào câu hỏi, "
                "KHÔNG viết mục 'Đánh giá nguy cơ'.")
    full = risk["model_used"] == "full"
    lines = [
        f"- Xác suất mắc tiểu đường theo mô hình (dùng ĐÚNG con số này): {percent(risk['probability'])}",
        "- Mô hình: " + ("đầy đủ (có HbA1c và đường huyết)" if full
                         else "sàng lọc (CHƯA có xét nghiệm nên độ chính xác thấp hơn)"),
        f"- Phân loại: {'nguy cơ CAO' if risk['label'] else 'nguy cơ THẤP'} so với ngưỡng {percent(risk['threshold'])}"
        + (" — gần ngưỡng (vùng xám)" if risk["grey_zone"] else ""),
    ]
    if risk.get("top_features"):
        lines.append("- Yếu tố ảnh hưởng nhiều nhất: " + "; ".join(
            f"{t['label']} = {t['value']} ({t['direction']})" for t in risk["top_features"][:4]))
    for w in risk.get("what_if", []):
        (feature, value), = w["changes"].items()
        lines.append(f"- What-if: nếu {feature} = {value} thì xác suất là {percent(w['after'])} "
                     f"(thay đổi {w['delta'] * 100:+.1f} điểm %)".replace(".", ","))
    return "\n".join(lines)


def lab_block(state: dict) -> str:
    checks = state.get("lab_checks") or []
    if not checks:
        return ""
    lines = []
    for c in checks:
        rng = format_threshold(c["op"], c["value"], c.get("value_max"), bool(c.get("max_exclusive")))
        status = "ĐẠT" if c["met"] else "không đạt"
        lines.append(f"- {c['test']} của bệnh nhân = {c['patient_value']} {c.get('patient_unit') or ''} "
                     f"(quy đổi {c['patient_value_converted']} {c['unit']}): {status} ngưỡng '{c['condition']}' ({rng} {c['unit']})")
    if any(c["feature"] == "blood_glucose_level" for c in checks):
        lines.append("- LƯU Ý: không rõ đường huyết của người dùng đo lúc đói hay bất kỳ. Phải nói rõ điều này; "
                     "ngưỡng 'lúc đói' chỉ áp dụng nếu người dùng đo khi nhịn ăn.")
    return "Đối chiếu chỉ số với ngưỡng trong nguồn:\n" + "\n".join(lines)


def evidence_block(evidence: list[dict]) -> str:
    if not evidence:
        return "(không có bằng chứng)"
    return "\n".join(f"[{e['n']}] {e['fact']}\n    Trích: \"{e['quote'][:300]}\"" for e in evidence)


def build_context(state: dict) -> str:
    parts = [f"Câu hỏi của người dùng: {(state.get('question') or '').strip() or '(chỉ nhập hồ sơ, muốn biết nguy cơ)'}",
             "KẾT QUẢ MÔ HÌNH:\n" + risk_block(state)]
    if lab_block(state):
        parts.append(lab_block(state))
    if state.get("follow_up"):
        parts.append(f"Cần nhắc người dùng: {state['follow_up']}")
    parts.append("BẰNG CHỨNG (chỉ được dùng các mục này, trích dẫn bằng [n]):\n" + evidence_block(state.get("evidence") or []))
    return "\n\n".join(parts)


def strip_sources(text: str) -> str:
    return re.split(r"\n\s*(\*\*|#+\s*)?(Nguồn|Tài liệu tham khảo)\b", text, maxsplit=1)[0].strip()


DRAFT_INTERVAL = 0.25  # giây giữa hai lần cập nhật bản nháp lên giao diện
# Model 9B hay để sót trích dẫn giữ chỗ "[n]"; xóa bằng code thay vì để Critic bắt viết lại cả bài (tốn ~15s mà thường
# viết lại y nguyên)
PLACEHOLDER = re.compile(r"\s*\[(n|\?|…|\.\.\.|x)\]", re.I)


def _draft_writer():
    """Hàm gửi sự kiện "custom" của LangGraph; None khi không chạy trong đồ thị (ví dụ gọi node trực tiếp)."""
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:
        return None


def advisor_node(state: dict, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("Advisor")
    user = build_context(state)
    critique = state.get("critique") or {}
    revising = bool(critique.get("issues")) and state.get("draft")
    if revising:
        user += ("\n\nBẢN NHÁP TRƯỚC:\n" + state["draft"] + "\n\nCritic yêu cầu sửa các lỗi sau, hãy viết lại toàn bộ câu trả lời:\n"
                 + "\n".join(f"- {issue}" for issue in critique["issues"]))
    messages = [{"role": "system", "content": prompts.ADVISOR}, {"role": "user", "content": user}]
    writer = _draft_writer()
    if writer is not None and hasattr(llm, "stream"):
        last = [0.0]

        def on_text(text: str) -> None:
            if time.perf_counter() - last[0] >= DRAFT_INTERVAL:
                last[0] = time.perf_counter()
                writer({"draft": strip_sources(text)})

        response = llm.stream(messages, think=THINK["advisor"], usage=step.usage, temperature=0.2, on_text=on_text)
    else:
        response = llm.chat(messages, think=THINK["advisor"], usage=step.usage, temperature=0.2)
    draft = PLACEHOLDER.sub("", strip_sources(response.message.content or ""))
    return {"draft": draft, "trace": [step.entry("viết lại theo góp ý Critic" if revising else "viết bản nháp",
                                                 words=len(draft.split()))]}
