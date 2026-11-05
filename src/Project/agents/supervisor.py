"""Supervisor: phân loại ý định (assess / ask / both) để định tuyến."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agents import prompts
from agents.base import Step, default_llm
from config import THINK


# Dấu hiệu của câu hỏi kiến thức đi kèm hồ sơ: assess → both để vẫn trả lời câu hỏi khi hồ sơ chưa đủ chạy mô hình
KNOWLEDGE_CUES = re.compile(r"dấu hiệu|triệu chứng|là gì|có phải|nên ăn|ăn gì|ăn uống|kiêng|phòng ngừa|phòng tránh"
                            r"|biến chứng|có nguy hiểm|thuốc gì|uống thuốc|điều trị|chữa|tập thể dục|vận động|nên làm gì"
                            r"|làm sao|làm thế nào|bằng cách nào", re.I)


class Intent(BaseModel):
    """Phân loại yêu cầu của người dùng."""

    intent: Literal["assess", "ask", "both"]
    reason: str = Field(description="Lý do ngắn gọn")


def supervisor_node(state: dict, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("Supervisor")
    form = {k: v for k, v in (state.get("form") or {}).items() if v not in (None, "")}
    has_profile = len(form) >= 3 or bool(state.get("history_patient"))
    question = (state.get("question") or "").strip()
    if not question:
        intent, reason = "assess", "chỉ có dữ liệu form"
    else:
        result = llm.structured(Intent, prompts.SUPERVISOR,
                                f"Người dùng đã nhập hồ sơ qua form: {'có' if has_profile else 'không'}\nTin nhắn: {question}",
                                think=THINK["supervisor"], usage=step.usage)
        intent, reason = result.intent, result.reason
        if has_profile and intent == "ask":
            intent, reason = "both", reason + " (có hồ sơ nên đánh giá kèm)"
        elif intent == "assess" and KNOWLEDGE_CUES.search(question):
            intent, reason = "both", reason + " (kèm câu hỏi kiến thức)"
    route = "Knowledge" if intent == "ask" else "Intake"
    return {"intent": intent, "counters": {"knowledge_rounds": 0, "critic_rounds": 0, "labs_requested": False},
            "evidence": [], "lab_checks": [], "knowledge_notes": [], "risk_request": None, "ask_user": None,
            "trace": [step.entry(f"intent={intent} → {route}", reason=reason)]}
