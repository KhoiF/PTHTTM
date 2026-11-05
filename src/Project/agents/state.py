"""Trạng thái dùng chung (blackboard) mà các agent đọc/ghi trong LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class CaseState(TypedDict, total=False):
    question: str                 # tin nhắn của người dùng
    form: dict                    # dữ liệu nhập từ form (có thể rỗng)
    history_patient: dict         # hồ sơ từ các lượt trước (hội thoại nhiều lượt)
    options: dict                 # cờ ablation: use_critic, use_kg
    intent: str                   # assess | ask | both
    patient: dict                 # hồ sơ đã chuẩn hóa
    missing_fields: list[str]
    input_errors: list[str]
    ask_user: str | None          # câu hỏi lại người dùng; có giá trị thì kết thúc lượt
    follow_up: str | None         # đề nghị bổ sung (không chặn câu trả lời)
    risk: dict                    # kết quả của Risk Agent
    risk_request: str | None      # Risk nhờ Intake tìm thêm dữ liệu
    knowledge_notes: list[str]    # gợi ý cho vòng Knowledge tiếp theo
    knowledge_done: bool          # False thì Knowledge tự lặp thêm một vòng
    evidence: list[dict]          # bằng chứng đánh số [n]
    lab_checks: list[dict]        # so chỉ số xét nghiệm với ngưỡng chẩn đoán trong KG
    draft: str
    critique: dict
    answer: str
    counters: Annotated[dict, lambda a, b: {**a, **b}]
    trace: Annotated[list[dict], operator.add]
