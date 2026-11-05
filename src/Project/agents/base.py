"""Tiện ích chung cho các agent: LLM dùng chung và ghi trace."""

from __future__ import annotations

import time
from functools import lru_cache

from agents.llm import LLM, Usage


@lru_cache
def default_llm() -> LLM:
    return LLM()


class Step:
    """Đo thời gian/token của một agent và tạo bản ghi trace."""

    def __init__(self, agent: str):
        self.agent = agent
        self.usage = Usage()
        self.start = time.perf_counter()

    def entry(self, decision: str, **detail) -> dict:
        return {"agent": self.agent, "decision": decision, "detail": detail, "usage": self.usage.as_dict(),
                "seconds": round(time.perf_counter() - self.start, 2)}


def percent(p: float) -> str:
    """0.6556 → '65,6%' (cách viết thập phân tiếng Việt)."""
    return f"{p * 100:.1f}%".replace(".", ",")
