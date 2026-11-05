"""Lắp 6 agent thành đồ thị LangGraph có định tuyến động và vòng phản hồi.

    Supervisor ─ask─────────────────────────────┐
        └─assess/both→ Intake ─thiếu dữ liệu→ (hỏi lại người dùng)
                         ↑ └→ Risk ─vùng xám, thiếu xét nghiệm→ Intake
                         └──────┘   └→ Knowledge ⟲ (chưa đủ bằng chứng)
                                        └→ Advisor ⇄ Critic → Finalize

Chạy thử từ thư mục src/Project:
    python -m agents.graph --question "Tôi nữ 52 tuổi, cao 1m58, nặng 70kg, HbA1c 6.7%..."
    python -m agents.graph --scenario full_grey_hba1c
"""

from __future__ import annotations

import argparse
import json
import re
import time
from functools import partial

from langgraph.graph import END, START, StateGraph

from agents.advisor import advisor_node
from agents.base import Step
from agents.critic import critic_node
from agents.intake import intake_node
from agents.knowledge import knowledge_node
from agents.risk import risk_node
from agents.state import CaseState
from agents.supervisor import supervisor_node
from config import EVAL_DIR, MAX_CRITIC_ROUNDS

DISCLAIMER = ("> ⚠️ Kết quả chỉ mang tính tham khảo, được tạo tự động từ mô hình học máy và thông tin của Nhà thuốc "
              "Long Châu; không thay thế chẩn đoán của bác sĩ.")


LATEX_SYMBOLS = {r"\geq": "≥", r"\ge": "≥", r"\leq": "≤", r"\le": "≤", r"\times": "×", r"\approx": "≈", r"\%": "%"}


def strip_latex(text: str) -> str:
    r"""Model đôi khi viết ký hiệu kiểu LaTeX ($\ge$ 126); giao diện chat không hiển thị được nên đổi sang Unicode."""
    def symbols(match: re.Match) -> str:
        math = match.group(1)
        if "\\" not in math:            # "$...$" không chứa lệnh LaTeX: giữ nguyên
            return match.group(0)
        for command, char in LATEX_SYMBOLS.items():
            math = re.sub(re.escape(command) + r"(?![a-zA-Z])", char, math)
        return math.strip()
    return re.sub(r"\$([^$\n]{1,40})\$", symbols, text)


def finalize_node(state: dict) -> dict:
    step = Step("Finalize")
    if state.get("ask_user"):
        return {"answer": state["ask_user"], "trace": [step.entry("trả câu hỏi bổ sung cho người dùng")]}
    draft = re.sub(r"\s*\[(n|\?|…|\.\.\.|x)\]", "", state.get("draft") or "", flags=re.I)
    draft = strip_latex(draft)
    evidence = state.get("evidence") or []
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", draft) if 1 <= int(n) <= len(evidence)})
    parts = [draft]
    follow_up = state.get("follow_up")
    if follow_up and follow_up[:40] not in draft:
        parts.append(f"**Lưu ý:** {follow_up}")
    if not cited and evidence:
        # Critic hết lượt sửa mà bản nháp vẫn không có [n]: vẫn cho người dùng thấy nguồn đã tra cứu, ghi đúng như vậy
        urls = list(dict.fromkeys(e["source_url"] for e in evidence if e.get("source_url")))[:5]
        titles = {e["source_url"]: e.get("title") or e["source_url"] for e in evidence}
        parts.append("**Nguồn đã tra cứu (Nhà thuốc Long Châu):**\n" + "\n".join(f"- [{titles[u]}]({u})" for u in urls))
    if cited:
        by_url: dict[str, list[int]] = {}
        for n in cited:
            by_url.setdefault(evidence[n - 1]["source_url"], []).append(n)
        titles = {e["source_url"]: e.get("title") or e["source_url"] for e in evidence}
        sources = [f"{''.join(f'[{n}]' for n in ns)} [{titles[url]}]({url})" for url, ns in by_url.items()]
        parts.append("**Nguồn (Nhà thuốc Long Châu):**\n" + "\n".join(f"- {s}" for s in sources))
    parts.append(DISCLAIMER)
    decision = f"gắn {len(cited)} nguồn" if cited or not evidence else "không có [n] → liệt kê nguồn đã tra cứu"
    return {"answer": "\n\n".join(parts), "trace": [step.entry(decision, cited=cited)]}


def _after_supervisor(state):
    return "knowledge" if state["intent"] == "ask" else "intake"


def _after_intake(state):
    if state.get("ask_user"):
        return "finalize"
    if state.get("missing_fields") or state.get("input_errors"):
        return "knowledge"  # intent both: thiếu hồ sơ nhưng vẫn trả lời câu hỏi kiến thức
    return "risk"


def _after_risk(state):
    return "intake" if state.get("risk_request") else "knowledge"


def _after_knowledge(state):
    return "advisor" if state.get("knowledge_done", True) else "knowledge"


def _after_advisor(state):
    return "critic" if (state.get("options") or {}).get("use_critic", True) else "finalize"


def _after_critic(state):
    critique = state.get("critique") or {}
    if not critique.get("passed") and state["counters"].get("critic_rounds", 0) < MAX_CRITIC_ROUNDS:
        return "advisor"
    return "finalize"


def build_graph(llm=None):
    graph = StateGraph(CaseState)
    for name, fn in [("supervisor", supervisor_node), ("intake", intake_node), ("risk", risk_node),
                     ("knowledge", knowledge_node), ("advisor", advisor_node), ("critic", critic_node)]:
        graph.add_node(name, partial(fn, llm=llm))
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _after_supervisor, ["intake", "knowledge"])
    graph.add_conditional_edges("intake", _after_intake, ["risk", "knowledge", "finalize"])
    graph.add_conditional_edges("risk", _after_risk, ["intake", "knowledge"])
    graph.add_conditional_edges("knowledge", _after_knowledge, ["knowledge", "advisor"])
    graph.add_conditional_edges("advisor", _after_advisor, ["critic", "finalize"])
    graph.add_conditional_edges("critic", _after_critic, ["advisor", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


def run_case(question: str = "", form: dict | None = None, history_patient: dict | None = None,
             options: dict | None = None, llm=None, app=None) -> dict:
    app = app or build_graph(llm)
    start = time.perf_counter()
    state = app.invoke({"question": question, "form": form or {}, "history_patient": history_patient or {},
                        "options": options or {}, "counters": {}, "trace": []}, config={"recursion_limit": 40})
    state["total_seconds"] = round(time.perf_counter() - start, 1)
    return state


def stream_case(question: str = "", form: dict | None = None, history_patient: dict | None = None,
                options: dict | None = None, llm=None, app=None):
    """Như run_case nhưng trả về trạng thái sau mỗi agent (dùng cho giao diện). Trong lúc Advisor viết, trả thêm các
    trạng thái có khóa "draft_preview" là phần câu trả lời đã sinh, để giao diện hiện chữ dần."""
    app = app or build_graph(llm)
    start = time.perf_counter()
    state = {}
    for mode, chunk in app.stream({"question": question, "form": form or {}, "history_patient": history_patient or {},
                                   "options": options or {}, "counters": {}, "trace": []},
                                  config={"recursion_limit": 40}, stream_mode=["values", "custom"]):
        if mode == "values":
            state = chunk
            state["total_seconds"] = round(time.perf_counter() - start, 1)
            yield state
        elif isinstance(chunk, dict) and "draft" in chunk:
            yield {**state, "draft_preview": chunk["draft"]}


def route_of(state: dict) -> str:
    return " → ".join(t["agent"] for t in state.get("trace", []))


def load_scenarios() -> dict:
    path = EVAL_DIR / "scenarios.jsonl"
    return {s["id"]: s for s in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if s}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="")
    parser.add_argument("--form", default="{}", help="hồ sơ dạng JSON")
    parser.add_argument("--scenario", help="id trong eval/scenarios.jsonl")
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--vector-rag", action="store_true", help="thay KG bằng tìm kiếm vector thường")
    args = parser.parse_args()
    question, form = args.question, json.loads(args.form)
    if args.scenario:
        scenario = load_scenarios()[args.scenario]
        question, form = scenario.get("question", ""), scenario.get("form", {})
    state = run_case(question, form, options={"use_critic": not args.no_critic, "use_kg": not args.vector_rag})
    print(f"\nĐƯỜNG ĐI: {route_of(state)}  ({state['total_seconds']}s)\n")
    for t in state["trace"]:
        print(f"  [{t['agent']:10s}] {t['decision']}  ({t['seconds']}s, {t['usage']['calls']} lượt LLM)")
        for issue in t["detail"].get("issues", []):
            print(f"               - {issue}")
    print("\n" + "=" * 80 + "\n" + state["answer"])


if __name__ == "__main__":
    main()
