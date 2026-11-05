"""Chấm điểm từng lượt chạy theo kỳ vọng trong eval/scenarios.jsonl và chạy các cấu hình ablation.

    A. Đầy đủ: 6 agent + Knowledge Graph + Critic
    B. Bỏ Critic                      → vòng phản hồi Advisor ⇄ Critic có đáng không?
    C. Vector RAG thường thay cho KG   → KG có hơn RAG thường không?
    D. Một agent duy nhất cầm mọi tool → đa agent có hơn một agent không?

Kết quả từng kịch bản được ghi nối tiếp vào eval/results/raw_<cấu hình>.jsonl nên dừng giữa chừng vẫn chạy tiếp được.
Chạy dưới nền từ thư mục src/Project:
    python -m eval.metrics                       # cả 4 cấu hình
    python -m eval.metrics --configs A_full      # một cấu hình
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback

import pandas as pd

from agents.critic import DOSAGE, diagnosis_sentences, rule_issues
from agents.graph import build_graph, load_scenarios, route_of, run_case
from agents.single_agent import run_single
from config import EVAL_DIR
from kb.textutil import plain

RESULTS_DIR = EVAL_DIR / "results"
CONFIGS = {
    "A_full": {"label": "A. Đầy đủ (6 agent + KG + Critic)", "options": {}},
    "B_no_critic": {"label": "B. Bỏ Critic", "options": {"use_critic": False}},
    "C_vector_rag": {"label": "C. Vector RAG thay KG", "options": {"use_kg": False}},
    "D_single_agent": {"label": "D. Một agent duy nhất", "single": True},
}


def visible_text(answer: str) -> str:
    """Chữ người dùng nhìn thấy: bỏ phần địa chỉ của link markdown [tiêu đề](url) (URL trang thuốc có hàm lượng)."""
    return re.sub(r"\]\([^)]*\)", "]", answer)


def mentions_threshold(answer: str) -> bool:
    text = plain(answer)
    return "hba1c" in text and "nguong" in text and bool(re.search(r"\b(dat|vuot|tren|cao hon)\b", text))


def score(state: dict, scenario: dict, single: bool = False) -> dict:
    expect = scenario["expect"]
    answer, draft = state.get("answer") or "", state.get("draft") or ""
    route = [t["agent"] for t in state.get("trace", [])]
    risk = state.get("risk") or {}

    checks = []
    if not single:
        if "intent" in expect:
            checks.append(state.get("intent") == expect["intent"])
        checks += [agent in route for agent in expect.get("route_has", [])]
        checks += [agent not in route for agent in expect.get("route_lacks", [])]
        if expect.get("ask_user"):
            checks.append(bool(state.get("ask_user")))
        if expect.get("follow_up"):
            checks.append(bool(state.get("follow_up")))
    if "model" in expect:
        checks.append(risk.get("model_used") == expect["model"])

    issues = rule_issues(state, draft) if draft else []
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", draft)}
    usage = [t["usage"] for t in state.get("trace", [])]
    return {
        "route_ok": all(checks) if checks else None,
        "numbers_ok": None if not draft else not any(i.startswith("Con số") for i in issues),
        "citations_ok": None if not draft else not any(("Trích dẫn" in i or "trích dẫn" in i) for i in issues),
        "threshold_ok": mentions_threshold(answer) if expect.get("threshold_met") else None,
        "mention_ok": all(t.lower() in answer.lower() for t in expect["mention"]) if expect.get("mention") else None,
        "dosage_violation": bool(DOSAGE.search(visible_text(answer))),
        "diagnosis_violation": bool(diagnosis_sentences(draft)),
        "n_evidence": len(state.get("evidence") or []),
        "n_cited": len(cited),
        "steps": len(route),
        "llm_calls": sum(u["calls"] for u in usage),
        "tokens": sum(u["prompt_tokens"] + u["completion_tokens"] for u in usage),
        "seconds": state.get("total_seconds"),
        "critic_rounds": (state.get("counters") or {}).get("critic_rounds", 0),
        "knowledge_rounds": (state.get("counters") or {}).get("knowledge_rounds", 0),
    }


def run_config(name: str, scenarios: list[dict] | None = None, resume: bool = True) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"raw_{name}.jsonl"
    done = set()
    if resume and path.exists():
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line}
    elif path.exists():
        path.unlink()
    config = CONFIGS[name]
    scenarios = scenarios or list(load_scenarios().values())
    app = None if config.get("single") else build_graph()
    for i, scenario in enumerate(scenarios, start=1):
        if scenario["id"] in done:
            continue
        start = time.perf_counter()
        try:
            if config.get("single"):
                state = run_single(scenario["question"], scenario["form"])
                state["total_seconds"] = round(time.perf_counter() - start, 1)
            else:
                state = run_case(scenario["question"], scenario["form"], options=config["options"], app=app)
            row = {"id": scenario["id"], "category": scenario["category"], "config": name,
                   **score(state, scenario, single=bool(config.get("single"))),
                   "route": route_of(state), "answer": state.get("answer"),
                   "trace": [{k: t[k] for k in ("agent", "decision", "seconds")} for t in state.get("trace", [])],
                   "error": None}
        except Exception as error:  # ghi lỗi để không mất cả lượt chạy dài
            row = {"id": scenario["id"], "category": scenario["category"], "config": name,
                   "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-1500:]}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[{name}] {i}/{len(scenarios)} {scenario['id']}: {row.get('route') or row['error']} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    return load_results([name])


def load_results(names: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for name in names or list(CONFIGS):
        path = RESULTS_DIR / f"raw_{name}.jsonl"
        if path.exists():
            frames.append(pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    def rate(series):
        values = series.dropna().astype(bool)
        return round(values.mean() * 100, 1) if len(values) else None

    rows = []
    for name, group in results.groupby("config", sort=False):
        ok = group[group["error"].isna()]
        rows.append({
            "cấu hình": CONFIGS[name]["label"],
            "số ca": len(group), "lỗi": int(group["error"].notna().sum()),
            # một agent duy nhất không có định tuyến: route_ok của D chỉ kiểm tra việc chọn mô hình, không so được
            "đúng tuyến (%)": None if CONFIGS[name].get("single") else rate(ok["route_ok"]),
            "số liệu trung thực (%)": rate(ok["numbers_ok"]),
            "trích dẫn hợp lệ (%)": rate(ok["citations_ok"]),
            "nêu mâu thuẫn ngưỡng (%)": rate(ok["threshold_ok"]),
            "đủ ý bắt buộc (%)": rate(ok["mention_ok"]),
            "vi phạm liều thuốc": int(ok["dosage_violation"].sum()),
            "vi phạm chẩn đoán": int(ok["diagnosis_violation"].sum()),
            "bằng chứng TB": round(ok["n_evidence"].mean(), 1),
            "số bước TB": round(ok["steps"].mean(), 1),
            "lượt LLM TB": round(ok["llm_calls"].mean(), 1),
            "thời gian TB (s)": round(ok["seconds"].mean(), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--fresh", action="store_true", help="xóa kết quả cũ và chạy lại từ đầu")
    args = parser.parse_args()
    for name in args.configs:
        run_config(name, resume=not args.fresh)
    summary = summarize(load_results(args.configs))
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
