#!/usr/bin/env python3
"""Assignment 07: diabetes_large Knowledge Graph, RAG và agent cục bộ."""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes_prediction_dataset.csv"
MODEL_NAME = "Qwen/Qwen3-0.6B"
COLUMNS = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi",
           "HbA1c_level", "blood_glucose_level", "diabetes"]
GROUP_COLUMNS = ["gender", "smoking_history", "age_group", "hypertension", "heart_disease"]
AGE_BINS = [-1, 17, 39, 59, float("inf")]
AGE_LABELS = ["0-17", "18-39", "40-59", "60+"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(f"CSV thiếu cột: {sorted(set(COLUMNS) - set(df.columns))}")
    if df[COLUMNS].isna().any().any():
        raise ValueError("CSV có giá trị thiếu; cần xử lý trước khi xây KB.")
    if not set(df["diabetes"].unique()).issubset({0, 1}):
        raise ValueError("Nhãn diabetes phải là 0 hoặc 1.")
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    return df


def cohort_stats(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in GROUP_COLUMNS:
        group = df.groupby(column, observed=True)["diabetes"].agg(["size", "sum"]).reset_index()
        group.columns = ["value", "count", "positive"]
        group["dimension"] = column
        group["rate"] = group["positive"] / group["count"]
        frames.append(group[["dimension", "value", "count", "positive", "rate"]])
    return pd.concat(frames, ignore_index=True)


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("dataset:diabetes_large", kind="dataset", count=len(df))
    graph.add_node("outcome:0", kind="outcome", value=0)
    graph.add_node("outcome:1", kind="outcome", value=1)
    for row in cohort_stats(df).itertuples(index=False):
        node_id = f"cohort:{row.dimension}:{row.value}"
        graph.add_node(node_id, kind="cohort", dimension=row.dimension, value=str(row.value),
                       count=int(row.count), positive=int(row.positive), rate=float(row.rate))
        graph.add_edge("dataset:diabetes_large", node_id, relation="has_cohort")
        graph.add_edge(node_id, "outcome:1", relation="positive_count", count=int(row.positive))
        graph.add_edge(node_id, "outcome:0", relation="negative_count", count=int(row.count - row.positive))
    return graph


def graph_triples(graph: nx.DiGraph, limit: int = 8) -> list[tuple[str, str, str]]:
    triples = [(u, data["relation"], v) for u, v, data in graph.edges(data=True)]
    examples = []
    for relation in ("has_cohort", "positive_count", "negative_count"):
        match = next((triple for triple in triples if triple[1] == relation), None)
        if match:
            examples.append(match)
    return (examples + [triple for triple in triples if triple not in examples])[:limit]


def query_graph(graph: nx.DiGraph, dimension: str, value: str) -> dict | None:
    node = graph.nodes.get(f"cohort:{dimension}:{value}")
    return dict(node) if node else None


def build_documents(df: pd.DataFrame) -> list[dict[str, str]]:
    positive = int(df["diabetes"].sum())
    docs = [{"id": "DIABETES:overall", "text":
             f"Dataset diabetes_large có {len(df)} bản ghi, {positive} nhãn diabetes=1, "
             f"{len(df)-positive} nhãn diabetes=0; tỷ lệ nhãn 1 là {positive/len(df):.2%}. "
             "Các trường: gender, age, hypertension, heart_disease, smoking_history, bmi, "
             "HbA1c_level, blood_glucose_level, diabetes. Đây là dữ liệu quan sát, không phải chẩn đoán."}]
    for row in cohort_stats(df).itertuples(index=False):
        docs.append({"id": f"DIABETES:{row.dimension}:{row.value}",
                     "text": (f"Nhóm {row.dimension}={row.value}: {int(row.count)} bản ghi, "
                              f"{int(row.positive)} nhãn diabetes=1, tỷ lệ {row.rate:.2%}. "
                              "Tỷ lệ mô tả dataset, không chứng minh quan hệ nhân quả.")})
    return docs


def build_index(documents: list[dict[str, str]]):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform([doc["text"] for doc in documents])
    return vectorizer, matrix


def retrieve(question: str, documents: list[dict[str, str]], vectorizer, matrix,
             top_k: int = 4) -> list[dict]:
    scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [{**documents[i], "score": round(float(scores[i]), 4)} for i in indices if scores[i] >= 0.10]


@lru_cache(maxsize=1)
def load_local_model():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Cần cài transformers>=4.51 và accelerate để dùng Qwen3 cục bộ.") from exc
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
    except OSError as exc:
        raise RuntimeError("Không tải được Qwen3-0.6B. Kiểm tra mạng ở lần đầu hoặc cache model cục bộ.") from exc
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return tokenizer, model, device


def generate_grounded(question: str, evidence: list[dict]) -> str:
    import torch

    tokenizer, model, device = load_local_model()
    context = "\n".join(f"[{item['id']}] {item['text']}" for item in evidence)
    messages = [
        {"role": "system", "content": "Trả lời ngắn bằng tiếng Việt và chỉ dùng NGỮ CẢNH. "
         "Không chẩn đoán, tư vấn điều trị hoặc nói rằng nhóm dữ liệu gây ra bệnh. Nếu thiếu dữ kiện, nói không đủ dữ liệu."},
        {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=180, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def _cohort_from_question(question: str) -> tuple[str, str] | None:
    q = question.lower()
    for label in AGE_LABELS:
        if label in q:
            return "age_group", label
    if "trên 60" in q or "60 tuổi trở lên" in q:
        return "age_group", "60+"
    for term, value in (("female", "Female"), ("nữ", "Female"), ("male", "Male"), ("nam", "Male"),
                        ("other", "Other")):
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", q):
            return "gender", value
    for value in ("not current", "no info", "current", "former", "never", "ever"):
        if value in q:
            canonical = next((v for v in ["not current", "No Info", "current", "former", "never", "ever"]
                              if v.lower() == value), value)
            return "smoking_history", canonical
    if "hút thuốc hiện tại" in q:
        return "smoking_history", "current"
    if "chưa từng hút" in q:
        return "smoking_history", "never"
    if "tăng huyết áp" in q or "hypertension" in q:
        return "hypertension", "1"
    if "bệnh tim" in q or "heart_disease" in q:
        return "heart_disease", "1"
    return None


class DiabetesAgent:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = load_data() if df is None else df
        self.graph = build_graph(self.df)
        self.documents = build_documents(self.df)
        self.vectorizer, self.matrix = build_index(self.documents)

    def ask(self, question: str, generate: bool = True) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống.")
        q = question.lower()
        cohort = _cohort_from_question(question)
        statistical = any(word in q for word in ("tỷ lệ", "bao nhiêu", "số lượng", "phần trăm", "prevalence", "thống kê"))
        if any(word in q for word in ("chẩn đoán", "điều trị", "thuốc nào", "tôi có bị")):
            return {"answer": "Dataset không đủ cơ sở để chẩn đoán hoặc tư vấn điều trị.",
                    "sources": [], "tools": []}
        if any(word in q for word in ("các trường", "các cột", "schema", "thuộc tính nào")):
            result = {"answer": "Các cột trong CSV: " + ", ".join(COLUMNS) + ".",
                      "sources": ["DIABETES:overall"], "tools": ["csv_schema"]}
        elif cohort and statistical:
            dimension, value = cohort
            node = query_graph(self.graph, dimension, value)
            if node is None:
                result = {"answer": f"Không có nhóm {dimension}={value} trong CSV.",
                          "sources": [], "tools": ["knowledge_graph"]}
            else:
                result = {"answer": (f"Nhóm {dimension}={value} có {node['count']:,} bản ghi; "
                                     f"{node['positive']:,} nhãn diabetes=1; tỷ lệ {node['rate']:.2%}."),
                          "sources": [f"DIABETES:{dimension}:{value}"], "tools": ["knowledge_graph"]}
        elif statistical and any(word in q for word in ("diabetes", "tiểu đường", "dataset", "dữ liệu")):
            n = len(self.df)
            positive = int(self.df["diabetes"].sum())
            result = {"answer": f"CSV có {n:,} bản ghi; {positive:,} nhãn diabetes=1; tỷ lệ {positive/n:.2%}.",
                      "sources": ["DIABETES:overall"], "tools": ["csv_statistics"]}
        elif any(word in q for word in ("diabetes", "tiểu đường", "dataset", "dữ liệu", "bmi", "hba1c", "glucose", "smoking")):
            evidence = retrieve(question, self.documents, self.vectorizer, self.matrix)
            if not evidence:
                result = {"answer": "Không tìm thấy bằng chứng phù hợp trong CSV.", "sources": [],
                          "tools": ["rag_retrieval"]}
            else:
                answer = generate_grounded(question, evidence) if generate else "\n".join(item["text"] for item in evidence)
                result = {"answer": answer or "Không đủ dữ liệu để trả lời.",
                          "sources": [item["id"] for item in evidence],
                          "tools": ["rag_retrieval", "local_llm"] if generate else ["rag_retrieval"]}
        else:
            result = {"answer": "Câu hỏi nằm ngoài phạm vi dataset diabetes_large.", "sources": [], "tools": []}
        return result


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Diabetes KB + RAG</title><style>body{font:16px system-ui;background:#10232c;color:#eef8fa;max-width:850px;margin:40px auto;padding:0 16px}textarea,button{font:inherit;border-radius:8px;padding:12px}textarea{width:100%;box-sizing:border-box;background:#1b3942;color:white;border:1px solid #528087;min-height:90px}button{margin-top:10px;background:#72d4be;border:0;cursor:pointer}pre{white-space:pre-wrap;background:#1b3942;padding:16px;border-radius:8px}small{color:#b7d3d5}</style></head><body><h1>Diabetes · Knowledge Graph + RAG</h1><p>Hỏi về thống kê trong dataset. Kết quả không dùng để chẩn đoán.</p><textarea id="q">Tỷ lệ diabetes ở nữ là bao nhiêu?</textarea><br><button id="send">Hỏi agent</button><pre id="answer">Sẵn sàng.</pre><small id="meta"></small><script>document.getElementById('send').onclick=async()=>{let a=document.getElementById('answer'),m=document.getElementById('meta');a.textContent='Đang xử lý…';m.textContent='';try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});let d=await r.json();if(!r.ok)throw Error(d.error);a.textContent=d.answer;m.textContent='Nguồn: '+d.sources.join(', ')+' | Công cụ: '+d.tools.join(', ')}catch(e){a.textContent=e.message}}</script></body></html>"""


def create_app(agent: DiabetesAgent | None = None) -> Flask:
    app = Flask(__name__)
    agent = agent or DiabetesAgent()

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.post("/api/chat")
    def chat():
        try:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload.get("question"), str):
                raise ValueError("Trường question phải là chuỗi.")
            return jsonify(agent.ask(payload["question"]))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question", help="Câu hỏi gửi agent")
    mode.add_argument("--web", action="store_true", help="Mở web localhost")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    agent = DiabetesAgent()
    if args.web:
        create_app(agent).run(host="127.0.0.1", port=args.port, debug=False)
    else:
        print(json.dumps(agent.ask(args.question), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
