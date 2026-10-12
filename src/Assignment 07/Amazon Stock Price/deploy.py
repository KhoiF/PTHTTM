#!/usr/bin/env python3
"""Assignment 07: Amazon Knowledge Graph, RAG và agent cục bộ."""

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
DATA_PATH = BASE_DIR / "data" / "AMZN.csv"
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH_PATTERN = re.compile(r"(?<!\d)(\d{4}-\d{2})(?!-\d)")
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?![-\d])")
METRICS = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "adj close": "Adj Close", "volume": "Volume"}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    expected = {"Date", *METRICS.values()}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV thiếu cột: {sorted(expected - set(df.columns))}")
    if df[list(expected)].isna().any().any() or df["Date"].duplicated().any():
        raise ValueError("CSV có giá trị thiếu hoặc ngày giao dịch bị trùng.")
    return df.sort_values("Date").reset_index(drop=True)


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("AMZN", kind="stock", name="Amazon")
    for row in df.itertuples(index=False):
        date = row.Date.strftime("%Y-%m-%d")
        year, month = date[:4], date[:7]
        year_id, month_id, day_id = f"year:{year}", f"month:{month}", f"day:{date}"
        graph.add_node(year_id, kind="year", value=year)
        graph.add_node(month_id, kind="month", value=month)
        graph.add_node(day_id, kind="session", date=date,
                       Open=float(row.Open), High=float(row.High), Low=float(row.Low),
                       Close=float(row.Close), Adj_Close=float(row._5), Volume=int(row.Volume))
        graph.add_edge("AMZN", year_id, relation="has_year")
        graph.add_edge(year_id, month_id, relation="has_month")
        graph.add_edge(month_id, day_id, relation="has_session")
    return graph


def graph_triples(graph: nx.DiGraph, limit: int = 6) -> list[tuple[str, str, str]]:
    triples = [(u, data["relation"], v) for u, v, data in graph.edges(data=True)]
    examples = []
    for relation in ("has_year", "has_month", "has_session"):
        match = next((triple for triple in triples if triple[1] == relation), None)
        if match:
            examples.append(match)
    return (examples + [triple for triple in triples if triple not in examples])[:limit]


def build_documents(df: pd.DataFrame) -> list[dict[str, str]]:
    first_date = df["Date"].min().strftime("%Y-%m-%d")
    last_date = df["Date"].max().strftime("%Y-%m-%d")
    docs = [{
        "id": "AMZN:overview",
        "text": (f"Tóm tắt dữ liệu Amazon AMZN: {len(df)} phiên giao dịch từ {first_date} đến {last_date}. "
                 "Dataset gồm Date, Open, High, Low, Close, Adj Close và Volume. "
                 "Đây là giá lịch sử trong CSV, không phải giá hiện tại hoặc dự báo."),
    }]
    for row in df.itertuples(index=False):
        date = row.Date.strftime("%Y-%m-%d")
        docs.append({
            "id": f"AMZN:{date}",
            "text": (f"Amazon AMZN phiên {date}: Open {row.Open:.4f} USD; High {row.High:.4f} USD; "
                     f"Low {row.Low:.4f} USD; Close {row.Close:.4f} USD; "
                     f"Adj Close {row._5:.4f} USD; Volume {int(row.Volume)} cổ phiếu."),
        })
    for month, part in df.groupby(df["Date"].dt.to_period("M")):
        docs.append({
            "id": f"AMZN:month:{month}",
            "text": (f"Amazon AMZN tháng {month}: {len(part)} phiên; giá Close trung bình "
                     f"{part['Close'].mean():.4f} USD, thấp nhất {part['Close'].min():.4f} USD, "
                     f"cao nhất {part['Close'].max():.4f} USD; tổng Volume {int(part['Volume'].sum())}."),
        })
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


def query_graph(graph: nx.DiGraph, date: str) -> dict | None:
    node = graph.nodes.get(f"day:{date}")
    return dict(node) if node else None


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
        {"role": "system", "content": "Trả lời ngắn bằng tiếng Việt. Chỉ dùng dữ kiện trong NGỮ CẢNH. "
         "Không dự báo giá, không suy diễn tin tức hay giá hiện tại. Nếu thiếu dữ kiện, nói không đủ dữ liệu."},
        {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=180, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def _metric_from_question(question: str) -> str:
    q = question.lower()
    if "khối lượng" in q or "volume" in q:
        return "Volume"
    if "adj close" in q:
        return "Adj Close"
    for word, metric in METRICS.items():
        if word in q:
            return metric
    return "Close"


def _range_answer(question: str, df: pd.DataFrame, dates: list[str]) -> dict:
    start, end = sorted(dates[:2])
    try:
        pd.Timestamp(start)
        pd.Timestamp(end)
    except ValueError:
        raise ValueError("Ngày không hợp lệ; dùng YYYY-MM-DD.") from None
    part = df.loc[df["Date"].between(start, end)]
    if part.empty:
        return {"answer": f"Không có phiên giao dịch từ {start} đến {end} trong CSV.",
                "sources": [], "tools": ["csv_statistics"]}
    metric = _metric_from_question(question)
    values = part[metric]
    unit = "cổ phiếu" if metric == "Volume" else "USD"
    answer = (f"Từ {start} đến {end} có {len(part)} phiên. {metric}: trung bình {values.mean():,.4f} {unit}; "
              f"thấp nhất {values.min():,.4f} {unit}; cao nhất {values.max():,.4f} {unit}.")
    return {"answer": answer, "sources": [f"AMZN.csv#Date={start}..{end}"],
            "tools": ["csv_statistics"]}


class AmazonAgent:
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
        dates = DATE_PATTERN.findall(question)
        if len(dates) >= 2:
            result = _range_answer(question, self.df, dates)
        elif len(dates) == 1:
            try:
                pd.Timestamp(dates[0])
            except ValueError:
                raise ValueError("Ngày không hợp lệ; dùng YYYY-MM-DD.") from None
            node = query_graph(self.graph, dates[0])
            if node is None:
                result = {"answer": f"Không có phiên {dates[0]} trong CSV.", "sources": [],
                          "tools": ["knowledge_graph"]}
            else:
                metric = _metric_from_question(question)
                value = node["Adj_Close" if metric == "Adj Close" else metric]
                unit = "cổ phiếu" if metric == "Volume" else "USD"
                result = {"answer": f"Ngày {dates[0]}, {metric} của AMZN là {value:,.4f} {unit}.",
                          "sources": [f"AMZN:{dates[0]}"], "tools": ["knowledge_graph"]}
        elif MONTH_PATTERN.search(question):
            month = MONTH_PATTERN.search(question).group(1)
            if f"month:{month}" not in self.graph:
                result = {"answer": f"Không có tháng {month} trong CSV.", "sources": [],
                          "tools": ["knowledge_graph"]}
            else:
                part = self.df.loc[self.df["Date"].dt.strftime("%Y-%m").eq(month)]
                metric = _metric_from_question(question)
                values = part[metric]
                unit = "cổ phiếu" if metric == "Volume" else "USD"
                result = {"answer": (f"Tháng {month} có {len(part)} phiên; {metric} trung bình "
                                     f"{values.mean():,.4f} {unit}, thấp nhất {values.min():,.4f} {unit}, "
                                     f"cao nhất {values.max():,.4f} {unit}."),
                          "sources": [f"AMZN:month:{month}"], "tools": ["knowledge_graph", "csv_statistics"]}
        elif YEAR_PATTERN.search(question):
            year = YEAR_PATTERN.search(question).group(1)
            if f"year:{year}" not in self.graph:
                result = {"answer": f"Không có năm {year} trong CSV.", "sources": [],
                          "tools": ["knowledge_graph"]}
            else:
                part = self.df.loc[self.df["Date"].dt.year.eq(int(year))]
                metric = _metric_from_question(question)
                values = part[metric]
                unit = "cổ phiếu" if metric == "Volume" else "USD"
                result = {"answer": (f"Năm {year} có {len(part)} phiên; {metric} trung bình "
                                     f"{values.mean():,.4f} {unit}, thấp nhất {values.min():,.4f} {unit}, "
                                     f"cao nhất {values.max():,.4f} {unit}."),
                          "sources": [f"AMZN.csv#Year={year}"], "tools": ["knowledge_graph", "csv_statistics"]}
        elif "mới nhất" in q or "cuối cùng" in q:
            row = self.df.iloc[-1]
            date = row["Date"].strftime("%Y-%m-%d")
            result = {"answer": f"Phiên cuối trong CSV là {date}; Close {row['Close']:,.4f} USD. Đây không phải giá hiện tại.",
                      "sources": [f"AMZN:{date}"], "tools": ["knowledge_graph"]}
        elif any(word in q for word in ("amazon", "amzn", "giá", "cổ phiếu", "phiên", "volume", "đóng cửa", "dữ liệu")):
            evidence = retrieve(question, self.documents, self.vectorizer, self.matrix)
            if not evidence:
                result = {"answer": "Không tìm thấy bằng chứng phù hợp trong CSV.", "sources": [],
                          "tools": ["rag_retrieval"]}
            else:
                answer = generate_grounded(question, evidence) if generate else "\n".join(item["text"] for item in evidence)
                result = {"answer": answer or "Không đủ dữ liệu để trả lời.",
                          "sources": [item["id"] for item in evidence], "tools": ["rag_retrieval", "local_llm"] if generate else ["rag_retrieval"]}
        else:
            result = {"answer": "Câu hỏi nằm ngoài phạm vi dataset Amazon Stock Price.", "sources": [], "tools": []}
        return result


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Amazon KB + RAG</title><style>body{font:16px system-ui;background:#101b2b;color:#eef4ff;max-width:850px;margin:40px auto;padding:0 16px}textarea,button{font:inherit;border-radius:8px;padding:12px}textarea{width:100%;box-sizing:border-box;background:#192b42;color:white;border:1px solid #49617d;min-height:90px}button{margin-top:10px;background:#f7b452;border:0;cursor:pointer}pre{white-space:pre-wrap;background:#192b42;padding:16px;border-radius:8px}small{color:#b9c9dc}</style></head><body><h1>Amazon · Knowledge Graph + RAG</h1><p>Hỏi về các phiên giao dịch trong CSV đến 05/12/2023.</p><textarea id="q">Giá Close ngày 2023-12-05 là bao nhiêu?</textarea><br><button id="send">Hỏi agent</button><pre id="answer">Sẵn sàng.</pre><small id="meta"></small><script>document.getElementById('send').onclick=async()=>{let a=document.getElementById('answer'),m=document.getElementById('meta');a.textContent='Đang xử lý…';m.textContent='';try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});let d=await r.json();if(!r.ok)throw Error(d.error);a.textContent=d.answer;m.textContent='Nguồn: '+d.sources.join(', ')+' | Công cụ: '+d.tools.join(', ')}catch(e){a.textContent=e.message}}</script></body></html>"""


def create_app(agent: AmazonAgent | None = None) -> Flask:
    app = Flask(__name__)
    agent = agent or AmazonAgent()

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
    parser.add_argument("--port", type=int, default=8007)
    args = parser.parse_args()
    agent = AmazonAgent()
    if args.web:
        create_app(agent).run(host="127.0.0.1", port=args.port, debug=False)
    else:
        print(json.dumps(agent.ask(args.question), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
