"""REST API + giao diện web cho Ứng dụng 1 - Dự đoán tiểu đường (Assignment 02, mục 2.4).

Nạp weight đã lưu ở mục 10 của notebook (`models/*.sav` + `models/metadata.json`). Mỗi `.sav`
là một Pipeline trọn gói (StandardScaler + model) nên dữ liệu mới đi qua đúng phép chuẩn hoá đã
fit lúc huấn luyện.

Tuỳ chọn: truy vấn đồ thị tri thức Neo4j để trả tư vấn chi tiết khi dự đoán là dương tính.
Không có Neo4j thì dự đoán vẫn chạy bình thường.

Chạy:
    cd "src/Assignment 02/diabetes"
    python REST_API.py          # http://127.0.0.1:8080
"""
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PORT = 8080

# --- Weight đã huấn luyện ---
with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES = METADATA["deploy_features"]
DEFAULT_MODEL = METADATA.get("default_model", "knn")
FEATURE_MEANS = METADATA.get("feature_means", {})

LOADED_MODELS = {}
for key, info in METADATA["models"].items():
    path = MODEL_DIR / info["file"]
    try:
        with open(path, "rb") as f:
            LOADED_MODELS[key] = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] Chưa có weight {path} - hãy chạy notebook mục 10 trước.")

# --- Neo4j (tuỳ chọn) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

neo4j_driver = None
if NEO4J_PASSWORD:
    try:
        from neo4j import GraphDatabase

        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception as exc:
        print(f"[WARN] Không khởi tạo được Neo4j driver: {exc}")

KNOWLEDGE_GROUPS = {
    "HAS_DIET_ADVICE": "Chế độ ăn",
    "HAS_LIFESTYLE_ACTION": "Lối sống và vận động",
    "HAS_DRUG_INFO": "Thông tin thuốc",
    "HAS_COMPLICATION": "Biến chứng cần lưu ý",
}

KNOWLEDGE_QUERY = """
MATCH (d:Disease {id: 'diabetes'})-[relation:HAS_DIET_ADVICE|HAS_LIFESTYLE_ACTION|HAS_DRUG_INFO|HAS_COMPLICATION]->(item)
OPTIONAL MATCH (item)-[:SOURCED_FROM]->(article:Article)
RETURN type(relation) AS relation_type,
       coalesce(item.title, item.name) AS title,
       item.content AS content,
       collect(DISTINCT {title: article.title, url: article.url}) AS sources
ORDER BY relation_type, title
"""


def get_diabetes_knowledge():
    """Đọc tri thức y khoa đã nhóm sẵn từ Neo4j."""
    if neo4j_driver is None:
        return [], "Neo4j password has not been configured (NEO4J_PASSWORD)."

    try:
        grouped = defaultdict(list)
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            for record in session.run(KNOWLEDGE_QUERY):
                grouped[record["relation_type"]].append({
                    "title": record["title"],
                    "content": record["content"],
                    "sources": [s for s in record["sources"] if s.get("title")],
                })

        return [
            {"group_label": label, "items": grouped[rel_type]}
            for rel_type, label in KNOWLEDGE_GROUPS.items()
            if rel_type in grouped
        ], None
    except Exception as exc:
        return [], str(exc)


@app.route("/")
def home():
    return render_template(
        "index.html",
        models=[
            {"key": key, "name": info["display_name"], "metrics": info["test_metrics"]}
            for key, info in METADATA["models"].items()
        ],
        default_model=DEFAULT_MODEL,
    )


@app.route("/diabetes/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "features": FEATURES,
        "default_model": DEFAULT_MODEL,
        "models": METADATA["models"],
    })


@app.route("/diabetes/v1/predict", methods=["POST"])
def predict():
    if not LOADED_MODELS:
        return jsonify({"error": "Chưa có weight. Hãy chạy notebook mục 10 để sinh file .sav."}), 500

    payload = request.get_json(silent=True) or {}
    model_key = payload.get("model", DEFAULT_MODEL)
    if model_key not in LOADED_MODELS:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400

    # Chỉ số bỏ trống được điền bằng trung bình của bộ huấn luyện
    row = {}
    for name in FEATURES:
        value = payload.get(name)
        if value in (None, ""):
            if name not in FEATURE_MEANS:
                return jsonify({"error": f"Thiếu giá trị bắt buộc cho '{name}'."}), 400
            value = FEATURE_MEANS[name]
        try:
            row[name] = float(value)
        except (TypeError, ValueError):
            return jsonify({"error": f"Giá trị của '{name}' không phải là số."}), 400

    model = LOADED_MODELS[model_key]
    sample = pd.DataFrame([row])[FEATURES]
    prediction = int(model.predict(sample)[0])
    confidence = float(np.max(model.predict_proba(sample)[0]))

    knowledge, knowledge_error = ([], None)
    if prediction == 1:
        knowledge, knowledge_error = get_diabetes_knowledge()

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "model": model_key,
        "model_name": METADATA["models"][model_key]["display_name"],
        "input": row,
        "knowledge": knowledge,
        "knowledge_error": knowledge_error,
    })


if __name__ == "__main__":
    print(f"Ứng dụng 1 - Dự đoán tiểu đường | http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
