"""REST API + giao diện web cho hệ thống dự đoán nguy cơ tiểu đường (Phần I - Bài tập 1).

Nạp weight đã lưu ở mục 18 của notebook (`models/*.sav` + `models/metadata.json`) nên dữ liệu
mới đi qua đúng pipeline tiền xử lý như lúc huấn luyện.

Chạy:
    cd "src/Assignment 01/diabetes"
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

# --- Models ---
with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES = METADATA["deploy_features"]
DEFAULT_MODEL = METADATA.get("default_model", "knn")

LOADED_MODELS = {}
for key, info in METADATA["models"].items():
    model_path = MODEL_DIR / info["file"]
    try:
        with open(model_path, "rb") as f:
            LOADED_MODELS[key] = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] Chưa có weight {model_path} - hãy chạy notebook mục 18 trước.")

# --- Neo4j Config (tuỳ chọn) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

neo4j_driver = None
if NEO4J_PASSWORD:
    try:
        from neo4j import GraphDatabase

        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
    except Exception as exc:  # thiếu package hoặc không kết nối được
        print(f"[WARN] Không khởi tạo được Neo4j driver: {exc}")

KNOWLEDGE_GROUPS = {
    "HAS_DIET_ADVICE": "Chế độ ăn",
    "HAS_LIFESTYLE_ACTION": "Lối sống và vận động",
    "HAS_DRUG_INFO": "Thông tin thuốc",
    "HAS_COMPLICATION": "Biến chứng cần lưu ý",
}


def get_diabetes_knowledge():
    """Read grouped diabetes knowledge and article sources from Neo4j."""
    if neo4j_driver is None:
        return [], "Neo4j password has not been configured (NEO4J_PASSWORD)."

    query = """
    MATCH (d:Disease {id: 'diabetes'})-[relation:HAS_DIET_ADVICE|HAS_LIFESTYLE_ACTION|HAS_DRUG_INFO|HAS_COMPLICATION]->(item)
    OPTIONAL MATCH (item)-[:SOURCED_FROM]->(article:Article)
    RETURN type(relation) AS relation_type,
           coalesce(item.title, item.name) AS title,
           item.content AS content,
           item.duration AS duration,
           item.frequency AS frequency,
           collect(DISTINCT {title: article.title, url: article.url}) AS sources
    ORDER BY relation_type, title
    """
    try:
        grouped_items = defaultdict(list)
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            for record in session.run(query):
                sources = [s for s in record["sources"] if s.get("title")]
                grouped_items[record["relation_type"]].append({
                    "title": record["title"],
                    "content": record["content"],
                    "duration": record["duration"],
                    "frequency": record["frequency"],
                    "sources": sources,
                })

        result = []
        for rel_type, label in KNOWLEDGE_GROUPS.items():
            if rel_type in grouped_items:
                result.append({"group_label": label, "items": grouped_items[rel_type]})
        return result, None
    except Exception as e:
        return [], str(e)


@app.route("/", methods=["GET"])
def home():
    if (BASE_DIR / "templates" / "index.html").exists():
        return render_template(
            "index.html",
            models=[
                {"key": key, "name": info["display_name"], "metrics": info["test_metrics"]}
                for key, info in METADATA["models"].items()
            ],
            default_model=DEFAULT_MODEL,
        )
    return "Diabetes Prediction API is running"


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
        return jsonify({"error": "Model not found. Hãy chạy notebook mục 18 để sinh weight."}), 500

    features = request.get_json(silent=True) or {}
    model_key = features.get("model", DEFAULT_MODEL)
    if model_key not in LOADED_MODELS:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400

    # Mặc định cho các chỉ số người dùng bỏ trống: trung bình của bộ dữ liệu huấn luyện
    defaults = {"Pregnancies": 3, "SkinThickness": 29, "Insulin": 155}
    try:
        row = {}
        for name in FEATURES:
            value = features.get(name)
            if value in (None, ""):
                if name not in defaults:
                    raise KeyError(name)
                value = defaults[name]
            row[name] = float(value)
    except (KeyError, TypeError, ValueError):
        required = [f for f in FEATURES if f not in defaults]
        return jsonify({
            "error": f"Missing or invalid required fields ({', '.join(required)})"
        }), 400

    model = LOADED_MODELS[model_key]
    sample = pd.DataFrame([row])[FEATURES]
    prediction = model.predict(sample)
    confidence = model.predict_proba(sample)

    has_diabetes = int(prediction[0]) == 1

    knowledge, knowledge_error = [], None
    if has_diabetes:
        knowledge, knowledge_error = get_diabetes_knowledge()

    return jsonify({
        "prediction": int(prediction[0]),
        "confidence": str(round(float(np.amax(confidence[0])) * 100, 2)),
        "model": model_key,
        "model_name": METADATA["models"][model_key]["display_name"],
        "knowledge": knowledge,
        "knowledge_error": knowledge_error,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        use_reloader=False,
    )
