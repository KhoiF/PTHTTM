"""Flask web app cho hệ thống dự đoán giá nhà (Phần II - Bài tập 1).

Nạp weight đã lưu ở mục 18 của notebook (`models/*.sav`, `models/preprocessor.sav`,
`models/metadata.json`). App KHÔNG fit lại scaler/encoder lúc chạy — dữ liệu mới đi qua đúng
preprocessor đã fit trên tập huấn luyện.

Chạy:
    cd "src/Assignment 01/house_price"
    python flask_app.py          # http://127.0.0.1:8081
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES = METADATA["features"]
NUMERIC_FEATURES = METADATA["numeric_features"]
CATEGORY_FEATURES = METADATA["category_features"]
DEFAULT_MODEL = METADATA.get("default_model", "decision_tree")

with open(MODEL_DIR / METADATA["preprocessor"], "rb") as f:
    PREPROCESSOR = pickle.load(f)

LOADED_MODELS = {}
for key, info in METADATA["models"].items():
    model_path = MODEL_DIR / info["file"]
    try:
        with open(model_path, "rb") as f:
            LOADED_MODELS[key] = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] Chưa có weight {model_path} - hãy chạy notebook mục 18 trước.")

# Nhãn tiếng Việt cho các giá trị phân loại có trong dataset
CATEGORY_LABELS = {
    "Legal status": {
        "Have certificate": "Đã có sổ đỏ/sổ hồng",
        "Sale contract": "Hợp đồng mua bán",
        "Unknown": "Chưa rõ",
    },
    "Furniture state": {
        "Full": "Đầy đủ (Full)",
        "Basic": "Cơ bản (Basic)",
        "Unknown": "Chưa rõ",
    },
}


def category_choices(column):
    """Danh sách (value, nhãn hiển thị) cho một cột phân loại."""
    labels = CATEGORY_LABELS.get(column, {})
    return [
        {"value": value, "label": labels.get(value, value)}
        for value in METADATA["category_options"][column]
    ]


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=[
            {"key": key, "name": info["display_name"], "metrics": info["test_metrics"]}
            for key, info in METADATA["models"].items()
        ],
        default_model=DEFAULT_MODEL,
        legal_options=category_choices("Legal status"),
        furniture_options=category_choices("Furniture state"),
    )


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "features": FEATURES,
        "default_model": DEFAULT_MODEL,
        "models": METADATA["models"],
        "category_options": METADATA["category_options"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not LOADED_MODELS:
        return jsonify({
            "success": False,
            "error": "Chưa có weight. Hãy chạy notebook mục 18 trước.",
        }), 500

    data = request.get_json(silent=True) or {}
    model_key = data.get("model", DEFAULT_MODEL)
    if model_key not in LOADED_MODELS:
        return jsonify({"success": False, "error": f"Unknown model '{model_key}'."}), 400

    try:
        row = {}
        for name in NUMERIC_FEATURES:
            value = data.get(name)
            if value in (None, ""):
                raise ValueError(f"Thiếu giá trị cho '{name}'.")
            row[name] = float(value)
        for name in CATEGORY_FEATURES:
            row[name] = data.get(name) or "Unknown"

        sample = pd.DataFrame([row])[FEATURES]
        sample_processed = PREPROCESSOR.transform(sample)
        predicted_price = float(LOADED_MODELS[model_key].predict(sample_processed)[0])

        return jsonify({
            "success": True,
            "price": round(predicted_price, 2),
            "unit": METADATA["target_unit"],
            "model": model_key,
            "model_name": METADATA["models"][model_key]["display_name"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
