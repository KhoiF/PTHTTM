"""REST API + giao diện web cho Ứng dụng 2 - Dự đoán giá nhà (Assignment 02, mục 3.4).

Nạp `preprocessor.sav` (đã fit trên dữ liệu huấn luyện) và `models/*.sav` đã lưu ở mục 10 của
notebook. App KHÔNG fit lại scaler/encoder lúc chạy — đây là điểm mục 5.1 của báo cáo nhấn mạnh
để tránh rò rỉ dữ liệu khi inference.

Chạy:
    cd "src/Assignment 02/house_price"
    python REST_API.py          # http://127.0.0.1:8081
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PORT = 8081

with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES = METADATA["features"]
NUMERIC_FEATURES = METADATA["numeric_features"]
CATEGORY_FEATURES = METADATA["category_features"]
DEFAULT_MODEL = METADATA.get("default_model", "decision_tree")
TARGET_UNIT = METADATA.get("target_unit", "tỷ VNĐ")

with open(MODEL_DIR / METADATA["preprocessor"], "rb") as f:
    PREPROCESSOR = pickle.load(f)

LOADED_MODELS = {}
for key, info in METADATA["models"].items():
    path = MODEL_DIR / info["file"]
    try:
        with open(path, "rb") as f:
            LOADED_MODELS[key] = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] Chưa có weight {path} - hãy chạy notebook mục 10 trước.")

# Nhãn tiếng Việt cho các giá trị phân loại có trong dataset
CATEGORY_LABELS = {
    "Legal status": {
        "Have certificate": "Đã có sổ đỏ / sổ hồng",
        "Sale contract": "Hợp đồng mua bán",
        "Unknown": "Chưa rõ",
    },
    "Furniture state": {
        "Full": "Nội thất đầy đủ",
        "Basic": "Nội thất cơ bản",
        "Unknown": "Chưa rõ",
    },
}


def category_choices(column):
    """Danh sách (giá trị, nhãn hiển thị) cho một cột phân loại."""
    labels = CATEGORY_LABELS.get(column, {})
    return [
        {"value": value, "label": labels.get(value, value)}
        for value in METADATA["category_options"][column]
    ]


@app.route("/")
def home():
    return render_template(
        "index.html",
        models=[
            {"key": key, "name": info["display_name"], "metrics": info["test_metrics"]}
            for key, info in METADATA["models"].items()
        ],
        default_model=DEFAULT_MODEL,
        legal_options=category_choices("Legal status"),
        furniture_options=category_choices("Furniture state"),
        unit=TARGET_UNIT,
    )


@app.route("/house/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "features": FEATURES,
        "default_model": DEFAULT_MODEL,
        "models": METADATA["models"],
        "category_options": METADATA["category_options"],
    })


@app.route("/house/v1/predict", methods=["POST"])
def predict():
    if not LOADED_MODELS:
        return jsonify({
            "success": False,
            "error": "Chưa có weight. Hãy chạy notebook mục 10 để sinh file .sav.",
        }), 500

    payload = request.get_json(silent=True) or {}
    model_key = payload.get("model", DEFAULT_MODEL)
    if model_key not in LOADED_MODELS:
        return jsonify({"success": False, "error": f"Unknown model '{model_key}'."}), 400

    row = {}
    for name in NUMERIC_FEATURES:
        value = payload.get(name)
        if value in (None, ""):
            return jsonify({"success": False, "error": f"Thiếu giá trị cho '{name}'."}), 400
        try:
            row[name] = float(value)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": f"Giá trị của '{name}' không phải là số."}), 400
    for name in CATEGORY_FEATURES:
        row[name] = payload.get(name) or "Unknown"

    sample = pd.DataFrame([row])[FEATURES]
    sample_processed = PREPROCESSOR.transform(sample)
    price = float(LOADED_MODELS[model_key].predict(sample_processed)[0])

    return jsonify({
        "success": True,
        "price": round(price, 2),
        "unit": TARGET_UNIT,
        "model": model_key,
        "model_name": METADATA["models"][model_key]["display_name"],
        "metrics": METADATA["models"][model_key]["test_metrics"],
        "input": row,
    })


if __name__ == "__main__":
    print(f"Ứng dụng 2 - Dự đoán giá nhà | http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
