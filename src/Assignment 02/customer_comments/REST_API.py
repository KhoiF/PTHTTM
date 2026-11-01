"""REST API + giao diện web cho Ứng dụng 3 - Hành vi khách hàng (Assignment 02, mục 4.4).

Nạp `tfidf_vectorizer.sav` cùng `models/*.sav` đã lưu ở mục 10 của notebook. API nhận chuỗi
Review Text, chạy `clean_text`, transform bằng vectorizer rồi trả về nhãn và xác suất khuyến
nghị.

`clean_text` phải GIỐNG HỆT hàm trong notebook — nếu khác, văn bản lúc inference được chuẩn hoá
khác lúc huấn luyện và vector TF-IDF sẽ lệch.

Chạy:
    cd "src/Assignment 02/customer_comments"
    python REST_API.py          # http://127.0.0.1:8082
"""
import json
import pickle
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PORT = 8082

with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

DEFAULT_MODEL = METADATA.get("default_model", "logistic_regression")
LABELS = METADATA.get("labels", {"0": "Không khuyến nghị", "1": "Khuyến nghị"})

with open(MODEL_DIR / METADATA["vectorizer"], "rb") as f:
    VECTORIZER = pickle.load(f)

LOADED_MODELS = {}
for key, info in METADATA["models"].items():
    path = MODEL_DIR / info["file"]
    try:
        with open(path, "rb") as f:
            LOADED_MODELS[key] = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] Chưa có weight {path} - hãy chạy notebook mục 10 trước.")


def clean_text(text):
    """Chuẩn hoá văn bản trước khi TF-IDF (giống hệt hàm trong notebook)."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def top_contributing_terms(model, vector, limit=6):
    """Các n-gram đẩy dự đoán mạnh nhất - chỉ có với mô hình tuyến tính."""
    if not hasattr(model, "coef_"):
        return []

    coefficients = model.coef_[0]
    vocabulary = VECTORIZER.get_feature_names_out()
    contributions = [
        (vocabulary[index], float(vector[0, index] * coefficients[index]))
        for index in vector.nonzero()[1]
    ]
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    return [
        {"term": term, "weight": round(weight, 4),
         "direction": "positive" if weight > 0 else "negative"}
        for term, weight in contributions[:limit]
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
    )


@app.route("/recommendation/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "default_model": DEFAULT_MODEL,
        "models": METADATA["models"],
        "vectorizer_params": METADATA.get("vectorizer_params", {}),
    })


@app.route("/recommendation/v1/predict", methods=["POST"])
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

    # Ghép Title và Review Text đúng như lúc huấn luyện
    raw_text = " ".join(filter(None, [
        str(payload.get("Title") or "").strip(),
        str(payload.get("Review Text") or payload.get("text") or "").strip(),
    ])).strip()

    if not raw_text:
        return jsonify({"success": False, "error": "Thiếu nội dung 'Review Text'."}), 400

    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify({
            "success": False,
            "error": "Review không còn từ nào sau khi làm sạch (chỉ nhận tiếng Anh).",
        }), 400

    vector = VECTORIZER.transform([cleaned])
    model = LOADED_MODELS[model_key]
    prediction = int(model.predict(vector)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(vector)[0][1])
    else:
        probability = None

    matched = int(vector.nnz)

    return jsonify({
        "success": True,
        "prediction": prediction,
        "label": LABELS[str(prediction)],
        "probability": round(probability * 100, 2) if probability is not None else None,
        "model": model_key,
        "model_name": METADATA["models"][model_key]["display_name"],
        "metrics": METADATA["models"][model_key]["test_metrics"],
        "cleaned_text": cleaned,
        "matched_terms": matched,
        "top_terms": top_contributing_terms(model, vector),
    })


if __name__ == "__main__":
    print(f"Ứng dụng 3 - Hành vi khách hàng | http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
