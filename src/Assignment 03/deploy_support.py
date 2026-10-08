#!/usr/bin/env python3
"""Shared CLI and localhost UI for Assignment 03 dataset deploy apps."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPECS = {
    "diabetes": {
        "label": "Diabetes (Pima)",
        "models": ["NumPy MLP (8-32-16-1)"],
        "fields": [
            ("Pregnancies", "number"), ("Glucose", "number"), ("BloodPressure", "number"),
            ("SkinThickness", "number"), ("Insulin", "number"), ("BMI", "number"),
            ("DiabetesPedigreeFunction", "number"), ("Age", "number"),
        ],
    },
    "diabetes_large": {
        "label": "Diabetes Prediction", "models": ["Logistic Regression", "Gaussian Naive Bayes", "Decision Tree", "NumPy MLP (64-32-16)"],
        "fields": [("gender", "select:Female|Male|Other"), ("age", "number"), ("hypertension", "select:0|1"), ("heart_disease", "select:0|1"), ("smoking_history", "select:never|No Info|current|former|ever|not current"), ("bmi", "number"), ("HbA1c_level", "number"), ("blood_glucose_level", "number")],
    },
    "house_price": {
        "label": "House Price", "models": ["Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor", "NumPy MLP (128-64-32)"],
        "fields": [("status", "text"), ("bed", "number"), ("bath", "number"), ("acre_lot", "number"), ("city", "text"), ("state", "text"), ("zip_code", "text"), ("house_size", "number"), ("prev_sold_date", "text"), ("brokered_by", "text")],
    },
    "customer_comments": {
        "label": "Customer Review Recommendation", "models": ["Logistic Regression", "Multinomial Naive Bayes", "Random Forest", "NumPy MLP (1000-64-16-1)"],
        "fields": [("text", "textarea")],
    },
}


class HousePriceFeatureEngineer:
    def __init__(
        self,
        numeric_cols,
        low_card_categorical_cols,
        frequency_cols,
        date_col="prev_sold_date",
        missing_token="__missing__",
    ):
        self.numeric_cols = numeric_cols
        self.low_card_categorical_cols = low_card_categorical_cols
        self.frequency_cols = frequency_cols
        self.date_col = date_col
        self.missing_token = missing_token

    def fit(self, X):
        X = X.copy()
        self.numeric_medians_ = {}
        for col in self.numeric_cols:
            values = pd.to_numeric(X[col], errors="coerce")
            median = values.median()
            self.numeric_medians_[col] = 0.0 if pd.isna(median) else float(median)

        self.low_card_categories_ = {}
        for col in self.low_card_categorical_cols:
            values = X[col].astype("string").fillna(self.missing_token).astype(str)
            self.low_card_categories_[col] = values.value_counts().index.tolist()

        self.frequency_maps_ = {}
        self.frequency_defaults_ = {}
        train_size = len(X)
        for col in self.frequency_cols:
            values = X[col].astype("string").fillna(self.missing_token).astype(str)
            counts = values.value_counts()
            self.frequency_maps_[col] = counts.to_dict()
            self.frequency_defaults_[col] = 1.0

        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        max_year = dates.dt.year.max()
        self.reference_year_ = int(max_year) if pd.notna(max_year) else 2026

        features = self._build_features(X)
        self.feature_columns_ = features.columns.tolist()
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        features = self._build_features(X.copy())
        features = features.reindex(columns=self.feature_columns_, fill_value=0.0)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features.astype(np.float32)

    def _build_features(self, X):
        features = pd.DataFrame(index=X.index)

        for col in self.numeric_cols:
            values = pd.to_numeric(X[col], errors="coerce")
            missing = values.isna().astype(np.float32)
            filled = values.fillna(self.numeric_medians_[col]).astype(np.float64)
            clipped_nonnegative = np.clip(filled, 0, None)

            features[col] = filled
            features[f"{col}_missing"] = missing
            if col in {"acre_lot", "house_size"}:
                features[f"log1p_{col}"] = np.log1p(clipped_nonnegative)

        for col in self.frequency_cols:
            values = X[col].astype("string").fillna(self.missing_token).astype(str)
            counts = values.map(self.frequency_maps_[col]).astype(float)
            counts = counts.fillna(self.frequency_defaults_[col])
            features[f"{col}_count"] = counts
            features[f"{col}_log_count"] = np.log1p(counts)
            features[f"{col}_frequency"] = counts / max(sum(self.frequency_maps_[col].values()), 1)

        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        year = dates.dt.year.astype("float64")
        month = dates.dt.month.astype("float64")
        missing_date = dates.isna().astype(np.float32)

        features["prev_sold_year"] = year.fillna(self.reference_year_)
        features["prev_sold_month"] = month.fillna(0.0)
        features["years_since_prev_sold"] = (self.reference_year_ - year).clip(lower=0).fillna(0.0)
        features["prev_sold_missing"] = missing_date

        for col in self.low_card_categorical_cols:
            values = X[col].astype("string").fillna(self.missing_token).astype(str)
            known_categories = set(self.low_card_categories_[col])
            for category in self.low_card_categories_[col]:
                features[f"{col}={category}"] = (values == category).astype(np.float32)
            features[f"{col}=__unknown__"] = (~values.isin(known_categories)).astype(np.float32)

        return features

def _bundle(dataset: str) -> dict[str, Any]:
    import joblib
    filename = "sklearn_bundle.joblib" if dataset == "diabetes_large" else "model_bundle.joblib"
    path = Path(__file__).resolve().parent / dataset / "models" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the matching notebook to create model artifacts first.")
    if dataset == "house_price":
        setattr(sys.modules["__main__"], "HousePriceFeatureEngineer", HousePriceFeatureEngineer)
    return joblib.load(path)


def _numpy_mlp_forward(x: np.ndarray, model_dir: Path, metadata: dict, probability: bool) -> float:
    state = np.load(model_dir / "numpy_mlp_weights.npz")
    current = np.asarray(x, dtype=np.float64)
    n_layers = len(metadata["numpy_layer_sizes"]) - 1
    for idx in range(n_layers):
        current = current @ state[f"weight_{idx}"] + state[f"bias_{idx}"]
        if idx < n_layers - 1:
            current = np.maximum(current, 0)
    if probability:
        current = 1 / (1 + np.exp(-np.clip(current, -500, 500)))
    return float(np.asarray(current).reshape(-1)[0])


def _diabetes_artifacts(dataset: str):
    path = Path(__file__).resolve().parent / dataset / "models"
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    weights = np.load(path / "diabetes_improved_weights.npz")
    return path, metadata, weights


def _numpy_forward(x: np.ndarray, model: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)).reshape(-1)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(x)).reshape(-1)
    current = np.asarray(x, dtype=np.float64)
    for w, b in zip(model["weights"][:-1], model["biases"][:-1]):
        current = np.maximum(current @ w + b, 0)
    logits = current @ model["weights"][-1] + model["biases"][-1]
    return (1 / (1 + np.exp(-np.clip(logits, -500, 500)))).reshape(-1)


def predict(dataset: str, payload: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    if dataset == "diabetes":
        model_dir, metadata, weights = _diabetes_artifacts(dataset)
        x = np.asarray([[float(payload[name]) for name in metadata["features"]]], dtype=np.float64)
        if not np.isfinite(x).all(): raise ValueError("All feature values must be finite numbers.")
        for pos, idx in enumerate(metadata["zero_as_missing_indices"]):
            if x[0, idx] == 0: x[0, idx] = weights["zero_imputation_medians"][pos]
        x = (x - weights["mean"]) / weights["std"]
        params = {name: weights[f"param_{name}"] for name in ("W1", "b1", "W2", "b2", "W3", "b3")}
        current = np.maximum(x @ params["W1"] + params["b1"], 0)
        current = np.maximum(current @ params["W2"] + params["b2"], 0)
        probability = float((1 / (1 + np.exp(-np.clip(current @ params["W3"] + params["b3"], -500, 500))))[0, 0])
        return {"model": metadata["target"] + " NumPy MLP", "probability": probability, "prediction": int(probability >= metadata["threshold"]), "class_name": "Diabetes" if probability >= metadata["threshold"] else "No diabetes"}

    model_dir = Path(__file__).resolve().parent / dataset / "models"
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    bundle = _bundle(dataset)
    model_name = model_name or SPECS[dataset]["models"][0]
    if model_name not in SPECS[dataset]["models"]: raise ValueError(f"Unknown model: {model_name}")
    if dataset == "diabetes_large":
        import pandas as pd
        row = pd.DataFrame([{k: payload[k] for k in metadata["features"]}])
        x = bundle["preprocessor"].transform(row)
        if model_name == "NumPy MLP (64-32-16)":
            probability = _numpy_mlp_forward(x, model_dir, metadata, probability=True)
        else:
            model = bundle["baseline_models"][model_name]
            probability = float(model.predict_proba(x)[0, 1])
        threshold = metadata["thresholds"].get(model_name, 0.5)
        return {"model": model_name, "probability": probability, "threshold": threshold, "prediction": int(probability >= threshold), "class_name": "Diabetes" if probability >= threshold else "No diabetes"}
    if dataset == "house_price":
        import pandas as pd
        row = pd.DataFrame([{k: payload.get(k) for k in metadata["input_columns"]}])
        x = bundle["feature_engineer"].transform(row)
        x = bundle["scaler"].transform(x).astype(np.float32)
        if model_name == "NumPy MLP (128-64-32)":
            log_price = _numpy_mlp_forward(x, model_dir, metadata, probability=False)
        else:
            model = bundle["baseline_models"][model_name]
            log_price = float(np.asarray(model.predict(x)).reshape(-1)[0])
        price = float(np.expm1(np.clip(log_price, 0, bundle["max_log_prediction"])))
        return {"model": model_name, "predicted_price": price, "currency": "USD"}
    if dataset == "customer_comments":
        text = str(payload.get("text", ""))
        cleaned = re.sub(r"<[^>]+>|https?://\S+|www\.\S+", " ", text.lower())
        cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned: raise ValueError("Please enter a non-empty review.")
        x = bundle["tfidf"].transform([cleaned])
        if model_name == "NumPy MLP (1000-64-16-1)":
            probability = _numpy_mlp_forward(x.toarray().astype(np.float32), model_dir, metadata, probability=True)
        else:
            model = bundle["baseline_models"][model_name]
            probability = float(model.predict_proba(x)[0, 1])
        return {"model": model_name, "probability": probability, "prediction": int(probability >= 0.5), "class_name": "Recommended" if probability >= 0.5 else "Not recommended"}
    raise ValueError("Unknown dataset")


def run(dataset: str, default_port: int) -> int:
    spec = SPECS[dataset]
    parser = argparse.ArgumentParser(description=f"Assignment 03 deploy: {spec['label']}")
    parser.add_argument("--model", choices=spec["models"], default=spec["models"][0])
    parser.add_argument("--input", help="JSON object containing input features.")
    parser.add_argument("--web", action="store_true", help="Start localhost web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    args=parser.parse_args()
    if args.web:
        from flask import Flask, jsonify, render_template_string, request
        app=Flask(__name__)
        fields=json.dumps(spec["fields"])
        models=json.dumps(spec["models"])
        page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'''+spec["label"]+'''</title><style>body{font:16px system-ui;background:#0b1220;color:#e8eef9;margin:0}main{max-width:760px;margin:36px auto;padding:26px;background:#141f32;border:1px solid #293a56;border-radius:20px}h1{margin-top:0}label{display:block;margin:14px 0 6px;color:#b7c7df}input,textarea,select{box-sizing:border-box;width:100%;padding:11px;border-radius:9px;border:1px solid #405472;background:#0d1727;color:#fff;font:inherit}textarea{min-height:130px}button{margin-top:18px;border:0;border-radius:9px;padding:12px 18px;background:#83e2bd;color:#092017;font-weight:800;cursor:pointer}.result-panel{display:grid;gap:12px;margin-top:22px;padding:20px;border:1px solid #354967;border-radius:16px;background:linear-gradient(145deg,#101d2f,#0c1725);box-shadow:0 14px 35px #03081366}.result-head{display:flex;align-items:center;gap:10px;color:#a8bdd8;font-size:.78rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}.result-dot{width:9px;height:9px;border-radius:50%;background:#83e2bd;box-shadow:0 0 16px #83e2bd}.result-value{font-size:clamp(1.5rem,5vw,2.4rem);font-weight:850;line-height:1.1;color:#f4f8ff;overflow-wrap:anywhere}.result-meta{color:#9fb1cb;font-size:.92rem}.meter{height:9px;border-radius:99px;background:#26364e;overflow:hidden}.meter-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,#4fd1a1,#a7f3c5);transition:width .35s}.result-error{border-color:#713a4b;background:#291622;color:#ffc4cc}.result-wait{color:#c3d2e7}.actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap}button{transition:transform .15s,filter .15s}button:hover{filter:brightness(1.08);transform:translateY(-1px)}button:disabled{opacity:.6;cursor:wait;transform:none}</style><main><small>ASSIGNMENT 03 · LOCAL INFERENCE</small><h1>'''+spec["label"]+'''</h1><label>Model</label><select id="model"></select><div id="fields"></div><div class="actions"><button id="predict-button" onclick="predict()">✦ Dự đoán</button><span class="result-meta">Kết quả sẽ hiển thị tại đây</span></div><section id="result" class="result-panel"><div class="result-head"><i class="result-dot"></i> Sẵn sàng</div><div class="result-meta">Nhập dữ liệu rồi nhấn Dự đoán.</div></section></main><script>const fields='''+fields+''',models='''+models+''';const ms=document.querySelector('#model'),fs=document.querySelector('#fields');models.forEach(x=>ms.add(new Option(x,x)));fields.forEach(([n,t])=>{let l=document.createElement('label');l.textContent=n;fs.appendChild(l);let e=t==='textarea'?document.createElement('textarea'):t.startsWith('select:')?document.createElement('select'):document.createElement('input');e.id='f_'+n;if(t==='number'){e.type='number';e.step='any';e.value='0'}if(t.startsWith('select:'))t.slice(7).split('|').forEach(v=>e.add(new Option(v,v)));fs.appendChild(e)});async function predict(){const box=document.querySelector('#result'),button=document.querySelector('#predict-button');let p={};fields.forEach(([n])=>p[n]=document.querySelector('#f_'+n).value);button.disabled=true;box.innerHTML='<div class=\"result-head\"><i class=\"result-dot\"></i> Đang dự đoán</div><div class=\"result-meta\">Đang xử lý dữ liệu và chạy mô hình…</div>';try{let r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:ms.value,input:p})});let d=await r.json();if(!r.ok)throw new Error(d.error||'Không thể dự đoán.');box.className='result-panel';box.innerHTML='';let head=document.createElement('div');head.className='result-head';head.innerHTML='<i class=\"result-dot\"></i> Kết quả dự đoán';box.appendChild(head);let value=document.createElement('div');value.className='result-value';value.textContent=d.predicted_price!==undefined?new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(d.predicted_price):d.class_name;box.appendChild(value);let meta=document.createElement('div');meta.className='result-meta';meta.textContent='Mô hình: '+d.model+(d.probability!==undefined?' · Xác suất: '+(100*d.probability).toFixed(1)+'%':'');box.appendChild(meta);if(d.probability!==undefined){let meter=document.createElement('div');meter.className='meter';let fill=document.createElement('div');fill.className='meter-fill';fill.style.width=(100*d.probability)+'%';meter.appendChild(fill);box.appendChild(meter)}}catch(e){box.className='result-panel result-error';box.textContent='Không thể dự đoán: '+e.message}finally{button.disabled=false}}</script></html>'''
        @app.get('/')
        def home(): return render_template_string(page)
        @app.post('/api/predict')
        def api_predict():
            try:
                data=request.get_json(force=True); return jsonify(predict(dataset,data.get('input',{}),data.get('model')))
            except Exception as exc: return jsonify({"error":str(exc)}),400
        app.run(host=args.host,port=args.port,debug=False,use_reloader=False)
        return 0
    if not args.input: parser.error("Provide --input JSON or use --web")
    try: result=predict(dataset,json.loads(args.input),args.model)
    except Exception as exc: raise SystemExit(str(exc)) from exc
    print(json.dumps(result,ensure_ascii=False)); return 0
