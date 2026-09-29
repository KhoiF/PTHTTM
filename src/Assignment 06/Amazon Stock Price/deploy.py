#!/usr/bin/env python3
"""CLI va web app localhost du bao gia dong cua Amazon bang RNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_CACHE: dict[tuple[str, str], object] = {}


def parse_prices(raw: str) -> list[float]:
    try:
        prices = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--prices chi duoc chua cac so, cach nhau bang dau phay.") from exc
    if not prices:
        raise argparse.ArgumentTypeError("--prices khong duoc de trong.")
    if any(not math.isfinite(value) for value in prices):
        raise argparse.ArgumentTypeError("--prices phai la cac so huu han.")
    if any(value < 0 for value in prices):
        raise argparse.ArgumentTypeError("Gia khong duoc am.")
    return prices


def build_torch_model(config: dict):
    import torch.nn as nn

    class PriceRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.RNN(
                input_size=config["input_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                nonlinearity="tanh",
                batch_first=True,
            )
            self.dropout = nn.Dropout(config["dropout"])
            self.output = nn.Linear(config["hidden_size"], 1)

        def forward(self, inputs):
            sequence_output, _ = self.rnn(inputs)
            return self.output(self.dropout(sequence_output[:, -1, :]))

    return PriceRNN()


def build_keras_model(config: dict, lookback: int):
    from tensorflow import keras

    inputs = keras.Input(shape=(lookback, config["input_size"]), name="price_sequence")
    hidden = keras.layers.SimpleRNN(
        config["hidden_size"], activation="tanh", name="simple_rnn"
    )(inputs)
    hidden = keras.layers.Dropout(config["dropout"], name="dropout")(hidden)
    outputs = keras.layers.Dense(1, name="price_output")(hidden)
    return keras.Model(inputs, outputs, name="amazon_price_rnn")


def predict_pytorch(sequence: np.ndarray, metadata: dict) -> tuple[float, str]:
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache_key = ("pytorch", str(device))
    model = MODEL_CACHE.get(cache_key)
    if model is None:
        model = build_torch_model(metadata["model_config"]).to(device)
        checkpoint_path = MODEL_DIR / metadata["artifacts"]["pytorch"]
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        MODEL_CACHE[cache_key] = model
    tensor = torch.from_numpy(sequence).float().to(device)
    with torch.no_grad():
        prediction = model(tensor).cpu().numpy().reshape(-1)[0]
    return float(prediction), str(device)


def predict_keras(sequence: np.ndarray, metadata: dict) -> tuple[float, str]:
    import tensorflow as tf

    cache_key = ("keras", "tensorflow")
    model = MODEL_CACHE.get(cache_key)
    if model is None:
        model = build_keras_model(metadata["model_config"], metadata["lookback"])
        model.load_weights(MODEL_DIR / metadata["artifacts"]["keras"])
        MODEL_CACHE[cache_key] = model
    prediction = model.predict(sequence, verbose=0).reshape(-1)[0]
    devices = tf.config.list_physical_devices("GPU")
    return float(prediction), "tensorflow-gpu" if devices else "tensorflow-cpu"


def inverse_scale(value: float, scaler: dict) -> float:
    return value * scaler["scale"] + scaler["mean"]


def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Khong tim thay models/metadata.json. Hay chay notebook huan luyen truoc."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def forecast(prices: Sequence[float], framework: str, metadata: dict | None = None) -> dict:
    metadata = metadata or load_metadata()
    lookback = int(metadata["lookback"])
    if framework not in {"pytorch", "keras"}:
        raise ValueError("Framework phai la 'pytorch' hoac 'keras'.")
    if len(prices) != lookback:
        raise ValueError(f"Can dung {lookback} gia, nhung nhan duoc {len(prices)}.")
    if any(not math.isfinite(float(value)) for value in prices):
        raise ValueError("Tat ca gia phai la so huu han.")
    if any(float(value) < 0 for value in prices):
        raise ValueError("Gia khong duoc am.")

    scaler = metadata["scaler"]
    values = np.asarray(prices, dtype=np.float32)
    scaled = (values - scaler["mean"]) / scaler["scale"]
    sequence = scaled.reshape(1, lookback, 1).astype(np.float32)
    if framework == "pytorch":
        scaled_prediction, device = predict_pytorch(sequence, metadata)
    else:
        scaled_prediction, device = predict_keras(sequence, metadata)
    prediction = inverse_scale(scaled_prediction, scaler)
    return {
        "framework": framework,
        "target": metadata["target"],
        "lookback": lookback,
        "predicted_price": round(float(prediction), 6),
        "device": device,
    }


def load_latest_prices(lookback: int) -> list[float]:
    data_path = BASE_DIR / "data" / "AMZN.csv"
    with data_path.open(encoding="utf-8", newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: row["Date"])
    if len(rows) < lookback:
        raise ValueError("Dataset khong du so luong gia cho cua so deploy.")
    return [float(row["Close"]) for row in rows[-lookback:]]


def create_web_app():
    from flask import Flask, jsonify, render_template_string, request

    metadata = load_metadata()
    lookback = int(metadata["lookback"])
    latest_prices = load_latest_prices(lookback)
    app = Flask(__name__)

    page = r'''<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Amazon RNN Forecast</title>
  <style>
    :root { color-scheme: dark; --bg:#08111f; --panel:#101d31; --line:#263b58; --text:#e8f0fc; --muted:#9fb0c8; --accent:#ffb44c; --accent2:#ff7a45; --ok:#65d6a8; --error:#ff7785; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:var(--text); background:radial-gradient(circle at 15% 10%,#18375c 0,transparent 32%),linear-gradient(145deg,#07101d,#0c1728 60%,#101b2b); }
    main { width:min(980px,calc(100% - 32px)); margin:42px auto; }
    .eyebrow { color:var(--accent); font-size:.78rem; letter-spacing:.18em; text-transform:uppercase; font-weight:800; }
    h1 { margin:.45rem 0; font-size:clamp(2rem,5vw,3.6rem); line-height:1; }
    .lead { color:var(--muted); max-width:720px; line-height:1.65; }
    .panel { margin-top:28px; padding:26px; border:1px solid var(--line); border-radius:22px; background:rgba(16,29,49,.9); box-shadow:0 24px 70px rgba(0,0,0,.28); }
    .row { display:grid; grid-template-columns:1fr auto; gap:16px; align-items:end; }
    label { display:block; margin-bottom:8px; color:var(--muted); font-weight:700; }
    select,textarea { width:100%; color:var(--text); background:#091525; border:1px solid var(--line); border-radius:12px; padding:12px; font:inherit; }
    select { min-width:170px; }
    textarea { min-height:190px; resize:vertical; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; line-height:1.55; }
    .actions { display:flex; gap:12px; flex-wrap:wrap; margin-top:18px; }
    button { border:0; border-radius:999px; padding:12px 20px; font-weight:800; cursor:pointer; }
    .primary { background:linear-gradient(110deg,var(--accent),var(--accent2)); color:#1b1004; }
    .secondary { background:#1a2b43; color:var(--text); border:1px solid var(--line); }
    button:disabled { opacity:.55; cursor:wait; }
    .result { margin-top:20px; min-height:94px; padding:18px; border-radius:15px; background:#0a1728; border:1px solid var(--line); }
    .price { color:var(--ok); font-size:2rem; font-weight:900; }
    .error { color:var(--error); font-weight:750; }
    .meta { color:var(--muted); margin-top:7px; }
    @media(max-width:650px){ .row{grid-template-columns:1fr;} main{margin-top:24px;} .panel{padding:18px;} }
  </style>
</head>
<body><main>
  <div class="eyebrow">Assignment 06 · Local inference</div>
  <h1>Amazon Price RNN</h1>
  <p class="lead">Dự báo giá đóng cửa của phiên kế tiếp từ {{ lookback }} giá Close gần nhất. Mọi inference chạy cục bộ bằng weights đã huấn luyện.</p>
  <section class="panel">
    <div class="row">
      <div><label for="prices">Chuỗi giá — cũ đến mới</label></div>
      <div><label for="framework">Framework</label><select id="framework"><option value="pytorch">PyTorch</option><option value="keras">Keras</option></select></div>
    </div>
    <textarea id="prices" spellcheck="false"></textarea>
    <div class="actions"><button class="secondary" id="latest">Nạp {{ lookback }} giá mới nhất</button><button class="primary" id="predict">Dự báo giá</button></div>
    <div class="result" id="result"><div class="meta">Sẵn sàng dự báo.</div></div>
  </section>
</main>
<script>
const latest = {{ latest_prices|tojson }};
const prices = document.getElementById('prices');
const result = document.getElementById('result');
function fillLatest(){ prices.value = latest.map(v => Number(v).toFixed(6)).join(', '); result.innerHTML='<div class="meta">Đã nạp dữ liệu mới nhất từ AMZN.csv.</div>'; }
document.getElementById('latest').addEventListener('click', fillLatest);
document.getElementById('predict').addEventListener('click', async () => {
  const button=document.getElementById('predict'); button.disabled=true; result.innerHTML='<div class="meta">Đang nạp model và dự báo…</div>';
  try {
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({framework:document.getElementById('framework').value,prices:prices.value})});
    const data=await response.json(); if(!response.ok) throw new Error(data.error||'Không thể dự báo.');
    result.innerHTML=`<div class="price">${Number(data.predicted_price).toFixed(6)} USD</div><div class="meta">${data.framework} · ${data.device} · lookback ${data.lookback}</div>`;
  } catch(error){ result.innerHTML=`<div class="error">${error.message}</div>`; } finally { button.disabled=false; }
});
fillLatest();
</script></body></html>'''

    @app.get("/")
    def index():
        return render_template_string(page, lookback=lookback, latest_prices=latest_prices)

    @app.post("/api/predict")
    def api_predict():
        try:
            payload = request.get_json(silent=True) or {}
            raw_prices = payload.get("prices", "")
            prices = parse_prices(raw_prices) if isinstance(raw_prices, str) else [float(v) for v in raw_prices]
            return jsonify(forecast(prices, payload.get("framework", "pytorch"), metadata))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def launch_web(host: str, port: int) -> None:
    app = create_web_app()
    print(f"Amazon web app: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=False)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Du bao gia Close Amazon cua phien tiep theo tu 60 gia gan nhat."
    )
    parser.add_argument(
        "--framework",
        choices=("pytorch", "keras"),
        help="Framework cua weights can dung.",
    )
    parser.add_argument(
        "--prices",
        type=parse_prices,
        help="Danh sach gia Close cach nhau bang dau phay, theo thu tu cu den moi.",
    )
    parser.add_argument("--web", action="store_true", help="Mo giao dien web tren localhost.")
    parser.add_argument("--host", default="127.0.0.1", help="Dia chi bind cua web app.")
    parser.add_argument("--port", type=int, default=8001, help="Cong cua web app (mac dinh: 8001).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.web:
        launch_web(args.host, args.port)
        return 0
    if args.framework is None or args.prices is None:
        parser.error("Che do CLI can ca --framework va --prices; hoac dung --web.")
    try:
        result = forecast(args.prices, args.framework)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
