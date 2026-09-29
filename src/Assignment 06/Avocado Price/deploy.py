#!/usr/bin/env python3
"""CLI va web app localhost du bao AveragePrice avocado bang RNN."""

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
GROUP_SEPARATOR = "|||"
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
    return keras.Model(inputs, outputs, name="avocado_price_rnn")


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


def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Khong tim thay models/metadata.json. Hay chay notebook huan luyen truoc."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def forecast(
    prices: Sequence[float],
    framework: str,
    region: str,
    avocado_type: str,
    metadata: dict | None = None,
) -> dict:
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

    group_key = f"{region}{GROUP_SEPARATOR}{avocado_type}"
    if group_key not in metadata["group_scalers"]:
        valid_regions = sorted({group["region"] for group in metadata["groups"]})
        preview = ", ".join(valid_regions[:12])
        raise ValueError(
            f"Khong ton tai nhom region={region!r}, type={avocado_type!r}. "
            f"Vi du region hop le: {preview}."
        )

    scaler = metadata["group_scalers"][group_key]
    values = np.asarray(prices, dtype=np.float32)
    scaled = (values - scaler["mean"]) / scaler["scale"]
    sequence = scaled.reshape(1, lookback, 1).astype(np.float32)
    if framework == "pytorch":
        scaled_prediction, device = predict_pytorch(sequence, metadata)
    else:
        scaled_prediction, device = predict_keras(sequence, metadata)
    prediction = scaled_prediction * scaler["scale"] + scaler["mean"]
    return {
        "framework": framework,
        "target": metadata["target"],
        "region": region,
        "type": avocado_type,
        "lookback": lookback,
        "predicted_price": round(float(prediction), 6),
        "device": device,
    }


def load_latest_prices(region: str, avocado_type: str, lookback: int) -> list[float]:
    data_path = BASE_DIR / "data" / "avocado.csv"
    with data_path.open(encoding="utf-8-sig", newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["region"] == region and row["type"] == avocado_type
        ]
    rows.sort(key=lambda row: row["Date"])
    if len(rows) < lookback:
        raise ValueError(f"Nhom {region} - {avocado_type} khong du {lookback} tuan du lieu.")
    return [float(row["AveragePrice"]) for row in rows[-lookback:]]


def create_web_app():
    from flask import Flask, jsonify, render_template_string, request

    metadata = load_metadata()
    lookback = int(metadata["lookback"])
    regions = sorted({group["region"] for group in metadata["groups"]})
    default_region = "TotalUS" if "TotalUS" in regions else regions[0]
    app = Flask(__name__)

    page = r'''<!doctype html>
<html lang="vi"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Avocado RNN Forecast</title>
  <style>
    :root { color-scheme:dark; --bg:#07130e; --panel:#10231a; --line:#294c39; --text:#ebf8ef; --muted:#a3bdae; --accent:#b9e769; --accent2:#55c88a; --ok:#c9f67d; --error:#ff8290; }
    * { box-sizing:border-box; } body { margin:0; min-height:100vh; font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:var(--text); background:radial-gradient(circle at 85% 8%,#285c3b 0,transparent 32%),linear-gradient(145deg,#06120c,#0b1d14 60%,#11271b); }
    main { width:min(1020px,calc(100% - 32px)); margin:42px auto; } .eyebrow { color:var(--accent); font-size:.78rem; letter-spacing:.18em; text-transform:uppercase; font-weight:800; }
    h1 { margin:.45rem 0; font-size:clamp(2rem,5vw,3.6rem); line-height:1; } .lead { color:var(--muted); max-width:760px; line-height:1.65; }
    .panel { margin-top:28px; padding:26px; border:1px solid var(--line); border-radius:22px; background:rgba(16,35,26,.91); box-shadow:0 24px 70px rgba(0,0,0,.3); }
    .grid { display:grid; grid-template-columns:1fr 1.6fr 1fr; gap:14px; margin-bottom:18px; } label { display:block; margin-bottom:8px; color:var(--muted); font-weight:700; }
    select,textarea { width:100%; color:var(--text); background:#081810; border:1px solid var(--line); border-radius:12px; padding:12px; font:inherit; }
    textarea { min-height:170px; resize:vertical; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; line-height:1.55; }
    .actions { display:flex; gap:12px; flex-wrap:wrap; margin-top:18px; } button { border:0; border-radius:999px; padding:12px 20px; font-weight:800; cursor:pointer; }
    .primary { background:linear-gradient(110deg,var(--accent),var(--accent2)); color:#10200f; } .secondary { background:#1a3828; color:var(--text); border:1px solid var(--line); }
    button:disabled { opacity:.55; cursor:wait; } .result { margin-top:20px; min-height:94px; padding:18px; border-radius:15px; background:#081910; border:1px solid var(--line); }
    .price { color:var(--ok); font-size:2rem; font-weight:900; } .error { color:var(--error); font-weight:750; } .meta { color:var(--muted); margin-top:7px; }
    @media(max-width:720px){ .grid{grid-template-columns:1fr;} main{margin-top:24px;} .panel{padding:18px;} }
  </style>
</head><body><main>
  <div class="eyebrow">Assignment 06 · Local inference</div><h1>Avocado Price RNN</h1>
  <p class="lead">Dự báo AveragePrice tuần kế tiếp từ {{ lookback }} tuần gần nhất. Chọn đúng region và type để dùng scaler tương ứng.</p>
  <section class="panel">
    <div class="grid">
      <div><label for="framework">Framework</label><select id="framework"><option value="pytorch">PyTorch</option><option value="keras">Keras</option></select></div>
      <div><label for="region">Region</label><select id="region">{% for region in regions %}<option value="{{ region }}" {% if region == default_region %}selected{% endif %}>{{ region }}</option>{% endfor %}</select></div>
      <div><label for="type">Type</label><select id="type"><option value="conventional">conventional</option><option value="organic">organic</option></select></div>
    </div>
    <label for="prices">{{ lookback }} AveragePrice — tuần cũ đến tuần mới</label><textarea id="prices" spellcheck="false"></textarea>
    <div class="actions"><button class="secondary" id="latest">Nạp {{ lookback }} tuần mới nhất</button><button class="primary" id="predict">Dự báo giá</button></div>
    <div class="result" id="result"><div class="meta">Sẵn sàng dự báo.</div></div>
  </section>
</main><script>
const prices=document.getElementById('prices'), result=document.getElementById('result');
async function fillLatest(){
  result.innerHTML='<div class="meta">Đang đọc dữ liệu…</div>';
  try { const params=new URLSearchParams({region:document.getElementById('region').value,type:document.getElementById('type').value}); const response=await fetch('/api/latest?'+params); const data=await response.json(); if(!response.ok) throw new Error(data.error); prices.value=data.prices.map(v=>Number(v).toFixed(4)).join(', '); result.innerHTML=`<div class="meta">Đã nạp ${data.region} — ${data.type}.</div>`; }
  catch(error){ result.innerHTML=`<div class="error">${error.message}</div>`; }
}
document.getElementById('latest').addEventListener('click',fillLatest);
document.getElementById('region').addEventListener('change',fillLatest); document.getElementById('type').addEventListener('change',fillLatest);
document.getElementById('predict').addEventListener('click',async()=>{
  const button=document.getElementById('predict'); button.disabled=true; result.innerHTML='<div class="meta">Đang nạp model và dự báo…</div>';
  try { const payload={framework:document.getElementById('framework').value,region:document.getElementById('region').value,type:document.getElementById('type').value,prices:prices.value}; const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const data=await response.json(); if(!response.ok) throw new Error(data.error||'Không thể dự báo.'); result.innerHTML=`<div class="price">${Number(data.predicted_price).toFixed(6)} USD</div><div class="meta">${data.region} · ${data.type} · ${data.framework} · ${data.device}</div>`; }
  catch(error){ result.innerHTML=`<div class="error">${error.message}</div>`; } finally { button.disabled=false; }
}); fillLatest();
</script></body></html>'''

    @app.get("/")
    def index():
        return render_template_string(
            page,
            lookback=lookback,
            regions=regions,
            default_region=default_region,
        )

    @app.get("/api/latest")
    def api_latest():
        try:
            region = request.args.get("region", default_region)
            avocado_type = request.args.get("type", "conventional")
            prices = load_latest_prices(region, avocado_type, lookback)
            return jsonify({"region": region, "type": avocado_type, "prices": prices})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/predict")
    def api_predict():
        try:
            payload = request.get_json(silent=True) or {}
            raw_prices = payload.get("prices", "")
            prices = parse_prices(raw_prices) if isinstance(raw_prices, str) else [float(v) for v in raw_prices]
            result = forecast(
                prices,
                payload.get("framework", "pytorch"),
                payload.get("region", default_region),
                payload.get("type", "conventional"),
                metadata,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def launch_web(host: str, port: int) -> None:
    app = create_web_app()
    print(f"Avocado web app: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=False)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Du bao AveragePrice avocado tu 12 muc gia hang tuan gan nhat."
    )
    parser.add_argument("--framework", choices=("pytorch", "keras"))
    parser.add_argument("--region", help="Region dung nhu trong avocado.csv.")
    parser.add_argument("--type", choices=("conventional", "organic"))
    parser.add_argument(
        "--prices",
        type=parse_prices,
        help="12 gia AveragePrice cach nhau bang dau phay, theo thu tu cu den moi.",
    )
    parser.add_argument("--web", action="store_true", help="Mo giao dien web tren localhost.")
    parser.add_argument("--host", default="127.0.0.1", help="Dia chi bind cua web app.")
    parser.add_argument("--port", type=int, default=8002, help="Cong cua web app (mac dinh: 8002).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.web:
        launch_web(args.host, args.port)
        return 0
    if any(value is None for value in (args.framework, args.region, args.type, args.prices)):
        parser.error(
            "Che do CLI can --framework, --region, --type va --prices; hoac dung --web."
        )
    try:
        result = forecast(args.prices, args.framework, args.region, args.type)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
