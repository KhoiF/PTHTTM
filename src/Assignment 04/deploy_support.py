#!/usr/bin/env python3
"""Shared local inference app for Assignment 04 CNN notebooks."""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

LABELS = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}
MODELS = ["NumPy CNN from scratch", "TensorFlow/Keras CNN", "PyTorch CNN"]
_CACHE: dict[tuple[str, str], Any] = {}


def paths(dataset: str):
    base = Path(__file__).resolve().parent / dataset
    model_dir = base / "models"
    if not (model_dir / "metadata.json").exists():
        raise FileNotFoundError(f"Missing {model_dir / 'metadata.json'}. Run the matching notebook to create model artifacts first.")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return base, model_dir, metadata


def load_model(dataset: str, framework: str):
    base, model_dir, metadata = paths(dataset)
    key = (dataset, framework)
    if key in _CACHE:
        return _CACHE[key]
    if framework == "NumPy CNN from scratch":
        model = np.load(model_dir / metadata["artifacts"]["numpy"])
    elif framework == "TensorFlow/Keras CNN":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_dir / metadata["artifacts"]["keras_model"], compile=False)
    elif framework == "PyTorch CNN":
        import torch
        model = torch.jit.load(str(model_dir / metadata["artifacts"]["torchscript"]), map_location="cpu").eval()
    else:
        raise ValueError(f"Unknown model: {framework}")
    _CACHE[key] = model
    return model


def _numpy_cnn_logits(state: Any, batch: np.ndarray) -> np.ndarray:
    def conv(images, key):
        w = state[f"{key}.weight"]
        b = state[f"{key}.bias"]
        kh, kw = w.shape[:2]
        padded = np.pad(images, ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(1, 2))
        patches = windows.transpose(0, 1, 2, 4, 5, 3)
        return np.maximum(np.einsum("nhwklc,klco->nhwo", patches, w, optimize=True) + b, 0)
    def pool(images):
        n, h, w, c = images.shape
        return images.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))
    current = pool(conv(batch, "conv1"))
    current = pool(conv(current, "conv2"))
    current = current.reshape(len(current), -1)
    current = np.maximum(current @ state["hidden.weight"] + state["hidden.bias"], 0)
    return current @ state["output.weight"] + state["output.bias"]


def infer(dataset: str, raw_image: np.ndarray, framework: str) -> dict[str, Any]:
    _, _, metadata = paths(dataset)
    expected = tuple(metadata["input_shape"])
    raw = np.asarray(raw_image, dtype=np.float32)
    if raw.shape != expected:
        raise ValueError(f"Expected image shape {expected}; received {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("Image contains non-finite pixels.")
    if raw.min() < 0 or raw.max() > 255:
        raise ValueError("Pixel values must be in the 0-255 range.")
    image = raw / 255.0
    mean = np.asarray(metadata["channel_mean"], dtype=np.float32).reshape((1, 1, -1))
    std = np.asarray(metadata["channel_std"], dtype=np.float32).reshape((1, 1, -1))
    image = (image - mean) / std
    batch = image[None, ...]
    model = load_model(dataset, framework)
    if framework == "NumPy CNN from scratch":
        logits = _numpy_cnn_logits(model, batch)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    elif framework == "TensorFlow/Keras CNN":
        import tensorflow as tf
        logits = model.predict(batch, verbose=0)
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
    else:
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2)).copy()).float()
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    scores = np.asarray(probabilities)[0]
    class_id = int(np.argmax(scores))
    return {"framework": framework, "class_id": class_id, "class_name": LABELS[dataset][class_id], "confidence": float(scores[class_id]), "probabilities": [float(x) for x in scores]}


def load_cifar_test_data(data_dir: Path):
    candidates = [data_dir / "cifar-10-batches-py" / "test_batch", data_dir / "cifar-10-python_extracted" / "cifar-10-batches-py" / "test_batch"]
    batch_path = next((p for p in candidates if p.is_file()), None)
    if batch_path is None:
        raise FileNotFoundError(f"CIFAR-10 test batch not found under {data_dir}. Run the notebook once to download the dataset.")
    with batch_path.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch.get(b"labels", batch.get(b"fine_labels")), dtype=int)
    return images, labels


def run(dataset: str, port: int) -> int:
    parser = argparse.ArgumentParser(description=f"Assignment 04 {dataset} CNN local inference")
    parser.add_argument("--web", action="store_true", help="Start localhost drawing/sample app")
    parser.add_argument("--framework", choices=MODELS, default=MODELS[0])
    parser.add_argument("--image", help="Path to an image file for CLI inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    base, _, metadata = paths(dataset)
    if args.web:
        from flask import Flask, jsonify, render_template_string, request
        app = Flask(__name__)
        options = "".join(f'<option>{name}</option>' for name in MODELS)
        if dataset == "mnist":
            widget = '<p>Vẽ số 0–9 bằng nét trắng trên nền đen.</p><canvas id="pad" width="280" height="280"></canvas><div><button id="clear">Xóa</button></div>'
            script = '''const c=document.querySelector('#pad'),ctx=c.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,280,280);ctx.strokeStyle='#fff';ctx.lineWidth=24;ctx.lineCap='round';let drawing=false;function pos(e){let r=c.getBoundingClientRect();return[(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height]}c.onpointerdown=e=>{drawing=true;c.setPointerCapture(e.pointerId);let [x,y]=pos(e);ctx.beginPath();ctx.moveTo(x,y)};c.onpointermove=e=>{if(!drawing)return;let[x,y]=pos(e);ctx.lineTo(x,y);ctx.stroke()};c.onpointerup=()=>drawing=false;document.querySelector('#clear').onclick=()=>{ctx.fillStyle='#000';ctx.fillRect(0,0,280,280)};function pixels(){let t=document.createElement('canvas');t.width=t.height=28;let x=t.getContext('2d');x.drawImage(c,0,0,28,28);return Array.from(x.getImageData(0,0,28,28).data).filter((_,i)=>i%4===0)}'''
        else:
            widget = '<div class="layout"><div class="sample"><canvas id="sample" width="320" height="320"></canvas><div class="result-meta" id="truth"></div></div><div><button id="another">↻ Ảnh ngẫu nhiên khác</button></div></div>'
            script = '''let sample=[];async function another(){let r=await fetch('/api/sample');let d=await r.json();sample=d.image;document.querySelector('#truth').textContent='Nhãn thật trong dataset: '+d.label;let c=document.querySelector('#sample'),x=c.getContext('2d'),im=x.createImageData(32,32);let flat=sample.flat(2);for(let i=0;i<1024;i++){im.data[i*4]=flat[i*3];im.data[i*4+1]=flat[i*3+1];im.data[i*4+2]=flat[i*3+2];im.data[i*4+3]=255}let t=document.createElement('canvas');t.width=t.height=32;t.getContext('2d').putImageData(im,0,0);x.imageSmoothingEnabled=false;x.clearRect(0,0,320,320);x.drawImage(t,0,0,320,320)}document.querySelector('#another').onclick=another;another();function pixels(){return sample}'''
        page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Assignment 04 '''+dataset+'''</title><style>body{font:16px system-ui;background:#101827;color:#eaf0fb;margin:0}main{max-width:760px;margin:36px auto;padding:clamp(20px,5vw,34px);background:linear-gradient(145deg,#1a2638,#141f30);border:1px solid #2a3a52;border-radius:22px;box-shadow:0 24px 70px #050914aa}h1{margin:8px 0 24px;font-size:clamp(1.8rem,5vw,2.5rem)}small{color:#8de7c2;font-weight:800;letter-spacing:.12em}canvas{width:280px;height:280px;image-rendering:pixelated;background:#000;border:2px solid #50627c;border-radius:8px;touch-action:none}.sample canvas{width:320px;height:320px}select{padding:10px;background:#101827;color:#fff;border:1px solid #50627c;border-radius:8px}button{padding:11px 17px;margin:10px 8px 10px 0;border:0;border-radius:10px;background:linear-gradient(110deg,#85e4bd,#a4f0c9);color:#10251d;font-weight:800;cursor:pointer;transition:transform .15s,filter .15s}button:hover{filter:brightness(1.06);transform:translateY(-1px)}button:disabled{opacity:.6;cursor:wait;transform:none}.result-card{display:grid;gap:13px;margin-top:18px;padding:20px;border:1px solid #354967;border-radius:16px;background:linear-gradient(145deg,#111d2f,#0d1726);box-shadow:0 14px 35px #05091466}.result-kicker{color:#9db4d2;font-size:.76rem;font-weight:850;letter-spacing:.13em;text-transform:uppercase}.result-class{font-size:clamp(2rem,7vw,3.3rem);font-weight:900;line-height:1;color:#f1f7ff}.result-meta{color:#aabbd2;font-size:.92rem}.confidence-track,.class-track{height:9px;overflow:hidden;border-radius:99px;background:#26364d}.confidence-fill,.class-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,#51d5a5,#aff3c9);transition:width .35s}.top-classes{display:grid;gap:9px;margin-top:3px}.class-row{display:grid;grid-template-columns:minmax(88px,1fr) 2fr 48px;align-items:center;gap:10px;color:#c6d3e6;font-size:.86rem}.class-fill{background:linear-gradient(90deg,#6497ff,#77d9d2)}.result-error{border-color:#713a4b;background:#291622;color:#ffc4cc}.result-wait{color:#c4d2e5}.layout{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;align-items:start}.sample{display:grid;gap:8px}.sample canvas{width:min(320px,70vw);height:min(320px,70vw)}@media(max-width:600px){main{margin:12px;border-radius:16px}.layout{grid-template-columns:1fr}.sample canvas{width:min(280px,76vw);height:min(280px,76vw)}}</style><main><small>ASSIGNMENT 04 · LOCAL INFERENCE</small><h1>'''+("MNIST digit recognizer" if dataset=="mnist" else "CIFAR-10 image classifier")+'''</h1><label>Model: <select id="model">'''+options+'''</select></label>'''+widget+'''<div><button id="predict">Dự đoán</button></div><section id="result" class="result-card"><div class="result-kicker">Sẵn sàng</div><div class="result-meta">Chọn hoặc vẽ ảnh, sau đó nhấn Dự đoán.</div></section></main><script>const classNames='''+json.dumps(LABELS[dataset])+''';'''+script+'''document.querySelector('#predict').onclick=async()=>{const box=document.querySelector('#result'),button=document.querySelector('#predict');button.disabled=true;box.className='result-card result-wait';box.innerHTML='<div class=\"result-kicker\">Đang dự đoán</div><div class=\"result-meta\">Mô hình đang phân tích ảnh…</div>';try{let r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({framework:document.querySelector('#model').value,image:pixels()})});let d=await r.json();if(!r.ok)throw new Error(d.error||'Không thể dự đoán.');box.className='result-card';box.innerHTML='';let kicker=document.createElement('div');kicker.className='result-kicker';kicker.textContent='Kết quả dự đoán';box.appendChild(kicker);let name=document.createElement('div');name.className='result-class';name.textContent=d.class_name;box.appendChild(name);let meta=document.createElement('div');meta.className='result-meta';meta.textContent='Độ tin cậy '+(100*d.confidence).toFixed(1)+'% · '+d.framework;box.appendChild(meta);let track=document.createElement('div');track.className='confidence-track';let fill=document.createElement('div');fill.className='confidence-fill';fill.style.width=(100*d.confidence)+'%';track.appendChild(fill);box.appendChild(track);let list=document.createElement('div');list.className='top-classes';d.probabilities.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p).slice(0,3).forEach(item=>{let row=document.createElement('div');row.className='class-row';let label=document.createElement('span');label.textContent=classNames[item.i];let bar=document.createElement('div');bar.className='class-track';let barFill=document.createElement('div');barFill.className='class-fill';barFill.style.width=(100*item.p)+'%';bar.appendChild(barFill);let percent=document.createElement('span');percent.textContent=(100*item.p).toFixed(1)+'%';row.append(label,bar,percent);list.appendChild(row)});box.appendChild(list)}catch(e){box.className='result-card result-error';box.textContent='Không thể dự đoán: '+e.message}finally{button.disabled=false}}</script></html>'''
        @app.get("/")
        def home(): return render_template_string(page)
        @app.post("/api/predict")
        def api_predict():
            try:
                data = request.get_json(force=True)
                shape = tuple(metadata["input_shape"])
                pixels = np.asarray(data.get("image"), dtype=np.float32)
                image = pixels.reshape(shape) if pixels.size == int(np.prod(shape)) else pixels
                return jsonify(infer(dataset, image, data.get("framework", MODELS[0])))
            except Exception as exc: return jsonify({"error": str(exc)}), 400
        if dataset == "cifar10":
            @app.get("/api/sample")
            def api_sample():
                try:
                    images, labels = load_cifar_test_data(base / "data")
                    index = random.randrange(len(images))
                    return jsonify({"image": images[index].tolist(), "label": LABELS[dataset][int(labels[index])]})
                except Exception as exc: return jsonify({"error": str(exc)}), 400
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return 0
    if not args.image:
        parser.error("Use --web or provide --image PATH")
    try:
        from PIL import Image
        shape = tuple(metadata["input_shape"])
        mode = "L" if shape[-1] == 1 else "RGB"
        image = Image.open(args.image).convert(mode).resize((shape[1], shape[0]))
        array = np.asarray(image)
        if array.ndim == 2: array = array[..., None]
        print(json.dumps(infer(dataset, array, args.framework), ensure_ascii=False))
    except Exception as exc: raise SystemExit(str(exc)) from exc
    return 0
