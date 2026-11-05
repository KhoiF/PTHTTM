"""Giao diện Gradio cho hệ multi-agent dự đoán nguy cơ tiểu đường.

Chạy từ thư mục src/Project:
    python -m app.app
rồi mở http://127.0.0.1:8010
"""

from __future__ import annotations

import html
import threading
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyvis.network import Network

from agents.base import percent
from agents.graph import DISCLAIMER, build_graph, stream_case
from kb.schema import format_threshold
from ml.predictor import load_metadata

PORT = 8010
ASSETS = Path(__file__).parent / "assets"
SMOKING_CHOICES = [("Không rõ", "No Info"), ("Chưa từng hút", "never"), ("Đã bỏ", "former"),
                   ("Đang hút", "current"), ("Từng hút", "ever"), ("Hiện không hút", "not current")]
REL_VI = {"HAS_SYMPTOM": "triệu chứng", "HAS_RISK_FACTOR": "yếu tố nguy cơ", "HAS_COMPLICATION": "biến chứng",
          "DIAGNOSED_BY": "chẩn đoán bằng", "TREATED_BY": "điều trị bằng", "PREVENTED_BY": "phòng ngừa bằng",
          "BELONGS_TO": "thuộc nhóm", "HAS_SIDE_EFFECT": "tác dụng phụ", "CONTRAINDICATED_IN": "chống chỉ định"}
# Màu nút đích trong đồ thị tri thức theo loại quan hệ (nút nguồn luôn màu navy)
REL_COLORS = {"HAS_SYMPTOM": "#f59e0b", "HAS_RISK_FACTOR": "#8b5cf6", "HAS_COMPLICATION": "#f43f5e",
              "DIAGNOSED_BY": "#0ea5e9", "TREATED_BY": "#10b981", "PREVENTED_BY": "#14b8a6", "BELONGS_TO": "#64748b",
              "HAS_SIDE_EFFECT": "#ec4899", "CONTRAINDICATED_IN": "#ef4444"}
AGENT_STYLE = {"Supervisor": ("🧭", "#6366f1"), "Intake": ("📝", "#0ea5e9"), "Risk": ("📈", "#f43f5e"),
               "Knowledge": ("📚", "#10b981"), "Advisor": ("💬", "#f59e0b"), "Critic": ("🛡️", "#8b5cf6"),
               "Finalize": ("✅", "#14b8a6"), "SingleAgent": ("🤖", "#64748b")}

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.sky, neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Be Vietnam Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f3f6fa", body_background_fill_dark="#0b1220",
    block_background_fill="#ffffff", block_background_fill_dark="#111a2e",
    block_border_width="1px", block_border_color="#e2e8f0", block_border_color_dark="#1e293b",
    block_shadow="0 1px 2px rgba(15,23,42,.04), 0 6px 20px rgba(15,23,42,.05)", block_shadow_dark="none",
    block_label_text_color="#475569", block_label_text_color_dark="#94a3b8", block_title_text_weight="600",
    input_background_fill="#f8fafc", input_background_fill_dark="#0f172a",
    input_border_color="#e2e8f0", input_border_color_dark="#1e293b", input_border_color_focus="#14b8a6",
    button_primary_background_fill="linear-gradient(135deg, #0d9488, #0891b2)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0f766e, #0e7490)",
    button_primary_background_fill_dark="linear-gradient(135deg, #0d9488, #0891b2)",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    checkbox_label_background_fill_selected="#ccfbf1", checkbox_label_background_fill_selected_dark="#134e4a",
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_text_color="#334155", block_title_text_color_dark="#cbd5e1", block_title_padding="0",
    block_title_radius="0", block_title_text_size="*text_sm",
)

APP_CSS = """
:root { --ink:#0f172a; --muted:#64748b; --card:#ffffff; --soft:#f8fafc; --line:#e2e8f0; --teal:#0d9488; }
.dark { --ink:#e2e8f0; --muted:#94a3b8; --card:#111a2e; --soft:#0f172a; --line:#1e293b; --teal:#2dd4bf; }
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
.card, .card .styler { background: var(--block-background-fill) !important; }
.card { border-radius: 18px !important; }
#chat, #chat .wrapper, #chat .bubble-wrap { border: none !important; box-shadow: none !important; }
#chat .placeholder-content { display: grid; place-items: center; text-align: center; padding: 0 56px; color: var(--muted); }
#chat .placeholder h3 { color: var(--ink); margin-bottom: 6px; }
footer { display: none !important; }

/* Banner đầu trang */
.hero { background: radial-gradient(1200px 300px at 85% -40%, rgba(45,212,191,.35), transparent 60%),
        linear-gradient(120deg, #0b1f3a 0%, #0f3d4c 55%, #0d6e6e 100%);
        border-radius: 20px; padding: 28px 32px; color: #fff; box-shadow: 0 10px 30px rgba(11,31,58,.25); }
.hero .eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 600; color: #5eead4; }
.hero h1 { margin: 6px 0 8px; font-size: 1.9rem; font-weight: 700; line-height: 1.25; color: #fff; }
.hero p { margin: 0; max-width: 820px; color: #cbd5e1; font-size: .95rem; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.chip { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 999px;
        padding: 6px 14px; font-size: .82rem; color: #e2e8f0; backdrop-filter: blur(4px); }
.chip b { color: #fff; font-weight: 600; }

/* Tiêu đề từng thẻ */
.card-title { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 1.02rem; color: var(--ink); }
.card-title .icon { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
                    background: linear-gradient(135deg, #ccfbf1, #e0f2fe); font-size: 1rem; }
.dark .card-title .icon { background: linear-gradient(135deg, #134e4a, #0c4a6e); }
.card-sub { color: var(--muted); font-size: .82rem; margin: 2px 0 0 40px; }
.group-label { font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
               color: var(--teal); margin: 6px 0 -4px; }

/* Đường đi của agent */
.route { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 6px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
        font-size: .82rem; font-weight: 600; color: var(--c); background: color-mix(in srgb, var(--c) 12%, transparent);
        border: 1px solid color-mix(in srgb, var(--c) 35%, transparent); }
.arrow { color: var(--muted); font-size: .8rem; }
.pulse { animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; } }
.meta { color: var(--muted); font-size: .82rem; margin: 4px 0 14px; }
.timeline { list-style: none; margin: 0; padding: 0; }
.timeline li { position: relative; display: flex; gap: 14px; padding: 0 0 14px 0; }
.timeline li::before { content: ""; position: absolute; left: 17px; top: 36px; bottom: 0; width: 2px; background: var(--line); }
.timeline li:last-child::before { display: none; }
.timeline .dot { flex: 0 0 36px; height: 36px; border-radius: 12px; display: grid; place-items: center; font-size: 1rem;
                 background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent); }
.timeline .body { flex: 1; background: var(--soft); border: 1px solid var(--line); border-radius: 12px; padding: 9px 14px; }
.timeline .head { display: flex; justify-content: space-between; gap: 10px; font-size: .88rem; }
.timeline .head b { color: var(--c); }
.timeline .head span { color: var(--muted); font-size: .78rem; white-space: nowrap; }
.timeline .body p { margin: 3px 0 0; color: var(--ink); font-size: .86rem; }

/* Thẻ nguy cơ */
.risk-card { display: grid; grid-template-columns: auto 1fr; gap: 22px; align-items: center; padding: 6px 4px 12px; }
.risk-value { font-size: 2.6rem; font-weight: 700; line-height: 1; color: var(--c); }
.risk-value small { display: block; font-size: .78rem; font-weight: 500; color: var(--muted); margin-top: 6px; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600;
         color: var(--c); background: color-mix(in srgb, var(--c) 13%, transparent); }
.risk-info { color: var(--ink); font-size: .88rem; }
.risk-info div { margin-top: 6px; color: var(--muted); }
.bar { position: relative; height: 10px; border-radius: 999px; background: var(--line); margin-top: 12px; }
.bar .fill { position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--c); }
.bar .mark { position: absolute; top: -5px; width: 2px; height: 20px; background: var(--ink); border-radius: 2px; }
.bar-legend { display: flex; justify-content: space-between; font-size: .74rem; color: var(--muted); margin-top: 6px; }

/* Bằng chứng */
.ev { display: grid; grid-template-columns: 34px 1fr; gap: 12px; padding: 12px 14px; margin-bottom: 10px;
      background: var(--soft); border: 1px solid var(--line); border-radius: 14px; }
.ev .n { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; font-weight: 700;
         font-size: .82rem; color: #fff; background: linear-gradient(135deg, #0d9488, #0891b2); }
.ev .fact { font-weight: 600; color: var(--ink); font-size: .9rem; }
.ev blockquote { margin: 6px 0; padding: 2px 0 2px 12px; border-left: 3px solid var(--teal); color: var(--muted); font-size: .85rem; }
.ev a { font-size: .8rem; color: var(--teal); text-decoration: none; }
.ev a:hover { text-decoration: underline; }
.checks { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .86rem; margin-top: 6px; }
.checks th { text-align: left; color: var(--muted); font-weight: 600; font-size: .76rem; text-transform: uppercase;
             letter-spacing: .05em; padding: 6px 10px; border-bottom: 1px solid var(--line); }
.checks td { padding: 8px 10px; border-bottom: 1px solid var(--line); color: var(--ink); }
.ok { color: #059669; font-weight: 600; } .no { color: var(--muted); }
details summary { cursor: pointer; color: var(--muted); font-size: .84rem; margin: 10px 0 4px; }
.section-h { font-weight: 650; color: var(--ink); margin: 16px 0 8px; font-size: .95rem; }

/* Đồ thị tri thức */
.kg-frame { width: 100%; height: 470px; border: 1px solid var(--line); border-radius: 14px; background: #fff; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: .8rem; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                       background: var(--c); margin-right: 6px; vertical-align: -1px; }

.empty { color: var(--muted); font-size: .9rem; padding: 18px 4px; }
.disclaimer { background: color-mix(in srgb, #f59e0b 10%, transparent); border: 1px solid color-mix(in srgb, #f59e0b 35%, transparent);
              border-radius: 14px; padding: 12px 16px; color: var(--ink); font-size: .85rem; }
"""

GRAPH = build_graph()


def kg_size() -> tuple[int, int] | None:
    """Số thực thể và quan hệ của KG trong Neo4j (không tính đoạn văn, nguồn và liên kết kỹ thuật)."""
    try:
        from kb.load_neo4j import get_driver, stats
        with get_driver() as driver:
            nodes, rels = stats(driver)
        entities = int(nodes[~nodes["type"].isin(["Chunk", "Source"])]["count"].sum())
        relations = int(rels[~rels["relation"].isin(["MENTIONS", "FROM"])]["count"].sum())
        return entities, relations
    except Exception:
        return None


NUMBER_FIELDS = {"age": "Tuổi", "bmi": "BMI", "HbA1c_level": "HbA1c", "blood_glucose_level": "Đường huyết"}


def to_number(text) -> float | None:
    """'6,2' hoặc '6.2' → 6.2; ô trống hoặc không phải số → None."""
    text = str(text if text is not None else "").strip().replace(",", ".")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def form_dict(gender, age, bmi, hypertension, heart_disease, smoking, hba1c, glucose) -> dict:
    values = {"gender": gender, "age": to_number(age), "bmi": to_number(bmi),
              "hypertension": None if hypertension is None else int(hypertension == "Có"),
              "heart_disease": None if heart_disease is None else int(heart_disease == "Có"),
              "smoking_history": smoking if smoking != "No Info" else None,
              "HbA1c_level": to_number(hba1c), "blood_glucose_level": to_number(glucose)}
    return {k: v for k, v in values.items() if v not in (None, "")}


def invalid_numbers(age, bmi, hba1c, glucose) -> list[str]:
    raw = dict(zip(NUMBER_FIELDS, (age, bmi, hba1c, glucose)))
    return [NUMBER_FIELDS[k] for k, v in raw.items() if str(v or "").strip() and to_number(v) is None]


def _esc(value) -> str:
    return html.escape(str(value))


def _pill(agent: str) -> str:
    icon, color = AGENT_STYLE.get(agent, ("•", "#64748b"))
    return f'<span class="pill" style="--c:{color}">{icon} {_esc(agent)}</span>'


def trace_html(state: dict, running: bool = False) -> str:
    trace = state.get("trace") or []
    if not trace:
        return '<div class="empty">⏳ Đang khởi động các agent…</div>' if running else \
               '<div class="empty">Chưa có lượt chạy nào. Gửi một câu hỏi để xem các agent phối hợp.</div>'
    pills = '<span class="arrow">→</span>'.join(_pill(t["agent"]) for t in trace)
    if running:
        pills += '<span class="arrow">→</span><span class="pill pulse" style="--c:#64748b">⏳ đang xử lý</span>'
    calls = sum(t["usage"]["calls"] for t in trace)
    meta = (f'Đang chạy · {len(trace)} bước · {calls} lượt LLM' if running else
            f'Hoàn tất trong <b>{state.get("total_seconds", 0)}s</b> · {len(trace)} bước · {calls} lượt LLM')
    items = []
    for t in trace:
        icon, color = AGENT_STYLE.get(t["agent"], ("•", "#64748b"))
        items.append(f'<li style="--c:{color}"><span class="dot">{icon}</span><div class="body">'
                     f'<div class="head"><b>{_esc(t["agent"])}</b><span>{t["seconds"]}s · {t["usage"]["calls"]} LLM</span></div>'
                     f'<p>{_esc(t["decision"])}</p></div></li>')
    return f'<div class="route">{pills}</div><div class="meta">{meta}</div><ol class="timeline">{"".join(items)}</ol>'


def risk_card(state: dict) -> str:
    risk = state.get("risk")
    if not risk:
        if state.get("trace"):
            return '<div class="empty">Câu hỏi kiến thức: không cần chạy mô hình dự đoán nguy cơ.</div>'
        return '<div class="empty">Điền hồ sơ hoặc mô tả sức khỏe trong khung chat để xem nguy cơ.</div>'
    probability, threshold = risk["probability"], risk["threshold"]
    if risk.get("grey_zone"):
        color, status = "#f59e0b", "Vùng xám (gần ngưỡng)"
    elif risk.get("label"):
        color, status = "#f43f5e", "Nguy cơ cao"
    else:
        color, status = "#10b981", "Nguy cơ thấp"
    model = "Mô hình đầy đủ (có HbA1c / đường huyết)" if risk["model_used"] == "full" else \
            "Mô hình sàng lọc (chưa có xét nghiệm)"
    return (f'<div class="risk-card" style="--c:{color}">'
            f'<div class="risk-value">{percent(probability)}<small>xác suất mắc tiểu đường</small></div>'
            f'<div class="risk-info"><span class="badge">{status}</span>'
            f'<div>{model} · ngưỡng phân loại {percent(threshold)}</div>'
            f'<div class="bar"><div class="fill" style="width:{probability * 100:.1f}%"></div>'
            f'<div class="mark" style="left:calc({threshold * 100:.1f}% - 1px)" title="ngưỡng"></div></div>'
            f'<div class="bar-legend"><span>0%</span><span>▲ ngưỡng {percent(threshold)}</span><span>100%</span></div>'
            f'</div></div>')


def risk_figure(state: dict):
    risk = state.get("risk")
    if not risk or not risk.get("top_features"):
        return None
    features = list(reversed(risk["top_features"]))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#f43f5e" if f["shap"] > 0 else "#10b981" for f in features]
    ax.barh([f"{f['label']} = {f['value']}" for f in features], [f["shap"] for f in features], color=colors, height=0.6)
    ax.axvline(0, color="#94a3b8", lw=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=9, length=0)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Yếu tố ảnh hưởng nhiều nhất (SHAP)", loc="left", fontsize=11, color="#0f172a", fontweight="bold")
    ax.set_xlabel("đỏ: làm tăng nguy cơ · xanh: làm giảm nguy cơ", fontsize=8.5, color="#64748b")
    fig.tight_layout()
    return fig


def kg_html(state: dict) -> str:
    triples = [e["triple"] for e in state.get("evidence") or [] if e.get("triple")]
    if not triples:
        return '<div class="empty">Chưa có quan hệ nào từ Knowledge Graph cho câu hỏi này.</div>'
    net = Network(height="450px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    # Bố cục ổn định xong mới vẽ và tự căn khung; lực hút về tâm giữ các cụm rời gần nhau để nhãn đủ lớn
    net.set_options("""{"physics": {"solver": "forceAtlas2Based",
                                    "forceAtlas2Based": {"gravitationalConstant": -110, "centralGravity": 0.012,
                                                         "springLength": 150, "springConstant": 0.06,
                                                         "avoidOverlap": 0.6},
                                    "stabilization": {"iterations": 300, "fit": true}},
                        "edges": {"smooth": {"type": "continuous"}},
                        "interaction": {"hover": true}}""")
    heads = {head for head, _, _ in triples}
    font = {"face": "Arial, Helvetica, sans-serif", "size": 13, "color": "#0f172a", "strokeWidth": 4, "strokeColor": "#ffffff"}
    for head, relation, tail in triples:
        net.add_node(head, label=head, color="#0b3b5c", shape="dot", size=16, font=font)
        if tail not in heads:
            net.add_node(tail, label=tail, color=REL_COLORS.get(relation, "#64748b"), shape="dot", size=11, font=font)
        net.add_edge(head, tail, label=REL_VI.get(relation, relation), arrows="to", color="#cbd5e1",
                     font={"size": 10, "color": "#64748b", "strokeWidth": 0})
    used = dict.fromkeys(rel for _, rel, _ in triples)
    legend = "".join(f'<span style="--c:{REL_COLORS.get(rel, "#64748b")}">{_esc(REL_VI.get(rel, rel))}</span>' for rel in used)
    frame = f'<iframe class="kg-frame" srcdoc="{html.escape(net.generate_html())}"></iframe>'
    return frame + f'<div class="legend"><span style="--c:#0b3b5c">thực thể chính</span>{legend}</div>'


def evidence_html(state: dict) -> str:
    evidence = state.get("evidence") or []
    if not evidence:
        return '<div class="empty">Chưa có bằng chứng nào được truy xuất.</div>'
    parts = ['<div class="section-h">📚 Bằng chứng từ Nhà thuốc Long Châu</div>']
    for e in evidence:
        url = e.get("source_url") or ""
        title = e.get("title") or url
        parts.append(f'<div class="ev"><div class="n">{e["n"]}</div><div><div class="fact">{_esc(e["fact"])}</div>'
                     f'<blockquote>{_esc((e.get("quote") or "")[:320])}</blockquote>'
                     f'<a href="{_esc(url)}" target="_blank">↗ {_esc(title)}</a></div></div>')
    checks = state.get("lab_checks") or []
    if checks:
        rows, seen = {"met": [], "other": []}, set()
        for c in sorted(checks, key=lambda c: not c["met"]):
            rng = format_threshold(c["op"], c["value"], c.get("value_max"), bool(c.get("max_exclusive")))
            key = (c["test"], c["condition"], rng, c["unit"])
            if key in seen:
                continue
            seen.add(key)
            status = '<span class="ok">● đạt</span>' if c["met"] else '<span class="no">○ không đạt</span>'
            rows["met" if c["met"] else "other"].append(
                f'<tr><td>{_esc(c["test"])}</td><td>{c["patient_value_converted"]} {_esc(c["unit"])}</td>'
                f'<td>{_esc(c["condition"])} ({_esc(rng)} {_esc(c["unit"])})</td><td>{status}</td></tr>')
        head = "<tr><th>Xét nghiệm</th><th>Của bạn</th><th>Ngưỡng theo Long Châu</th><th></th></tr>"
        parts.append('<div class="section-h">🧪 Đối chiếu chỉ số với ngưỡng chẩn đoán</div>')
        if rows["met"]:
            parts.append(f'<table class="checks">{head}{"".join(rows["met"])}</table>')
        if rows["other"]:
            parts.append(f'<details><summary>Các ngưỡng khác chưa đạt ({len(rows["other"])})</summary>'
                         f'<table class="checks">{head}{"".join(rows["other"])}</table></details>')
    return "".join(parts)


def respond(message, history, patient_state, gender, age, bmi, hypertension, heart_disease, smoking, hba1c, glucose):
    message = (message or "").strip()
    form = form_dict(gender, age, bmi, hypertension, heart_disease, smoking, hba1c, glucose)
    if bad := invalid_numbers(age, bmi, hba1c, glucose):
        gr.Warning(f"Bỏ qua ô không phải số: {', '.join(bad)}.")
    if not message and len(form) < 3:
        gr.Warning("Hãy nhập câu hỏi hoặc điền hồ sơ bên trái.")
        yield history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), patient_state, ""
        return
    shown = message or "Đánh giá nguy cơ theo hồ sơ đã nhập."
    history = history + [{"role": "user", "content": shown}, {"role": "assistant", "content": "⏳ Đang xử lý…"}]
    state = {}
    for state in stream_case(message, form, history_patient=patient_state, app=GRAPH):
        if state.get("draft_preview"):          # Advisor đang viết: hiện chữ dần, không vẽ lại dòng thời gian
            history[-1]["content"] = state["draft_preview"] + "\n\n*✍️ Advisor đang viết…*"
            yield history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), patient_state, ""
            continue
        if state.get("draft") and not state.get("answer"):
            history[-1]["content"] = state["draft"] + "\n\n*🛡️ Critic đang kiểm tra bản nháp…*"
        elif not state.get("draft"):
            done = " → ".join(t["agent"] for t in state.get("trace") or [])
            history[-1]["content"] = f"⏳ Đang xử lý… {done}"
        yield history, trace_html(state, running=True), gr.skip(), gr.skip(), gr.skip(), gr.skip(), patient_state, ""
    history[-1]["content"] = state.get("answer") or "Xin lỗi, hệ thống chưa tạo được câu trả lời."
    new_patient = state.get("patient") or patient_state
    yield (history, trace_html(state), risk_card(state), risk_figure(state), kg_html(state), evidence_html(state),
           new_patient, "")


def assess_profile(_message, *args):
    """Nút "Đánh giá theo hồ sơ": bỏ qua ô tin nhắn, chỉ dùng dữ liệu form."""
    yield from respond("", *args)


def hero_html() -> str:
    meta = load_metadata()
    full = meta["feature_sets"]["full"]["test_metrics"]
    size = kg_size()
    chips = [f"<b>XGBoost</b> · PR-AUC {full['PR-AUC']:.3f}".replace(".", ","),
             f"<b>Knowledge Graph</b> · {size[0]:,} thực thể · {size[1]:,} quan hệ".replace(",", ".") if size else
             "<b>Knowledge Graph</b> · Nhà thuốc Long Châu",
             "<b>6 agent</b> · LangGraph", "<b>LLM local</b> · Ollama"]
    return ('<div class="hero"><div class="eyebrow">Hệ đa agent · ML/DL + Knowledge Graph</div>'
            '<h1>🩺 Trợ lý đánh giá nguy cơ tiểu đường</h1>'
            '<p>Mô hình học máy ước lượng nguy cơ từ hồ sơ sức khỏe; Knowledge Graph dựng từ Nhà thuốc Long Châu giải thích '
            'kết quả và đối chiếu ngưỡng chẩn đoán; sáu agent (Supervisor, Intake, Risk, Knowledge, Advisor, Critic) '
            'phối hợp và tự kiểm tra câu trả lời.</p>'
            f'<div class="chips">{"".join(f"<span class=chip>{c}</span>" for c in chips)}</div></div>')


def card_title(icon: str, title: str, sub: str = "") -> str:
    return (f'<div class="card-title"><span class="icon">{icon}</span>{title}</div>'
            + (f'<div class="card-sub">{sub}</div>' if sub else ""))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Trợ lý nguy cơ tiểu đường") as demo:
        gr.HTML(hero_html())
        patient_state = gr.State({})
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                with gr.Group(elem_classes="card"):
                    gr.HTML(card_title("🧾", "Hồ sơ sức khỏe", "Không bắt buộc, có thể kể trong khung chat"))
                    gr.HTML('<div class="group-label">Thông tin chung</div>')
                    gender = gr.Radio([("Nữ", "Female"), ("Nam", "Male")], label="Giới tính")
                    with gr.Row():
                        age = gr.Textbox(label="Tuổi", placeholder="vd. 45", max_lines=1)
                        bmi = gr.Textbox(label="BMI", placeholder="vd. 24,5", max_lines=1)
                    with gr.Row():
                        hypertension = gr.Radio(["Có", "Không"], label="Tăng huyết áp")
                        heart_disease = gr.Radio(["Có", "Không"], label="Bệnh tim")
                    smoking = gr.Dropdown(SMOKING_CHOICES, value="No Info", label="Tiền sử hút thuốc")
                    gr.HTML('<div class="group-label">Xét nghiệm (nếu có)</div>')
                    with gr.Row():
                        hba1c = gr.Textbox(label="HbA1c (%)", placeholder="vd. 6,2", max_lines=1)
                        glucose = gr.Textbox(label="Đường huyết (mg/dL)", placeholder="vd. 140", max_lines=1)
                assess_btn = gr.Button("📊 Đánh giá theo hồ sơ", variant="primary", size="lg")
            with gr.Column(scale=7):
                with gr.Group(elem_classes="card"):
                    gr.HTML(card_title("💬", "Trò chuyện với trợ lý",
                                       "Hỏi về nguy cơ của bạn, triệu chứng, xét nghiệm, chế độ ăn hay thuốc"))
                    chatbot = gr.Chatbot(
                        elem_id="chat", show_label=False, height=470, layout="bubble",
                        avatar_images=(str(ASSETS / "user.svg"), str(ASSETS / "assistant.svg")),
                        placeholder="### Xin chào 👋\nHãy mô tả hồ sơ sức khỏe của bạn hoặc đặt một câu hỏi về bệnh "
                                    "tiểu đường. Mỗi câu trả lời đều kèm nguồn từ Nhà thuốc Long Châu.")
                    with gr.Row():
                        message = gr.Textbox(placeholder="Ví dụ: Tôi nữ 52 tuổi, BMI 28, HbA1c 6.7%… nguy cơ thế nào?",
                                             show_label=False, scale=6, container=False)
                        send = gr.Button("Gửi ➤", variant="primary", scale=1, min_width=90)
                gr.Examples(["Triệu chứng của bệnh tiểu đường tuýp 2 là gì?",
                             "Tôi nam 50 tuổi, BMI 29, không tăng huyết áp, không bệnh tim, không hút thuốc. "
                             "Tôi có nguy cơ tiểu đường không?",
                             "Tôi là nữ, 52 tuổi, cao 1m58 nặng 70kg, có cao huyết áp, không bệnh tim, không hút thuốc. "
                             "HbA1c 6.7%, đường huyết 150 mg/dL. Nguy cơ của tôi thế nào?",
                             "Metformin có tác dụng phụ gì?"], inputs=message, label="Gợi ý câu hỏi")
        with gr.Group(elem_classes="card"):
            gr.HTML(card_title("🔍", "Bên trong hệ thống", "Cách các agent đi đến câu trả lời gần nhất"))
            with gr.Tabs():
                with gr.Tab("🧭 Đường đi của agent"):
                    trace = gr.HTML(trace_html({}))
                with gr.Tab("📈 Nguy cơ & SHAP"):
                    risk_view = gr.HTML(risk_card({}))
                    risk_plot = gr.Plot(show_label=False)
                with gr.Tab("🕸️ Đồ thị tri thức"):
                    graph_view = gr.HTML('<div class="empty">Đồ thị con của KG sẽ hiện ở đây sau khi có câu trả lời.</div>')
                with gr.Tab("📚 Bằng chứng"):
                    evidence = gr.HTML('<div class="empty">Câu trích từ Long Châu và phần đối chiếu ngưỡng sẽ hiện ở đây.</div>')
        gr.HTML(f'<div class="disclaimer">{_esc(DISCLAIMER.lstrip("> "))}</div>')

        inputs = [message, chatbot, patient_state, gender, age, bmi, hypertension, heart_disease, smoking, hba1c, glucose]
        outputs = [chatbot, trace, risk_view, risk_plot, graph_view, evidence, patient_state, message]
        send.click(respond, inputs, outputs)
        message.submit(respond, inputs, outputs)
        assess_btn.click(assess_profile, inputs, outputs)
    return demo


def warm_up() -> None:
    """Nạp sẵn LLM và model embedding vào bộ nhớ để câu hỏi đầu tiên không phải chờ (chạy nền khi mở app)."""
    try:
        from agents.llm import LLM
        llm = LLM()
        llm.client.chat(model=llm.model, messages=[{"role": "user", "content": "xin chào"}], keep_alive="30m",
                        options={"num_predict": 1, "num_ctx": llm.num_ctx})   # cùng num_ctx để Ollama không nạp lại
        llm.embed(["khởi động"])
    except Exception as error:
        print(f"Không nạp sẵn được model: {error}")


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    build_ui().launch(server_name="127.0.0.1", server_port=PORT, theme=THEME, css=APP_CSS,
                      allowed_paths=[str(ASSETS)])
