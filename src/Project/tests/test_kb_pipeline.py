from pathlib import Path

from kb.extract import refine_threshold, section_tasks, validate
from kb.import_html import detect_url, is_blocked
from kb.normalize import build_graph
from kb.parse import chunk_page, parse_page
from kb.seed import load_seeds

FIXTURE = Path(__file__).parent / "fixtures" / "sample_benh.html"
SEED = next(s for s in load_seeds() if s.url.endswith("tieu-duong-tuyp-2.html"))


def _page():
    return parse_page(FIXTURE.read_text(encoding="utf-8"), SEED)


def test_import_detects_url_and_cloudflare_block():
    assert detect_url(FIXTURE.read_text(encoding="utf-8")) == SEED.url
    assert is_blocked("<title>Attention Required! | Cloudflare</title>")


def test_parse_keeps_content_sections_and_drops_boilerplate():
    page = _page()
    headings = [s["heading"] for s in page["sections"]]
    assert page["main_entities"] == ["Đái tháo đường tuýp 2"]
    assert any("Triệu chứng" in h for h in headings)
    assert not any("liên quan" in h.lower() for h in headings)
    assert all("hotline" not in s["text"].lower() for s in page["sections"])


def test_section_rules_map_headings_to_relations():
    chunks = {c["heading"]: c for c in chunk_page(_page())}
    symptom = section_tasks(chunks["Triệu chứng tiểu đường type 2"])
    assert symptom == [{"kind": "guided", "relation": "HAS_SYMPTOM", "direction": "out", "types": ["Symptom"]}]
    kinds = [t.get("relation") or t["kind"] for t in section_tasks(chunks["Phương pháp chẩn đoán & điều trị tiểu đường type 2"])]
    assert kinds == ["DIAGNOSED_BY", "TREATED_BY", "threshold"]


def test_drug_page_skips_dosage_and_separates_contraindication():
    base = {"page_type": "thuoc", "main_entities": ["Metformin"], "text": "nội dung", "id": "x", "url": "u", "title": "t"}
    assert section_tasks({**base, "heading": "Liều dùng Metformin"}) == []
    assert section_tasks({**base, "heading": "Chống chỉ định"})[0]["relation"] == "CONTRAINDICATED_IN"
    assert section_tasks({**base, "heading": "Chỉ định"})[0]["relation"] == "TREATED_BY"


def test_validation_rejects_fabricated_evidence_and_wrong_domain():
    text = "Người bệnh có thể khát nước nhiều, đi tiểu nhiều lần."
    record = {"head": "Đái tháo đường", "head_type": "Disease", "relation": "HAS_SYMPTOM", "tail": "Khát nước nhiều",
              "tail_type": "Symptom", "evidence": "khát nước nhiều", "extracted": "Khát nước nhiều"}
    assert validate(record, text) is None
    assert "câu trích" in validate({**record, "evidence": "đau ngực dữ dội"}, text)
    assert "sai miền" in validate({**record, "tail_type": "Drug"}, text)


def test_validation_rejects_generic_entity_names():
    text = "Bệnh tiểu đường gây ra nhiều biến chứng nguy hiểm như bệnh thận, bệnh võng mạc."
    record = {"head": "Đái tháo đường", "head_type": "Disease", "relation": "HAS_COMPLICATION", "tail": "Biến chứng nguy hiểm",
              "tail_type": "Disease", "evidence": text, "extracted": "Biến chứng nguy hiểm"}
    assert validate(record, text) == "tên quá chung chung"
    assert validate({**record, "tail": "Bệnh thận", "extracted": "Bệnh thận"}, text) is None
    numeric = {**record, "relation": "DIAGNOSED_BY", "tail": "140 đến 199 mg/dL", "tail_type": "Test",
               "extracted": "140 đến 199 mg/dL", "evidence": "từ 140 đến 199 mg/dL", "head": "Tiền tiểu đường"}
    assert validate(numeric, "Tiền tiểu đường được chẩn đoán ở mức từ 140 đến 199 mg/dL.").startswith("tên là con số")


def test_priority_rules_fix_misleading_disease_headings():
    def tasks(heading):
        chunk = {"page_type": "benh", "heading": heading, "text": "Nội dung ngắn.", "main_entities": ["Đái tháo đường"]}
        return [(t["relation"], tuple(t["types"])) for t in section_tasks(chunk) if t["kind"] == "guided"]
    assert tasks("Bệnh đái tháo đường có làm tăng nguy cơ mắc bệnh tim mạch không?") == [("HAS_COMPLICATION", ("Disease",))]
    assert tasks("Những thói quen sinh hoạt có thể giúp bạn hạn chế diễn tiến của bệnh") == [("TREATED_BY", ("Lifestyle",))]
    assert tasks("Phương pháp phòng ngừa bệnh tiểu đường") == [("PREVENTED_BY", ("Lifestyle",))]
    assert tasks("Yếu tố làm tăng nguy cơ mắc phải bệnh tiểu đường type 2") == [("HAS_RISK_FACTOR", ("RiskFactor", "Disease"))]


def test_normalize_merges_synonyms_without_embeddings():
    records = [
        {"head": "Tiểu đường type 2", "head_type": "Disease", "relation": "HAS_RISK_FACTOR", "tail": "Cao huyết áp",
         "tail_type": "RiskFactor", "evidence": "e1", "source_url": "u1", "chunk_id": "c1", "task": "generic", "confidence": 0.7},
        {"head": "Đái tháo đường tuýp 2", "head_type": "Disease", "relation": "HAS_COMPLICATION", "tail": "Tăng huyết áp",
         "tail_type": "Disease", "evidence": "e2", "source_url": "u2", "chunk_id": "c2", "task": "guided", "confidence": 0.9},
    ]
    graph = build_graph(records)
    ids = {e["id"] for e in graph["entities"]}
    assert ids == {"Disease:dai thao duong tuyp 2", "Disease:tang huyet ap"}


def test_refine_threshold_reads_operator_and_upper_bound_from_evidence():
    def th(op, value, evidence, value_max=None):
        return refine_threshold({"op": op, "value": value, "value_max": value_max, "unit": "mg/dL", "evidence": evidence})
    fasting = th(">=", 100.0, "≥100 mg/dL và <126 mg/dL (nhưng nghiệm pháp dung nạp glucose bình thường): Rối loạn đường huyết lúc đói.")
    assert (fasting["op"], fasting["value_max"], fasting["max_exclusive"]) == ("range", 126.0, True)
    assert th("<=", 140.0, "Dưới 140 mg/dL (7,8 mmol/L) là trong phạm vi bình thường")["op"] == "<"
    assert th(">", 6.5, "HbA1c từ 6,5% trở lên: Tiểu đường type 2")["op"] == ">="
    assert th(">=", 126.0, "Đường huyết đói: Lớn hơn hoặc bằng 126mg/dL (hay 7.0 mmol/L).")["op"] == ">="
    assert th(">=", 180.0, "vượt quá 180 mg/dL sau khi ăn")["op"] == ">"
    kept = th("range", 5.7, "HbA1c 5,7 đến 6,4%: Tiền đái tháo đường", value_max=6.4)
    assert (kept["op"], kept["value_max"], kept.get("max_exclusive")) == ("range", 6.4, None)


def test_merge_decisions_apply_to_review_band():
    import numpy as np
    from kb.normalize import build_graph
    from kb.textutil import name_key

    class Embedder:  # cosine("Bỏ hút thuốc", "Ngừng hút thuốc lá") = 0,93: nằm trong vùng cần duyệt tay
        def embed(self, names):
            vectors = {"Bỏ hút thuốc": [1.0, 0.0], "Ngừng hút thuốc lá": [0.93, 0.3676]}
            return np.array([vectors.get(n, [0.0, 1.0]) for n in names])

    records = [{"head": "Đái tháo đường", "head_type": "Disease", "relation": "PREVENTED_BY", "tail": tail,
                "tail_type": "Lifestyle", "evidence": tail, "source_url": "u", "task": "guided", "confidence": 0.9,
                "chunk_id": "c"} for tail in ("Bỏ hút thuốc", "Ngừng hút thuốc lá")]
    undecided = build_graph([dict(r) for r in records], Embedder())
    assert len(undecided["merge_candidates"]) == 1 and len(undecided["entities"]) == 3
    key = ("Lifestyle", *sorted((name_key("Bỏ hút thuốc"), name_key("Ngừng hút thuốc lá"))))
    merged = build_graph([dict(r) for r in records], Embedder(), decisions={key: True})
    assert not merged["merge_candidates"] and len(merged["entities"]) == 2
    apart = build_graph([dict(r) for r in records], Embedder(), decisions={key: False})
    assert not apart["merge_candidates"] and len(apart["entities"]) == 3

