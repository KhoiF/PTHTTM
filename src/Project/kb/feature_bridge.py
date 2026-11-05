"""Cầu nối DB ↔ KB: 8 cột của dataset thành nút Feature, nối MAPS_TO tới thực thể tương ứng trong KG.

Đây là chỗ hai nhánh trên sơ đồ gặp nhau: SHAP nói "HbA1c đẩy nguy cơ lên" → Feature(HbA1c_level)
→ Test(HbA1c) → các ngưỡng DIAGNOSED_BY kèm câu trích từ Long Châu.
"""

from __future__ import annotations

from kb.normalize import find_entities
from kb.textutil import name_key

FEATURES = {
    "HbA1c_level": {"label": "HbA1c", "unit": "%", "targets": [("Test", ["hba1c"])],
                    "fallback": ("Test", "HbA1c")},
    # Dataset chỉ có một số đo đường huyết (lúc đói hoặc bất kỳ), không phải số đo 2 giờ sau khi uống 75 g glucose:
    # bỏ các xét nghiệm và ngưỡng của nghiệm pháp dung nạp / sau ăn. Xét nghiệm có tên đã nói rõ lúc đói/bất kỳ thì
    # luôn giữ ("keep"); xét nghiệm tên chung ("Glucose") thì xét thêm câu trích của ngưỡng.
    "blood_glucose_level": {"label": "Đường huyết", "unit": "mg/dL", "targets": [("Test", ["duong huyet", "glucose"])],
                            "exclude": ["dung nap", "sau an", "sau khi an", "2 gio", "75 g", "75g", "ogtt", "sau khi uong"],
                            "keep": ["luc doi", "huyet doi", "bat ky", "ngau nhien", "fpg"],
                            "max_targets": 8, "fallback": ("Test", "Đường huyết"),
                            "note": "Dataset không ghi rõ đường huyết đo lúc đói hay bất kỳ; khi so với ngưỡng phải nêu giả định."},
    "bmi": {"label": "BMI", "unit": "kg/m²", "targets": [("RiskFactor", ["thua can", "beo phi"]), ("Disease", ["beo phi"])],
            "fallback": ("RiskFactor", "Thừa cân, béo phì")},
    "hypertension": {"label": "Tăng huyết áp", "targets": [("Disease", ["tang huyet ap"]), ("RiskFactor", ["tang huyet ap"])],
                     "fallback": ("Disease", "Tăng huyết áp")},
    "heart_disease": {"label": "Bệnh tim", "targets": [("Disease", ["tim mach", "benh tim", "mach vanh"])],
                      "fallback": ("Disease", "Bệnh tim mạch")},
    "smoking_history": {"label": "Tiền sử hút thuốc", "targets": [("RiskFactor", ["hut thuoc"])],
                        "fallback": ("RiskFactor", "Hút thuốc lá")},
    "age": {"label": "Tuổi", "unit": "năm", "targets": [("RiskFactor", ["tuoi"])], "fallback": ("RiskFactor", "Tuổi cao")},
    "gender": {"label": "Giới tính", "targets": [], "fallback": None},
}
MAX_TARGETS = 6


def add_feature_bridge(graph: dict) -> dict:
    """Thêm nút Feature và quan hệ MAPS_TO vào graph (tạo nút đích nếu KG chưa có)."""
    entities = {e["id"]: e for e in graph["entities"]}
    for column, spec in FEATURES.items():
        fid = f"Feature:{column}"
        entities[fid] = {"id": fid, "type": "Feature", "name": spec["label"], "name_key": column, "aliases": [column],
                         "mention_count": 0, "sources": [], "column": column, "unit": spec.get("unit"),
                         "note": spec.get("note")}
        targets = []
        for etype, patterns in spec["targets"]:
            targets += [e["id"] for e in find_entities({"entities": list(entities.values())}, etype, patterns)
                        if not any(x in e["name_key"] for x in spec.get("exclude", []))]
        targets = list(dict.fromkeys(targets))[:spec.get("max_targets", MAX_TARGETS)]
        if not targets and spec["fallback"]:
            etype, name = spec["fallback"]
            nid = f"{etype}:{name_key(name)}"
            entities.setdefault(nid, {"id": nid, "type": etype, "name": name, "name_key": name_key(name), "aliases": [],
                                      "mention_count": 0, "sources": [], "bridge_created": True})
            targets = [nid]
        for target in targets:
            graph["relations"].append({"head": fid, "relation": "MAPS_TO", "tail": target, "props": {},
                                       "evidence": [], "source_urls": [], "confidence": 1.0, "tasks": ["bridge"]})
    graph["entities"] = list(entities.values())
    return graph
