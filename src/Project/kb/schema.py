"""Ontology đóng của Knowledge Graph và các schema Pydantic dùng làm "tool" để LLM nộp kết quả trích xuất."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ENTITY_TYPES = ["Disease", "Symptom", "RiskFactor", "Test", "Drug", "DrugClass", "SideEffect", "Lifestyle"]
EntityType = Literal["Disease", "Symptom", "RiskFactor", "Test", "Drug", "DrugClass", "SideEffect", "Lifestyle"]

TYPE_LABELS_VI = {
    "Disease": "bệnh / tình trạng bệnh lý",
    "Symptom": "triệu chứng",
    "RiskFactor": "yếu tố nguy cơ",
    "Test": "xét nghiệm / chỉ số",
    "Drug": "hoạt chất thuốc",
    "DrugClass": "nhóm thuốc",
    "SideEffect": "tác dụng phụ",
    "Lifestyle": "biện pháp lối sống (ăn uống, vận động...)",
}

# quan hệ: (miền đầu, miền đuôi, mô tả)
RELATIONS: dict[str, tuple[set[str], set[str], str]] = {
    "HAS_SYMPTOM": ({"Disease"}, {"Symptom"}, "bệnh có triệu chứng"),
    "HAS_RISK_FACTOR": ({"Disease"}, {"RiskFactor", "Disease"}, "bệnh có yếu tố nguy cơ / nguyên nhân"),
    "HAS_COMPLICATION": ({"Disease"}, {"Disease"}, "bệnh gây biến chứng"),
    "DIAGNOSED_BY": ({"Disease"}, {"Test"}, "bệnh được chẩn đoán bằng xét nghiệm"),
    "TREATED_BY": ({"Disease"}, {"Drug", "DrugClass", "Lifestyle"}, "bệnh được điều trị bằng"),
    "PREVENTED_BY": ({"Disease"}, {"Lifestyle"}, "bệnh được phòng ngừa bằng"),
    "BELONGS_TO": ({"Drug"}, {"DrugClass"}, "hoạt chất thuộc nhóm thuốc"),
    "HAS_SIDE_EFFECT": ({"Drug", "DrugClass"}, {"SideEffect"}, "thuốc có tác dụng phụ"),
    "CONTRAINDICATED_IN": ({"Drug", "DrugClass"}, {"Disease"}, "thuốc chống chỉ định ở"),
}
RelationType = Literal["HAS_SYMPTOM", "HAS_RISK_FACTOR", "HAS_COMPLICATION", "DIAGNOSED_BY", "TREATED_BY",
                       "PREVENTED_BY", "BELONGS_TO", "HAS_SIDE_EFFECT", "CONTRAINDICATED_IN"]
STRUCTURAL_RELATIONS = {"MAPS_TO", "MENTIONS", "FROM"}

# Thực thể chính của các trang /benh/ (theo cột topic trong seed_urls.csv)
MAIN_DISEASE_BY_TOPIC = {
    "đái tháo đường tổng quan": "Đái tháo đường",
    "tiểu đường tuýp 1": "Đái tháo đường tuýp 1",
    "tiểu đường tuýp 2": "Đái tháo đường tuýp 2",
    "tiểu đường không phụ thuộc insulin": "Đái tháo đường tuýp 2",
    "tiểu đường ở trẻ em": "Đái tháo đường ở trẻ em",
    "hạ đường huyết": "Hạ đường huyết",
    "bệnh tim mạch": "Bệnh tim mạch",
}

# Luật tên mục → (quan hệ, chiều, loại thực thể cần trích). Chiều "out": thực thể chính là đầu quan hệ;
# "in": thực thể được trích là đầu (ví dụ mục "Chỉ định" của trang thuốc: Bệnh -TREATED_BY-> Thuốc).
# Thứ tự quan trọng: "chống chỉ định" phải đứng trước "chỉ định".
DISEASE_SECTION_RULES = [
    (r"bien chung|nguy hiem", "HAS_COMPLICATION", "out", ["Disease"]),
    (r"trieu chung|dau hieu|bieu hien", "HAS_SYMPTOM", "out", ["Symptom"]),
    (r"nguyen nhan|nguy co|doi tuong|tac nhan|yeu to", "HAS_RISK_FACTOR", "out", ["RiskFactor", "Disease"]),
    (r"chan doan|xet nghiem|chi so", "DIAGNOSED_BY", "out", ["Test"]),
    (r"phong ngua|phong tranh|che do|sinh hoat|loi song", "PREVENTED_BY", "out", ["Lifestyle"]),
    (r"dieu tri|thuoc|phac do", "TREATED_BY", "out", ["Drug", "DrugClass", "Lifestyle"]),
]
# Xét trước DISEASE_SECTION_RULES; khớp thì chỉ dùng luật này. Từ khóa chung ("nguy cơ", "sinh hoạt") gán nhầm các mục:
# "X có làm tăng nguy cơ mắc Y không?" nói về biến chứng của X, không phải yếu tố nguy cơ;
# "Thói quen sinh hoạt giúp hạn chế diễn tiến của bệnh" là cách kiểm soát bệnh, không phải phòng ngừa.
DISEASE_PRIORITY_RULES = [
    (r"\bco lam tang nguy co mac\b.*\bkhong\b", "HAS_COMPLICATION", "out", ["Disease"]),
    (r"han che dien tien", "TREATED_BY", "out", ["Lifestyle"]),
]
DRUG_SECTION_RULES = [
    (r"lieu dung|cach dung|qua lieu|quen lieu", None, None, None),  # không lưu liều dùng vào KG
    (r"chong chi dinh", "CONTRAINDICATED_IN", "out", ["Disease"]),
    (r"chi dinh|cong dung", "TREATED_BY", "in", ["Disease"]),
    (r"tac dung phu|tac dung khong mong muon|phan ung co hai", "HAS_SIDE_EFFECT", "out", ["SideEffect"]),
    (r"duoc luc hoc|co che|nhom thuoc|phan loai", "BELONGS_TO", "out", ["DrugClass"]),
]


class ExtractedEntity(BaseModel):
    name: str = Field(description="Tên thực thể ngắn gọn bằng tiếng Việt, đúng như trong đoạn văn")
    type: EntityType = Field(description="Loại thực thể")
    evidence: str = Field(description="Cụm từ hoặc câu trích NGUYÊN VĂN từ đoạn văn có nhắc tới thực thể")


class EntityList(BaseModel):
    """Danh sách thực thể trích được từ đoạn văn."""

    entities: list[ExtractedEntity] = Field(default_factory=list)


class ExtractedTriple(BaseModel):
    head: str = Field(description="Tên thực thể đầu")
    head_type: EntityType
    relation: RelationType
    tail: str = Field(description="Tên thực thể đuôi")
    tail_type: EntityType
    evidence: str = Field(description="Câu trích NGUYÊN VĂN từ đoạn văn thể hiện quan hệ")


class TripleList(BaseModel):
    """Danh sách quan hệ (bộ ba) trích được từ đoạn văn."""

    triples: list[ExtractedTriple] = Field(default_factory=list)


class ExtractedThreshold(BaseModel):
    condition: str = Field(description="Tình trạng được chẩn đoán, ví dụ 'Đái tháo đường', 'Tiền đái tháo đường'")
    test: str = Field(description="Tên xét nghiệm, ví dụ 'HbA1c', 'Đường huyết lúc đói'")
    op: Literal[">=", ">", "<=", "<", "range"] = Field(description="Phép so sánh; 'range' nếu là khoảng từ value đến value_max")
    value: float
    value_max: float | None = Field(default=None, description="Chỉ dùng khi op = 'range'")
    unit: Literal["%", "mg/dL", "mmol/L"]
    evidence: str = Field(description="Câu trích NGUYÊN VĂN chứa con số")

    @field_validator("op", mode="before")
    @classmethod
    def _normalize_op(cls, value):
        mapping = {"≥": ">=", "=>": ">=", "≤": "<=", "=<": "<=", "từ": "range", "khoảng": "range", "between": "range"}
        return mapping.get(str(value).strip(), str(value).strip())

    @field_validator("unit", mode="before")
    @classmethod
    def _normalize_unit(cls, value):
        text = str(value).strip().lower().replace(" ", "")
        return {"%": "%", "mg/dl": "mg/dL", "mmol/l": "mmol/L", "mmol": "mmol/L", "mg": "mg/dL"}.get(text, value)

    @model_validator(mode="after")
    def _fill_range_max(self):
        if self.op == "range" and self.value_max is None:
            numbers = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", self.evidence)]
            later = [n for n in numbers if n > self.value]
            if self.value in numbers and later:
                self.value_max = later[0]
            else:
                self.op = ">="
        return self


def format_threshold(op: str, value: float, value_max: float | None = None, max_exclusive: bool = False) -> str:
    """Ngưỡng dạng chữ: '≥ 126', '5.7–6.4', '≥ 100 và < 126' (khoảng có cận trên không lấy dấu bằng)."""
    if op == "range" and value_max is not None:
        return f"≥ {value:g} và < {value_max:g}" if max_exclusive else f"{value:g}–{value_max:g}"
    return f"{ {'>=': '≥', '<=': '≤'}.get(op, op)} {value:g}"


class ThresholdList(BaseModel):
    """Danh sách ngưỡng chẩn đoán trích được từ đoạn văn."""

    thresholds: list[ExtractedThreshold] = Field(default_factory=list)
