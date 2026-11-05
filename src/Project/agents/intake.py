"""Intake Agent: gom hồ sơ từ form + tin nhắn, kiểm tra hợp lệ, hỏi lại khi thiếu."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from agents import prompts
from agents.base import Step, default_llm
from config import THINK
from ml.predictor import FEATURE_LABELS, clean_patient, missing_required

MMOLL_TO_MGDL = 18.0


class PatientExtraction(BaseModel):
    """Thông tin sức khỏe trích từ tin nhắn (null nếu không nêu)."""

    gender: Literal["Male", "Female"] | None = None
    age: float | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    hypertension: bool | None = None
    heart_disease: bool | None = None
    smoking_history: Literal["never", "former", "current", "ever", "not current"] | None = None
    HbA1c_level: float | None = None
    blood_glucose: float | None = None
    blood_glucose_unit: Literal["mg/dL", "mmol/L"] | None = None


class LabExtraction(BaseModel):
    """Kết quả xét nghiệm trích từ tin nhắn (null nếu không nêu)."""

    HbA1c_level: float | None = None
    blood_glucose: float | None = None
    blood_glucose_unit: Literal["mg/dL", "mmol/L"] | None = None


def _to_fields(data: dict) -> dict:
    fields = {k: data.get(k) for k in ("gender", "age", "bmi", "smoking_history", "HbA1c_level")}
    for key in ("hypertension", "heart_disease"):
        if data.get(key) is not None:
            fields[key] = int(bool(data[key]))
    if fields.get("bmi") is None and data.get("height_cm") and data.get("weight_kg"):
        height = data["height_cm"] / 100 if data["height_cm"] > 3 else data["height_cm"]
        fields["bmi"] = round(data["weight_kg"] / height ** 2, 1)
    glucose = data.get("blood_glucose")
    if glucose:
        mmol = data.get("blood_glucose_unit") == "mmol/L" or glucose < 35  # < 35 chắc chắn là mmol/L
        fields["blood_glucose_level"] = round(glucose * MMOLL_TO_MGDL) if mmol else glucose
    return {k: v for k, v in fields.items() if v is not None}


HBA1C_RE = re.compile(r"(?:hba1c|hb\s*a1c|a1c)\D{0,15}?(\d+(?:[.,]\d+)?)\s*%?", re.I)
GLUCOSE_RE = re.compile(r"(?:đường huyết|đường máu|glucose)[^\d]{0,25}?(\d+(?:[.,]\d+)?)\s*(mg/dl|mmol/l|mmol)?", re.I)


AGE_RE = re.compile(r"(\d{1,3})\s*tuổi", re.I)
BMI_RE = re.compile(r"\bbmi\D{0,6}?(\d{1,3}(?:[.,]\d+)?)", re.I)
# Giới tính: nêu thẳng, hoặc suy ra từ cách gọi người thân kèm "tôi/mình/em" ("ông tôi", "mẹ mình"...)
MALE_RE = re.compile(r"(?<!việt )\bnam\b|\bnam giới\b|\bđàn ông\b|\b(?:ông|bố|cha|ba|chồng|anh trai|em trai|con trai|chú|cậu"
                     r"|bác trai)\s+(?:tôi|mình|em)\b", re.I)
FEMALE_RE = re.compile(r"\bnữ\b|\bphụ nữ\b|\b(?:bà|mẹ|má|vợ|chị gái|em gái|con gái|cô|dì|bác gái)\s+(?:tôi|mình|em)\b", re.I)


def regex_profile(text: str) -> dict:
    """Bắt tuổi, BMI, giới tính bằng biểu thức chính quy, dùng khi LLM bỏ sót (không ghi đè kết quả của LLM)."""
    found = {}
    if (m := AGE_RE.search(text)):
        found["age"] = float(m.group(1))
    if (m := BMI_RE.search(text)):
        found["bmi"] = float(m.group(1).replace(",", "."))
    male, female = bool(MALE_RE.search(text)), bool(FEMALE_RE.search(text))
    if male != female:
        found["gender"] = "Male" if male else "Female"
    return found


def regex_labs(text: str) -> dict:
    """Bắt HbA1c và đường huyết bằng biểu thức chính quy (không phụ thuộc LLM)."""
    found = {}
    if (m := HBA1C_RE.search(text)):
        found["HbA1c_level"] = float(m.group(1).replace(",", "."))
    if (m := GLUCOSE_RE.search(text)):
        unit = "mmol/L" if (m.group(2) or "").lower().startswith("mmol") else None
        found.update(_to_fields({"blood_glucose": float(m.group(1).replace(",", ".")), "blood_glucose_unit": unit}))
    return found


def _ask_message(missing: list[str], errors: list[str]) -> str:
    lines = []
    if missing:
        lines.append("Để đánh giá nguy cơ, mình cần thêm: " + ", ".join(FEATURE_LABELS[f] for f in missing) + ".")
    if errors:
        lines.append("Một số giá trị chưa hợp lệ: " + "; ".join(errors) + ".")
    lines.append("Nếu có, bạn cũng có thể gửi kết quả HbA1c và đường huyết để đánh giá chính xác hơn.")
    return " ".join(lines)


def intake_node(state: dict, llm=None) -> dict:
    llm = llm or default_llm()
    step = Step("Intake")
    question = (state.get("question") or "").strip()
    patient = dict(state.get("patient") or state.get("history_patient") or {})
    patient.update({k: v for k, v in (state.get("form") or {}).items() if v not in (None, "")})

    if state.get("risk_request") == "labs":
        # Risk nhờ tìm lại kết quả xét nghiệm (model nhỏ có thể bỏ sót ở lượt đầu)
        found = {}
        if question:
            labs = llm.structured(LabExtraction, prompts.INTAKE_LABS, question, think=THINK["intake"], usage=step.usage)
            found = {k: v for k, v in _to_fields(labs.model_dump()).items() if patient.get(k) in (None, "")}
        patient.update(found)
        if found:
            decision = f"tìm thấy thêm {list(found)} → Risk tính lại"
            follow_up = None
        else:
            decision = "không có xét nghiệm → giữ mô hình screening, đề nghị bổ sung"
            follow_up = ("Bạn chưa cung cấp HbA1c hoặc đường huyết. Hãy làm xét nghiệm và gửi kết quả để mình đánh giá "
                         "bằng mô hình đầy đủ, chính xác hơn.")
        clean, _ = clean_patient(patient)
        return {"patient": clean, "risk_request": None, "follow_up": follow_up,
                "trace": [step.entry(decision, found=found)]}

    extracted = {}
    if question:
        result = llm.structured(PatientExtraction, prompts.INTAKE, question, think=THINK["intake"], usage=step.usage)
        extracted = _to_fields(result.model_dump())
        for key, value in {**regex_profile(question), **regex_labs(question)}.items():
            extracted.setdefault(key, value)
        patient.update(extracted)
    clean, errors = clean_patient(patient)
    missing = missing_required(clean)
    if missing or errors:
        message = _ask_message(missing, errors)
        if state.get("intent") == "both":
            return {"patient": clean, "missing_fields": missing, "input_errors": errors, "follow_up": message,
                    "trace": [step.entry("thiếu dữ liệu → bỏ qua Risk, vẫn trả lời câu hỏi kiến thức",
                                         missing=missing, errors=errors, extracted=extracted)]}
        return {"patient": clean, "missing_fields": missing, "input_errors": errors, "ask_user": message,
                "trace": [step.entry("thiếu dữ liệu → hỏi lại người dùng", missing=missing, errors=errors,
                                     extracted=extracted)]}
    return {"patient": clean, "missing_fields": [], "input_errors": [],
            "trace": [step.entry("hồ sơ hợp lệ → Risk", extracted=extracted)]}
