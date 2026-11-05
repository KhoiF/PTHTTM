"""Dự đoán nguy cơ, giải thích SHAP và phân tích what-if cho một hồ sơ bệnh nhân.

Mô hình full cần đủ HbA1c và đường huyết; thiếu một trong hai thì dùng mô hình screening.
SHAP giải thích mô hình cây gốc (thang log-odds, trước bước hiệu chỉnh isotonic).
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import GREY_MARGIN, MODELS_DIR
from ml.preprocess import FULL_FEATURES, GENDERS, LAB_FEATURES, SMOKING, VALID_RANGES

REQUIRED = ["gender", "age", "bmi", "hypertension", "heart_disease"]
FEATURE_LABELS = {
    "gender": "Giới tính",
    "age": "Tuổi",
    "hypertension": "Tăng huyết áp",
    "heart_disease": "Bệnh tim",
    "smoking_history": "Tiền sử hút thuốc",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c",
    "blood_glucose_level": "Đường huyết",
}
MISSING_TOKENS = {"", "none", "null", "nan", "không rõ", "khong ro"}
# Isotonic cho ra đúng 0/1 ở hai đầu; không bao giờ báo nguy cơ 0% hay 100%
PROB_FLOOR, PROB_CEIL = 0.01, 0.99


@lru_cache
def load_bundle(set_name: str) -> dict:
    return joblib.load(MODELS_DIR / f"{set_name}.sav")


@lru_cache
def load_metadata() -> dict:
    return json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))


@lru_cache
def _explainer(set_name: str):
    import shap

    return shap.TreeExplainer(load_bundle(set_name)["base"].named_steps["clf"])


def _to_number(value):
    if value is None or (isinstance(value, str) and value.strip().lower() in MISSING_TOKENS):
        return None
    return float(str(value).replace(",", "."))


def clean_patient(patient: dict) -> tuple[dict, list[str]]:
    """Chuẩn hóa kiểu dữ liệu và kiểm tra miền giá trị. Trả về (hồ sơ sạch, danh sách lỗi)."""
    clean, errors = {}, []
    gender = str(patient.get("gender") or "").strip().lower()
    gender_map = {"female": "Female", "nữ": "Female", "nu": "Female", "male": "Male", "nam": "Male"}
    clean["gender"] = gender_map.get(gender)

    smoking = patient.get("smoking_history")
    clean["smoking_history"] = smoking if smoking in SMOKING else "No Info"

    for field in ("age", "bmi", "HbA1c_level", "blood_glucose_level"):
        try:
            value = _to_number(patient.get(field))
        except ValueError:
            errors.append(f"{FEATURE_LABELS[field]} không phải là số: {patient.get(field)!r}")
            value = None
        # Mô hình ngôn ngữ đôi khi điền 0 cho xét nghiệm không có; coi là thiếu
        if field in LAB_FEATURES and value == 0:
            value = None
        if value is not None:
            low, high = VALID_RANGES[field]
            if not low <= value <= high:
                errors.append(f"{FEATURE_LABELS[field]} = {value:g} nằm ngoài miền hợp lệ [{low:g}, {high:g}]")
        clean[field] = value

    for field in ("hypertension", "heart_disease"):
        value = patient.get(field)
        if isinstance(value, str):
            value = {"1": 1, "0": 0, "có": 1, "co": 1, "không": 0, "khong": 0, "true": 1, "false": 0}.get(
                value.strip().lower())
        clean[field] = int(value) if value in (0, 1, True, False) else None

    if clean["gender"] not in GENDERS and patient.get("gender") not in (None, ""):
        errors.append(f"Giới tính không hợp lệ: {patient.get('gender')!r}")
    return clean, errors


def missing_required(patient: dict) -> list[str]:
    return [f for f in REQUIRED if patient.get(f) is None]


def choose_feature_set(patient: dict) -> str:
    return "full" if all(patient.get(f) is not None for f in LAB_FEATURES) else "screening"


def _frame(patient: dict, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{f: patient[f] for f in features}])


def predict_risk(patient: dict, model: str = "auto") -> dict:
    clean, errors = clean_patient(patient)
    missing = missing_required(clean)
    if errors or missing:
        return {"ok": False, "errors": errors, "missing_fields": missing}
    set_name = choose_feature_set(clean) if model == "auto" else model
    if set_name == "full" and choose_feature_set(clean) != "full":
        set_name = "screening"
    bundle = load_bundle(set_name)
    probability = float(bundle["calibrated"].predict_proba(_frame(clean, bundle["features"]))[0, 1])
    probability = min(max(probability, PROB_FLOOR), PROB_CEIL)
    threshold = float(bundle["threshold"])
    return {
        "ok": True,
        "probability": round(probability, 4),
        "label": int(probability >= threshold),
        "threshold": round(threshold, 4),
        "grey_zone": abs(probability - threshold) < GREY_MARGIN,
        "model_used": set_name,
        "model_name": bundle["model_name"],
        "missing_labs": [f for f in LAB_FEATURES if clean.get(f) is None],
        "patient": clean,
    }


def explain(patient: dict, k: int = 5, model: str = "auto") -> list[dict]:
    clean, errors = clean_patient(patient)
    if errors or missing_required(clean):
        return []
    set_name = choose_feature_set(clean) if model == "auto" else model
    bundle = load_bundle(set_name)
    pre = bundle["base"].named_steps["pre"]
    X = pre.transform(_frame(clean, bundle["features"]))
    values = np.asarray(_explainer(set_name).shap_values(X))
    if values.ndim == 3:  # Random Forest trả về theo từng lớp
        values = values[..., 1]
    values = values.reshape(-1)

    totals: dict[str, float] = {}
    for name, value in zip(pre.get_feature_names_out(), values):
        original = next((f for f in bundle["features"] if name == f or name.startswith(f + "_")), name)
        totals[original] = totals.get(original, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
    return [{
        "feature": feature,
        "label": FEATURE_LABELS.get(feature, feature),
        "value": clean.get(feature),
        "shap": round(value, 4),
        "direction": "tăng nguy cơ" if value > 0 else "giảm nguy cơ",
    } for feature, value in ranked]


def what_if(patient: dict, changes: dict, model: str = "auto") -> dict:
    before = predict_risk(patient, model)
    if not before["ok"]:
        return before
    modified = {**before["patient"], **changes}
    after = predict_risk(modified, before["model_used"])
    if not after["ok"]:
        return after
    return {
        "ok": True,
        "changes": changes,
        "before": before["probability"],
        "after": after["probability"],
        "delta": round(after["probability"] - before["probability"], 4),
        "model_used": before["model_used"],
    }


__all__ = ["FEATURE_LABELS", "FULL_FEATURES", "REQUIRED", "clean_patient", "explain", "load_metadata",
           "missing_required", "predict_risk", "what_if"]
