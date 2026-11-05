"""Công cụ ML cho Risk Agent, gắn sẵn hồ sơ bệnh nhân hiện tại để LLM chỉ cần truyền tham số tùy chọn."""

from __future__ import annotations

from ml import predictor
from ml.preprocess import FULL_FEATURES

EDITABLE = {"bmi", "smoking_history", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease"}


def make_ml_tools(patient: dict) -> list:
    def predict_risk(model: str = "auto") -> dict:
        """Dự đoán xác suất mắc tiểu đường của bệnh nhân hiện tại bằng mô hình học máy đã huấn luyện.

        Args:
            model: "auto" (tự chọn), "full" (cần HbA1c và đường huyết) hoặc "screening" (không cần xét nghiệm)
        """
        model = model if model in {"auto", "full", "screening"} else "auto"
        result = predictor.predict_risk(patient, model)
        result.pop("patient", None)
        return result

    def explain_risk(k: int = 5) -> dict:
        """Giải thích dự đoán: các chỉ số ảnh hưởng nhiều nhất tới nguy cơ (giá trị SHAP, dương = làm tăng nguy cơ).

        Args:
            k: Số chỉ số cần liệt kê (1-8)
        """
        k = max(1, min(int(k or 5), 8))
        return {"top_features": predictor.explain(patient, k=k)}

    def what_if(feature: str, new_value: float) -> dict:
        """Ước tính nguy cơ thay đổi thế nào nếu một chỉ số có thể điều chỉnh thay đổi (vd. giảm BMI, bỏ thuốc lá).

        Args:
            feature: Một trong: bmi, smoking_history, HbA1c_level, blood_glucose_level, hypertension, heart_disease
            new_value: Giá trị mới (smoking_history dùng chuỗi như "former", "never")
        """
        if feature not in EDITABLE:
            return {"error": f"chỉ thay đổi được: {sorted(EDITABLE)}"}
        return predictor.what_if(patient, {feature: new_value})

    return [predict_risk, explain_risk, what_if]


__all__ = ["FULL_FEATURES", "make_ml_tools"]
