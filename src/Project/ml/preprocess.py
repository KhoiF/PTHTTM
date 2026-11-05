"""Làm sạch dữ liệu và dựng bộ tiền xử lý cho hai bộ đặc trưng full / screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, RANDOM_STATE

TARGET = "diabetes"
CATEGORICAL = ["gender", "smoking_history"]
LAB_FEATURES = ["HbA1c_level", "blood_glucose_level"]
FULL_FEATURES = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi",
                 "HbA1c_level", "blood_glucose_level"]
SCREENING_FEATURES = [f for f in FULL_FEATURES if f not in LAB_FEATURES]
FEATURE_SETS = {"full": FULL_FEATURES, "screening": SCREENING_FEATURES}

GENDERS = ["Female", "Male"]
SMOKING = ["never", "No Info", "current", "former", "ever", "not current"]
# Miền giá trị hợp lệ lấy từ dataset, Intake dùng để kiểm tra hồ sơ
VALID_RANGES = {
    "age": (0.08, 80.0),
    "bmi": (10.0, 96.0),
    "HbA1c_level": (3.5, 9.0),
    "blood_glucose_level": (80.0, 300.0),
}


@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_clean(path=DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Đọc CSV, loại dòng trùng và 18 dòng gender=Other; trả kèm thống kê làm sạch."""
    raw = pd.read_csv(path)
    stats = {"rows_raw": len(raw), "duplicates": int(raw.duplicated().sum())}
    df = raw.drop_duplicates()
    stats["gender_other"] = int((df["gender"] == "Other").sum())
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    stats["rows_clean"] = len(df)
    stats["positive_rate"] = float(df[TARGET].mean())
    return df, stats


def split_70_15_15(df: pd.DataFrame) -> Split:
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int).to_numpy()
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=RANDOM_STATE, stratify=y_tv)
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in features if c in CATEGORICAL]
    numeric = [c for c in features if c not in CATEGORICAL]
    categories = [GENDERS if c == "gender" else SMOKING for c in categorical]
    return ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False), categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
