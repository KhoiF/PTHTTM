"""Hàm huấn luyện + hiệu chỉnh + đánh giá dùng chung cho mọi mô hình (chỉ phụ thuộc scikit-learn)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from ml.preprocess import Split, make_preprocessor


def best_f1_threshold(y, prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y, prob, threshold) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, prob),
        "PR-AUC": average_precision_score(y, prob),
        "F1": f1_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "Brier": brier_score_loss(y, prob),
    }


def fit_and_evaluate(estimator, features: list[str], split: Split) -> dict:
    """Huấn luyện trên train, hiệu chỉnh isotonic trên validation, chọn ngưỡng F1 trên validation, đo trên test."""
    start = time.perf_counter()
    base = Pipeline([("pre", make_preprocessor(features)), ("clf", estimator)])
    base.fit(split.X_train[features], split.y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(split.X_val[features], split.y_val)
    train_s = time.perf_counter() - start

    val_prob = calibrated.predict_proba(split.X_val[features])[:, 1]
    threshold = best_f1_threshold(split.y_val, val_prob)
    test_prob = calibrated.predict_proba(split.X_test[features])[:, 1]
    raw_test_prob = base.predict_proba(split.X_test[features])[:, 1]
    return {
        "base": base,
        "calibrated": calibrated,
        "threshold": threshold,
        "val_PR-AUC": average_precision_score(split.y_val, val_prob),
        "test_prob": test_prob,
        "metrics": evaluate(split.y_test, test_prob, threshold),
        "brier_raw": brier_score_loss(split.y_test, raw_test_prob),
        "train_s": train_s,
    }
