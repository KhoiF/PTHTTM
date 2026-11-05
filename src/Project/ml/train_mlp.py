"""Huấn luyện MLP PyTorch trong tiến trình riêng.

PyTorch và XGBoost mỗi bên mang một bản libomp; nạp cả hai vào cùng tiến trình gây deadlock
trên macOS, nên notebook ml/train.ipynb gọi file này bằng subprocess. Kết quả ghi vào ml/models/mlp_results.npz.
"""

from __future__ import annotations

import json

import numpy as np

from config import MODELS_DIR, RANDOM_STATE
from ml.evaluation import fit_and_evaluate
from ml.mlp import TorchMLPClassifier
from ml.preprocess import FEATURE_SETS, load_clean, split_70_15_15


def main() -> None:
    df, _ = load_clean()
    split = split_70_15_15(df)
    arrays, summary = {}, {}
    for set_name, features in FEATURE_SETS.items():
        result = fit_and_evaluate(TorchMLPClassifier(random_state=RANDOM_STATE), features, split)
        arrays[f"{set_name}_test_prob"] = result["test_prob"]
        summary[set_name] = {k: result[k] for k in ("threshold", "val_PR-AUC", "metrics", "brier_raw", "train_s")}
    np.savez(MODELS_DIR / "mlp_results.npz", summary=json.dumps(summary), **arrays)


if __name__ == "__main__":
    main()
