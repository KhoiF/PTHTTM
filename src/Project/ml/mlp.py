"""MLP PyTorch bọc theo giao diện scikit-learn (fit / predict_proba) cho phần "DL" trên sơ đồ."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layers=(64, 32, 16), dropout=0.1, lr=1e-3, epochs=40, batch_size=1024,
                 patience=6, weight_decay=1e-4, random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _build(self, n_in: int) -> nn.Module:
        layers, prev = [], n_in
        for width in self.hidden_layers:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])

        # Tách 10% để dừng sớm
        idx = rng.permutation(len(X))
        cut = int(len(X) * 0.9)
        tr, va = idx[:cut], idx[cut:]
        X_tr, y_tr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        X_va, y_va = torch.from_numpy(X[va]), torch.from_numpy(y[va])

        self.model_ = self._build(X.shape[1])
        pos_weight = torch.tensor([(y_tr == 0).sum().item() / max((y_tr == 1).sum().item(), 1)])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best, best_state, waited = float("inf"), None, 0
        self.history_ = []
        for _ in range(self.epochs):
            self.model_.train()
            order = torch.randperm(len(X_tr))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optim.zero_grad()
                loss = loss_fn(self.model_(X_tr[batch]).squeeze(1), y_tr[batch])
                loss.backward()
                optim.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_va).squeeze(1), y_va).item()
            self.history_.append(val_loss)
            if val_loss < best - 1e-4:
                best, waited = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)
        p = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
