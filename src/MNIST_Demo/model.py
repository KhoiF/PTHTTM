"""Kiến trúc CNN dùng chung cho notebook huấn luyện và ứng dụng dự đoán."""

from __future__ import annotations

import torch
from torch import nn


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
NUM_CLASSES = 10


class MNISTCNN(nn.Module):
    """CNN nhỏ gọn cho ảnh MNIST grayscale kích thước 28 x 28."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def count_trainable_parameters(model: nn.Module) -> int:
    """Trả về số tham số có thể huấn luyện của mô hình."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
