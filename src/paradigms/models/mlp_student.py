"""Compact student MLP architecture for Fashion-MNIST distillation experiments.

Designed as a single-hidden-layer perceptron to serve as a parameter-efficient
student model compared to the convolutional teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StudentMLP(nn.Module):
    """Configurable multi-layer perceptron for student distillation.

    The hidden dimension is parameterized to facilitate capacity-sweep experiments
    (e.g., across 8, 16, 32, 64, 128 units). Outputs raw logits without softmax
    to allow downstream temperature division and numerically stable distillation losses.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing unnormalized class logits.

        Args:
            x: Input tensor of shape (B, 1, 28, 28) or (B, 784).

        Returns:
            Raw logits tensor of shape (B, 10).
        """
        return self.net(x)
