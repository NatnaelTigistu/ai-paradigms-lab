"""Teacher CNN architecture for Fashion-MNIST distillation.

Designed as a convolutional feature extractor with two conv-pool stages followed
by a dense classifier with dropout. Serves as the high-capacity teacher network
producing soft targets for distillation to smaller student architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherCNN(nn.Module):
    """Convolutional teacher model for Fashion-MNIST classification.

    Outputs raw logits to prevent premature probability saturation and ensure
    loss functions (e.g. CrossEntropyLoss, KL-divergence for distillation) can
    apply temperature scaling or log-softmax in a numerically stable manner.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        # Conv stages extract spatial hierarchies; 28x28 -> 14x14 -> 7x7.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 64 channels * 7 * 7 spatial grid = 3136 flattened dimensions.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing unnormalized class logits.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Raw logits tensor of shape (B, 10).
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
