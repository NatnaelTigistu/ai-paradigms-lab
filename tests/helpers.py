"""Shared test helpers and synthetic dataset generation."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def make_synthetic(n: int, seed: int) -> TensorDataset:
    """Construct a fast, deterministic synthetic dataset for trainer and loss unit tests.

    Each class y has a distinct bright horizontal band (image[0, 2*y : 2*y+2, :] += 3.0)
    over low-amplitude noise, providing a trivially learnable task without network overhead.
    """
    rng = torch.Generator().manual_seed(seed)
    images = []
    labels = []
    for i in range(n):
        y = i % 10
        img = 0.1 * torch.randn(1, 28, 28, generator=rng)
        img[0, 2 * y : 2 * y + 2, :] += 3.0
        images.append(img)
        labels.append(y)

    x = torch.stack(images, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)
