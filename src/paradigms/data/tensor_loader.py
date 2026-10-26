"""In-memory tensor data loader for fast, reproducible CPU training."""

from __future__ import annotations

import math
from typing import Any, Iterator, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from paradigms.data.fashion_mnist import MEAN, STD
from paradigms.seed import make_generator


class TensorLoader:
    """In-memory data loader yielding sliced batches of pre-materialized tensors."""

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if len(X) != len(Y):
            raise ValueError(f"X and Y length mismatch: {len(X)} vs {len(Y)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.X = X
        self.Y = Y
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.generator = make_generator(seed) if shuffle else None
        self.n_samples = len(X)

    @classmethod
    def from_tensors(
        cls,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> TensorLoader:
        """Construct TensorLoader directly from pre-existing tensors."""
        return cls(X, Y, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def __len__(self) -> int:
        if self.n_samples == 0:
            return 0
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.n_samples == 0:
            return

        indices = torch.randperm(self.n_samples, generator=self.generator) if self.shuffle else None

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            if indices is not None:
                batch_idx = indices[start:end]
                yield self.X[batch_idx], self.Y[batch_idx]
            else:
                yield self.X[start:end], self.Y[start:end]


def make_fast_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> TensorLoader:
    """Materialize a Fashion-MNIST Dataset or Subset into a fast TensorLoader."""
    if isinstance(dataset, Subset):
        idx = torch.as_tensor(dataset.indices, dtype=torch.long)
        base_ds = dataset.dataset
        raw_x = base_ds.data[idx]
        raw_y = base_ds.targets[idx]
    elif hasattr(dataset, "data") and hasattr(dataset, "targets"):
        raw_x = dataset.data
        raw_y = dataset.targets
    else:
        # Fallback for generic datasets: materialize via single-batch loader
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        raw_x, raw_y = next(iter(loader))
        return TensorLoader(raw_x, raw_y, batch_size=batch_size, shuffle=shuffle, seed=seed)

    X = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1)
    Y = raw_y.clone().detach().long()
    return TensorLoader(X, Y, batch_size=batch_size, shuffle=shuffle, seed=seed)
