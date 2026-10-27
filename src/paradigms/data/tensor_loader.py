"""In-memory tensor data loader for fast, reproducible CPU training."""

from __future__ import annotations

import math
from typing import Any, Iterator, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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


def materialize(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a Dataset into contiguous (X, Y) float and long tensors."""
    from paradigms.recursive.datasets import PseudoLabeledDataset

    if isinstance(dataset, PseudoLabeledDataset):
        X, _ = materialize(Subset(dataset.base, dataset.indices))
        Y = torch.as_tensor(dataset.labels, dtype=torch.long)
        return X, Y

    if isinstance(dataset, ConcatDataset):
        if len(dataset.datasets) == 0:
            return torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long)
        parts = [materialize(part) for part in dataset.datasets]
        X = torch.cat([p[0] for p in parts], dim=0)
        Y = torch.cat([p[1] for p in parts], dim=0)
        return X, Y

    if isinstance(dataset, Subset):
        base_ds = dataset.dataset
        if hasattr(base_ds, "data") and hasattr(base_ds, "targets"):
            idx = torch.as_tensor(dataset.indices, dtype=torch.long)
            raw_x = base_ds.data[idx]
            raw_y = base_ds.targets
            if isinstance(raw_y, torch.Tensor):
                raw_y = raw_y[idx]
            else:
                raw_y = torch.as_tensor([raw_y[i] for i in idx], dtype=torch.long)
            X = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1)
            Y = raw_y.clone().detach().long() if isinstance(raw_y, torch.Tensor) else torch.as_tensor(raw_y, dtype=torch.long)
            return X, Y
        else:
            base_x, base_y = materialize(base_ds)
            idx = torch.as_tensor(dataset.indices, dtype=torch.long)
            return base_x[idx], base_y[idx]

    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        raw_x = dataset.data
        raw_y = dataset.targets
        X = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1)
        if isinstance(raw_y, torch.Tensor):
            Y = raw_y.clone().detach().long()
        else:
            Y = torch.as_tensor(raw_y, dtype=torch.long)
        return X, Y

    if len(dataset) == 0:
        return torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    raw_x, raw_y = next(iter(loader))
    if not isinstance(raw_x, torch.Tensor):
        raw_x = torch.as_tensor(raw_x)
    if not isinstance(raw_y, torch.Tensor):
        raw_y = torch.as_tensor(raw_y, dtype=torch.long)
    return raw_x, raw_y


def make_fast_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> TensorLoader:
    """Materialize any supported Dataset into a fast in-memory TensorLoader."""
    X, Y = materialize(dataset)
    return TensorLoader(X, Y, batch_size=batch_size, shuffle=shuffle, seed=seed)
