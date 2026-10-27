"""Dataset wrappers for recursive self-training with pseudo-labels."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PseudoLabeledDataset(Dataset):
    """Dataset wrapping a base dataset with verified pseudo-labels."""

    def __init__(
        self,
        base: Dataset,
        indices: Sequence[int],
        labels: Sequence[int] | torch.Tensor | np.ndarray,
    ) -> None:
        if len(indices) != len(labels):
            raise ValueError(
                f"indices length ({len(indices)}) and labels length ({len(labels)}) must match"
            )
        self.base = base
        self.indices = list(indices)
        self.labels = [int(l) for l in labels]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        # WHY: the model's own verified predictions replace true labels.
        item = self.base[self.indices[idx]]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, int(self.labels[idx])
