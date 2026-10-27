"""Candidate prediction and confidence generation for unlabeled pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from paradigms.data.tensor_loader import make_fast_loader
from paradigms.training.evaluate import predict_logits


@dataclass
class Candidates:
    """Aligned predictions and confidences for candidate sample indices."""

    indices: np.ndarray
    preds: np.ndarray
    confs: np.ndarray


def generate(
    model: nn.Module,
    base: Dataset,
    indices: Sequence[int] | np.ndarray,
    batch_size: int = 512,
) -> Candidates:
    """Run model inference over subset indices and return aligned predictions and confidences.

    Args:
        model: PyTorch model predicting logits.
        base: Underlying base dataset.
        indices: Subset indices to evaluate.
        batch_size: Inference batch size.

    Returns:
        Candidates dataclass with aligned np.ndarray fields in original input order.
    """
    indices_list = list(indices)
    if len(indices_list) == 0:
        return Candidates(
            indices=np.array([], dtype=np.int64),
            preds=np.array([], dtype=np.int64),
            confs=np.array([], dtype=np.float32),
        )

    subset = Subset(base, indices_list)
    loader = make_fast_loader(subset, batch_size=batch_size, shuffle=False)
    logits, _ = predict_logits(model, loader, device="cpu")

    probs = torch.softmax(logits, dim=-1)
    confs_t, preds_t = torch.max(probs, dim=-1)

    return Candidates(
        indices=np.asarray(indices_list, dtype=np.int64),
        preds=preds_t.cpu().numpy().astype(np.int64),
        confs=confs_t.cpu().numpy().astype(np.float32),
    )
