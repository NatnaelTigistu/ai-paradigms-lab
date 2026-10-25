"""Pluggable loss interface and standard loss functions for model training.

Loss Contract:
    LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]
    loss_fn(student_logits, labels, inputs) -> scalar Tensor

All training losses in the lab adhere to this 3-argument signature:
1. `student_logits`: Raw logits output by the model being trained of shape (B, C).
2. `labels`: Ground-truth integer class labels of shape (B,).
3. `inputs`: Raw input batch tensors `x` of shape (B, 1, 28, 28).

Hard-label cross-entropy ignores `inputs`. Distillation losses (implemented in later
phases) use `inputs` to query frozen teacher models for soft probability targets.
Decoupling the loss function from the training loop guarantees that baseline models
and distilled models execute the exact same optimization steps, batching, and
checkpointing routines, ensuring rigorous and fair experimental comparisons.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def hard_label_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard hard-label supervised cross-entropy loss.

    Ignores the `inputs` argument to conform to the universal LossFn contract.

    Args:
        logits: Unnormalized class predictions of shape (B, C).
        labels: Ground-truth target indices of shape (B,).
        inputs: Optional raw input batch tensor (ignored).

    Returns:
        Scalar cross-entropy loss tensor.
    """
    return F.cross_entropy(logits, labels)
