"""Distillation loss formulation and factory for teacher-student knowledge transfer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from paradigms.training.losses import LossFn


def distillation_loss_from_logits(
    s_logits: torch.Tensor,
    t_logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    T: float,
    alpha: float,
) -> torch.Tensor:
    """Compute knowledge distillation loss combining hard cross-entropy and soft KL.

    Loss Formulation:
        L = alpha * CE(s_logits, labels) + (1 - alpha) * (T^2 * KL(p_s, p_t))

    Why T^2 scaling:
        Dividing logits by temperature T scales the gradient of the KL divergence
        by 1 / T^2. Multiplying by T^2 ensures that the magnitude of gradients
        derived from soft teacher targets remains commensurate with hard-label
        cross-entropy gradients across varying temperatures.

    Why conditional skipping:
        - When alpha == 0: Cross-entropy is completely skipped. Labels are never
          touched, allowing distillation on unlabeled data (or dummy labels like -1).
        - When alpha == 1: KL divergence is completely skipped and teacher logits are
          unused, guaranteeing exact numerical equivalence with hard-label training.

    Args:
        s_logits: Raw student logits of shape (B, C).
        t_logits: Raw teacher logits of shape (B, C), or None if alpha == 1.
        labels: Ground-truth target labels of shape (B,), or None if alpha == 0.
        T: Distillation temperature scalar (must be > 0).
        alpha: Weight for hard-label cross-entropy in [0, 1].

    Raises:
        ValueError: If T <= 0 or alpha not in [0, 1].

    Returns:
        Scalar loss tensor.
    """
    if T <= 0:
        raise ValueError(f"Temperature T must be positive, got {T}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    if alpha == 1.0:
        if labels is None:
            raise ValueError("labels cannot be None when alpha == 1.0")
        return F.cross_entropy(s_logits, labels)

    if alpha == 0.0:
        if t_logits is None:
            raise ValueError("t_logits cannot be None when alpha < 1.0")
        log_p_s = F.log_softmax(s_logits / T, dim=-1)
        p_t = F.softmax(t_logits.detach() / T, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return kl

    # Mixed loss: 0 < alpha < 1
    if labels is None:
        raise ValueError("labels cannot be None when alpha > 0")
    if t_logits is None:
        raise ValueError("t_logits cannot be None when alpha < 1")

    ce = F.cross_entropy(s_logits, labels)
    log_p_s = F.log_softmax(s_logits / T, dim=-1)
    p_t = F.softmax(t_logits.detach() / T, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    return alpha * ce + (1.0 - alpha) * kl


def make_distillation_loss(
    teacher: nn.Module,
    T: float,
    alpha: float,
    device: str = "cpu",
) -> LossFn:
    """Create a pluggable LossFn closure binding a frozen teacher network.

    Prepares the teacher model for distillation by transferring it to the target
    device, placing it in eval mode, and disabling parameter gradients. The returned
    callable adheres to the LossFn contract: (s_logits, labels, inputs) -> Tensor.

    Args:
        teacher: Teacher PyTorch neural network.
        T: Temperature scaling parameter (> 0).
        alpha: Balance factor between hard CE and soft KL in [0, 1].
        device: Device for teacher inference.

    Returns:
        LossFn callable compatible with `trainer.fit`.
    """
    if T <= 0:
        raise ValueError(f"Temperature T must be positive, got {T}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    def loss_fn(
        s_logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if alpha == 1.0:
            return distillation_loss_from_logits(
                s_logits=s_logits,
                t_logits=None,
                labels=labels,
                T=T,
                alpha=1.0,
            )

        if inputs is None:
            raise ValueError("inputs batch required for distillation loss when alpha < 1.0")

        # Guarantee eval mode in case external code mutated teacher state
        teacher.eval()
        with torch.no_grad():
            t_logits = teacher(inputs.to(device))

        return distillation_loss_from_logits(
            s_logits=s_logits,
            t_logits=t_logits,
            labels=labels,
            T=T,
            alpha=alpha,
        )

    return loss_fn
