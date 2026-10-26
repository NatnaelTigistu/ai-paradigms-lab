"""Cached teacher inference and masked knowledge distillation loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from paradigms.data.fashion_mnist import FashionData, MEAN, STD


def cache_teacher_logits(
    teacher: nn.Module,
    data: FashionData,
    path: str | Path = "checkpoints/v2/teacher_logits.pt",
    batch_size: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute and cache teacher logits for all 60,000 train images in dataset order.

    If cached tensor already exists at `path`, loads and returns it. Otherwise,
    evaluates the frozen teacher in eval mode without gradients and persists
    the resulting (60000, 10) tensor.

    Args:
        teacher: Trained PyTorch teacher model.
        data: FashionData instance managing Fashion-MNIST dataset.
        path: Filepath for cached tensor storage.
        batch_size: Inference batch size for logits computation.
        device: Device on which teacher inference executes.

    Returns:
        torch.Tensor of shape (60000, 10) containing raw teacher logits.
    """
    p = Path(path)
    if p.exists():
        return torch.load(p, map_location="cpu")

    teacher.eval()
    teacher.to(device)

    raw_data = data.full_train.data
    X = ((raw_data.float() / 255.0 - MEAN) / STD).unsqueeze(1)
    n_samples = len(X)

    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = X[i : i + batch_size].to(device)
            batch_logits = teacher(batch_x).detach().cpu()
            logits_list.append(batch_logits)

    all_logits = torch.cat(logits_list, dim=0)
    if all_logits.shape != (n_samples, 10):
        raise ValueError(
            f"Expected cached logits shape ({n_samples}, 10), got {all_logits.shape}"
        )

    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_logits, p)
    return all_logits


def masked_distillation_loss(
    s_logits: torch.Tensor,
    labels: torch.Tensor,
    t_logits: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    """Compute masked distillation loss supporting partially or fully unlabeled batches.

    A label value of -1 indicates an unlabelled sample.
    Per sample i:
        kd_i = T * T * sum_c p_t(c) * (log p_t(c) - log p_s(c))
        ce_i = cross_entropy(s_i, y_i) if labelled else 0
        loss = mean_i( alpha * labelled_i * ce_i + (1 - alpha * labelled_i) * kd_i )

    Labelled samples contribute alpha * CE + (1 - alpha) * KD.
    Unlabelled samples contribute pure KD (1 * KD).

    Args:
        s_logits: Student logits of shape (B, C).
        labels: Target labels of shape (B,), with -1 denoting unlabelled.
        t_logits: Teacher logits of shape (B, C).
        T: Distillation temperature (> 0).
        alpha: Weight for hard cross-entropy in [0, 1].

    Raises:
        ValueError: If T <= 0 or alpha not in [0, 1].

    Returns:
        Scalar loss tensor.
    """
    if T <= 0:
        raise ValueError(f"Temperature T must be positive, got {T}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    labelled = labels >= 0

    # Soft teacher and student distributions scaled by T
    log_p_s = F.log_softmax(s_logits / T, dim=-1)
    p_t = F.softmax(t_logits.detach() / T, dim=-1)
    # PyTorch F.kl_div with log_target=False computes target * (log(target) - input)
    kd = (T * T) * F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)

    # Cross-entropy for labelled samples (clamp min=0 to prevent negative index errors)
    safe_labels = labels.clamp(min=0)
    ce_all = F.cross_entropy(s_logits, safe_labels, reduction="none")
    ce = torch.where(labelled, ce_all, torch.zeros_like(kd))

    labelled_f = labelled.float()
    sample_loss = (alpha * labelled_f * ce) + ((1.0 - alpha * labelled_f) * kd)
    return sample_loss.mean()
