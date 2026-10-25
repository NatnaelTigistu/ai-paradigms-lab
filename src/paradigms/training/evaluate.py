"""Evaluation utilities and metric computation routines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Compute classification metrics directly from unnormalized logits and targets.

    Pure function that does not depend on any model instance.

    Args:
        logits: Unnormalized class predictions of shape (N, C).
        labels: Ground-truth target indices of shape (N,).
        num_classes: Number of distinct classes to track.

    Returns:
        Dictionary containing:
            - "accuracy": Overall classification accuracy in [0.0, 1.0].
            - "loss": Mean cross-entropy loss over all samples.
            - "per_class_accuracy": List of accuracies per class (or None if unobserved).
            - "num_samples": Total number of evaluated samples N.
    """
    num_samples = len(labels)
    if num_samples == 0:
        return {
            "accuracy": 0.0,
            "loss": 0.0,
            "per_class_accuracy": [None] * num_classes,
            "num_samples": 0,
        }

    preds = torch.argmax(logits, dim=-1)
    correct = preds == labels
    accuracy = float(correct.float().mean().item())
    loss = float(F.cross_entropy(logits, labels).item())

    per_class_accuracy: list[float | None] = []
    for c in range(num_classes):
        mask = labels == c
        count = int(mask.sum().item())
        if count == 0:
            per_class_accuracy.append(None)
        else:
            acc_c = float(correct[mask].float().mean().item())
            per_class_accuracy.append(acc_c)

    return {
        "accuracy": accuracy,
        "loss": loss,
        "per_class_accuracy": per_class_accuracy,
        "num_samples": num_samples,
    }


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model inference across a DataLoader and accumulate logits on CPU.

    Preserves the caller's model.training state upon completion. Generalizes
    to both labeled and unlabeled inputs, serving evaluation and future pseudo-labeling.

    Args:
        model: PyTorch model producing raw class logits.
        loader: DataLoader providing input batches.
        device: Execution device ("cpu" by default).

    Returns:
        Tuple of (all_logits, all_labels) concatenated on CPU.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else torch.empty(len(x), dtype=torch.long)
        else:
            x = batch
            y = torch.empty(len(x), dtype=torch.long)

        x = x.to(device)
        batch_logits = model(x).detach().cpu()

        logits_list.append(batch_logits)
        labels_list.append(y.detach().cpu())

    model.train(was_training)

    if not logits_list:
        return torch.empty(0, 10), torch.empty(0, dtype=torch.long)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return all_logits, all_labels


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    num_classes: int = 10,
) -> dict[str, Any]:
    """Evaluate model performance on a DataLoader using cross-entropy and accuracy.

    Args:
        model: Model under evaluation.
        loader: DataLoader representing validation or test samples.
        device: Device for inference.
        num_classes: Expected number of classes.

    Returns:
        Dictionary of metrics produced by `metrics_from_logits`.
    """
    logits, labels = predict_logits(model, loader, device=device)
    return metrics_from_logits(logits, labels, num_classes=num_classes)
