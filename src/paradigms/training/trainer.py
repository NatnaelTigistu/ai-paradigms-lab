"""Generic model trainer decoupled from loss functions and model architectures."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from paradigms.training.evaluate import evaluate
from paradigms.training.losses import LossFn


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    loss_fn: LossFn,
    training_cfg: dict[str, Any],
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, Any]:
    """Train a model using a pluggable loss function and best-checkpoint selection.

    Architectural Rationale:
    1. Agnostic to Loss Formulation: Both hard-label baseline training and knowledge
       distillation utilize this identical trainer, altering only `loss_fn`. This
       ensures all optimizer settings, epoch counts, batch ordering, and validation
       schedules remain perfectly controlled and comparable.
    2. Best-by-Dev Checkpointing: Checkpoints are selected strictly based on peak
       development accuracy (`dev_acc`). At no point during training is the protected
       test set accessed. Ties in dev accuracy preserve the earlier epoch (strict `>`).
    3. Uniform Validation Metric: Validation loss (`dev_loss`) is always evaluated via
       unadulterated cross-entropy, even when the training loss is distillation-based,
       providing a standardized objective benchmark across paradigms.
    4. Explicit RNG Control: `fit()` avoids seeding internally. Callers explicitly
       seed the environment before building models and loaders to guarantee determinism.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader yielding training batches (x, y).
        dev_loader: DataLoader yielding development validation batches.
        loss_fn: Callable conforming to LossFn: loss_fn(logits, y, x) -> Tensor.
        training_cfg: Configuration dictionary containing:
            - "epochs": int, total training epochs.
            - "lr": float, learning rate.
            - "optimizer": str, "adam" or "sgd".
            - "weight_decay": float, L2 regularization factor.
        device: Device on which training takes place (default "cpu").
        verbose: If True, prints epoch progress summary.

    Raises:
        ValueError: If an unsupported optimizer is specified in `training_cfg`.

    Returns:
        Dictionary containing:
            - "history": List of epoch record dicts with keys:
                epoch, train_loss, dev_loss, dev_acc, seconds.
            - "best_epoch": Integer epoch index of the highest dev accuracy.
            - "best_dev_acc": Highest dev accuracy attained.
    """
    model.to(device)

    opt_name = str(training_cfg.get("optimizer", "")).lower()
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer '{training_cfg.get('optimizer')}'. Expected 'adam' or 'sgd'."
        )

    epochs = int(training_cfg["epochs"])
    best_acc: float = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch: int = -1
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss: float = 0.0
        n_samples: int = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        train_loss = total_loss / n_samples if n_samples > 0 else 0.0
        dev = evaluate(model, dev_loader, device=device)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev["loss"],
            "dev_acc": dev["accuracy"],
            "seconds": elapsed,
        }
        history.append(record)

        # Strict '>' ensures ties retain the earliest epoch
        if dev["accuracy"] > best_acc:
            best_acc = dev["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Dev Loss: {dev['loss']:.4f} | "
                f"Dev Acc: {dev['accuracy']:.4f} | "
                f"{elapsed:.2f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_dev_acc": best_acc,
    }
