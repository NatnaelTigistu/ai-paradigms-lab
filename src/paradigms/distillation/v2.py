"""Phase 1 v2 student training routine with cached logits, tensor loaders, and CPU optimizations."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from paradigms.data.fashion_mnist import FashionData, MEAN, STD
from paradigms.data.tensor_loader import TensorLoader
from paradigms.distillation.cached import cache_teacher_logits
from paradigms.distillation.experiment import effective_epochs
from paradigms.models.registry import build_model, count_parameters
from paradigms.seed import set_seed


def run_student_v2(
    cfg: Any,
    data: FashionData,
    cache: Any,
    *,
    seed: int,
    mode: str,
    fraction: float,
    hidden: int = 32,
    T: Optional[float] = None,
    alpha: Optional[float] = None,
    wd: float = 0.0,
    base_epochs: int = 30,
    ckpt_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute student training in Phase 1 v2 across hard, distilled, or transfer modes.

    Optimized for CPU efficiency:
    - Sets single-thread PyTorch execution to eliminate thread sync overhead on small MLPs.
    - Precomputes teacher softmax probabilities p_t once per run for temperature T.
    - Evaluates development accuracy in a single inference pass under torch.inference_mode.
    - Avoids deepcopy overhead by cloning tensor parameters only on dev accuracy improvement.

    Args:
        cfg: Configuration dictionary or namespace.
        data: FashionData manager.
        cache: Precomputed teacher logits tensor of shape (60000, 10), or file path.
        seed: Random seed for model initialization and batch shuffling.
        mode: One of "hard", "distilled", or "distilled_transfer".
        fraction: Fraction of training data to use in (0.0, 1.0].
        hidden: Hidden dimension for student MLP (default 32).
        T: Distillation temperature (for distilled and distilled_transfer).
        alpha: Loss weighting factor in [0, 1].
        wd: Weight decay for Adam optimizer.
        base_epochs: Base epoch budget for full data (default 30).
        ckpt_path: Optional path to save best dev model checkpoint.
        verbose: Verbose trainer output flag.

    Returns:
        Result dictionary containing model performance, hyperparameters, and history.
    """
    valid_modes = ("hard", "distilled", "distilled_transfer")
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {valid_modes}")

    if mode == "distilled_transfer" and fraction >= 1.0:
        raise ValueError("distilled_transfer mode is only supported when fraction < 1.0")

    if mode in ("distilled", "distilled_transfer"):
        if T is None or alpha is None:
            raise ValueError(f"T and alpha must be provided for mode '{mode}'")

    # Enforce single-thread execution for fast small-MLP inference/training on CPU
    torch.set_num_threads(1)

    # Load cache if provided as filepath
    if isinstance(cache, (str, Path)):
        cached_logits = torch.load(cache, map_location="cpu")
    elif isinstance(cache, torch.Tensor):
        cached_logits = cache
    else:
        cached_logits = None

    # Step 1 & 2: Set seed, build student model BEFORE building loader (matching v1 RNG sequence)
    set_seed(seed)
    model = build_model("student_mlp", hidden=hidden)

    # Step 3: Determine split seed and train subset indices
    if hasattr(cfg, "data") and hasattr(cfg.data, "split_seed"):
        split_seed = int(cfg.data.split_seed)
    elif isinstance(cfg, dict):
        split_seed = int(cfg.get("data", {}).get("split_seed", 42))
    else:
        split_seed = 42

    if fraction == 1.0:
        subset_indices = list(data.splits["train"])
    else:
        subset = data.train_indices_subset(fraction, seed=split_seed)
        subset_indices = list(subset.indices)

    # Step 4: Construct contiguous tensors
    all_train_indices = list(data.splits["train"])

    if mode == "hard":
        target_indices = subset_indices
        raw_x = data.full_train.data[target_indices]
        X = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1).contiguous()
        raw_y = data.full_train.targets[target_indices].long().contiguous()

        # packed has shape (N, 11) with label in col 0 to adhere to spec
        t_dummy = torch.zeros(len(target_indices), 10, dtype=torch.float32)
        packed = torch.cat([raw_y.unsqueeze(1).float(), t_dummy], dim=1).contiguous()
        epochs = effective_epochs(base_epochs, fraction)
    elif mode == "distilled":
        target_indices = subset_indices
        raw_x = data.full_train.data[target_indices]
        X = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1).contiguous()
        raw_y = data.full_train.targets[target_indices].long().contiguous()

        if cached_logits is None:
            raise ValueError("cached teacher logits required for distilled mode")

        t_logits = cached_logits[target_indices]
        # Precompute teacher softmax probabilities at temperature T once per run
        p_t = F.softmax(t_logits / float(T), dim=-1).contiguous()
        packed = torch.cat([raw_y.unsqueeze(1).float(), p_t], dim=1).contiguous()
        epochs = effective_epochs(base_epochs, fraction)
    else:  # mode == "distilled_transfer"
        target_indices = all_train_indices
        raw_x = data.full_train.data[target_indices]
        X = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1).contiguous()

        subset_set = set(subset_indices)
        raw_y = torch.full((len(target_indices), 1), -1.0, dtype=torch.float32)
        idx_to_pos = {idx: pos for pos, idx in enumerate(target_indices)}
        for s_idx in subset_indices:
            pos = idx_to_pos[s_idx]
            raw_y[pos, 0] = float(data.full_train.targets[s_idx].item())

        if cached_logits is None:
            raise ValueError("cached teacher logits required for distilled_transfer mode")

        t_logits = cached_logits[target_indices]
        p_t = F.softmax(t_logits / float(T), dim=-1).contiguous()
        packed = torch.cat([raw_y, p_t], dim=1).contiguous()
        epochs = base_epochs

    train_loader = TensorLoader(X, packed, batch_size=128, shuffle=True, seed=seed)

    # Pre-materialize dev split as contiguous tensors for high-speed evaluation
    dev_indices = list(data.splits["dev"])
    raw_x_dev = data.full_train.data[dev_indices]
    X_dev = ((raw_x_dev.float() / 255.0 - MEAN) / STD).unsqueeze(1).contiguous()
    Y_dev = data.full_train.targets[dev_indices].long().contiguous()

    # Step 5: Optimizer setup
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=float(wd))

    # Step 6: Training loop with fast dev evaluation and clone checkpointing
    t0 = time.time()
    best_acc: float = -1.0
    best_epoch: int = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    T_val = float(T) if T is not None else 1.0
    alpha_val = float(alpha) if alpha is not None else 1.0
    T_sq = T_val * T_val

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss: float = 0.0
        n_samples: int = 0

        for x_batch, packed_batch in train_loader:
            s_logits = model(x_batch)

            if mode == "hard":
                labels = packed_batch[:, 0].long()
                loss = F.cross_entropy(s_logits, labels)
            else:
                labels = packed_batch[:, 0].long()
                p_t_batch = packed_batch[:, 1:]
                labelled = labels >= 0

                log_p_s = F.log_softmax(s_logits / T_val, dim=-1)
                kd = T_sq * F.kl_div(log_p_s, p_t_batch, reduction="none").sum(dim=-1)

                safe_labels = labels.clamp(min=0)
                ce_all = F.cross_entropy(s_logits, safe_labels, reduction="none")
                ce = torch.where(labelled, ce_all, torch.zeros_like(kd))

                labelled_f = labelled.float()
                sample_loss = (alpha_val * labelled_f * ce) + ((1.0 - alpha_val * labelled_f) * kd)
                loss = sample_loss.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            b_len = len(x_batch)
            total_loss += loss.item() * b_len
            n_samples += b_len

        train_loss = total_loss / n_samples if n_samples > 0 else 0.0

        # Fast dev evaluation under inference_mode
        with torch.inference_mode():
            dev_logits = model(X_dev)
            dev_loss = float(F.cross_entropy(dev_logits, Y_dev).item())
            dev_acc = float((dev_logits.argmax(dim=-1) == Y_dev).float().mean().item())

        elapsed_ep = time.time() - t_ep
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc,
            "seconds": elapsed_ep,
        })

        # Strict > ensures ties preserve the earliest epoch
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | "
                f"Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc * 100:.2f}% | {elapsed_ep:.2f}s"
            )

    elapsed_seconds = time.time() - t0

    # Step 7: Load best state and compute per-class accuracy
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.inference_mode():
        dev_logits = model(X_dev)
        final_preds = dev_logits.argmax(dim=-1)
        final_dev_acc = float((final_preds == Y_dev).float().mean().item())

        per_class_accuracy: list[float | None] = []
        correct = final_preds == Y_dev
        for c in range(10):
            mask = Y_dev == c
            if mask.any():
                per_class_accuracy.append(float(correct[mask].float().mean().item()))
            else:
                per_class_accuracy.append(None)

    if ckpt_path is not None:
        p_ckpt = Path(ckpt_path)
        p_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p_ckpt)

    cfg_dict = dict(cfg) if isinstance(cfg, dict) else (cfg.to_dict() if hasattr(cfg, "to_dict") else {})

    return {
        "role": f"student_{mode}",
        "model": "student_mlp",
        "seed": seed,
        "mode": mode,
        "fraction": fraction,
        "hidden": hidden,
        "T": T,
        "alpha": alpha,
        "wd": float(wd),
        "epochs_used": epochs,
        "seconds": elapsed_seconds,
        "device_info": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "processor": platform.processor(),
            "num_threads": torch.get_num_threads(),
        },
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "dev_accuracy": final_dev_acc,
        "dev_per_class_accuracy": per_class_accuracy,
        "history": history,
        "config": cfg_dict,
    }
