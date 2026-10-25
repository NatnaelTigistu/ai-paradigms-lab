"""Reusable experiment orchestrator for hard-label and knowledge-distilled students."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from paradigms.data.fashion_mnist import FashionData, make_loader
from paradigms.distillation.loss import make_distillation_loss
from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.registry import build_model, count_parameters
from paradigms.seed import set_seed
from paradigms.training.evaluate import evaluate
from paradigms.training.losses import hard_label_loss
from paradigms.training.trainer import fit
from paradigms.utils.io import load_json, save_json
from paradigms.utils.stats import mean_std


def effective_epochs(base_epochs: int, fraction: float) -> int:
    """Compute normalized training epochs for low-data fraction regimes.

    Formula:
        int(min(400, max(base_epochs, round(base_epochs / fraction))))

    Why scale epochs:
        When training on reduced data fractions (e.g. 1%, 5%, 10%), each epoch
        contains fewer optimization updates. Scaling the epoch budget inversely
        with dataset fraction ensures all data slices receive a comparable number
        of gradient steps, isolating representation quality from optimization volume.
        Capped at 400 epochs to prevent runaway computation.
    """
    if fraction <= 0 or fraction > 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return int(min(400, max(base_epochs, round(base_epochs / fraction))))


def load_teacher(
    seed: int,
    ckpt_dir: str | Path = "checkpoints/teacher",
    device: str = "cpu",
) -> nn.Module:
    """Load pre-trained TeacherCNN weights from checkpoint, freeze parameters, and set to eval.

    Args:
        seed: Random seed identifier corresponding to teacher checkpoint.
        ckpt_dir: Directory containing `seed{seed}.pt`.
        device: Target execution device.

    Returns:
        Frozen, eval-mode TeacherCNN instance.
    """
    ckpt_path = Path(ckpt_dir) / f"seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found at: {ckpt_path}")

    teacher = TeacherCNN()
    state_dict = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(state_dict)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def run_student(
    cfg: Any,
    data: FashionData,
    *,
    seed: int,
    hidden: int,
    fraction: float,
    mode: str,
    T: Optional[float] = None,
    alpha: Optional[float] = None,
    teacher: Optional[nn.Module] = None,
    scale_epochs: bool = False,
    ckpt_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a single student training and evaluation run across a given seed.

    Args:
        cfg: Configuration object containing training hyperparameters.
        data: FashionData dataset manager.
        seed: Random seed for model initialization and batch shuffling.
        hidden: Hidden layer dimension for StudentMLP.
        fraction: Fraction of training partition to utilize.
        mode: Training mode: "hard" (standard cross-entropy) or "distilled" (teacher KD).
        T: Distillation temperature (required if mode=="distilled").
        alpha: Weight for hard cross-entropy (required if mode=="distilled").
        teacher: Pre-loaded teacher model (required if mode=="distilled").
        scale_epochs: Whether to adjust training epochs via `effective_epochs`.
        ckpt_path: Optional path to save best model weights.
        verbose: Whether to log epoch-by-epoch trainer progress.

    Returns:
        Structured dictionary recording run configuration, metrics, and history.
    """
    if mode not in ("hard", "distilled"):
        raise ValueError(f"Unknown mode '{mode}'. Expected 'hard' or 'distilled'.")

    set_seed(seed)
    model = build_model("student_mlp", hidden=hidden)

    # Use fixed split_seed for subsetting so all random seeds observe the identical data slice
    if fraction == 1.0:
        train_ds = data.get("train")
    else:
        train_ds = data.train_indices_subset(fraction, seed=int(cfg.data.split_seed))

    dev_ds = data.get("dev")

    base_epochs = int(cfg.training.epochs)
    epochs = effective_epochs(base_epochs, fraction) if scale_epochs else base_epochs
    batch_size = int(cfg.training.get("batch_size", 128))

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed)
    dev_loader = make_loader(dev_ds, batch_size=batch_size, shuffle=False, seed=seed)

    if mode == "hard":
        loss_fn = hard_label_loss
    else:
        if teacher is None:
            raise ValueError("teacher model must be provided when mode == 'distilled'")
        if T is None or alpha is None:
            raise ValueError("T and alpha must be provided when mode == 'distilled'")
        loss_fn = make_distillation_loss(teacher, T=T, alpha=alpha, device="cpu")

    training_cfg = {**cfg["training"], "epochs": epochs}
    out = fit(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        loss_fn=loss_fn,
        training_cfg=training_cfg,
        device="cpu",
        verbose=verbose,
    )

    final = evaluate(model, dev_loader, device="cpu")

    if ckpt_path is not None:
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p)

    return {
        "role": f"student_{mode}",
        "model": "student_mlp",
        "seed": seed,
        "hidden": hidden,
        "fraction": fraction,
        "mode": mode,
        "T": T,
        "alpha": alpha,
        "epochs_used": epochs,
        "params": count_parameters(model),
        "best_epoch": out["best_epoch"],
        "dev_accuracy": float(final["accuracy"]),
        "dev_per_class_accuracy": final["per_class_accuracy"],
        "history": out["history"],
        "config": dict(cfg),
    }


def run_group(
    cfg: Any,
    data: FashionData,
    seeds: list[int],
    out_dir: str | Path,
    ckpt_dir: Optional[str | Path] = None,
    resume: bool = False,
    **kw: Any,
) -> dict[str, Any]:
    """Execute training across a seed cohort, recording per-seed and summary JSON files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ckpt_path_dir = Path(ckpt_dir) if ckpt_dir is not None else None
    if ckpt_path_dir is not None:
        ckpt_path_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for seed in seeds:
        seed_json = out_path / f"seed{seed}.json"
        if resume and seed_json.exists():
            record = load_json(seed_json)
        else:
            seed_ckpt = (ckpt_path_dir / f"seed{seed}.pt") if ckpt_path_dir is not None else None
            record = run_student(
                cfg=cfg,
                data=data,
                seed=seed,
                ckpt_path=seed_ckpt,
                **kw,
            )
            save_json(record, seed_json)
        results.append(record)

    dev_accs = [r["dev_accuracy"] for r in results]
    dev_mean, dev_std = mean_std(dev_accs)
    params = results[0]["params"] if results else count_parameters(build_model("student_mlp", hidden=kw.get("hidden", 32)))

    summary = {
        "dev_accuracy_mean": dev_mean,
        "dev_accuracy_std": dev_std,
        "seeds": seeds,
        "params": params,
    }
    save_json(summary, out_path / "summary.json")

    print("\n" + "=" * 50)
    print(f"{'Seed':<8} | {'Best Epoch':<12} | {'Dev Accuracy':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['seed']:<8} | {r['best_epoch']:<12} | {r['dev_accuracy'] * 100:.2f}%")
    print("-" * 50)
    print(f"Mean ± Std: {dev_mean * 100:.2f}% ± {dev_std * 100:.2f}%")
    print("=" * 50 + "\n")

    return summary
