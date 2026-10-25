"""Script to train hard-label StudentMLP baseline as the distillation control."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from paradigms.config import apply_overrides, load_config
from paradigms.data.fashion_mnist import FashionData, make_loader
from paradigms.models.registry import build_model, count_parameters
from paradigms.seed import set_seed
from paradigms.training.evaluate import evaluate
from paradigms.training.losses import hard_label_loss
from paradigms.training.trainer import fit
from paradigms.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hard-label StudentMLP baseline on Fashion-MNIST."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds (default: seeds defined in config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training.epochs.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="Student MLP hidden layer dimensionality (default: 32).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (default: 1.0).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/student_hard",
        help="Base output directory for results (default: experiments/student_hard).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/student_hard",
        help="Base output directory for model checkpoints (default: checkpoints/student_hard).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose training progress for each epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg = apply_overrides(cfg, {"training.epochs": args.epochs})

    if "num_threads" in cfg:
        torch.set_num_threads(int(cfg["num_threads"]))
    elif torch.get_num_threads() > 8:
        torch.set_num_threads(4)

    seeds: list[int] = (
        args.seeds if args.seeds is not None else list(cfg.get("seeds", [0, 1, 2, 3, 4]))
    )

    tag = f"h{args.hidden}_f{args.fraction}"
    out_dir = Path(args.out_dir) / tag
    ckpt_dir = Path(args.ckpt_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg.training.get("batch_size", 128))

    # Initialize data partitions
    data = FashionData(
        root=cfg.data.root,
        split_seed=int(cfg.data.split_seed),
        dev_size=int(cfg.data.dev_size),
    )

    # Use fixed split_seed for subsetting so all training seeds see the identical data slice
    if args.fraction == 1.0:
        train_ds = data.get("train")
    else:
        train_ds = data.train_indices_subset(args.fraction, seed=int(cfg.data.split_seed))

    dev_ds = data.get("dev")

    results: list[dict[str, Any]] = []

    for seed in seeds:
        if args.verbose:
            print(f"\n--- Training Hard Student | Tag: {tag} | Seed {seed} ---")

        set_seed(seed)
        model = build_model("student_mlp", hidden=args.hidden)
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed)
        dev_loader = make_loader(dev_ds, batch_size=batch_size, shuffle=False, seed=seed)

        out = fit(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            loss_fn=hard_label_loss,
            training_cfg=cfg["training"],
            device="cpu",
            verbose=args.verbose,
        )

        final = evaluate(model, dev_loader, device="cpu")

        # Save checkpoint
        ckpt_path = ckpt_dir / f"seed{seed}.pt"
        torch.save(model.state_dict(), ckpt_path)

        record = {
            "role": "student_hard",
            "model": "student_mlp",
            "seed": seed,
            "hidden": args.hidden,
            "fraction": args.fraction,
            "params": count_parameters(model),
            "best_epoch": out["best_epoch"],
            "dev_accuracy": float(final["accuracy"]),
            "dev_per_class_accuracy": final["per_class_accuracy"],
            "history": out["history"],
            "config": dict(cfg),
        }
        save_json(record, out_dir / f"seed{seed}.json")
        results.append(record)

    # Summary across seeds
    dev_accs = [r["dev_accuracy"] for r in results]
    dev_mean = float(np.mean(dev_accs)) if dev_accs else 0.0
    dev_std = float(np.std(dev_accs, ddof=0)) if dev_accs else 0.0
    params = results[0]["params"] if results else count_parameters(build_model("student_mlp", hidden=args.hidden))

    summary = {
        "dev_accuracy_mean": dev_mean,
        "dev_accuracy_std": dev_std,
        "seeds": seeds,
        "params": params,
    }
    save_json(summary, out_dir / "summary.json")

    print("\n" + "=" * 50)
    print(f"{'Seed':<8} | {'Best Epoch':<12} | {'Dev Accuracy':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['seed']:<8} | {r['best_epoch']:<12} | {r['dev_accuracy'] * 100:.2f}%")
    print("-" * 50)
    print(f"Mean ± Std: {dev_mean * 100:.2f}% ± {dev_std * 100:.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
