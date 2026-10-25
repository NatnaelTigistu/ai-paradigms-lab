"""Script to train hard-label StudentMLP baseline using the generic experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from paradigms.config import apply_overrides, load_config
from paradigms.data.fashion_mnist import FashionData
from paradigms.distillation.experiment import run_group


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
        "--scale-epochs",
        action="store_true",
        help="Scale epochs inversely with fraction using effective_epochs.",
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
        "--resume",
        action="store_true",
        help="Resume experiment by loading existing seed JSON results.",
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

    data = FashionData(
        root=cfg.data.root,
        split_seed=int(cfg.data.split_seed),
        dev_size=int(cfg.data.dev_size),
    )

    run_group(
        cfg=cfg,
        data=data,
        seeds=seeds,
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        resume=args.resume,
        mode="hard",
        hidden=args.hidden,
        fraction=args.fraction,
        scale_epochs=args.scale_epochs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
