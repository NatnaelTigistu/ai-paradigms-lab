"""Script to train knowledge-distilled StudentMLP models against a frozen teacher CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from paradigms.config import apply_overrides, load_config
from paradigms.data.fashion_mnist import FashionData
from paradigms.distillation.experiment import load_teacher, run_group
from paradigms.utils.io import load_json
from paradigms.utils.stats import paired_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train knowledge-distilled StudentMLP on Fashion-MNIST."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML configuration file (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--dist-config",
        type=str,
        default="configs/distillation.yaml",
        help="Path to distillation YAML configuration file (default: configs/distillation.yaml).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds (default: seeds from config).",
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
        help="Student MLP hidden dimension (default: 32).",
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
        "--T",
        type=float,
        default=None,
        help="Distillation temperature (default: from config, e.g. 4.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Distillation alpha weight for hard loss (default: from config, e.g. 0.5).",
    )
    parser.add_argument(
        "--teacher-seed",
        type=int,
        default=None,
        help="Seed identifying the teacher checkpoint (default: from config).",
    )
    parser.add_argument(
        "--teacher-ckpt-dir",
        type=str,
        default=None,
        help="Directory containing teacher checkpoints.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/student_distilled",
        help="Base output directory for results (default: experiments/student_distilled).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/student_distilled",
        help="Base output directory for model checkpoints (default: checkpoints/student_distilled).",
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

    # Load and merge base and distillation configs
    base_cfg = load_config(args.config)
    dist_cfg = load_config(args.dist_config)
    cfg = apply_overrides(base_cfg, dist_cfg)

    if args.epochs is not None:
        cfg = apply_overrides(cfg, {"training.epochs": args.epochs})

    if "num_threads" in cfg:
        torch.set_num_threads(int(cfg["num_threads"]))
    elif torch.get_num_threads() > 8:
        torch.set_num_threads(4)

    seeds: list[int] = (
        args.seeds if args.seeds is not None else list(cfg.get("seeds", [0, 1, 2, 3, 4]))
    )

    T = float(args.T) if args.T is not None else float(cfg.distillation.default_T)
    alpha = float(args.alpha) if args.alpha is not None else float(cfg.distillation.default_alpha)
    teacher_seed = (
        int(args.teacher_seed)
        if args.teacher_seed is not None
        else int(cfg.distillation.teacher_seed)
    )
    teacher_ckpt_dir = (
        args.teacher_ckpt_dir
        if args.teacher_ckpt_dir is not None
        else str(cfg.distillation.teacher_ckpt_dir)
    )

    tag = f"h{args.hidden}_f{args.fraction}_T{T}_a{alpha}"
    out_dir = Path(args.out_dir) / tag
    ckpt_dir = Path(args.ckpt_dir) / tag

    # Load frozen teacher
    teacher = load_teacher(seed=teacher_seed, ckpt_dir=teacher_ckpt_dir, device="cpu")

    data = FashionData(
        root=cfg.data.root,
        split_seed=int(cfg.data.split_seed),
        dev_size=int(cfg.data.dev_size),
    )

    summary = run_group(
        cfg=cfg,
        data=data,
        seeds=seeds,
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        resume=args.resume,
        mode="distilled",
        hidden=args.hidden,
        fraction=args.fraction,
        scale_epochs=args.scale_epochs,
        T=T,
        alpha=alpha,
        teacher=teacher,
        verbose=args.verbose,
    )

    # Check for matched hard student results for comparison
    hard_tag = f"h{args.hidden}_f{args.fraction}"
    hard_dir = Path("experiments/student_hard") / hard_tag
    teacher_summary_path = Path("experiments/teacher/summary.json")

    hard_records = []
    dist_records = []

    for s in seeds:
        dist_f = out_dir / f"seed{s}.json"
        hard_f = hard_dir / f"seed{s}.json"
        if dist_f.exists() and hard_f.exists():
            dist_records.append(load_json(dist_f))
            hard_records.append(load_json(hard_f))

    if len(dist_records) == len(seeds) and len(hard_records) == len(seeds):
        hard_accs = [r["dev_accuracy"] for r in hard_records]
        dist_accs = [r["dev_accuracy"] for r in dist_records]

        print("\n" + "=" * 65)
        print("PAIRED COMPARISON: Hard Student vs Distilled Student")
        print("-" * 65)
        print(f"{'Seed':<6} | {'Hard Dev Acc':<14} | {'Distilled Dev Acc':<18} | {'Diff':<10}")
        print("-" * 65)
        for s, h_acc, d_acc in zip(seeds, hard_accs, dist_accs):
            diff = (d_acc - h_acc) * 100
            print(f"{s:<6} | {h_acc * 100:6.2f}%        | {d_acc * 100:6.2f}%           | {diff:+6.2f}%")
        print("-" * 65)

        stats = paired_comparison(dist_accs, hard_accs)
        mean_diff = stats["mean_diff"] * 100
        std_diff = stats["std_diff"] * 100
        wins = stats["wins"]
        n = stats["n"]
        print(f"Mean Diff: {mean_diff:+.2f}% ± {std_diff:.2f}% | Wins: {wins}/{n}")
        print("=" * 65 + "\n")

    # Three-row summary table
    if teacher_summary_path.exists() and (hard_dir / "summary.json").exists():
        t_summary = load_json(teacher_summary_path)
        h_summary = load_json(hard_dir / "summary.json")

        print("=" * 65)
        print("OVERALL PARADIGM COMPARISON (DEV ACCURACY)")
        print("-" * 65)
        print(f"{'Model':<22} | {'Params':<10} | {'Dev Acc (Mean ± Std)':<22}")
        print("-" * 65)
        print(
            f"{'Teacher (CNN)':<22} | {t_summary['params']:>10,d} | "
            f"{t_summary['dev_accuracy_mean'] * 100:6.2f}% ± {t_summary['dev_accuracy_std'] * 100:4.2f}%"
        )
        print(
            f"{'Hard Student (MLP)':<22} | {h_summary['params']:>10,d} | "
            f"{h_summary['dev_accuracy_mean'] * 100:6.2f}% ± {h_summary['dev_accuracy_std'] * 100:4.2f}%"
        )
        print(
            f"{'Distilled Student':<22} | {summary['params']:>10,d} | "
            f"{summary['dev_accuracy_mean'] * 100:6.2f}% ± {summary['dev_accuracy_std'] * 100:4.2f}%"
        )
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
