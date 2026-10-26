"""Phase 1: Knowledge distillation parameter sweeps across temperature, alpha, fraction, and size.

Executes sequential one-at-a-time sweeps on Fashion-MNIST using strictly the development
split (dev) for model selection and validation.

Stages:
  1. Temperature: Sweeps T in [1, 2, 4, 8, 16] with h=32, f=1.0, alpha=default_alpha.
     Selects best_T by highest mean dev accuracy.
  2. Alpha: Sweeps alpha in [0.0, 0.3, 0.5, 0.7, 1.0] at T=best_T.
     Runtime sanity check: verifies alpha=1.0 matches hard baseline within 1e-9.
     Selects best_alpha by highest mean dev accuracy among alpha < 1.0.
     Saves best_config.json with T, alpha, tables, and selection metadata.
  3. Fraction: Sweeps f in [0.01, 0.05, 0.10, 0.25, 1.0] for both hard and distilled
     modes with h=32, scale_epochs=True, T=best_T, alpha=best_alpha.
  4. Size: Sweeps h in [8, 16, 32, 64, 128] for both hard and distilled modes with
     f=1.0, scale_epochs=False, T=best_T, alpha=best_alpha.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from paradigms.config import apply_overrides, load_config
from paradigms.data.fashion_mnist import FashionData
from paradigms.distillation.experiment import load_teacher, run_group
from paradigms.utils.io import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 Knowledge Distillation sweeps on Fashion-MNIST."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["temperature", "alpha", "fraction", "size", "all"],
        default="all",
        help="Stage to execute (default: all).",
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
        help="Optional override for training.epochs (e.g. for smoke testing).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip groups where summary.json exists (default: True).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/distillation",
        help="Base output directory for experiment results (default: experiments/distillation).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/distillation",
        help="Base output directory for model checkpoints (default: checkpoints/distillation).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML config (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--dist-config",
        type=str,
        default="configs/distillation.yaml",
        help="Path to distillation YAML config (default: configs/distillation.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose trainer logging.",
    )
    return parser.parse_args()


def execute_group(
    cfg: Any,
    data: FashionData,
    seeds: list[int],
    out_dir: Path,
    ckpt_dir: Path,
    mode: str,
    hidden: int,
    fraction: float,
    scale_epochs: bool,
    T: Optional[float] = None,
    alpha: Optional[float] = None,
    teacher: Optional[nn.Module] = None,
    resume: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a seed cohort group or load existing summary if resume=True."""
    summary_file = out_dir / "summary.json"
    if resume and summary_file.exists():
        print(f"Skipping {out_dir} - summary.json already exists.")
        return load_json(summary_file)

    return run_group(
        cfg=cfg,
        data=data,
        seeds=seeds,
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        resume=resume,
        hidden=hidden,
        fraction=fraction,
        mode=mode,
        T=T,
        alpha=alpha,
        teacher=teacher,
        scale_epochs=scale_epochs,
        verbose=verbose,
    )


def run_temperature_stage(
    cfg: Any,
    data: FashionData,
    teacher: nn.Module,
    seeds: list[int],
    out_base: Path,
    ckpt_base: Path,
    resume: bool,
    verbose: bool,
) -> tuple[float, list[dict[str, Any]]]:
    """Execute Stage 'temperature': sweeps T in sweeps.temperature.

    Returns:
        (best_T, temperature_table)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: TEMPERATURE SWEEP")
    print("=" * 60)

    t_values = cfg.distillation.sweeps.temperature
    default_alpha = float(cfg.distillation.default_alpha)
    results_table: list[dict[str, Any]] = []

    stage_out = out_base / "temperature"
    stage_ckpt = ckpt_base / "temperature"

    for T_val in t_values:
        tag = f"T{T_val}"
        group_out = stage_out / tag / "distilled"
        group_ckpt = stage_ckpt / tag / "distilled"
        print(f"\n---> Running Temperature: T={T_val} (alpha={default_alpha})")
        summary = execute_group(
            cfg=cfg,
            data=data,
            seeds=seeds,
            out_dir=group_out,
            ckpt_dir=group_ckpt,
            mode="distilled",
            hidden=32,
            fraction=1.0,
            scale_epochs=False,
            T=float(T_val),
            alpha=default_alpha,
            teacher=teacher,
            resume=resume,
            verbose=verbose,
        )
        results_table.append(
            {
                "T": T_val,
                "dev_mean": float(summary["dev_accuracy_mean"]),
                "dev_std": float(summary["dev_accuracy_std"]),
            }
        )

    # Find best_T by highest mean dev accuracy
    best_item = max(results_table, key=lambda x: x["dev_mean"])
    best_T = best_item["T"]

    print("\n" + "-" * 45)
    print("TEMPERATURE SWEEP SUMMARY TABLE")
    print("-" * 45)
    print(f"{'Setting (T)':<15} | {'Mean Dev Acc ± Std':<25}")
    print("-" * 45)
    for r in results_table:
        is_best = " (BEST)" if r["T"] == best_T else ""
        print(f"T={r['T']:<13} | {r['dev_mean'] * 100:.2f}% ± {r['dev_std'] * 100:.2f}%{is_best}")
    print("-" * 45)
    print(f"Selected best_T = {best_T} (dev acc: {best_item['dev_mean'] * 100:.2f}%)\n")

    return best_T, results_table


def run_alpha_stage(
    cfg: Any,
    data: FashionData,
    teacher: nn.Module,
    seeds: list[int],
    best_T: float,
    out_base: Path,
    ckpt_base: Path,
    resume: bool,
    verbose: bool,
) -> tuple[float, list[dict[str, Any]]]:
    """Execute Stage 'alpha': sweeps alpha in sweeps.alpha with T=best_T.

    Includes runtime sanity check verifying alpha=1.0 matches hard baseline within 1e-9.
    Selects best_alpha among alphas excluding 1.0.

    Returns:
        (best_alpha, alpha_table)
    """
    print("\n" + "=" * 60)
    print(f"STAGE 2: ALPHA SWEEP (Fixed T={best_T})")
    print("=" * 60)

    alpha_values = cfg.distillation.sweeps.alpha
    results_table: list[dict[str, Any]] = []

    stage_out = out_base / "alpha"
    stage_ckpt = ckpt_base / "alpha"

    for a_val in alpha_values:
        tag = f"a{a_val}"
        group_out = stage_out / tag / "distilled"
        group_ckpt = stage_ckpt / tag / "distilled"
        print(f"\n---> Running Alpha: alpha={a_val} (T={best_T})")
        summary = execute_group(
            cfg=cfg,
            data=data,
            seeds=seeds,
            out_dir=group_out,
            ckpt_dir=group_ckpt,
            mode="distilled",
            hidden=32,
            fraction=1.0,
            scale_epochs=False,
            T=float(best_T),
            alpha=float(a_val),
            teacher=teacher,
            resume=resume,
            verbose=verbose,
        )
        results_table.append(
            {
                "alpha": a_val,
                "dev_mean": float(summary["dev_accuracy_mean"]),
                "dev_std": float(summary["dev_accuracy_std"]),
            }
        )

    # RUNTIME SANITY CHECK: alpha=1.0 must match hard baseline (experiments/student_hard/h32_f1.0) per seed within 1e-9
    print("\nVerifying runtime sanity check for alpha=1.0 group against hard baseline...")
    hard_baseline_dir = Path("experiments/student_hard/h32_f1.0")
    a1_dir = stage_out / "a1.0" / "distilled"
    for s in seeds:
        a1_seed_file = a1_dir / f"seed{s}.json"
        hard_seed_file = hard_baseline_dir / f"seed{s}.json"
        if not a1_seed_file.exists():
            raise FileNotFoundError(f"Missing alpha=1.0 seed file: {a1_seed_file}")
        if not hard_seed_file.exists():
            print(f"Notice: Hard baseline {hard_seed_file} not found; skipping sanity check for seed {s}.")
            continue

        a1_rec = load_json(a1_seed_file)
        hard_rec = load_json(hard_seed_file)

        # Check if epochs match (if smoke testing with --epochs 1, baseline may differ)
        if a1_rec.get("epochs_used") == hard_rec.get("epochs_used"):
            diff = abs(float(a1_rec["dev_accuracy"]) - float(hard_rec["dev_accuracy"]))
            if diff > 1e-9:
                msg = (
                    f"RUNTIME SANITY CHECK FAILED for seed {s}: "
                    f"distilled a=1.0 dev acc ({a1_rec['dev_accuracy']}) != "
                    f"hard baseline ({hard_rec['dev_accuracy']}), diff={diff:.2e} > 1e-9"
                )
                print(f"ERROR: {msg}")
                raise RuntimeError(msg)
            print(f"  [PASS] Seed {s}: |a=1.0 - hard| = {diff:.2e} <= 1e-9")
        else:
            print(
                f"  [NOTICE] Seed {s}: Epoch counts differ ({a1_rec.get('epochs_used')} vs "
                f"{hard_rec.get('epochs_used')}), skipping strict 1e-9 equality check."
            )

    print("RUNTIME SANITY CHECK PASSED: alpha=1.0 matches hard baseline.")

    # Select best_alpha among alphas excluding 1.0
    candidates = [r for r in results_table if float(r["alpha"]) < 1.0]
    best_item = max(candidates, key=lambda x: x["dev_mean"])
    best_alpha = best_item["alpha"]

    print("\n" + "-" * 45)
    print("ALPHA SWEEP SUMMARY TABLE")
    print("-" * 45)
    print(f"{'Setting (alpha)':<15} | {'Mean Dev Acc ± Std':<25}")
    print("-" * 45)
    for r in results_table:
        is_best = " (BEST)" if r["alpha"] == best_alpha else ""
        print(f"alpha={r['alpha']:<9} | {r['dev_mean'] * 100:.2f}% ± {r['dev_std'] * 100:.2f}%{is_best}")
    print("-" * 45)
    print(f"Selected best_alpha = {best_alpha} (dev acc: {best_item['dev_mean'] * 100:.2f}%)\n")

    return best_alpha, results_table


def run_fraction_stage(
    cfg: Any,
    data: FashionData,
    teacher: nn.Module,
    seeds: list[int],
    best_T: float,
    best_alpha: float,
    out_base: Path,
    ckpt_base: Path,
    resume: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Execute Stage 'fraction': sweeps data fractions with scale_epochs=True.

    Runs both hard and distilled modes for each fraction.
    """
    print("\n" + "=" * 60)
    print(f"STAGE 3: DATA FRACTION SWEEP (T={best_T}, alpha={best_alpha})")
    print("=" * 60)

    fraction_values = cfg.distillation.sweeps.fraction
    results_table: list[dict[str, Any]] = []

    stage_out = out_base / "fraction"
    stage_ckpt = ckpt_base / "fraction"

    for f_val in fraction_values:
        tag = f"f{f_val}"
        print(f"\n---> Running Data Fraction: {f_val} (scale_epochs=True)")
        row: dict[str, Any] = {"fraction": f_val}

        for mode in ("hard", "distilled"):
            group_out = stage_out / tag / mode
            group_ckpt = stage_ckpt / tag / mode
            summary = execute_group(
                cfg=cfg,
                data=data,
                seeds=seeds,
                out_dir=group_out,
                ckpt_dir=group_ckpt,
                mode=mode,
                hidden=32,
                fraction=float(f_val),
                scale_epochs=True,
                T=float(best_T) if mode == "distilled" else None,
                alpha=float(best_alpha) if mode == "distilled" else None,
                teacher=teacher if mode == "distilled" else None,
                resume=resume,
                verbose=verbose,
            )
            row[f"{mode}_mean"] = float(summary["dev_accuracy_mean"])
            row[f"{mode}_std"] = float(summary["dev_accuracy_std"])

        row["gain"] = row["distilled_mean"] - row["hard_mean"]
        results_table.append(row)

    print("\n" + "-" * 75)
    print("DATA FRACTION SWEEP SUMMARY TABLE")
    print("-" * 75)
    print(f"{'Fraction':<10} | {'Hard Mean ± Std':<22} | {'Distilled Mean ± Std':<22} | {'KD Gain':<10}")
    print("-" * 75)
    for r in results_table:
        hard_str = f"{r['hard_mean'] * 100:.2f}% ± {r['hard_std'] * 100:.2f}%"
        dist_str = f"{r['distilled_mean'] * 100:.2f}% ± {r['distilled_std'] * 100:.2f}%"
        gain_str = f"{r['gain'] * 100:+.2f}%"
        print(f"{r['fraction']:<10} | {hard_str:<22} | {dist_str:<22} | {gain_str:<10}")
    print("-" * 75 + "\n")

    return results_table


def run_size_stage(
    cfg: Any,
    data: FashionData,
    teacher: nn.Module,
    seeds: list[int],
    best_T: float,
    best_alpha: float,
    out_base: Path,
    ckpt_base: Path,
    resume: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Execute Stage 'size': sweeps student hidden dimension with fraction=1.0.

    Runs both hard and distilled modes for each size.
    """
    print("\n" + "=" * 60)
    print(f"STAGE 4: STUDENT SIZE SWEEP (T={best_T}, alpha={best_alpha})")
    print("=" * 60)

    hidden_values = cfg.distillation.sweeps.hidden
    results_table: list[dict[str, Any]] = []

    stage_out = out_base / "size"
    stage_ckpt = ckpt_base / "size"

    for h_val in hidden_values:
        tag = f"h{h_val}"
        print(f"\n---> Running Student Hidden Size: {h_val}")
        row: dict[str, Any] = {"hidden": h_val}

        for mode in ("hard", "distilled"):
            group_out = stage_out / tag / mode
            group_ckpt = stage_ckpt / tag / mode
            summary = execute_group(
                cfg=cfg,
                data=data,
                seeds=seeds,
                out_dir=group_out,
                ckpt_dir=group_ckpt,
                mode=mode,
                hidden=int(h_val),
                fraction=1.0,
                scale_epochs=False,
                T=float(best_T) if mode == "distilled" else None,
                alpha=float(best_alpha) if mode == "distilled" else None,
                teacher=teacher if mode == "distilled" else None,
                resume=resume,
                verbose=verbose,
            )
            row[f"{mode}_mean"] = float(summary["dev_accuracy_mean"])
            row[f"{mode}_std"] = float(summary["dev_accuracy_std"])

        row["gain"] = row["distilled_mean"] - row["hard_mean"]
        results_table.append(row)

    print("\n" + "-" * 75)
    print("STUDENT SIZE SWEEP SUMMARY TABLE")
    print("-" * 75)
    print(f"{'Hidden':<10} | {'Hard Mean ± Std':<22} | {'Distilled Mean ± Std':<22} | {'KD Gain':<10}")
    print("-" * 75)
    for r in results_table:
        hard_str = f"{r['hard_mean'] * 100:.2f}% ± {r['hard_std'] * 100:.2f}%"
        dist_str = f"{r['distilled_mean'] * 100:.2f}% ± {r['distilled_std'] * 100:.2f}%"
        gain_str = f"{r['gain'] * 100:+.2f}%"
        print(f"{r['hidden']:<10} | {hard_str:<22} | {dist_str:<22} | {gain_str:<10}")
    print("-" * 75 + "\n")

    return results_table


def main() -> None:
    args = parse_args()

    # Load and merge configurations
    base_cfg = load_config(args.config)
    dist_cfg = load_config(args.dist_config)
    cfg = apply_overrides(base_cfg, dist_cfg)

    if args.epochs is not None:
        cfg = apply_overrides(cfg, {"training.epochs": args.epochs})

    # Thread management for optimal CPU throughput
    if "num_threads" in cfg:
        torch.set_num_threads(int(cfg["num_threads"]))
    elif torch.get_num_threads() > 8:
        torch.set_num_threads(4)

    seeds: list[int] = (
        args.seeds if args.seeds is not None else list(cfg.get("seeds", [0, 1, 2, 3, 4]))
    )

    out_base = Path(args.out_dir)
    ckpt_base = Path(args.ckpt_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    data = FashionData(
        root=cfg.data.root,
        split_seed=int(cfg.data.split_seed),
        dev_size=int(cfg.data.dev_size),
    )

    teacher_seed = int(cfg.distillation.teacher_seed)
    teacher_ckpt_dir = str(cfg.distillation.teacher_ckpt_dir)
    teacher = load_teacher(seed=teacher_seed, ckpt_dir=teacher_ckpt_dir, device="cpu")

    best_config_file = out_base / "best_config.json"
    best_T: Optional[float] = None
    best_alpha: Optional[float] = None
    temperature_table: list[dict[str, Any]] = []
    alpha_table: list[dict[str, Any]] = []

    if best_config_file.exists():
        existing_best = load_json(best_config_file)
        best_T = existing_best.get("T")
        best_alpha = existing_best.get("alpha")
        temperature_table = existing_best.get("temperature_table", [])
        alpha_table = existing_best.get("alpha_table", [])

    stage = args.stage

    # 1. Temperature Stage
    if stage in ("temperature", "all"):
        best_T, temperature_table = run_temperature_stage(
            cfg=cfg,
            data=data,
            teacher=teacher,
            seeds=seeds,
            out_base=out_base,
            ckpt_base=ckpt_base,
            resume=args.resume,
            verbose=args.verbose,
        )

    # 2. Alpha Stage
    if stage in ("alpha", "all"):
        if best_T is None or not temperature_table:
            # Try to resolve best_T and temperature_table from temperature folder
            t_stage_dir = out_base / "temperature"
            if t_stage_dir.exists():
                t_candidates = []
                for p in t_stage_dir.glob("T*/distilled/summary.json"):
                    t_str = p.parent.parent.name[1:]
                    val = float(t_str) if "." in t_str else int(t_str)
                    summary = load_json(p)
                    t_candidates.append({
                        "T": val,
                        "dev_mean": float(summary["dev_accuracy_mean"]),
                        "dev_std": float(summary["dev_accuracy_std"]),
                    })
                if t_candidates:
                    t_candidates.sort(key=lambda x: x["T"])
                    if not temperature_table:
                        temperature_table = t_candidates
                    if best_T is None:
                        best_T = max(t_candidates, key=lambda x: x["dev_mean"])["T"]
            if best_T is None:
                best_T = float(cfg.distillation.default_T)
                print(f"Notice: defaulting best_T to {best_T}")

        best_alpha, alpha_table = run_alpha_stage(
            cfg=cfg,
            data=data,
            teacher=teacher,
            seeds=seeds,
            best_T=best_T,
            out_base=out_base,
            ckpt_base=ckpt_base,
            resume=args.resume,
            verbose=args.verbose,
        )

        best_config = {
            "T": best_T,
            "alpha": best_alpha,
            "selected_on": "dev",
            "temperature_table": temperature_table,
            "alpha_table": alpha_table,
        }
        save_json(best_config, best_config_file)
        print(f"Saved distillation hyperparameters to: {best_config_file}")

    # Ensure best_T and best_alpha are loaded for fraction and size stages
    if stage in ("fraction", "size"):
        if best_config_file.exists():
            cfg_data = load_json(best_config_file)
            best_T = cfg_data["T"]
            best_alpha = cfg_data["alpha"]
        else:
            if best_T is None:
                best_T = float(cfg.distillation.default_T)
            if best_alpha is None:
                best_alpha = float(cfg.distillation.default_alpha)
            print(f"Notice: best_config.json not found; using fallback T={best_T}, alpha={best_alpha}")

    # 3. Fraction Stage
    if stage in ("fraction", "all"):
        assert best_T is not None and best_alpha is not None
        run_fraction_stage(
            cfg=cfg,
            data=data,
            teacher=teacher,
            seeds=seeds,
            best_T=best_T,
            best_alpha=best_alpha,
            out_base=out_base,
            ckpt_base=ckpt_base,
            resume=args.resume,
            verbose=args.verbose,
        )

    # 4. Size Stage
    if stage in ("size", "all"):
        assert best_T is not None and best_alpha is not None
        run_size_stage(
            cfg=cfg,
            data=data,
            teacher=teacher,
            seeds=seeds,
            best_T=best_T,
            best_alpha=best_alpha,
            out_base=out_base,
            ckpt_base=ckpt_base,
            resume=args.resume,
            verbose=args.verbose,
        )

    print("\nDistillation sweeps completed successfully.")


if __name__ == "__main__":
    main()
