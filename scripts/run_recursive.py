#!/usr/bin/env python3
"""CLI script to execute a recursive self-training loop run."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np

from paradigms.config import Config, load_config
from paradigms.data import FashionData
from paradigms.recursive import make_bundle, run_loop
from paradigms.utils.io import save_json


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and Paths into standard JSON serializable types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single recursive loop experiment with gating and checkpointing."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/recursive.yaml"],
        help="Path(s) to YAML configuration files to merge.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the experiment run.")
    parser.add_argument(
        "--verifier",
        choices=["oracle", "agreement", "self_conf", "none"],
        default="oracle",
        help="Verification strategy.",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable the acceptance gate (always accept candidates).",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds.")
    parser.add_argument(
        "--epochs-initial",
        type=int,
        default=None,
        help="Override Round 0 initial training epochs.",
    )
    parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=None,
        help="Override per-round training epochs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Run identifier tag (defaults to f'{verifier}_{gate/nogate}').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/recursive",
        help="Root directory for saving experiment JSON results.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/recursive",
        help="Root directory for saving model checkpoints.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose training loss per epoch.",
    )
    return parser.parse_args()


def load_merged_configs(paths: list[str]) -> Config:
    """Load and recursively merge multiple YAML configuration files."""
    merged = Config()
    for p in paths:
        path_obj = Path(p)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        cfg = load_config(path_obj)
        for k, v in cfg.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k].update(v)
            else:
                merged[k] = v
    return merged


def print_round_table(log: list[dict[str, Any]], total_seconds: float) -> None:
    """Print standard per-round progress table."""
    header = (
        f"{'round':>5} | {'verified':>8} | {'selected':>8} | "
        f"{'precision':>9} | {'cand gate':>9} | {'decision':<22} | {'current monitor acc':>19}"
    )
    print("\n" + header)
    print("-" * len(header))

    for row in log:
        r = row["round"]
        verified = row["n_verified"]
        selected = row["n_selected"]
        prec = (
            f"{row['selected_precision']:.4f}"
            if row.get("selected_precision") is not None
            else "-"
        )
        cand_gate = f"{row['cand_gate_acc']:.4f}"
        decision = row["reason"]
        mon_acc = f"{row['current_monitor_acc']:.4f}"

        print(
            f"{r:>5} | {verified:>8} | {selected:>8} | "
            f"{prec:>9} | {cand_gate:>9} | {decision:<22} | {mon_acc:>19}"
        )

    print("-" * len(header))
    print(f"Total seconds: {total_seconds:.2f}s\n")


def main() -> None:
    args = parse_args()

    cfg = load_merged_configs(args.config)

    # Ensure nested Config dot access
    if "recursive" not in cfg:
        cfg.recursive = Config()

    rcfg = cfg.recursive
    if args.rounds is not None:
        rcfg.rounds = args.rounds
    if args.epochs_initial is not None:
        rcfg.epochs_initial = args.epochs_initial
    if args.epochs_per_round is not None:
        rcfg.epochs_per_round = args.epochs_per_round

    gate_enabled = not args.no_gate
    tag = args.tag or f"{args.verifier}_{'gate' if gate_enabled else 'nogate'}"

    out_json = Path(args.out_dir) / tag / f"seed{args.seed}.json"
    ckpt_seed_dir = Path(args.ckpt_dir) / tag / f"seed{args.seed}"

    data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else getattr(cfg, "data", {})
    root = data_cfg.get("root", "./data") if hasattr(data_cfg, "get") else getattr(data_cfg, "root", "./data")
    split_seed = int(data_cfg.get("split_seed", 42) if hasattr(data_cfg, "get") else getattr(data_cfg, "split_seed", 42))
    dev_size = int(data_cfg.get("dev_size", 10000) if hasattr(data_cfg, "get") else getattr(data_cfg, "dev_size", 10000))

    data = FashionData(root=root, split_seed=split_seed, dev_size=dev_size)
    bundle = make_bundle(data, cfg, seed=args.seed)

    print(f"Starting recursive loop: verifier={args.verifier}, gate={gate_enabled}, seed={args.seed}, tag={tag}")
    t0 = time.time()
    result = run_loop(
        cfg,
        bundle,
        seed=args.seed,
        verifier_name=args.verifier,
        gate_enabled=gate_enabled,
        ckpt_dir=ckpt_seed_dir,
        verbose=args.verbose,
    )
    total_time = time.time() - t0

    # Ensure total_seconds in result is populated
    result["total_seconds"] = total_time

    # Print summary table
    print_round_table(result["log"], total_time)

    # Save output JSON
    save_data = {
        **_to_serializable(result),
        "config": _to_serializable(dict(cfg)),
    }
    save_json(save_data, out_json)
    print(f"Saved run results to: {out_json}")
    print(f"Checkpoints saved in: {ckpt_seed_dir}")


if __name__ == "__main__":
    main()
