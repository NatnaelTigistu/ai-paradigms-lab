#!/usr/bin/env python3
"""Run recursive self-improvement experiment grid across verifiers and gate settings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure repository root and scripts directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paradigms.config import Config, load_config
from paradigms.data import FashionData
from paradigms.recursive import make_bundle, run_loop
from paradigms.utils.io import load_json, save_json
from run_recursive import _to_serializable, load_merged_configs, print_round_table

CONFIG_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "oracle_gate",
        "verifier": "oracle",
        "gate": True,
        "description": "Best case: perfect oracle verifier with gate",
    },
    {
        "name": "control_none",
        "verifier": "none",
        "gate": True,
        "description": "Compute control: continuous training on labeled data only",
    },
    {
        "name": "agreement_gate",
        "verifier": "agreement",
        "gate": True,
        "description": "Realistic: independent model agreement verifier with gate",
    },
    {
        "name": "selfconf_gate",
        "verifier": "self_conf",
        "gate": True,
        "description": "Self-confidence verifier with gate safety net",
    },
    {
        "name": "selfconf_nogate",
        "verifier": "self_conf",
        "gate": False,
        "description": "Self-confidence verifier without gate (ablation)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recursive self-improvement experiment grid."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/recursive.yaml"],
        help="Path(s) to YAML configuration files to merge.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seed list to execute (defaults to config seeds [0, 1, 2]).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=str,
        default=None,
        help="Filter to run only specified config name(s).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume and skip already completed runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/recursive",
        help="Root directory for saving experiment JSON outputs.",
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
        help="Print verbose training output per epoch.",
    )
    return parser.parse_args()


def summarize_config_results(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute and format summary statistics across seeds for a single config."""
    final_accs = [float(r["final_monitor_acc"]) for r in results]
    acc_rounds = [int(r["n_accepted_rounds"]) for r in results]

    precisions: list[float] = []
    for r in results:
        round_precs = [
            float(row["selected_precision"])
            for row in r["log"][1:]
            if row.get("n_selected", 0) > 0
        ]
        if round_precs:
            precisions.append(float(np.mean(round_precs)))

    mean_acc = float(np.mean(final_accs))
    std_acc = float(np.std(final_accs, ddof=1)) if len(final_accs) > 1 else 0.0

    mean_rounds = float(np.mean(acc_rounds))
    std_rounds = float(np.std(acc_rounds, ddof=1)) if len(acc_rounds) > 1 else 0.0

    if precisions:
        mean_prec = float(np.mean(precisions))
        std_prec = float(np.std(precisions, ddof=1)) if len(precisions) > 1 else 0.0
        prec_str = f"{mean_prec:.4f} +/- {std_prec:.4f}"
    else:
        mean_prec = None
        std_prec = None
        prec_str = "N/A (0 selected)"

    print(f"\n{'='*70}")
    print(f"Summary for: {name} (N={len(results)} seeds)")
    print(f"{'='*70}")
    print(f"  Final Monitor Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"  Mean Accepted Rounds:   {mean_rounds:.1f} +/- {std_rounds:.1f}")
    print(f"  Mean Pseudo-label Prec: {prec_str}")
    print(f"{'='*70}\n")

    return {
        "config_name": name,
        "n_seeds": len(results),
        "mean_final_monitor_acc": mean_acc,
        "std_final_monitor_acc": std_acc,
        "mean_accepted_rounds": mean_rounds,
        "std_accepted_rounds": std_rounds,
        "mean_pseudo_label_precision": mean_prec,
        "std_pseudo_label_precision": std_prec,
    }


def main() -> None:
    args = parse_args()
    cfg = load_merged_configs(args.config)

    seeds = args.seeds
    if seeds is None:
        rcfg = cfg.get("recursive", {})
        seeds = list(rcfg.get("seeds", [0, 1, 2]))

    configs_to_run = CONFIG_DEFINITIONS
    if args.only is not None:
        valid_names = {c["name"] for c in CONFIG_DEFINITIONS}
        for name in args.only:
            if name not in valid_names:
                raise ValueError(f"Unknown config name '{name}'. Valid: {sorted(valid_names)}")
        configs_to_run = [c for c in CONFIG_DEFINITIONS if c["name"] in args.only]

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "./data")
    split_seed = int(data_cfg.get("split_seed", 42))
    dev_size = int(data_cfg.get("dev_size", 10000))
    data = FashionData(root=root, split_seed=split_seed, dev_size=dev_size)

    for cdef in configs_to_run:
        cname = cdef["name"]
        verifier = cdef["verifier"]
        gate = cdef["gate"]
        desc = cdef["description"]

        print(f"\n>>> Running Config: {cname} ({desc}) | Seeds: {seeds}")
        config_results: list[dict[str, Any]] = []

        for seed in seeds:
            out_json = Path(args.out_dir) / cname / f"seed{seed}.json"
            ckpt_seed_dir = Path(args.ckpt_dir) / cname / f"seed{seed}"

            if args.resume and out_json.exists():
                print(f"  [Resume] Loading existing run: {out_json}")
                res = load_json(out_json)
                config_results.append(res)
                continue

            print(f"\n--- Running {cname} (seed {seed}) ---")
            bundle = make_bundle(data, cfg, seed=seed)
            t0 = time.time()
            res = run_loop(
                cfg,
                bundle,
                seed=seed,
                verifier_name=verifier,
                gate_enabled=gate,
                ckpt_dir=ckpt_seed_dir,
                verbose=args.verbose,
            )
            elapsed = time.time() - t0
            res["total_seconds"] = elapsed

            print_round_table(res["log"], elapsed)

            save_data = {
                **_to_serializable(res),
                "config": _to_serializable(dict(cfg)),
            }
            save_json(save_data, out_json)
            print(f"  Saved run results to: {out_json}")
            config_results.append(res)

        summarize_config_results(cname, config_results)


if __name__ == "__main__":
    main()
