#!/usr/bin/env python3
"""Final one-shot evaluation on the official Fashion-MNIST test set for the recursive loop.

This script represents the sole location where the recursive loop touches the protected test partition.
Checkpoints and hyperparameters are frozen prior to this execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

# Add project root and scripts to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradigms.config import load_config
from paradigms.data.fashion_mnist import FashionData, make_loader
from paradigms.models.registry import build_model
from paradigms.training.evaluate import evaluate
from paradigms.utils.io import save_json
from paradigms.utils.stats import mean_std, paired_comparison

RECURSIVE_CONFIG_ORDER: list[str] = [
    "oracle_gate",
    "agreement_gate",
    "selfconf_gate",
    "selfconf_nogate",
    "control_none",
]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 recursive loop final one-shot official test set evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML config (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--recursive-config",
        type=str,
        default="configs/recursive.yaml",
        help="Path to recursive YAML config (default: configs/recursive.yaml).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/recursive",
        help="Root directory for recursive model checkpoints (default: checkpoints/recursive).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="experiments/final/recursive_test.json",
        help="Destination path for final test evaluation results (default: experiments/final/recursive_test.json).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to evaluate (default: [0, 1, 2]).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if target output file already exists.",
    )
    parser.add_argument(
        "--readme-path",
        type=str,
        default="README.md",
        help="Path to README.md to update with final results (default: README.md).",
    )
    return parser.parse_args(args)


def check_refusal(out_file: Path, force: bool) -> None:
    """Refuse to run if target output exists unless --force is specified."""
    if out_file.exists():
        if not force:
            print(
                f"\n[REFUSAL] Final evaluation file already exists: '{out_file}'.\n"
                "The protected official test set may only be evaluated once per experiment phase.\n"
                "To force an overwrite, explicitly pass the --force flag.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                "\n"
                + "!" * 70
                + "\nWARNING: --force specified. Overwriting existing final test results at:\n"
                f"  {out_file}\n"
                + "!" * 70
                + "\n"
            )


def check_git_status() -> str:
    """Warn if working directory has uncommitted modifications and return HEAD commit hash."""
    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        status_text = status_proc.stdout.strip()
        if status_text:
            print(
                "\n"
                + "=" * 70
                + "\nWARNING: Git working tree contains uncommitted changes!\n"
                "Official test evaluation should ideally run against a clean commit.\n"
                + "=" * 70
                + "\n"
            )
    except Exception as e:
        print(f"Notice: git status check encountered: {e}")

    try:
        rev_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return rev_proc.stdout.strip()
    except Exception:
        return "unknown"


def update_readme_with_results(readme_path: str | Path, final_data: dict[str, Any]) -> None:
    """Update README.md with formatted Phase 2 recursive loop test results table."""
    p = Path(readme_path)
    if not p.exists():
        print(f"Notice: README not found at {p}, skipping update.")
        return

    content = p.read_text(encoding="utf-8")
    summary = final_data["summary"]
    paired_dict = final_data["paired_final_minus_round0"]

    lines = [
        "| Configuration | Round 0 Test Acc (Mean ± Std) | Final Test Acc (Mean ± Std) | Paired Gain (Final - R0) | Wins / Seeds |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for cfg in RECURSIVE_CONFIG_ORDER:
        if cfg not in summary:
            continue
        info = summary[cfg]
        p_info = paired_dict.get(cfg, {})
        r0_str = f"{info['round0_mean'] * 100:.2f}% ± {info['round0_std'] * 100:.2f}%"
        fn_str = f"{info['final_mean'] * 100:.2f}% ± {info['final_std'] * 100:.2f}%"
        gain_str = f"{p_info.get('mean_diff', 0.0) * 100:+.2f}% ± {p_info.get('std_diff', 0.0) * 100:.2f}%"
        wins_str = f"{p_info.get('wins', 0)} / {p_info.get('n', 0)}"
        lines.append(f"| `{cfg}` | {r0_str} | {fn_str} | {gain_str} | {wins_str} |")

    table_md = "\n".join(lines)

    phase2_section = (
        "## Phase 2: Recursive Self-Improvement Loop\n\n"
        "Phase 2: complete\n\n"
        "### Phase 2 Final Official Test Results\n\n"
        f"{table_md}\n"
    )

    # Append or replace Phase 2 section
    if "## Phase 2: Recursive Self-Improvement Loop" in content:
        parts = content.split("## Phase 2: Recursive Self-Improvement Loop")
        new_content = parts[0].rstrip() + "\n\n" + phase2_section
    else:
        new_content = content.rstrip() + "\n\n" + phase2_section

    p.write_text(new_content, encoding="utf-8")
    print(f"Updated {p} with final Phase 2 recursive loop results table.")


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    out_file = Path(args.out_file)
    ckpt_dir = Path(args.ckpt_dir)

    # 1. Refusal guard
    check_refusal(out_file, args.force)

    # 2. Check git status
    git_commit = check_git_status()

    # 3. Load configurations and test dataset
    base_cfg = load_config(args.config)
    seeds = list(args.seeds)

    data = FashionData(
        root=base_cfg.data.root,
        split_seed=int(base_cfg.data.split_seed),
        dev_size=int(base_cfg.data.dev_size),
    )
    test_ds = data.get("test", allow_test=True)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False, seed=0)

    print("\n" + "=" * 70)
    print("EXECUTING RECURSIVE LOOP FINAL OFFICIAL TEST EVALUATION")
    print(f"Git Commit: {git_commit}")
    print(f"Seeds:      {seeds}")
    print(f"Checkpoints:{ckpt_dir}")
    print(f"Output:     {out_file}")
    print("=" * 70 + "\n")

    results: list[dict[str, Any]] = []

    for cfg in RECURSIVE_CONFIG_ORDER:
        print(f"---> Evaluating Configuration: {cfg}")
        for seed in seeds:
            seed_dir = ckpt_dir / cfg / f"seed{seed}"
            r0_path = seed_dir / "round0.pt"
            fn_path = seed_dir / "final.pt"

            if not r0_path.exists():
                raise FileNotFoundError(f"Round 0 checkpoint missing: {r0_path}")
            if not fn_path.exists():
                raise FileNotFoundError(f"Final checkpoint missing: {fn_path}")

            # Round 0 model
            model_r0 = build_model("small_cnn")
            model_r0.load_state_dict(torch.load(r0_path, map_location="cpu"))
            model_r0.eval()
            eval_r0 = evaluate(model_r0, test_loader, device="cpu")

            # Final model
            model_fn = build_model("small_cnn")
            model_fn.load_state_dict(torch.load(fn_path, map_location="cpu"))
            model_fn.eval()
            eval_fn = evaluate(model_fn, test_loader, device="cpu")

            r0_acc = float(eval_r0["accuracy"])
            fn_acc = float(eval_fn["accuracy"])
            diff = fn_acc - r0_acc

            results.append(
                {
                    "config": cfg,
                    "seed": seed,
                    "round0_test_acc": r0_acc,
                    "final_test_acc": fn_acc,
                    "diff": diff,
                    "round0_per_class": eval_r0["per_class_accuracy"],
                    "final_per_class": eval_fn["per_class_accuracy"],
                }
            )
            print(
                f"  seed {seed}: R0 = {r0_acc * 100:.2f}%, "
                f"Final = {fn_acc * 100:.2f}%, Diff = {diff * 100:+.2f}%"
            )

    # Compute summary and paired gains per config
    summary: dict[str, dict[str, float]] = {}
    paired_final_minus_round0: dict[str, dict[str, Any]] = {}

    for cfg in RECURSIVE_CONFIG_ORDER:
        cfg_rows = [r for r in results if r["config"] == cfg]
        if not cfg_rows:
            continue
        r0_accs = [r["round0_test_acc"] for r in cfg_rows]
        fn_accs = [r["final_test_acc"] for r in cfg_rows]

        r0_m, r0_s = mean_std(r0_accs)
        fn_m, fn_s = mean_std(fn_accs)
        paired = paired_comparison(fn_accs, r0_accs)

        summary[cfg] = {
            "round0_mean": r0_m,
            "round0_std": r0_s,
            "final_mean": fn_m,
            "final_std": fn_s,
        }
        paired_final_minus_round0[cfg] = paired

    final_payload = {
        "git_commit": git_commit,
        "results": results,
        "summary": summary,
        "paired_final_minus_round0": paired_final_minus_round0,
    }

    # Save final JSON
    save_json(final_payload, out_file)
    print(f"\nSaved test results to {out_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 2 FINAL OFFICIAL TEST SET RESULTS")
    print("=" * 80)
    print(
        f"{'Configuration':<18} | {'Round 0 Acc':<16} | {'Final Acc':<16} | {'Paired Gain':<20} | {'Wins':<6}"
    )
    print("-" * 80)
    for cfg in RECURSIVE_CONFIG_ORDER:
        if cfg not in summary:
            continue
        info = summary[cfg]
        p_info = paired_final_minus_round0[cfg]
        r0_str = f"{info['round0_mean'] * 100:.2f}% ± {info['round0_std'] * 100:.2f}%"
        fn_str = f"{info['final_mean'] * 100:.2f}% ± {info['final_std'] * 100:.2f}%"
        gain_str = (
            f"{p_info['mean_diff'] * 100:+.2f}% ± {p_info['std_diff'] * 100:.2f}%"
        )
        wins_str = f"{p_info['wins']}/{p_info['n']}"
        print(f"{cfg:<18} | {r0_str:<16} | {fn_str:<16} | {gain_str:<20} | {wins_str:<6}")
    print("=" * 80 + "\n")

    # Update README.md
    update_readme_with_results(args.readme_path, final_payload)


if __name__ == "__main__":
    main()
