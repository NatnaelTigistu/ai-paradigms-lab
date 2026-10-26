"""Final one-shot evaluation on the official Fashion-MNIST test set for Phase 1.

Hyperparameters and model checkpoints are strictly frozen prior to this execution.
This script represents the sole location where Phase 1 touches the protected test partition.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from paradigms.config import load_config
from paradigms.data.fashion_mnist import FashionData, make_loader
from paradigms.models.registry import build_model, count_parameters
from paradigms.training.evaluate import evaluate
from paradigms.utils.io import load_json, save_json
from paradigms.utils.stats import mean_std, paired_comparison


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 final one-shot official test set evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML config (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--best-config",
        type=str,
        default="experiments/distillation/best_config.json",
        help="Path to best_config.json (default: experiments/distillation/best_config.json).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="experiments/final/phase1_test.json",
        help="Destination path for final test evaluation results (default: experiments/final/phase1_test.json).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to evaluate (default: from config [0, 1, 2, 3, 4]).",
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
    """Warn if working directory has uncommitted modifications."""
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
    """Update README.md with formatted Phase 1 test results table and complete status."""
    p = Path(readme_path)
    if not p.exists():
        print(f"Notice: README not found at {p}, skipping update.")
        return

    content = p.read_text(encoding="utf-8")
    summary = final_data["summary"]
    paired = final_data["paired_distilled_minus_hard"]

    lines = [
        "| Model | Parameters | Test Accuracy (Mean ± Std) |",
        "| :--- | :--- | :--- |",
    ]
    for model_name in ["Teacher (CNN)", "Hard Student (MLP)", "Distilled Student (MLP)"]:
        if model_name in summary:
            info = summary[model_name]
            params_str = f"{info['params']:,}"
            acc_str = f"{info['mean'] * 100:.2f}% ± {info['std'] * 100:.2f}%"
            lines.append(f"| {model_name} | {params_str} | {acc_str} |")

    table_md = "\n".join(lines)
    paired_md = (
        f"**Paired Comparison (Distilled - Hard)**: "
        f"Mean Diff = `{paired['mean_diff'] * 100:+.2f}% ± {paired['std_diff'] * 100:.2f}%`, "
        f"Wins = `{paired['wins']} / {paired['n']}` seeds."
    )

    new_status_section = (
        "## Status\n\n"
        "Phase 1: complete\n\n"
        "### Phase 1 Final Official Test Results\n\n"
        f"{table_md}\n\n"
        f"{paired_md}\n"
    )

    # Replace existing ## Status section
    if "## Status" in content:
        before_status = content.split("## Status")[0]
        new_content = before_status.rstrip() + "\n\n" + new_status_section
    else:
        new_content = content.rstrip() + "\n\n" + new_status_section

    p.write_text(new_content, encoding="utf-8")
    print(f"Updated {p} with final Phase 1 results table.")


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    out_file = Path(args.out_file)

    # 1. Check refusal guard
    check_refusal(out_file, args.force)

    # 2. Check git status and commit
    git_commit = check_git_status()

    # 3. Load configurations
    base_cfg = load_config(args.config)
    seeds: list[int] = (
        args.seeds if args.seeds is not None else list(base_cfg.get("seeds", [0, 1, 2, 3, 4]))
    )

    best_cfg_path = Path(args.best_config)
    if best_cfg_path.exists():
        best_config = load_json(best_cfg_path)
    else:
        best_config = {"T": 1, "alpha": 0.5, "selected_on": "dev"}

    # 4. Prepare protected official test set DataLoader
    data = FashionData(
        root=base_cfg.data.root,
        split_seed=int(base_cfg.data.split_seed),
        dev_size=int(base_cfg.data.dev_size),
    )
    test_ds = data.get("test", allow_test=True)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False, seed=0)

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("EXECUTING PHASE 1 FINAL OFFICIAL TEST EVALUATION")
    print(f"Git Commit: {git_commit}")
    print(f"Seeds: {seeds}")
    print("=" * 70 + "\n")

    for seed in seeds:
        print(f"---> Evaluating Seed {seed}...")

        # 1. Teacher CNN
        t_path = Path("checkpoints/teacher") / f"seed{seed}.pt"
        if not t_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint missing: {t_path}")
        teacher = build_model("teacher_cnn")
        teacher.load_state_dict(torch.load(t_path, map_location="cpu"))
        teacher.eval()
        t_eval = evaluate(teacher, test_loader, device="cpu")
        results.append(
            {
                "model": "Teacher (CNN)",
                "seed": seed,
                "params": count_parameters(teacher),
                "test_accuracy": float(t_eval["accuracy"]),
                "per_class_accuracy": t_eval["per_class_accuracy"],
            }
        )
        print(f"  Teacher (CNN)           : {t_eval['accuracy'] * 100:.2f}%")

        # 2. Hard Student MLP
        h_candidates = [
            Path("checkpoints/distillation/size/h32/hard") / f"seed{seed}.pt",
            Path("checkpoints/student_hard/h32_f1.0") / f"seed{seed}.pt",
        ]
        h_path = next((p for p in h_candidates if p.exists()), None)
        if h_path is None:
            raise FileNotFoundError(f"Hard student checkpoint missing for seed {seed}")
        hard_student = build_model("student_mlp", hidden=32)
        hard_student.load_state_dict(torch.load(h_path, map_location="cpu"))
        hard_student.eval()
        h_eval = evaluate(hard_student, test_loader, device="cpu")
        results.append(
            {
                "model": "Hard Student (MLP)",
                "seed": seed,
                "params": count_parameters(hard_student),
                "test_accuracy": float(h_eval["accuracy"]),
                "per_class_accuracy": h_eval["per_class_accuracy"],
            }
        )
        print(f"  Hard Student (MLP)      : {h_eval['accuracy'] * 100:.2f}%")

        # 3. Distilled Student MLP
        d_candidates = [
            Path("checkpoints/distillation/size/h32/distilled") / f"seed{seed}.pt",
            Path("checkpoints/distillation/temperature/T1/distilled") / f"seed{seed}.pt",
            Path("checkpoints/student_distilled/h32_f1.0_T4.0_a0.5") / f"seed{seed}.pt",
        ]
        d_path = next((p for p in d_candidates if p.exists()), None)
        if d_path is None:
            raise FileNotFoundError(f"Distilled student checkpoint missing for seed {seed}")
        dist_student = build_model("student_mlp", hidden=32)
        dist_student.load_state_dict(torch.load(d_path, map_location="cpu"))
        dist_student.eval()
        d_eval = evaluate(dist_student, test_loader, device="cpu")
        results.append(
            {
                "model": "Distilled Student (MLP)",
                "seed": seed,
                "params": count_parameters(dist_student),
                "test_accuracy": float(d_eval["accuracy"]),
                "per_class_accuracy": d_eval["per_class_accuracy"],
            }
        )
        print(f"  Distilled Student (MLP) : {d_eval['accuracy'] * 100:.2f}%")

    # 5. Summarize metrics
    summary: dict[str, Any] = {}
    for model_name in ["Teacher (CNN)", "Hard Student (MLP)", "Distilled Student (MLP)"]:
        accs = [r["test_accuracy"] for r in results if r["model"] == model_name]
        m, s = mean_std(accs)
        params = next(r["params"] for r in results if r["model"] == model_name)
        summary[model_name] = {
            "mean": float(m),
            "std": float(s),
            "params": params,
            "seeds": seeds,
        }

    # Paired comparison between Distilled and Hard
    dist_accs = [r["test_accuracy"] for r in results if r["model"] == "Distilled Student (MLP)"]
    hard_accs = [r["test_accuracy"] for r in results if r["model"] == "Hard Student (MLP)"]
    paired = paired_comparison(dist_accs, hard_accs)

    final_payload = {
        "git_commit": git_commit,
        "best_config": best_config,
        "results": results,
        "summary": summary,
        "paired_distilled_minus_hard": paired,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(final_payload, out_file)
    print(f"\nFinal test evaluation results saved to: {out_file}")

    # Print final summary table
    print("\n" + "=" * 65)
    print("PHASE 1 OFFICIAL TEST SET EVALUATION SUMMARY")
    print("=" * 65)
    print(f"{'Model':<25} | {'Params':<10} | {'Test Accuracy (Mean ± Std)':<25}")
    print("-" * 65)
    for model_name, info in summary.items():
        print(
            f"{model_name:<25} | {info['params']:<10,d} | "
            f"{info['mean'] * 100:.2f}% ± {info['std'] * 100:.2f}%"
        )
    print("-" * 65)
    print("Paired Comparison (Distilled - Hard):")
    print(f"  Mean Diff : {paired['mean_diff'] * 100:+.2f}% ± {paired['std_diff'] * 100:.2f}%")
    print(f"  Wins      : {paired['wins']} / {paired['n']} seeds")
    print("=" * 65 + "\n")

    # Update README.md
    update_readme_with_results(args.readme_path, final_payload)


if __name__ == "__main__":
    main()
