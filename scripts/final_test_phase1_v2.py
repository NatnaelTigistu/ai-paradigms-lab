"""Final test set evaluation script for Phase 1 v2 models.

DO NOT RUN THIS SCRIPT until explicitly authorized.
Strictly guarded by one-shot execution checks and git hygiene warnings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from paradigms.config import load_config
from paradigms.data.fashion_mnist import FashionData, MEAN, STD
from paradigms.models.registry import build_model
from paradigms.utils.io import load_json, save_json
from paradigms.utils.stats import paired_comparison


def get_git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    try:
        proc = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return bool(proc.stdout.strip())
    except Exception:
        return True


def check_refusal(out_file: Path, force: bool = False) -> None:
    if out_file.exists() and not force:
        print(f"REFUSAL: Final test output already exists at {out_file}.")
        print("Final evaluation on the official test set is strictly one-shot.")
        print("Pass --force to explicitly overwrite if re-evaluation is authorized.")
        sys.exit(1)


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot final test evaluation for Phase 1 v2.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing final test evaluation")
    parser.add_argument("--out-file", default="experiments/v2/final_test.json", help="Output filepath")
    parser.add_argument("--ckpt-dir", default="checkpoints/v2/final", help="Directory of final checkpoints")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base configuration")
    return parser.parse_args(cli_args)


def main() -> None:
    args = parse_args()
    out_file = Path(args.out_file)

    check_refusal(out_file, force=args.force)

    if is_git_dirty():
        print("\n" + "!" * 70)
        print("WARNING: Working directory contains uncommitted changes!")
        print("Evaluation should strictly run on a clean git commit.")
        print("!" * 70 + "\n")

    git_commit = get_git_commit()
    print(f"Executing Phase 1 v2 Final Test Evaluation at git commit: {git_commit}")

    cfg = load_config(args.config)
    data = FashionData("./data")

    # Access official test set with explicit authorization
    test_ds = data.get("test", allow_test=True)
    raw_x_test = test_ds.data
    X_test = ((raw_x_test.float() / 255.0 - MEAN) / STD).unsqueeze(1).contiguous()
    Y_test = test_ds.targets.long().contiguous()

    ckpt_base = Path(args.ckpt_dir)
    seeds = [10, 11, 12, 13, 14]
    fractions = [0.02, 0.10, 1.0]

    results: dict[str, Any] = {
        "git_commit": git_commit,
        "previous_evaluation": "phase1-final",
        "fractions": {},
    }

    torch.set_num_threads(1)

    for frac in fractions:
        f_tag = f"f{frac}"
        frac_data: dict[str, Any] = {}

        # 1. Hard
        hard_accs = []
        for s in seeds:
            ckpt_path = ckpt_base / f_tag / "hard" / f"seed{s}.pt"
            model = build_model("student_mlp", hidden=32)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.eval()
            with torch.inference_mode():
                preds = model(X_test).argmax(dim=-1)
                acc = float((preds == Y_test).float().mean().item())
                hard_accs.append(acc)
        frac_data["hard"] = {"seeds": seeds, "accuracies": hard_accs}

        # 2. Distilled
        dist_accs = []
        for s in seeds:
            ckpt_path = ckpt_base / f_tag / "distilled" / f"seed{s}.pt"
            model = build_model("student_mlp", hidden=32)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.eval()
            with torch.inference_mode():
                preds = model(X_test).argmax(dim=-1)
                acc = float((preds == Y_test).float().mean().item())
                dist_accs.append(acc)
        frac_data["distilled"] = {"seeds": seeds, "accuracies": dist_accs}
        frac_data["distilled_paired"] = paired_comparison(dist_accs, hard_accs)

        # 3. Transfer (if frac < 1)
        if frac < 1.0:
            tr_accs = []
            for s in seeds:
                ckpt_path = ckpt_base / f_tag / "distilled_transfer" / f"seed{s}.pt"
                model = build_model("student_mlp", hidden=32)
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                model.eval()
                with torch.inference_mode():
                    preds = model(X_test).argmax(dim=-1)
                    acc = float((preds == Y_test).float().mean().item())
                    tr_accs.append(acc)
            frac_data["distilled_transfer"] = {"seeds": seeds, "accuracies": tr_accs}
            frac_data["transfer_paired"] = paired_comparison(tr_accs, hard_accs)

        results["fractions"][f_tag] = frac_data

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_file)
    print(f"\nFinal test evaluation written to: {out_file}")


if __name__ == "__main__":
    main()
