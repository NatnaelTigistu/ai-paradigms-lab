"""Orchestrator for Phase 1 v2 tuning and final comparison."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from paradigms.config import load_config
from paradigms.data.fashion_mnist import FashionData
from paradigms.distillation.cached import cache_teacher_logits
from paradigms.distillation.experiment import load_teacher
from paradigms.distillation.v2 import run_student_v2
from paradigms.utils.io import load_json, save_json
from paradigms.utils.stats import mean_std, paired_comparison


def compute_verdict(mean_diff: float, std_diff: float, wins: int, n: int) -> str:
    """Fixed verdict rule:

    WIN if mean diff > std and wins >= 4/5;
    LOSS if mean diff < -std and wins <= 1/5;
    otherwise TIE.
    """
    if mean_diff > std_diff and wins >= 4:
        return "WIN"
    elif mean_diff < -std_diff and wins <= 1:
        return "LOSS"
    else:
        return "TIE"


def run_single(
    cfg: Any,
    data: FashionData,
    cache: torch.Tensor,
    out_dir: Path,
    ckpt_dir: Path | None,
    *,
    seed: int,
    mode: str,
    fraction: float,
    hidden: int = 32,
    T: float | None = None,
    alpha: float | None = None,
    wd: float = 0.0,
    base_epochs: int = 30,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute or load a single student run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_json = out_dir / f"seed{seed}.json"

    if resume and seed_json.exists():
        return load_json(seed_json)

    ckpt_path = None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"seed{seed}.pt"

    rec = run_student_v2(
        cfg=cfg,
        data=data,
        cache=cache,
        seed=seed,
        mode=mode,
        fraction=fraction,
        hidden=hidden,
        T=T,
        alpha=alpha,
        wd=wd,
        base_epochs=base_epochs,
        ckpt_path=ckpt_path,
    )
    save_json(rec, seed_json)
    return rec


def run_group(
    cfg: Any,
    data: FashionData,
    cache: torch.Tensor,
    out_dir: Path,
    ckpt_dir: Path | None,
    seeds: list[int],
    *,
    mode: str,
    fraction: float,
    hidden: int = 32,
    T: float | None = None,
    alpha: float | None = None,
    wd: float = 0.0,
    base_epochs: int = 30,
    resume: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute a cohort of seeds and write summary.json."""
    records: list[dict[str, Any]] = []
    for s in seeds:
        rec = run_single(
            cfg=cfg,
            data=data,
            cache=cache,
            out_dir=out_dir,
            ckpt_dir=ckpt_dir,
            seed=s,
            mode=mode,
            fraction=fraction,
            hidden=hidden,
            T=T,
            alpha=alpha,
            wd=wd,
            base_epochs=base_epochs,
            resume=resume,
        )
        records.append(rec)

    dev_accs = [r["dev_accuracy"] for r in records]
    m, s = mean_std(dev_accs)
    summary = {
        "dev_accuracy_mean": m,
        "dev_accuracy_std": s,
        "seeds": seeds,
        "mode": mode,
        "fraction": fraction,
        "hidden": hidden,
        "T": T,
        "alpha": alpha,
        "wd": wd,
        "epochs": records[0]["epochs_used"],
        "params": records[0]["params"],
    }
    save_json(summary, out_dir / "summary.json")
    return summary, records


def run_tuning(
    cfg: Any,
    data: FashionData,
    cache: torch.Tensor,
    base_epochs: int = 30,
    resume: bool = True,
) -> dict[str, Any]:
    """Step 4: Coordinate descent tuning on dev set across fractions [0.02, 0.10, 1.0]."""
    fractions = [0.02, 0.10, 1.0]
    seeds = [0, 1, 2]
    best_hparams: dict[str, Any] = {"selected_on": "dev"}

    exp_tuning = Path("experiments/v2/tuning")
    ckpt_tuning = Path("checkpoints/v2/tuning")

    print("\n" + "=" * 70)
    print("STEP 4: HYPERPARAMETER TUNING (Dev Set Only, Seeds 0 1 2)")
    print("=" * 70)

    for frac in fractions:
        f_tag = f"f{frac}"
        print(f"\n>>> Tuning Fraction: {frac} ({f_tag}) <<<")
        f_dir = exp_tuning / f_tag
        f_ckpt = ckpt_tuning / f_tag

        tables: dict[str, Any] = {}

        # 1. Hard Student: wd in [0.0, 1e-4, 1e-3]
        print("\n--- 1. Hard Student wd sweep ---")
        wd_candidates = [0.0, 1e-4, 1e-3]
        best_wd = None
        best_wd_acc = -1.0
        wd_records = []

        for wd in wd_candidates:
            out_d = f_dir / "hard" / f"wd_{wd}"
            ckpt_d = f_ckpt / "hard" / f"wd_{wd}"
            summ, recs = run_group(
                cfg, data, cache, out_d, ckpt_d, seeds,
                mode="hard", fraction=frac, wd=wd, base_epochs=base_epochs, resume=resume
            )
            wd_records.append(summ)
            m = summ["dev_accuracy_mean"]
            print(f"  wd={wd:<6} | Dev Acc: {m * 100:.2f}% ± {summ['dev_accuracy_std'] * 100:.2f}%")
            if m > best_wd_acc:
                best_wd_acc = m
                best_wd = wd

        tables["hard_wd"] = wd_records
        print(f"  Selected wd* = {best_wd} (dev acc = {best_wd_acc * 100:.2f}%)")

        # 2. Distilled: T in [1, 2, 4] (fixed alpha=0.5, wd=best_wd)
        print("\n--- 2. Distilled T sweep (alpha=0.5, wd=wd*) ---")
        t_candidates = [1.0, 2.0, 4.0]
        best_T = None
        best_T_acc = -1.0
        T_records = []

        for T in t_candidates:
            out_d = f_dir / "distilled" / f"T_{int(T)}"
            ckpt_d = f_ckpt / "distilled" / f"T_{int(T)}"
            summ, recs = run_group(
                cfg, data, cache, out_d, ckpt_d, seeds,
                mode="distilled", fraction=frac, T=T, alpha=0.5, wd=best_wd, base_epochs=base_epochs, resume=resume
            )
            T_records.append(summ)
            m = summ["dev_accuracy_mean"]
            print(f"  T={T:<4} | Dev Acc: {m * 100:.2f}% ± {summ['dev_accuracy_std'] * 100:.2f}%")
            if m > best_T_acc:
                best_T_acc = m
                best_T = T

        tables["distilled_T"] = T_records
        print(f"  Selected T* = {best_T} (dev acc = {best_T_acc * 100:.2f}%)")

        # 3. Distilled: alpha in [0.2, 0.5, 0.8] (T=best_T, wd=best_wd)
        print(f"\n--- 3. Distilled alpha sweep (T={best_T}, wd=wd*) ---")
        alpha_candidates = [0.2, 0.5, 0.8]
        best_alpha = None
        best_alpha_acc = -1.0
        alpha_records = []

        for a in alpha_candidates:
            out_d = f_dir / "distilled" / f"alpha_{a}"
            ckpt_d = f_ckpt / "distilled" / f"alpha_{a}"
            summ, recs = run_group(
                cfg, data, cache, out_d, ckpt_d, seeds,
                mode="distilled", fraction=frac, T=best_T, alpha=a, wd=best_wd, base_epochs=base_epochs, resume=resume
            )
            alpha_records.append(summ)
            m = summ["dev_accuracy_mean"]
            print(f"  alpha={a:<4} | Dev Acc: {m * 100:.2f}% ± {summ['dev_accuracy_std'] * 100:.2f}%")
            if m > best_alpha_acc:
                best_alpha_acc = m
                best_alpha = a

        tables["distilled_alpha"] = alpha_records
        print(f"  Selected alpha* = {best_alpha} (dev acc = {best_alpha_acc * 100:.2f}%)")

        # 4. Transfer (only fraction < 1): alpha in [0.2, 0.5, 0.8] (T=best_T, wd=best_wd)
        best_alpha_t = None
        if frac < 1.0:
            print(f"\n--- 4. Transfer alpha sweep (T={best_T}, wd=wd*) ---")
            transfer_records = []
            best_alpha_t_acc = -1.0

            for a in alpha_candidates:
                out_d = f_dir / "transfer" / f"alpha_{a}"
                ckpt_d = f_ckpt / "transfer" / f"alpha_{a}"
                summ, recs = run_group(
                    cfg, data, cache, out_d, ckpt_d, seeds,
                    mode="distilled_transfer", fraction=frac, T=best_T, alpha=a, wd=best_wd, base_epochs=base_epochs, resume=resume
                )
                transfer_records.append(summ)
                m = summ["dev_accuracy_mean"]
                print(f"  alpha_transfer={a:<4} | Dev Acc: {m * 100:.2f}% ± {summ['dev_accuracy_std'] * 100:.2f}%")
                if m > best_alpha_t_acc:
                    best_alpha_t_acc = m
                    best_alpha_t = a

            tables["transfer_alpha"] = transfer_records
            print(f"  Selected alpha_transfer* = {best_alpha_t} (dev acc = {best_alpha_t_acc * 100:.2f}%)")

        best_hparams[f_tag] = {
            "fraction": frac,
            "wd": best_wd,
            "T": best_T,
            "alpha": best_alpha,
            "alpha_transfer": best_alpha_t,
            "tables": tables,
        }

    save_json(best_hparams, Path("experiments/v2/best_hparams.json"))
    print("\nSaved best hyperparameters to: experiments/v2/best_hparams.json")
    return best_hparams


def run_final_comparison(
    cfg: Any,
    data: FashionData,
    cache: torch.Tensor,
    best_hparams: dict[str, Any],
    base_epochs: int = 30,
    resume: bool = True,
) -> dict[str, Any]:
    """Step 5: Final comparison across fresh seeds [10, 11, 12, 13, 14] on dev set."""
    seeds = [10, 11, 12, 13, 14]
    fractions = [0.02, 0.10, 1.0]

    exp_final = Path("experiments/v2/final")
    ckpt_final = Path("checkpoints/v2/final")

    print("\n" + "=" * 70)
    print("STEP 5: FINAL COMPARISON (Fresh Seeds 10, 11, 12, 13, 14)")
    print("=" * 70)

    final_results: dict[str, Any] = {}

    for frac in fractions:
        f_tag = f"f{frac}"
        hp = best_hparams[f_tag]
        wd = hp["wd"]
        T = hp["T"]
        alpha = hp["alpha"]
        alpha_t = hp["alpha_transfer"]

        print(f"\n>>> Running Final Evaluation for Fraction: {frac} <<<")

        # 1. Hard
        hard_dir = exp_final / f_tag / "hard"
        hard_ckpt = ckpt_final / f_tag / "hard"
        hard_summ, hard_recs = run_group(
            cfg, data, cache, hard_dir, hard_ckpt, seeds,
            mode="hard", fraction=frac, wd=wd, base_epochs=base_epochs, resume=resume
        )

        # 2. Distilled
        dist_dir = exp_final / f_tag / "distilled"
        dist_ckpt = ckpt_final / f_tag / "distilled"
        dist_summ, dist_recs = run_group(
            cfg, data, cache, dist_dir, dist_ckpt, seeds,
            mode="distilled", fraction=frac, T=T, alpha=alpha, wd=wd, base_epochs=base_epochs, resume=resume
        )

        # Paired comparison: distilled - hard
        h_accs = [r["dev_accuracy"] for r in hard_recs]
        d_accs = [r["dev_accuracy"] for r in dist_recs]
        p_dist = paired_comparison(d_accs, h_accs)
        v_dist = compute_verdict(p_dist["mean_diff"], p_dist["std_diff"], p_dist["wins"], p_dist["n"])

        frac_res: dict[str, Any] = {
            "fraction": frac,
            "hard": hard_summ,
            "distilled": dist_summ,
            "distilled_minus_hard": {**p_dist, "verdict": v_dist},
        }

        print(f"  Hard Acc:      {hard_summ['dev_accuracy_mean'] * 100:.2f}% ± {hard_summ['dev_accuracy_std'] * 100:.2f}%")
        print(f"  Distilled Acc: {dist_summ['dev_accuracy_mean'] * 100:.2f}% ± {dist_summ['dev_accuracy_std'] * 100:.2f}%")
        print(f"  Dist - Hard:   Mean={p_dist['mean_diff'] * 100:+.2f}%, Std={p_dist['std_diff'] * 100:.2f}%, "
              f"Wins={p_dist['wins']}/{p_dist['n']} => VERDICT: {v_dist}")

        # 3. Transfer (if frac < 1.0)
        if frac < 1.0:
            tr_dir = exp_final / f_tag / "distilled_transfer"
            tr_ckpt = ckpt_final / f_tag / "distilled_transfer"
            tr_summ, tr_recs = run_group(
                cfg, data, cache, tr_dir, tr_ckpt, seeds,
                mode="distilled_transfer", fraction=frac, T=T, alpha=alpha_t, wd=wd, base_epochs=base_epochs, resume=resume
            )
            tr_accs = [r["dev_accuracy"] for r in tr_recs]
            p_tr = paired_comparison(tr_accs, h_accs)
            v_tr = compute_verdict(p_tr["mean_diff"], p_tr["std_diff"], p_tr["wins"], p_tr["n"])

            frac_res["distilled_transfer"] = tr_summ
            frac_res["transfer_minus_hard"] = {**p_tr, "verdict": v_tr}

            print(f"  Transfer Acc:  {tr_summ['dev_accuracy_mean'] * 100:.2f}% ± {tr_summ['dev_accuracy_std'] * 100:.2f}%")
            print(f"  Trans - Hard:  Mean={p_tr['mean_diff'] * 100:+.2f}%, Std={p_tr['std_diff'] * 100:.2f}%, "
                  f"Wins={p_tr['wins']}/{p_tr['n']} => VERDICT: {v_tr}")

        final_results[f_tag] = frac_res

    save_json(final_results, exp_final / "summary.json")
    print("\nSaved final comparison summary to: experiments/v2/final/summary.json")
    return final_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 v2 experiments.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path")
    parser.add_argument("--base-epochs", type=int, default=30, help="Base epochs for full data")
    parser.add_argument("--no-resume", action="store_true", help="Force re-run existing runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = FashionData("./data")
    teacher = load_teacher(0)
    cache = cache_teacher_logits(teacher, data, path="checkpoints/v2/teacher_logits.pt")

    t_start = time.time()
    best_hp = run_tuning(cfg, data, cache, base_epochs=args.base_epochs, resume=not args.no_resume)
    final_res = run_final_comparison(cfg, data, cache, best_hp, base_epochs=args.base_epochs, resume=not args.no_resume)
    t_end = time.time()

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {(t_end - t_start) / 60:.1f} minutes!")
    print("=" * 70)


if __name__ == "__main__":
    main()
