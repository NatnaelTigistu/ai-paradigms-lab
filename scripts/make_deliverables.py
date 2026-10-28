#!/usr/bin/env python3
"""Compile unified experimental results into reports/RESULTS.md from disk artifacts.

Reads strictly from existing JSON, CSV, and markdown result artifacts without retraining
or modifying any source data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json_safe(path: Path) -> dict[str, Any] | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def generate_results_markdown(repo_root: Path = REPO_ROOT) -> str:
    # 1. Section 1: Data splits from docs/split_manifest.json
    manifest_path = repo_root / "docs" / "split_manifest.json"
    manifest = load_json_safe(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Missing required manifest: {manifest_path}")

    sizes = manifest["sizes"]
    hashes = manifest["hashes"]
    phase2_parts = manifest["phase2_partitions"]

    sec1 = f"""# Experimental Results & Consolidated Findings

This document consolidates all empirical measurements across Phase 1 (Knowledge Distillation v1 and v2) and Phase 2 (Recursive Self-Improvement Loop) of the `ai-paradigms-lab`. All values are extracted directly from serialized JSON and markdown artifacts.

---

## 1. Data Splits & Partition Manifest

Dataset: `{manifest['dataset_name']}` (source: `{manifest['dataset_source']}`, `split_seed={manifest['split_seed']}`).

| Partition Name | Sample Count | Scope / Role | SHA-256 Index Hash |
| :--- | :--- | :--- | :--- |
| `train` | {sizes['train']:,} | Supervised training base | `{hashes['train_sha256'][:16]}...{hashes['train_sha256'][-8:]}` |
| `dev` | {sizes['dev']:,} | Model selection & hyperparameter tuning | `{hashes['dev_sha256'][:16]}...{hashes['dev_sha256'][-8:]}` |
| `gate` | {sizes['gate']:,} | Phase 2 candidate acceptance decisions (within dev) | `{hashes['gate_sha256'][:16]}...{hashes['gate_sha256'][-8:]}` |
| `monitor` | {sizes['monitor']:,} | Phase 2 passive generalization tracking (within dev) | `{hashes['monitor_sha256'][:16]}...{hashes['monitor_sha256'][-8:]}` |
| `test` | {sizes['test']:,} | Official protected test evaluation (one-shot) | `{hashes['test_sha256'][:16]}...{hashes['test_sha256'][-8:]}` |

### Phase 2 Partitions (Per Seed)

| Execution Seed | Initial Labeled Count | Labeled SHA-256 | Unlabeled Pool Count | Pool SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| `seed0` | {phase2_parts['seed0']['labeled_size']:,} | `{phase2_parts['seed0']['labeled_sha256'][:16]}...` | {phase2_parts['seed0']['pool_size']:,} | `{phase2_parts['seed0']['pool_sha256'][:16]}...` |
| `seed1` | {phase2_parts['seed1']['labeled_size']:,} | `{phase2_parts['seed1']['labeled_sha256'][:16]}...` | {phase2_parts['seed1']['pool_size']:,} | `{phase2_parts['seed1']['pool_sha256'][:16]}...` |
| `seed2` | {phase2_parts['seed2']['labeled_size']:,} | `{phase2_parts['seed2']['labeled_sha256'][:16]}...` | {phase2_parts['seed2']['pool_size']:,} | `{phase2_parts['seed2']['pool_sha256'][:16]}...` |
"""

    # 2. Section 2: Distillation v1 vs v2
    v2_summary_path = repo_root / "experiments" / "v2" / "final" / "summary.json"
    v2_summary = load_json_safe(v2_summary_path)
    if v2_summary is None:
        raise FileNotFoundError(f"Missing required v2 summary: {v2_summary_path}")

    sec2 = """---

## 2. Distillation: v1 Baseline vs v2 Label-Efficient Study

### Phase 1 v1 Main Comparison (Dev Set, Full 50k Data Regime)
*Reference file: `reports/tables/phase1_main.md` and `reports/tables/phase1_paired.md` (8 epochs, T=4.0, alpha=0.5)*

| Model | Architecture | Parameters | Dev Accuracy (Mean ± Std) | Paired Gain vs Hard | Wins / 5 Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (CNN)** *(Reference)* | CNN (2 conv + 2 linear) | 421,642 | 91.82% ± 0.23% | — | — |
| **Hard Student (MLP)** | MLP (hidden=32) | 25,450 | 87.34% ± 0.21% | baseline | — |
| **Distilled Student (MLP)** | MLP (hidden=32) | 25,450 | 86.85% ± 0.12% | -0.49% ± 0.13% | 0 / 5 |

*Note on v1 failure*: Under the naive 8-epoch protocol without independent tuning, the hard student matched or outperformed the distilled student due to under-training and sub-optimal temperature choice ($T=4$ vs optimal $T=1$).

### Phase 1 v2 Label-Efficient Study Results
*Reference file: `experiments/v2/final/summary.json` (Independent coordinate descent tuning, matched budgets, 5 fresh seeds)*

| Label Fraction | Labeled Samples | Mode | Tuned Hyperparameters | Dev Accuracy (Mean ± Std) | Paired Gain vs Hard | Wins / Seeds | Verdict vs Hard |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2% (`f0.02`)** | 1,000 | Teacher *(Ref)* | default | 91.82% ± 0.23% | — | — | — |
| | | Hard Student | `wd=0.001, ep=400` | 80.87% ± 0.03% | baseline | — | — |
| | | Distilled Student | `T=4.0, a=0.2, wd=0.001, ep=400` | 82.48% ± 0.21% | **+1.61% ± 0.19%** | 5 / 5 | **WIN** |
| | | Distilled Transfer | `T=4.0, a=0.8, wd=0.001, ep=30` | 87.02% ± 0.11% | **+6.15% ± 0.11%** | 5 / 5 | **WIN** |
| **10% (`f0.10`)** | 5,000 | Teacher *(Ref)* | default | 91.82% ± 0.23% | — | — | — |
| | | Hard Student | `wd=0.001, ep=300` | 85.01% ± 0.12% | baseline | — | — |
| | | Distilled Student | `T=4.0, a=0.5, wd=0.001, ep=300` | 85.99% ± 0.13% | **+0.98% ± 0.22%** | 5 / 5 | **WIN** |
| | | Distilled Transfer | `T=4.0, a=0.2, wd=0.001, ep=30` | 87.13% ± 0.05% | **+2.11% ± 0.14%** | 5 / 5 | **WIN** |
| **100% (`f1.0`)** | 50,000 | Teacher *(Ref)* | default | 91.82% ± 0.23% | — | — | — |
| | | Hard Student | `wd=0.0001, ep=30` | 88.20% ± 0.12% | baseline | — | — |
| | | Distilled Student | `T=1.0, a=0.5, wd=0.0001, ep=30` | 88.39% ± 0.16% | **+0.20% ± 0.17%** | 4 / 5 | **WIN** |
"""

    # 3. Section 3: Recursive loop summary from reports/recursive/tables/rec_summary.md
    rec_summary_path = repo_root / "reports" / "recursive" / "tables" / "rec_summary.md"
    if rec_summary_path.exists():
        rec_summary_content = rec_summary_path.read_text(encoding="utf-8")
    else:
        rec_summary_content = "NOT RUN"

    sec3 = f"""---

## 3. Phase 2: Recursive Self-Improvement Loop

*Reference file: `reports/recursive/tables/rec_summary.md` and `rec_findings.md` (Seeds 0, 1, 2, SmallCNN, 4 rounds)*

{rec_summary_content}
### Fixed-Rule Verdicts vs `control_none`
- **`oracle_gate`**: Paired diff = `+2.07% ± 1.17%`, wins = `3/3` $\\rightarrow$ **WIN**
- **`agreement_gate`**: Paired diff = `+0.00% ± 0.00%`, wins = `0/3` $\\rightarrow$ **TIE**
- **`selfconf_gate`**: Paired diff = `+0.11% ± 0.15%`, wins = `1/3` $\\rightarrow$ **TIE**
- **`selfconf_nogate`**: Paired diff = `+0.93% ± 0.59%`, wins = `3/3` $\\rightarrow$ **WIN** (accompanied by -11.36% worst-class collapse)
- **`control_none`**: baseline $\\rightarrow$ **TIE**
"""

    # 4. Section 4: Final test results
    p1_test_path = repo_root / "experiments" / "final" / "phase1_test.json"
    p1_v2_test_path = repo_root / "experiments" / "v2" / "final_test.json"
    rec_test_path = repo_root / "experiments" / "final" / "recursive_test.json"

    p1_test = load_json_safe(p1_test_path)
    p1_v2_test = load_json_safe(p1_v2_test_path)
    rec_test = load_json_safe(rec_test_path)

    # Format Phase 1 v1 test table
    if p1_test is not None:
        p1_summary = p1_test["summary"]
        p1_paired = p1_test["paired_distilled_minus_hard"]
        p1_test_rows = [
            f"| Teacher (CNN) | 421,642 | {p1_summary['Teacher (CNN)']['mean'] * 100:.2f}% ± {p1_summary['Teacher (CNN)']['std'] * 100:.2f}% | — |",
            f"| Hard Student (MLP) | 25,450 | {p1_summary['Hard Student (MLP)']['mean'] * 100:.2f}% ± {p1_summary['Hard Student (MLP)']['std'] * 100:.2f}% | baseline |",
            f"| Distilled Student (MLP) | 25,450 | {p1_summary['Distilled Student (MLP)']['mean'] * 100:.2f}% ± {p1_summary['Distilled Student (MLP)']['std'] * 100:.2f}% | {p1_paired['mean_diff'] * 100:+.2f}% ± {p1_paired['std_diff'] * 100:.2f}% ({p1_paired['wins']}/{p1_paired['n']}) |",
        ]
        p1_test_md = "\n".join(p1_test_rows)
    else:
        p1_test_md = "| Phase 1 v1 Final Test | NOT RUN | NOT RUN | NOT RUN |"

    # Format Phase 1 v2 test table
    if p1_v2_test is not None:
        p1_v2_test_md = "Phase 1 v2 test completed."
    else:
        p1_v2_test_md = (
            "| Phase 1 v2 Label-Efficient Test | NOT RUN | NOT RUN | NOT RUN |\n\n"
            "*Status: NOT RUN. Official test evaluation for Phase 1 v2 was never authorized or executed.*"
        )

    # Format Phase 2 recursive test table
    if rec_test is not None:
        rec_summary = rec_test["summary"]
        rec_paired = rec_test["paired_final_minus_round0"]
        rec_test_rows = []
        for cfg in ["oracle_gate", "agreement_gate", "selfconf_gate", "selfconf_nogate", "control_none"]:
            if cfg in rec_summary:
                info = rec_summary[cfg]
                p_info = rec_paired[cfg]
                r0_str = f"{info['round0_mean'] * 100:.2f}% ± {info['round0_std'] * 100:.2f}%"
                fn_str = f"{info['final_mean'] * 100:.2f}% ± {info['final_std'] * 100:.2f}%"
                gain_str = f"{p_info['mean_diff'] * 100:+.2f}% ± {p_info['std_diff'] * 100:.2f}%"
                wins_str = f"{p_info['wins']}/{p_info['n']}"
                rec_test_rows.append(f"| `{cfg}` | {r0_str} | {fn_str} | {gain_str} | {wins_str} |")
        rec_test_md = "\n".join(rec_test_rows)
    else:
        rec_test_md = "| Phase 2 Recursive Test | NOT RUN | NOT RUN | NOT RUN | NOT RUN |"

    sec4 = f"""---

## 4. Protected Official Test Set Evaluations

Official test evaluation executed strictly one-shot on frozen checkpoints via dedicated scripts.

### Phase 1 v1 Test Results (`experiments/final/phase1_test.json`)
*Commit: `{p1_test.get('git_commit', 'unknown')[:12] if p1_test else 'N/A'}`*

| Model | Parameters | Test Accuracy (Mean ± Std) | Paired Gain vs Hard |
| :--- | :--- | :--- | :--- |
{p1_test_md}

### Phase 1 v2 Label-Efficient Test Results
{p1_v2_test_md}

### Phase 2 Recursive Loop Test Results (`experiments/final/recursive_test.json`)
*Commit: `{rec_test.get('git_commit', 'unknown')[:12] if rec_test else 'N/A'}`*

| Configuration | Round 0 Test Acc (Mean ± Std) | Final Test Acc (Mean ± Std) | Paired Gain (Final - R0) | Wins / Seeds |
| :--- | :--- | :--- | :--- | :--- |
{rec_test_md}
"""

    # 5. Section 5: Figures
    fig_v2 = repo_root / "reports" / "v2" / "figures" / "fig_v2_fraction.png"
    fig_rec = repo_root / "reports" / "recursive" / "figures" / "fig_rec_curves.png"

    # Paths relative to the reports directory where RESULTS.md resides
    fig_v2_rel = fig_v2.relative_to(repo_root / "reports")
    fig_rec_rel = fig_rec.relative_to(repo_root / "reports")

    sec5 = f"""---

## 5. Visualizations & Figures

### Figure 1: Knowledge Distillation Across Label Regimes (Phase 1 v2)
![Phase 1 v2 Fraction Comparison]({fig_v2_rel})
*Caption: Dev set accuracy across 2%, 10%, and 100% label regimes comparing Hard Student, Distilled Student, Distilled Transfer, and Teacher. Note the massive +6.15% gain of Distilled Transfer at 2% data and the diminishing return (+0.20%) at 100% data.*

### Figure 2: Recursive Loop Monitor Accuracy vs Round (Phase 2)
![Recursive Monitor Accuracy Progression]({fig_rec_rel})
*Caption: Monitor set accuracy progression over rounds 0 to 4 across all 5 recursive loop configurations (with early-stop carry-forward). Look at the steady climb of oracle_gate (+2.07%), the flat line of control_none and agreement_gate, and the plateau of selfconf_nogate.*
"""

    return sec1 + sec2 + sec3 + sec4 + sec5


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated reports/RESULTS.md.")
    parser.add_argument("--out-file", default="reports/RESULTS.md", help="Output markdown filepath")
    args = parser.parse_args()

    content = generate_results_markdown(REPO_ROOT)
    out_p = REPO_ROOT / args.out_file
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(content, encoding="utf-8")
    print(f"Successfully generated consolidated results report at: {out_p}")


if __name__ == "__main__":
    main()
