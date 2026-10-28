# AI Paradigms Lab

`ai-paradigms-lab` is a Python and PyTorch research codebase designed to investigate three AI paradigms on Fashion-MNIST in a CPU-only environment: Phase 1 implements knowledge distillation from a teacher CNN (421k parameters) to a compact student MLP (25k parameters) alongside a matched hard-label control across label-efficiency regimes; Phase 2 explores a bounded recursive self-improvement loop featuring an independent verifier, class-capped curation, protected holdout evaluation, and automated rollback; and Phase 3 synthesizes both paradigms by distilling an improved teacher produced via recursive self-improvement.

---

## Deliverables & Documentation Index

- **Executive Report (1-2 pages)**: [reports/REPORT.md](file:///home/natnale/ai-paradigms-lab/reports/REPORT.md)
- **Failure Analysis & Retrospective**: [reports/FAILURE_ANALYSIS.md](file:///home/natnale/ai-paradigms-lab/reports/FAILURE_ANALYSIS.md)
- **Consolidated Empirical Results**: [reports/RESULTS.md](file:///home/natnale/ai-paradigms-lab/reports/RESULTS.md)
- **Distillation Implementation Guide**: [docs/DISTILLATION_IMPLEMENTATION.md](file:///home/natnale/ai-paradigms-lab/docs/DISTILLATION_IMPLEMENTATION.md)
- **Recursive Implementation Guide**: [docs/RECURSIVE_IMPLEMENTATION.md](file:///home/natnale/ai-paradigms-lab/docs/RECURSIVE_IMPLEMENTATION.md)
- **Presentation Outline & 3-Min Demo Script**: [reports/PRESENTATION_OUTLINE.md](file:///home/natnale/ai-paradigms-lab/reports/PRESENTATION_OUTLINE.md)
- **Deliverables & Status Table**: [docs/DELIVERABLES.md](file:///home/natnale/ai-paradigms-lab/docs/DELIVERABLES.md)
- **Reproducibility & Verification Audit**: [docs/REPRODUCIBILITY.md](file:///home/natnale/ai-paradigms-lab/docs/REPRODUCIBILITY.md)
- **Complete Reproduction Script**: [scripts/reproduce.sh](file:///home/natnale/ai-paradigms-lab/scripts/reproduce.sh)

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt && pip install -e .
```

---

## Quick Start (Main Results Reproduction)

Reproduce all primary findings and consolidated tables via 5 sequential commands (taken from `scripts/reproduce.sh`):

```bash
# 1. Verify data splits and compute SHA-256 partition hashes
python scripts/verify_splits.py

# 2. Run Phase 1 v2 label-efficient distillation study across data regimes
python scripts/run_v2_experiments.py

# 3. Run Phase 2 recursive self-improvement experiments across verifiers
python scripts/run_recursive_experiments.py

# 4. Compile Phase 2 recursive reports and progression plots
python scripts/make_recursive_report.py

# 5. Compile consolidated deliverables into reports/RESULTS.md
python scripts/make_deliverables.py
```

---

## Data Splits & Random Seeds

All partitions are generated deterministically by `scripts/verify_splits.py` using `split_seed=42` and verified in `docs/split_manifest.json`:

| Split Name | Samples | Scope / Role | Seed Protocol |
| :--- | :--- | :--- | :--- |
| `train` | 50,000 | Supervised training base | `split_seed=42` |
| `dev` | 10,000 | Model selection & hyperparameter tuning | `split_seed=42` |
| `gate` | 4,997 | Phase 2 candidate acceptance decisions (within dev) | `split_dev` 50/50 split |
| `monitor` | 5,003 | Phase 2 passive generalization tracking (within dev) | `split_dev` 50/50 split |
| `test` | 10,000 | Protected official test set (one-shot access only) | Torchvision test set |

### Evaluation Seeds Inventory
- **Phase 1 v1**: `seeds = [0, 1, 2, 3, 4]` (`configs/base.yaml`)
- **Phase 1 v2 (Label-Efficient Study)**: `seeds = [10, 11, 12, 13, 14]` (`scripts/run_v2_experiments.py`)
- **Phase 2 (Recursive Loop)**: `seeds = [0, 1, 2]` (`configs/recursive.yaml`)

---

## Repository Map

```
ai-paradigms-lab/
├── configs/                   # Base and recursive YAML configuration files
├── data/                      # Fashion-MNIST raw data and partition split JSONs
├── docs/                      # Architectural implementation guides and audit manifests
│   ├── DISTILLATION_IMPLEMENTATION.md
│   ├── RECURSIVE_IMPLEMENTATION.md
│   ├── REPRODUCIBILITY.md
│   ├── DELIVERABLES.md
│   └── split_manifest.json
├── experiments/               # Serialized experimental JSON metrics and benchmarks
│   ├── teacher/               # Phase 1 teacher CNN training records
│   ├── student_hard/          # Phase 1 hard student baseline records
│   ├── distillation/          # Phase 1 distillation parameter sweeps
│   ├── v2/                    # Phase 1 v2 label-efficient study outputs
│   ├── recursive/             # Phase 2 recursive self-improvement runs
│   └── final/                 # One-shot official test set evaluation records
├── reports/                   # Human-readable markdown reports and visualization figures
│   ├── REPORT.md              # Executive summary report (1-2 pages)
│   ├── FAILURE_ANALYSIS.md    # Systematic failure mode and root-cause analysis
│   ├── RESULTS.md             # Consolidated empirical results document
│   ├── PRESENTATION_OUTLINE.md# 10-slide presentation outline and demo script
│   ├── figures/               # Phase 1 visualization figures
│   ├── v2/                    # Phase 1 v2 figures and markdown tables
│   └── recursive/             # Phase 2 figures and markdown summary tables
├── scripts/                   # Executable CLI experiment runners and report compilers
├── src/paradigms/             # Core library source code (data, models, training, recursive)
└── tests/                     # Comprehensive automated pytest test suite
```

---

## Running the Automated Test Suite

Run the full pytest suite across unit, integration, and security audit tests:

```bash
# Run all unit tests
pytest -q

# Run static test-set access security audit
pytest tests/test_test_set_access.py -v

# Run markdown code symbol and link validator
python scripts/check_docs_refs.py
```

---

## Results Summary

### Phase 1 v2: Label-Efficient Distillation Summary
*Full table in `reports/RESULTS.md` (5 fresh seeds, matched tuning budgets)*
- **2% Data Regime (1,000 labels)**: Distilled Student achieved **82.48% vs 80.87%** for Hard Student (**+1.61% ± 0.19% gain, 5/5 wins**). Distilled Transfer achieved **87.02% (+6.15% gain)**.
- **10% Data Regime (5,000 labels)**: Distilled Student achieved **85.99% vs 85.01%** (**+0.98% ± 0.22% gain, 5/5 wins**). Distilled Transfer achieved **87.13% (+2.11% gain)**.
- **100% Data Regime (50,000 labels)**: Distilled Student achieved **88.39% vs 88.20%** (**+0.20% ± 0.17% gain**, paired t-test $p=0.0810$, marginal).

### Phase 2: Recursive Self-Improvement Summary
*Full table in `reports/RESULTS.md` (Seeds 0, 1, 2, SmallCNN, 4 rounds)*
- **`oracle_gate`**: **+2.19% ± 1.07% test gain** (83.89% to 86.08%, 3/3 wins, 0% noise). Proves steady recursive expansion ceiling.
- **`selfconf_nogate`**: **+1.08% ± 0.54% test gain**, but accumulated **8.66% label noise** and caused **-11.36% collapse on Class 4 (Coat)**.
- **`selfconf_gate`**: Acceptance gate caught class degradation, rejecting corrupted rounds and limiting noise to 2.17% (TIE with control).
- **`agreement_gate`**: Random Forest on raw flat pixels lacked spatial convolutional invariance; 0 rounds accepted (TIE with control).
- **`control_none`**: 0 rounds accepted; confirms that extra training without verified data yields zero gain.

---

## Status

Phase 1: complete

### Phase 1 Final Official Test Results

| Model | Parameters | Test Accuracy (Mean ± Std) |
| :--- | :--- | :--- |
| Teacher (CNN) | 421,642 | 91.35% ± 0.19% |
| Hard Student (MLP) | 25,450 | 86.45% ± 0.06% |
| Distilled Student (MLP) | 25,450 | 86.64% ± 0.25% |

**Paired Comparison (Distilled - Hard)**: Mean Diff = `+0.20% ± 0.27%`, Wins = `4 / 5` seeds.

---

## Phase 1 v2: Label-Efficient Distillation Study

A rigorous, label-efficient study evaluating distillation across low-data regimes (2%, 10%, 100% labelled data) with fair independent hyperparameter tuning and representation transfer over unlabelled samples.

- **Comprehensive Deep Dive & Guide**: [COMPREHENSIVE_GUIDE.md](file:///home/natnale/ai-paradigms-lab/docs/COMPREHENSIVE_GUIDE.md)
- **Motivation & Retrospective**: [WHY_RETRAIN.md](file:///home/natnale/ai-paradigms-lab/docs/phase1_v2/WHY_RETRAIN.md)
- **Experimental Protocol**: [PROTOCOL.md](file:///home/natnale/ai-paradigms-lab/docs/phase1_v2/PROTOCOL.md)
- **Results & Verdicts**: [RESULTS.md](file:///home/natnale/ai-paradigms-lab/docs/phase1_v2/RESULTS.md)

---

## Phase 2: Recursive Self-Improvement Loop

Phase 2: complete

- **Comprehensive Deep Dive & Guide**: [RECURSIVE_LOOP_GUIDE.md](file:///home/natnale/ai-paradigms-lab/docs/RECURSIVE_LOOP_GUIDE.md)
- **Empirical Findings & Verdicts**: [rec_findings.md](file:///home/natnale/ai-paradigms-lab/reports/recursive/tables/rec_findings.md)

### Phase 2 Final Official Test Results

| Configuration | Round 0 Test Acc (Mean ± Std) | Final Test Acc (Mean ± Std) | Paired Gain (Final - R0) | Wins / Seeds |
| :--- | :--- | :--- | :--- | :--- |
| `oracle_gate` | 83.89% ± 0.83% | 86.08% ± 0.26% | +2.19% ± 1.07% | 3 / 3 |
| `agreement_gate` | 83.89% ± 0.83% | 83.89% ± 0.83% | +0.00% ± 0.00% | 0 / 3 |
| `selfconf_gate` | 83.89% ± 0.83% | 83.97% ± 0.90% | +0.08% ± 0.12% | 1 / 3 |
| `selfconf_nogate` | 83.89% ± 0.83% | 84.97% ± 0.32% | +1.08% ± 0.54% | 3 / 3 |
| `control_none` | 83.89% ± 0.83% | 83.89% ± 0.83% | +0.00% ± 0.00% | 0 / 3 |
