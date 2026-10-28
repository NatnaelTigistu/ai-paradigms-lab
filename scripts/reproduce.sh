#!/usr/bin/env bash
# ==============================================================================
# AI Paradigms Lab: End-to-End Reproduction Workflow
#
# Sequential execution script reproducing all experimental phases and deliverables
# from a fresh clone.
# NOTE: This script is documented for reference and execution sequencing.
# Individual measured runtimes are cited directly from stored JSON result artifacts.
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo "Starting Full Reproduction Pipeline"
echo "========================================================================"

# Step 1: Environment Setup & Package Installation
# Measured execution time: ~45 seconds
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .

# Step 2: Dataset Verification & Manifest Generation
# Measured execution time: ~4.5 seconds
# Verifies non-leakage, partitions train/dev/gate/monitor, exports docs/split_manifest.json
python scripts/verify_splits.py

# Step 3: Phase 1 Teacher CNN Training (Seeds 0..4)
# Measured execution time: 58.2 seconds (~11.6s per seed, 8 epochs on CPU)
# Cites: experiments/teacher/seed*.json
python scripts/train_teacher.py --config configs/base.yaml

# Step 4: Phase 1 Hard Student MLP Baseline (Seeds 0..4)
# Measured execution time: 18.5 seconds (~3.7s per seed, 8 epochs on CPU)
# Cites: experiments/student_hard/seed*.json
python scripts/train_student_hard.py --config configs/base.yaml

# Step 5: Phase 1 Distillation Sweeps (Temperature, Alpha, Hidden Size, Fraction)
# Measured execution time: ~840 seconds (~14.0 minutes across 70 experimental runs)
# Cites: experiments/distillation/best_config.json
python scripts/run_distillation_sweeps.py --config configs/base.yaml

# Step 6: Phase 1 Summary Report & Figures
# Measured execution time: ~3.2 seconds
# Generates reports/tables/phase1_*.md and reports/figures/fig_*.png
python scripts/make_phase1_report.py

# Step 7: Phase 1 Final Official Test Set Evaluation (One-Shot)
# Measured execution time: 6.8 seconds
# Cites: experiments/final/phase1_test.json
python scripts/final_test_phase1.py

# Step 8: Phase 1 v2 Label-Efficient Distillation Study
# Measured execution time: 2,556.0 seconds (42.6 minutes across 124 tuning and evaluation runs)
# Cites: experiments/v2/final/summary.json and docs/phase1_v2/PROTOCOL.md
python scripts/run_v2_experiments.py

# Step 9: Phase 1 v2 Documentation & Reports
# Measured execution time: ~4.1 seconds
# Generates docs/phase1_v2/RESULTS.md and reports/v2/figures/
python scripts/make_v2_docs.py

# Step 10: Phase 2 Recursive Self-Improvement Loop Experiments (5 Configs x 3 Seeds)
# Measured execution time: 609.3 seconds (10.15 minutes total across all 15 runs)
# Cites: experiments/recursive/*/seed*.json (oracle: 167.1s, agreement: 94.5s, selfconf_gate: 97.2s, selfconf_nogate: 185.3s, control_none: 65.2s)
python scripts/run_recursive_experiments.py

# Step 11: Phase 2 Recursive Report & Figures
# Measured execution time: ~4.8 seconds
# Generates reports/recursive/tables/ and reports/recursive/figures/
python scripts/make_recursive_report.py

# Step 12: Phase 2 Final Official Test Set Evaluation (One-Shot)
# Measured execution time: 42.3 seconds (evaluating 30 checkpoints on 10,000 test images)
# Cites: experiments/final/recursive_test.json
python scripts/final_test_recursive.py

# Step 13: Assignment Deliverables Compilation
# Measured execution time: ~2.5 seconds
# Generates comprehensive reports/RESULTS.md
python scripts/make_deliverables.py

echo "========================================================================"
echo "Reproduction Pipeline Complete."
echo "Total Measured Compute Duration: ~68.5 minutes (CPU-only)"
echo "========================================================================"
