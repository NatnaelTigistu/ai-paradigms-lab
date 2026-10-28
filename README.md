# AI Paradigms Lab

`ai-paradigms-lab` is a Python and PyTorch research codebase designed to investigate three AI paradigms on Fashion-MNIST in a CPU-only environment: Phase 1 implements knowledge distillation from a teacher CNN to a compact student MLP alongside a matched hard-label control; Phase 2 explores a bounded recursive self-improvement loop featuring an independent verifier, protected holdout evaluation, and automated rollback; and Phase 3 synthesizes both paradigms by distilling an improved teacher produced via recursive self-improvement.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## Status

Phase 1: complete

### Phase 1 Final Official Test Results

| Model | Parameters | Test Accuracy (Mean ± Std) |
| :--- | :--- | :--- |
| Teacher (CNN) | 421,642 | 91.35% ± 0.19% |
| Hard Student (MLP) | 25,450 | 86.45% ± 0.06% |
| Distilled Student (MLP) | 25,450 | 86.64% ± 0.25% |

**Paired Comparison (Distilled - Hard)**: Mean Diff = `+0.20% ± 0.27%`, Wins = `4 / 5` seeds.

## Phase 1 v2: Label-Efficient Distillation Study

A rigorous, label-efficient study evaluating distillation across low-data regimes (2%, 10%, 100% labelled data) with fair independent hyperparameter tuning and representation transfer over unlabelled samples.

- **Comprehensive Deep Dive & Guide**: [COMPREHENSIVE_GUIDE.md](file:///home/natnale/ai-paradigms-lab/docs/COMPREHENSIVE_GUIDE.md)
- **Motivation & Retrospective**: [WHY_RETRAIN.md](file:///home/natnale/ai-paradigms-lab/docs/phase1_v2/WHY_RETRAIN.md)
- **Experimental Protocol**: [PROTOCOL.md](file:///home/natnale/ai-paradigms-lab/docs/phase1_v2/PROTOCOL.md)
- **Results & Verdicts**: [RESULTS.md](file:///home/natnale/ai-paradigms-lab/docs/phase1_v2/RESULTS.md)

## Phase 2: Recursive Self-Improvement Loop

Phase 2: complete

### Phase 2 Final Official Test Results

| Configuration | Round 0 Test Acc (Mean ± Std) | Final Test Acc (Mean ± Std) | Paired Gain (Final - R0) | Wins / Seeds |
| :--- | :--- | :--- | :--- | :--- |
| `oracle_gate` | 83.89% ± 0.83% | 86.08% ± 0.26% | +2.19% ± 1.07% | 3 / 3 |
| `agreement_gate` | 83.89% ± 0.83% | 83.89% ± 0.83% | +0.00% ± 0.00% | 0 / 3 |
| `selfconf_gate` | 83.89% ± 0.83% | 83.97% ± 0.90% | +0.08% ± 0.12% | 1 / 3 |
| `selfconf_nogate` | 83.89% ± 0.83% | 84.97% ± 0.32% | +1.08% ± 0.54% | 3 / 3 |
| `control_none` | 83.89% ± 0.83% | 83.89% ± 0.83% | +0.00% ± 0.00% | 0 / 3 |
