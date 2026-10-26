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
