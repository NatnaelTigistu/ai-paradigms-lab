# AI Paradigms Lab

`ai-paradigms-lab` is a Python and PyTorch research codebase designed to investigate three AI paradigms on Fashion-MNIST in a CPU-only environment: Phase 1 implements knowledge distillation from a teacher CNN to a compact student MLP alongside a matched hard-label control; Phase 2 explores a bounded recursive self-improvement loop featuring an independent verifier, protected holdout evaluation, and automated rollback; and Phase 3 synthesizes both paradigms by distilling an improved teacher produced via recursive self-improvement.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## Status

Phase 1: in progress
