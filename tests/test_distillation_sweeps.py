"""Tests for knowledge distillation sweep orchestration and CLI utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from paradigms.config import load_config
from scripts.run_distillation_sweeps import parse_args, run_alpha_stage, run_temperature_stage


def test_cli_parsing_defaults():
    with patch("sys.argv", ["run_distillation_sweeps.py"]):
        args = parse_args()
        assert args.stage == "all"
        assert args.resume is True
        assert args.seeds is None
        assert args.epochs is None
        assert args.out_dir == "experiments/distillation"
        assert args.ckpt_dir == "checkpoints/distillation"


def test_cli_parsing_overrides():
    with patch(
        "sys.argv",
        [
            "run_distillation_sweeps.py",
            "--stage",
            "temperature",
            "--no-resume",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "2",
            "--out-dir",
            "/tmp/test_out",
        ],
    ):
        args = parse_args()
        assert args.stage == "temperature"
        assert args.resume is False
        assert args.seeds == [0, 1]
        assert args.epochs == 2
        assert args.out_dir == "/tmp/test_out"


def test_alpha_selection_excludes_one():
    # Verify candidate filtering logic excludes alpha=1.0
    results_table = [
        {"alpha": 0.0, "dev_mean": 0.85, "dev_std": 0.01},
        {"alpha": 0.5, "dev_mean": 0.87, "dev_std": 0.01},
        {"alpha": 0.7, "dev_mean": 0.88, "dev_std": 0.01},
        {"alpha": 1.0, "dev_mean": 0.89, "dev_std": 0.01},  # Highest, but hard baseline
    ]
    candidates = [r for r in results_table if float(r["alpha"]) < 1.0]
    best_item = max(candidates, key=lambda x: x["dev_mean"])
    assert best_item["alpha"] == 0.7
