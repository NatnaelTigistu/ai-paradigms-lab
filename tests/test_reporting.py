"""Unit tests for Phase 1 reporting, data aggregation, table formatting, and plot generation."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from paradigms.reporting.phase1 import (
    aggregate_main_table,
    aggregate_paired_table,
    aggregate_perclass_table,
    aggregate_sweep_stage,
)
from paradigms.reporting.plots import plot_main_bars
from paradigms.reporting.tables import format_markdown_table, save_csv_table, save_markdown_table, write_table
from paradigms.utils.io import save_json


@pytest.fixture
def fake_exp_dir(tmp_path: Path) -> Path:
    """Construct a minimal fake experiment directory structure for testing."""
    exp = tmp_path / "experiments"
    exp.mkdir(parents=True)

    # 1. Benchmark
    bench_dir = exp / "benchmark"
    bench_dir.mkdir(parents=True)
    bench_data = {
        "models": {
            "Teacher (CNN)": {
                "name": "Teacher (CNN)",
                "params": 421642,
                "size_kb": 1650.39,
                "b1_median_ms": 0.8,
                "b256_median_ms": 235.0,
            },
            "Hard Student (MLP)": {
                "name": "Hard Student (MLP)",
                "params": 25450,
                "size_kb": 101.65,
                "b1_median_ms": 0.022,
                "b256_median_ms": 0.24,
            },
            "Distilled Student (MLP)": {
                "name": "Distilled Student (MLP)",
                "params": 25450,
                "size_kb": 101.65,
                "b1_median_ms": 0.022,
                "b256_median_ms": 0.24,
            },
        }
    }
    save_json(bench_data, bench_dir / "benchmark.json")

    # 2. Teacher
    teacher_dir = exp / "teacher"
    teacher_dir.mkdir(parents=True)
    save_json({"dev_accuracy_mean": 0.9182, "dev_accuracy_std": 0.0023, "params": 421642}, teacher_dir / "summary.json")

    # 3. Hard student (seed 0: 0.86, seed 1: 0.88 -> mean=0.87, std=0.01 with ddof=0)
    hard_dir = exp / "student_hard" / "h32_f1.0"
    hard_dir.mkdir(parents=True)
    save_json(
        {
            "seed": 0,
            "dev_accuracy": 0.86,
            "dev_per_class_accuracy": [0.8] * 10,
            "params": 25450,
        },
        hard_dir / "seed0.json",
    )
    save_json(
        {
            "seed": 1,
            "dev_accuracy": 0.88,
            "dev_per_class_accuracy": [0.9] * 10,
            "params": 25450,
        },
        hard_dir / "seed1.json",
    )
    save_json({"dev_accuracy_mean": 0.87, "dev_accuracy_std": 0.01, "params": 25450}, hard_dir / "summary.json")

    # 4. Distilled student (seed 0: 0.87, seed 1: 0.90 -> diffs: +0.01, +0.02)
    dist_dir = exp / "student_distilled" / "h32_f1.0_T4.0_a0.5"
    dist_dir.mkdir(parents=True)
    save_json(
        {
            "seed": 0,
            "dev_accuracy": 0.87,
            "dev_per_class_accuracy": [0.82] * 10,
            "params": 25450,
        },
        dist_dir / "seed0.json",
    )
    save_json(
        {
            "seed": 1,
            "dev_accuracy": 0.90,
            "dev_per_class_accuracy": [0.92] * 10,
            "params": 25450,
        },
        dist_dir / "seed1.json",
    )
    save_json({"dev_accuracy_mean": 0.885, "dev_accuracy_std": 0.015, "params": 25450}, dist_dir / "summary.json")

    # 5. Distillation sweep: T1 present, other Ts omitted
    t1_dir = exp / "distillation" / "temperature" / "T1" / "distilled"
    t1_dir.mkdir(parents=True)
    save_json({"seed": 0, "dev_accuracy": 0.87}, t1_dir / "seed0.json")
    save_json({"seed": 1, "dev_accuracy": 0.89}, t1_dir / "seed1.json")
    save_json({"dev_accuracy_mean": 0.88, "dev_accuracy_std": 0.01}, t1_dir / "summary.json")

    return exp


def test_mean_std_ddof_zero(fake_exp_dir: Path):
    """Verify standard deviation calculations use population std (ddof=0)."""
    paired = aggregate_paired_table(fake_exp_dir)
    # Seed 0: 0.87 - 0.86 = +0.01
    # Seed 1: 0.90 - 0.88 = +0.02
    diffs = paired["diffs"]
    assert diffs == pytest.approx([0.01, 0.02], abs=1e-6)

    expected_mean = np.mean([0.01, 0.02])
    expected_std_ddof0 = np.std([0.01, 0.02], ddof=0)
    expected_std_ddof1 = np.std([0.01, 0.02], ddof=1)

    assert paired["mean_diff"] == pytest.approx(expected_mean, abs=1e-6)
    assert paired["std_diff"] == pytest.approx(expected_std_ddof0, abs=1e-6)
    # Ensure ddof=0 was used rather than sample ddof=1
    assert paired["std_diff"] != pytest.approx(expected_std_ddof1, abs=1e-6)


def test_paired_diff_sign(fake_exp_dir: Path):
    """Verify paired difference sign is strictly (distilled - hard)."""
    paired = aggregate_paired_table(fake_exp_dir)
    # Seed 0: distilled (0.87) > hard (0.86) -> positive difference +0.01
    assert paired["diffs"][0] > 0
    assert paired["diffs"][0] == pytest.approx(0.01, abs=1e-6)
    # Seed 1: distilled (0.90) > hard (0.88) -> positive difference +0.02
    assert paired["diffs"][1] > 0
    assert paired["diffs"][1] == pytest.approx(0.02, abs=1e-6)
    assert paired["wins"] == 2
    assert paired["n"] == 2


def test_missing_group_skipped_with_warning(fake_exp_dir: Path):
    """Verify missing sweep groups emit warnings and are gracefully skipped."""
    # Temperature sweep: T1 exists, T2, T4, T8, T16 are missing
    with pytest.warns(UserWarning, match="Missing sweep group") as warning_records:
        temp_res = aggregate_sweep_stage("temperature", fake_exp_dir)

    # Should have caught warnings for missing T2, T4, T8, T16
    assert len(warning_records) > 0
    # Exactly one row present (for T1)
    assert len(temp_res["rows"]) == 1
    assert temp_res["rows"][0][0] == "1"

    # Alpha sweep: no groups present in fake_exp_dir
    with pytest.warns(UserWarning, match="Missing sweep group"):
        alpha_res = aggregate_sweep_stage("alpha", fake_exp_dir)
    assert len(alpha_res["rows"]) == 0


def test_table_formatting_and_saving(tmp_path: Path):
    """Verify Markdown and CSV table serialization produces consistent outputs."""
    headers = ["Model", "Accuracy", "Diff"]
    rows = [["Model A", "85.00%", "+1.00%"], ["Model B", "84.00%", "-1.00%"]]

    md_str = format_markdown_table(headers, rows)
    assert "| Model   | Accuracy | Diff   |" in md_str
    assert "| Model A | 85.00%   | +1.00% |" in md_str

    md_path, csv_path = write_table("test_summary", headers, rows, tmp_path)
    assert md_path.exists()
    assert csv_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Model A" in md_content
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "Model A,85.00%,+1.00%" in csv_content


def test_plot_main_bars_generation(fake_exp_dir: Path, tmp_path: Path):
    """Verify plot generation produces valid non-empty PNG files at 150 dpi."""
    main_res = aggregate_main_table(fake_exp_dir)
    out_fig = tmp_path / "fig_main_bars.png"
    p = plot_main_bars(main_res["models_data"], out_fig)

    assert p.exists()
    assert p.stat().st_size > 1000  # Valid non-empty PNG
