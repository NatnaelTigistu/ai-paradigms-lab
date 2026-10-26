"""Tests for model profiling, size computation, latency measurement, and benchmark_model."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from paradigms.benchmark.profiling import (
    benchmark_model,
    measure_latency,
    model_size_bytes,
)
from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP


def test_model_size_bytes() -> None:
    """Verify serialized model size is positive and larger for TeacherCNN than StudentMLP."""
    teacher = TeacherCNN()
    student = StudentMLP(hidden=32)

    t_size = model_size_bytes(teacher)
    s_size = model_size_bytes(student)

    assert t_size > 0
    assert s_size > 0
    assert t_size > s_size


def test_measure_latency_values_positive() -> None:
    """Verify latency metrics are strictly positive and formatted properly."""
    student = StudentMLP(hidden=32)
    metrics = measure_latency(student, batch_size=1, warmup=5, runs=10)

    assert metrics["median_ms"] > 0.0
    assert metrics["p95_ms"] > 0.0
    assert metrics["mean_ms"] > 0.0
    assert metrics["std_ms"] >= 0.0
    assert metrics["throughput_samples_per_s"] > 0.0


def test_thread_count_restoration() -> None:
    """Verify torch thread count is restored both on normal return and upon exception."""
    initial_threads = torch.get_num_threads()
    student = StudentMLP(hidden=32)

    # 1. Normal execution
    measure_latency(student, batch_size=1, threads=2, warmup=2, runs=5)
    assert torch.get_num_threads() == initial_threads

    # 2. Exceptional execution
    class ErrorModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("Forced inference failure")

    with pytest.raises(RuntimeError, match="Forced inference failure"):
        measure_latency(ErrorModel(), batch_size=1, threads=2, warmup=2, runs=5)

    assert torch.get_num_threads() == initial_threads


def test_teacher_b256_median_greater_than_student() -> None:
    """Verify TeacherCNN batch-256 latency exceeds StudentMLP batch-256 latency."""
    teacher = TeacherCNN()
    student = StudentMLP(hidden=32)

    t_lat = measure_latency(teacher, batch_size=256, warmup=2, runs=5)
    s_lat = measure_latency(student, batch_size=256, warmup=2, runs=5)

    assert t_lat["median_ms"] > s_lat["median_ms"]


def test_benchmark_model_keys() -> None:
    """Verify benchmark_model produces all expected keys with populated metrics."""
    student = StudentMLP(hidden=32)
    res = benchmark_model("StudentMLP", student, runs_b1=10, runs_b256=5)

    expected_keys = {"name", "params", "size_bytes", "size_kb", "latency_b1", "latency_b256"}
    assert expected_keys.issubset(res.keys())
    assert res["params"] == 25450
    assert res["size_kb"] > 0
    assert "median_ms" in res["latency_b1"]
    assert "median_ms" in res["latency_b256"]
