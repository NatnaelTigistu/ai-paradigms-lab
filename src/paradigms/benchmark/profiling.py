"""Benchmarking and profiling utilities: model size on disk, parameters, and CPU latency."""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from paradigms.models.registry import count_parameters


def model_size_bytes(model: nn.Module) -> int:
    """Measure serialized model checkpoint size on disk in bytes.

    Serializes the state dictionary into an in-memory BytesIO buffer, accurately
    reflecting the on-disk storage requirement without writing temporary files.

    Args:
        model: PyTorch module whose parameter state is to be measured.

    Returns:
        Integer size in bytes.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def measure_latency(
    model: nn.Module,
    batch_size: int,
    threads: int = 1,
    warmup: int = 50,
    runs: int = 1000,
) -> dict[str, float]:
    """Measure inference latency and throughput for a model on CPU.

    Measurement Rationale:
        1. Warm-up iterations: Running preliminary passes ensures JIT optimizations,
           memory allocations, and CPU frequency scaling settle before timing begins,
           eliminating one-time initialization overheads.
        2. Median statistic: Using the median rather than mean provides robustness
           against OS scheduling interruptions, interrupts, and thread preemption outliers.
        3. Single-threaded evaluation: Pinning to threads=1 removes multi-threaded
           OpenMP synchronization jitter and allows fair cross-machine comparisons.
        4. Guaranteed restoration: The previous PyTorch thread setting is restored in a
           finally block to prevent side-effects on subsequent training/evaluation loops.

    Args:
        model: PyTorch model to evaluate.
        batch_size: Number of input samples per forward pass.
        threads: Number of CPU threads dedicated to PyTorch inference (default: 1).
        warmup: Number of unmeasured initial warmup forward passes.
        runs: Number of timed forward passes to execute.

    Returns:
        Dictionary containing:
            - "median_ms": Median forward pass latency in milliseconds.
            - "p95_ms": 95th percentile latency in milliseconds.
            - "mean_ms": Mean latency in milliseconds.
            - "std_ms": Standard deviation of latency in milliseconds (ddof=0).
            - "throughput_samples_per_s": Inferred samples per second based on median.
    """
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        model.eval()
        x = torch.randn(batch_size, 1, 28, 28)

        with torch.inference_mode():
            for _ in range(warmup):
                model(x)

            times: list[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(old_threads)

    times_ms = np.asarray(times, dtype=np.float64) * 1000.0
    median_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms, ddof=0))
    median_sec = median_ms / 1000.0
    throughput = float(batch_size / median_sec) if median_sec > 0 else 0.0

    return {
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "throughput_samples_per_s": throughput,
    }


def benchmark_model(
    name: str,
    model: nn.Module,
    runs_b1: int = 1000,
    runs_b256: int = 100,
) -> dict[str, Any]:
    """Profile comprehensive model footprint including parameters, size, and latencies.

    Args:
        name: Identifying string for the model (e.g. "TeacherCNN").
        model: Model instance to evaluate.
        runs_b1: Repetitions for batch_size=1 latency measurement.
        runs_b256: Repetitions for batch_size=256 latency measurement.

    Returns:
        Dictionary with keys:
            - "name": Model identifier.
            - "params": Total trainable parameters.
            - "size_bytes": State dict size in bytes.
            - "size_kb": State dict size in kilobytes.
            - "latency_b1": Latency profiling dict for batch_size=1.
            - "latency_b256": Latency profiling dict for batch_size=256.
    """
    size_b = model_size_bytes(model)
    return {
        "name": name,
        "params": count_parameters(model),
        "size_bytes": size_b,
        "size_kb": round(size_b / 1024.0, 2),
        "latency_b1": measure_latency(model, batch_size=1, runs=runs_b1),
        "latency_b256": measure_latency(model, batch_size=256, runs=runs_b256),
    }
