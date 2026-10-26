"""Script to profile and compare Teacher CNN and Student MLP parameters, disk footprint, and CPU latency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from paradigms.benchmark.profiling import benchmark_model
from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP
from paradigms.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile model parameters, serialized disk size, and CPU latency."
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="checkpoints/teacher/seed0.pt",
        help="Path to Teacher CNN checkpoint (default: checkpoints/teacher/seed0.pt).",
    )
    parser.add_argument(
        "--hard-ckpt",
        type=str,
        default="checkpoints/student_hard/h32_f1.0/seed0.pt",
        help="Path to hard-label Student MLP checkpoint (default: checkpoints/student_hard/h32_f1.0/seed0.pt).",
    )
    parser.add_argument(
        "--distilled-ckpt",
        type=str,
        default="checkpoints/student_distilled/h32_f1.0_T4.0_a0.5/seed0.pt",
        help="Path to distilled Student MLP checkpoint (default: checkpoints/student_distilled/h32_f1.0_T4.0_a0.5/seed0.pt).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="experiments/benchmark/benchmark.json",
        help="Output path for benchmark JSON file (default: experiments/benchmark/benchmark.json).",
    )
    parser.add_argument(
        "--runs-b1",
        type=int,
        default=1000,
        help="Number of evaluation runs for batch_size=1 (default: 1000).",
    )
    parser.add_argument(
        "--runs-b256",
        type=int,
        default=100,
        help="Number of evaluation runs for batch_size=256 (default: 100).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of outer benchmark repetitions to compute median of medians (default: 3).",
    )
    return parser.parse_args()


def load_model_from_ckpt(model_cls: type[torch.nn.Module], ckpt_path: str, **kwargs: Any) -> torch.nn.Module:
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    model = model_cls(**kwargs)
    state = torch.load(p, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    print("\n" + "*" * 68)
    print("NOTE: For accurate and reproducible CPU latency profiling, ensure")
    print("the machine is otherwise idle and running on stable AC power.")
    print("*" * 68 + "\n")

    # Load models
    models: dict[str, torch.nn.Module] = {
        "Teacher (CNN)": load_model_from_ckpt(TeacherCNN, args.teacher_ckpt),
        "Hard Student (MLP)": load_model_from_ckpt(StudentMLP, args.hard_ckpt, hidden=32),
        "Distilled Student (MLP)": load_model_from_ckpt(StudentMLP, args.distilled_ckpt, hidden=32),
    }

    # Run repeated benchmark passes
    print(f"Running {args.repeats} complete benchmark passes (runs_b1={args.runs_b1}, runs_b256={args.runs_b256})...")
    all_runs: dict[str, list[dict[str, Any]]] = {name: [] for name in models}

    for repeat_idx in range(1, args.repeats + 1):
        print(f"  Pass {repeat_idx}/{args.repeats}...")
        for name, model in models.items():
            run_result = benchmark_model(
                name=name,
                model=model,
                runs_b1=args.runs_b1,
                runs_b256=args.runs_b256,
            )
            all_runs[name].append(run_result)

    # Compute final median of medians for each model
    results: dict[str, dict[str, Any]] = {}
    for name, runs in all_runs.items():
        b1_medians = [r["latency_b1"]["median_ms"] for r in runs]
        b256_medians = [r["latency_b256"]["median_ms"] for r in runs]

        b1_med = float(np.median(b1_medians))
        b256_med = float(np.median(b256_medians))

        first_run = runs[0]
        results[name] = {
            "name": name,
            "params": first_run["params"],
            "size_bytes": first_run["size_bytes"],
            "size_kb": first_run["size_kb"],
            "b1_median_ms": b1_med,
            "b256_median_ms": b256_med,
            "b1_throughput_samples_per_s": float(1.0 / (b1_med / 1000.0)),
            "b256_throughput_samples_per_s": float(256.0 / (b256_med / 1000.0)),
            "all_runs": runs,
        }

    # Calculate compression and speedup ratios against teacher
    teacher_res = results["Teacher (CNN)"]
    hard_res = results["Hard Student (MLP)"]
    dist_res = results["Distilled Student (MLP)"]

    ratios = {
        "teacher_vs_hard": {
            "param_ratio": teacher_res["params"] / hard_res["params"],
            "size_ratio": teacher_res["size_bytes"] / hard_res["size_bytes"],
            "b1_latency_ratio": teacher_res["b1_median_ms"] / hard_res["b1_median_ms"],
            "b256_latency_ratio": teacher_res["b256_median_ms"] / hard_res["b256_median_ms"],
        },
        "teacher_vs_distilled": {
            "param_ratio": teacher_res["params"] / dist_res["params"],
            "size_ratio": teacher_res["size_bytes"] / dist_res["size_bytes"],
            "b1_latency_ratio": teacher_res["b1_median_ms"] / dist_res["b1_median_ms"],
            "b256_latency_ratio": teacher_res["b256_median_ms"] / dist_res["b256_median_ms"],
        },
    }

    # Save benchmark artifact
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_payload = {
        "models": results,
        "ratios": ratios,
        "repeats": args.repeats,
        "runs_b1": args.runs_b1,
        "runs_b256": args.runs_b256,
    }
    save_json(benchmark_payload, out_path)

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Model':<25} | {'Params':<10} | {'Size KB':<10} | {'b1 Median ms':<13} | {'b256 Median ms':<15}")
    print("-" * 78)
    for name, r in results.items():
        print(
            f"{name:<25} | {r['params']:>10,d} | {r['size_kb']:>10.2f} | "
            f"{r['b1_median_ms']:>13.4f} | {r['b256_median_ms']:>15.4f}"
        )
    print("=" * 78)

    print("\n" + "-" * 78)
    print("COMPRESSION & SPEEDUP RATIOS (Teacher / Student)")
    print("-" * 78)
    th_r = ratios["teacher_vs_hard"]
    print(f"Parameters Compression Ratio : {th_r['param_ratio']:.2f}x smaller")
    print(f"Checkpoint Size Ratio        : {th_r['size_ratio']:.2f}x smaller")
    print(f"Batch-1 Latency Speedup      : {th_r['b1_latency_ratio']:.2f}x faster")
    print(f"Batch-256 Latency Speedup    : {th_r['b256_latency_ratio']:.2f}x faster")
    print("-" * 78 + "\n")


if __name__ == "__main__":
    main()
