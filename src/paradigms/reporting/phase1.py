"""Phase 1 data aggregation and reporting utilities.

Reads experimental results exclusively from stored JSON outputs to produce
standardized summary tables and plot-ready structured representations.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from paradigms.utils.io import load_json
from paradigms.utils.stats import mean_std, paired_comparison

FASHION_CLASSES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_seed_records(group_dir: Path) -> list[dict[str, Any]]:
    """Load and sort all per-seed JSON records within a specific group directory.

    Args:
        group_dir: Path containing `seed*.json` files.

    Returns:
        List of loaded dictionaries sorted by random seed identifier.
    """
    if not group_dir.exists():
        return []
    seed_files = sorted(
        group_dir.glob("seed*.json"),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
    )
    return [load_json(f) for f in seed_files]


def aggregate_main_table(exp_dir: str | Path = "experiments") -> dict[str, Any]:
    """Compile Phase 1 main comparison table across Teacher, Hard, and Distilled models.

    Reads parameter count, checkpoint size, and CPU inference latencies from
    `benchmark.json`, and development set accuracies from model summaries.

    Args:
        exp_dir: Base directory housing experiment artifacts.

    Returns:
        Dictionary containing headers, formatted table rows, and raw model metrics.
    """
    base = Path(exp_dir)
    bench_file = base / "benchmark" / "benchmark.json"
    bench_data = load_json(bench_file) if bench_file.exists() else {"models": {}}

    models_info = [
        {
            "key": "Teacher (CNN)",
            "label": "Teacher (CNN)",
            "summary_path": base / "teacher" / "summary.json",
        },
        {
            "key": "Hard Student (MLP)",
            "label": "Hard Student (MLP)",
            "summary_path": base / "student_hard" / "h32_f1.0" / "summary.json",
        },
        {
            "key": "Distilled Student (MLP)",
            "label": "Distilled Student (MLP)",
            "summary_path": base / "student_distilled" / "h32_f1.0_T4.0_a0.5" / "summary.json",
        },
    ]

    headers = [
        "Model",
        "Parameters",
        "Size (KB)",
        "Latency b=1 (ms)",
        "Latency b=256 (ms)",
        "Dev Accuracy (%)",
    ]
    rows: list[list[Any]] = []
    models_data: list[dict[str, Any]] = []

    for m in models_info:
        model_name = m["label"]
        b_entry = bench_data.get("models", {}).get(m["key"], {})

        params = b_entry.get("params")
        size_kb = b_entry.get("size_kb")
        b1_ms = b_entry.get("b1_median_ms")
        b256_ms = b_entry.get("b256_median_ms")

        # Load dev accuracy
        summary_file = m["summary_path"]
        dev_mean, dev_std = 0.0, 0.0
        if summary_file.exists():
            s_data = load_json(summary_file)
            dev_mean = float(s_data.get("dev_accuracy_mean", 0.0))
            dev_std = float(s_data.get("dev_accuracy_std", 0.0))
            if params is None:
                params = s_data.get("params")
        else:
            warnings.warn(f"Summary file not found for {model_name} at {summary_file}")

        params_str = f"{params:,}" if params is not None else "-"
        size_str = f"{size_kb:.2f}" if size_kb is not None else "-"
        b1_str = f"{b1_ms:.4f}" if b1_ms is not None else "-"
        b256_str = f"{b256_ms:.2f}" if b256_ms is not None else "-"
        acc_str = f"{dev_mean * 100:.2f}% ± {dev_std * 100:.2f}%"

        rows.append([model_name, params_str, size_str, b1_str, b256_str, acc_str])
        models_data.append(
            {
                "name": model_name,
                "params": params,
                "size_kb": size_kb,
                "b1_ms": b1_ms,
                "b256_ms": b256_ms,
                "dev_mean": dev_mean,
                "dev_std": dev_std,
            }
        )

    return {"headers": headers, "rows": rows, "models_data": models_data}


def aggregate_paired_table(exp_dir: str | Path = "experiments") -> dict[str, Any]:
    """Compile paired per-seed dev accuracy differences (Distilled - Hard).

    Args:
        exp_dir: Base directory housing experiment artifacts.

    Returns:
        Dictionary containing headers, per-seed comparison rows, and overall stats.
    """
    base = Path(exp_dir)
    hard_dir = base / "student_hard" / "h32_f1.0"
    dist_dir = base / "student_distilled" / "h32_f1.0_T4.0_a0.5"

    hard_records = {r["seed"]: r["dev_accuracy"] for r in load_seed_records(hard_dir)}
    dist_records = {r["seed"]: r["dev_accuracy"] for r in load_seed_records(dist_dir)}

    shared_seeds = sorted(set(hard_records.keys()).intersection(dist_records.keys()))
    if not shared_seeds:
        warnings.warn(f"No matching seeds between {hard_dir} and {dist_dir}")

    headers = ["Seed", "Hard Dev Acc (%)", "Distilled Dev Acc (%)", "Diff (%)"]
    rows: list[list[Any]] = []
    diffs: list[float] = []

    for s in shared_seeds:
        h_acc = float(hard_records[s])
        d_acc = float(dist_records[s])
        diff = d_acc - h_acc
        diffs.append(diff)
        rows.append(
            [
                f"Seed {s}",
                f"{h_acc * 100:.2f}%",
                f"{d_acc * 100:.2f}%",
                f"{diff * 100:+.2f}%",
            ]
        )

    if diffs:
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=0))
        wins = sum(1 for d in diffs if d > 0)
        n = len(diffs)
    else:
        mean_diff, std_diff, wins, n = 0.0, 0.0, 0, 0

    # Summary rows
    rows.append(
        [
            "Mean ± Std",
            "-",
            "-",
            f"{mean_diff * 100:+.2f}% ± {std_diff * 100:.2f}%",
        ]
    )
    rows.append(["Wins / Total", "-", "-", f"{wins} / {n}"])

    return {
        "headers": headers,
        "rows": rows,
        "shared_seeds": shared_seeds,
        "diffs": diffs,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "wins": wins,
        "n": n,
    }


def aggregate_perclass_table(exp_dir: str | Path = "experiments") -> dict[str, Any]:
    """Compile class-wise dev accuracies averaged across seeds for Hard vs Distilled.

    Args:
        exp_dir: Base directory housing experiment artifacts.

    Returns:
        Dictionary containing headers, class-by-class rows, and per-class vectors.
    """
    base = Path(exp_dir)
    hard_dir = base / "student_hard" / "h32_f1.0"
    dist_dir = base / "student_distilled" / "h32_f1.0_T4.0_a0.5"

    hard_records = load_seed_records(hard_dir)
    dist_records = load_seed_records(dist_dir)

    num_classes = 10
    hard_per_class: list[list[float]] = [[] for _ in range(num_classes)]
    dist_per_class: list[list[float]] = [[] for _ in range(num_classes)]

    for r in hard_records:
        pca = r.get("dev_per_class_accuracy", [])
        for c, val in enumerate(pca):
            if c < num_classes and val is not None:
                hard_per_class[c].append(float(val))

    for r in dist_records:
        pca = r.get("dev_per_class_accuracy", [])
        for c, val in enumerate(pca):
            if c < num_classes and val is not None:
                dist_per_class[c].append(float(val))

    headers = ["Class ID", "Class Name", "Hard Dev Acc (%)", "Distilled Dev Acc (%)", "Diff (%)"]
    rows: list[list[Any]] = []
    hard_means: list[float] = []
    dist_means: list[float] = []

    for c in range(num_classes):
        c_name = FASHION_CLASSES[c] if c < len(FASHION_CLASSES) else f"Class {c}"
        h_mean = float(np.mean(hard_per_class[c])) if hard_per_class[c] else 0.0
        d_mean = float(np.mean(dist_per_class[c])) if dist_per_class[c] else 0.0
        diff = d_mean - h_mean

        hard_means.append(h_mean)
        dist_means.append(d_mean)

        rows.append(
            [
                str(c),
                c_name,
                f"{h_mean * 100:.2f}%",
                f"{d_mean * 100:.2f}%",
                f"{diff * 100:+.2f}%",
            ]
        )

    return {
        "headers": headers,
        "rows": rows,
        "classes": FASHION_CLASSES,
        "hard_means": hard_means,
        "dist_means": dist_means,
    }


def aggregate_sweep_stage(stage: str, exp_dir: str | Path = "experiments") -> dict[str, Any]:
    """Aggregate sweep stage results with graceful skipping of missing groups.

    Args:
        stage: One of 'temperature', 'alpha', 'fraction', 'size'.
        exp_dir: Base directory housing experiment artifacts.

    Returns:
        Dictionary containing headers, rows, and structured plot data.
    """
    base = Path(exp_dir)
    stage_dir = base / "distillation" / stage

    # Define standard swept settings
    stage_configs = {
        "temperature": {
            "name": "Temperature (T)",
            "settings": [1, 2, 4, 8, 16],
            "tag_fn": lambda val: f"T{val}",
            "hard_fixed": base / "student_hard" / "h32_f1.0",
        },
        "alpha": {
            "name": "Alpha",
            "settings": [0.0, 0.3, 0.5, 0.7, 1.0],
            "tag_fn": lambda val: f"a{val}",
            "hard_fixed": base / "student_hard" / "h32_f1.0",
        },
        "fraction": {
            "name": "Fraction",
            "settings": [0.01, 0.05, 0.1, 0.25, 1.0],
            "tag_fn": lambda val: f"f{val}",
            "hard_fixed": None,
        },
        "size": {
            "name": "Hidden Size",
            "settings": [8, 16, 32, 64, 128],
            "tag_fn": lambda val: f"h{val}",
            "hard_fixed": None,
        },
    }

    if stage not in stage_configs:
        raise ValueError(f"Unknown sweep stage: '{stage}'.")

    sc = stage_configs[stage]
    setting_col = sc["name"]
    headers = [setting_col, "Hard Dev Acc (%)", "Distilled Dev Acc (%)", "Paired Mean Diff (%)"]
    rows: list[list[Any]] = []
    plot_data: list[dict[str, Any]] = []

    # Pre-load fixed hard baseline if applicable (for temperature and alpha)
    fixed_hard_records: dict[int, float] = {}
    fixed_hard_mean, fixed_hard_std = 0.0, 0.0
    if sc["hard_fixed"] is not None:
        recs = load_seed_records(sc["hard_fixed"])
        if recs:
            fixed_hard_records = {r["seed"]: float(r["dev_accuracy"]) for r in recs}
            h_accs = list(fixed_hard_records.values())
            fixed_hard_mean = float(np.mean(h_accs))
            fixed_hard_std = float(np.std(h_accs, ddof=0))

    for val in sc["settings"]:
        tag = sc["tag_fn"](val)
        tag_dir = stage_dir / tag

        if sc["hard_fixed"] is not None:
            # Distilled group is under tag/distilled
            dist_group = tag_dir / "distilled"
            if not dist_group.exists():
                warnings.warn(f"Missing sweep group: {dist_group}, skipping with warning.")
                continue

            dist_recs = load_seed_records(dist_group)
            if not dist_recs:
                warnings.warn(f"No seed JSONs in {dist_group}, skipping with warning.")
                continue

            d_accs = [float(r["dev_accuracy"]) for r in dist_recs]
            d_mean = float(np.mean(d_accs))
            d_std = float(np.std(d_accs, ddof=0))

            # Compute paired differences
            paired_diffs = []
            for r in dist_recs:
                s = r["seed"]
                if s in fixed_hard_records:
                    paired_diffs.append(float(r["dev_accuracy"]) - fixed_hard_records[s])
            p_mean = float(np.mean(paired_diffs)) if paired_diffs else (d_mean - fixed_hard_mean)

            h_str = f"{fixed_hard_mean * 100:.2f}% ± {fixed_hard_std * 100:.2f}%"
            d_str = f"{d_mean * 100:.2f}% ± {d_std * 100:.2f}%"
            p_str = f"{p_mean * 100:+.2f}%"

            rows.append([str(val), h_str, d_str, p_str])
            plot_data.append(
                {
                    "setting": val,
                    "hard_mean": fixed_hard_mean,
                    "hard_std": fixed_hard_std,
                    "dist_mean": d_mean,
                    "dist_std": d_std,
                    "paired_diff": p_mean,
                }
            )
        else:
            # Both hard and distilled groups exist under tag_dir
            hard_group = tag_dir / "hard"
            dist_group = tag_dir / "distilled"

            if not hard_group.exists() or not dist_group.exists():
                warnings.warn(
                    f"Missing group pair for {stage} {tag} ({hard_group} or {dist_group}), skipping."
                )
                continue

            hard_recs = {r["seed"]: float(r["dev_accuracy"]) for r in load_seed_records(hard_group)}
            dist_recs = {r["seed"]: float(r["dev_accuracy"]) for r in load_seed_records(dist_group)}

            if not hard_recs or not dist_recs:
                warnings.warn(f"Empty seed records for {stage} {tag}, skipping with warning.")
                continue

            h_accs = list(hard_recs.values())
            d_accs = list(dist_recs.values())
            h_mean, h_std = float(np.mean(h_accs)), float(np.std(h_accs, ddof=0))
            d_mean, d_std = float(np.mean(d_accs)), float(np.std(d_accs, ddof=0))

            shared_seeds = set(hard_recs.keys()).intersection(dist_recs.keys())
            paired_diffs = [dist_recs[s] - hard_recs[s] for s in shared_seeds]
            p_mean = float(np.mean(paired_diffs)) if paired_diffs else (d_mean - h_mean)

            # Get parameter count if available
            first_dist = load_seed_records(dist_group)[0]
            params = first_dist.get("params")

            h_str = f"{h_mean * 100:.2f}% ± {h_std * 100:.2f}%"
            d_str = f"{d_mean * 100:.2f}% ± {d_std * 100:.2f}%"
            p_str = f"{p_mean * 100:+.2f}%"

            rows.append([str(val), h_str, d_str, p_str])
            plot_data.append(
                {
                    "setting": val,
                    "hard_mean": h_mean,
                    "hard_std": h_std,
                    "dist_mean": d_mean,
                    "dist_std": d_std,
                    "paired_diff": p_mean,
                    "params": params,
                }
            )

    return {
        "headers": headers,
        "rows": rows,
        "plot_data": plot_data,
        "hard_fixed_mean": fixed_hard_mean,
        "hard_fixed_std": fixed_hard_std,
    }
