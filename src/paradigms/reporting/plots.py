"""Matplotlib plotting utilities for Phase 1 distillation experiments.

Adheres strictly to headless Agg backend, dpi 150, explicit axis labels,
dev set titles, error bars representing standard deviation, and no seaborn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Headless rendering backend
import matplotlib.pyplot as plt
import numpy as np


def plot_main_bars(main_data: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Plot bar chart comparing Dev Set accuracy across Teacher, Hard, and Distilled models.

    Args:
        main_data: List of model metric dicts containing 'name', 'dev_mean', 'dev_std'.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    names = [m["name"] for m in main_data]
    means = [m["dev_mean"] * 100 for m in main_data]
    stds = [m["dev_std"] * 100 for m in main_data]

    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[: len(names)], alpha=0.88)

    ax.set_title("Dev Set Accuracy: Teacher vs Hard vs Distilled Student", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
    ax.set_ylim(bottom=max(0.0, min(means) - 5), top=min(100.0, max(means) + 4))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + std + 0.3,
            f"{mean:.2f}%\n±{std:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="normal",
        )

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_temperature(
    plot_data: list[dict[str, Any]],
    hard_baseline: float,
    out_path: str | Path,
) -> Path:
    """Plot Distilled Student dev accuracy vs Temperature with hard baseline horizontal rule.

    Args:
        plot_data: List of dicts with 'setting', 'dist_mean', 'dist_std'.
        hard_baseline: Fixed hard baseline dev accuracy.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x_vals = [d["setting"] for d in plot_data]
    y_means = [d["dist_mean"] * 100 for d in plot_data]
    y_stds = [d["dist_std"] * 100 for d in plot_data]

    if x_vals:
        ax.errorbar(
            [str(x) for x in x_vals],
            y_means,
            yerr=y_stds,
            marker="o",
            color="#1b9e77",
            linewidth=2,
            capsize=4,
            label="Distilled Student (MLP)",
        )

    hard_pct = hard_baseline * 100
    ax.axhline(
        hard_pct,
        color="#d95f02",
        linestyle="--",
        linewidth=1.8,
        label=f"Hard Student Baseline ({hard_pct:.2f}%)",
    )

    ax.set_title("Distillation Temperature Sweep (Dev Set Accuracy)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Temperature (T)", fontsize=10)
    ax.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    if plot_data:
        ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_alpha(
    plot_data: list[dict[str, Any]],
    hard_baseline: float,
    out_path: str | Path,
) -> Path:
    """Plot Distilled Student dev accuracy vs Alpha with hard baseline horizontal rule.

    Args:
        plot_data: List of dicts with 'setting', 'dist_mean', 'dist_std'.
        hard_baseline: Fixed hard baseline dev accuracy.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x_vals = [d["setting"] for d in plot_data]
    y_means = [d["dist_mean"] * 100 for d in plot_data]
    y_stds = [d["dist_std"] * 100 for d in plot_data]

    if x_vals:
        ax.errorbar(
            [str(x) for x in x_vals],
            y_means,
            yerr=y_stds,
            marker="s",
            color="#1b9e77",
            linewidth=2,
            capsize=4,
            label="Distilled Student (MLP)",
        )

    hard_pct = hard_baseline * 100
    ax.axhline(
        hard_pct,
        color="#d95f02",
        linestyle="--",
        linewidth=1.8,
        label=f"Hard Student Baseline ({hard_pct:.2f}%)",
    )

    ax.set_title("Distillation Loss Alpha Sweep (Dev Set Accuracy)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Loss Weight Alpha", fontsize=10)
    ax.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    if plot_data:
        ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fraction(plot_data: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Plot Hard vs Distilled accuracy vs training-data fraction on log x-axis.

    Args:
        plot_data: List of dicts with 'setting', 'hard_mean', 'hard_std', 'dist_mean', 'dist_std'.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    if plot_data:
        x_vals = [float(d["setting"]) for d in plot_data]
        h_means = [d["hard_mean"] * 100 for d in plot_data]
        h_stds = [d["hard_std"] * 100 for d in plot_data]
        d_means = [d["dist_mean"] * 100 for d in plot_data]
        d_stds = [d["dist_std"] * 100 for d in plot_data]

        ax.errorbar(
            x_vals,
            h_means,
            yerr=h_stds,
            marker="o",
            color="#d95f02",
            linewidth=2,
            capsize=4,
            label="Hard Student (MLP)",
        )
        ax.errorbar(
            x_vals,
            d_means,
            yerr=d_stds,
            marker="s",
            color="#1b9e77",
            linewidth=2,
            capsize=4,
            label="Distilled Student (MLP)",
        )
        ax.set_xscale("log")

    ax.set_title("Training Data Fraction Scaling (Dev Set Accuracy)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Training Data Fraction (log scale)", fontsize=10)
    ax.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    if plot_data:
        ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_size(plot_data: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Plot Hard vs Distilled accuracy vs student hidden width with parameter count on secondary axis.

    Args:
        plot_data: List of dicts with 'setting', 'hard_mean', 'hard_std', 'dist_mean', 'dist_std', 'params'.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)

    if plot_data:
        x_vals = [int(d["setting"]) for d in plot_data]
        h_means = [d["hard_mean"] * 100 for d in plot_data]
        h_stds = [d["hard_std"] * 100 for d in plot_data]
        d_means = [d["dist_mean"] * 100 for d in plot_data]
        d_stds = [d["dist_std"] * 100 for d in plot_data]
        params = [d.get("params") for d in plot_data]

        l1 = ax1.errorbar(
            x_vals,
            h_means,
            yerr=h_stds,
            marker="o",
            color="#d95f02",
            linewidth=2,
            capsize=4,
            label="Hard Student Acc (%)",
        )
        l2 = ax1.errorbar(
            x_vals,
            d_means,
            yerr=d_stds,
            marker="s",
            color="#1b9e77",
            linewidth=2,
            capsize=4,
            label="Distilled Student Acc (%)",
        )

        ax1.set_xlabel("Student Hidden Width (Units)", fontsize=10)
        ax1.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Secondary y-axis for parameter count
        valid_params = [p for p in params if p is not None]
        if len(valid_params) == len(params):
            ax2 = ax1.twinx()
            l3 = ax2.plot(
                x_vals,
                params,
                color="slategray",
                linestyle=":",
                marker="x",
                label="Parameters",
            )
            ax2.set_ylabel("Parameters", fontsize=10, color="slategray")
            ax2.tick_params(axis="y", labelcolor="slategray")

            lines = [l1, l2, l3[0]]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="best", frameon=True)
        else:
            ax1.legend(loc="best", frameon=True)
    else:
        ax1.set_xlabel("Student Hidden Width (Units)", fontsize=10)
        ax1.set_ylabel("Dev Set Accuracy (%)", fontsize=10)

    ax1.set_title("Student Model Capacity Sweep (Dev Set Accuracy)", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_perclass(perclass_data: dict[str, Any], out_path: str | Path) -> Path:
    """Plot grouped bar chart showing per-class dev accuracy for Hard vs Distilled students.

    Args:
        perclass_data: Dict containing 'classes', 'hard_means', 'dist_means'.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    classes = perclass_data["classes"]
    hard_means = [m * 100 for m in perclass_data["hard_means"]]
    dist_means = [m * 100 for m in perclass_data["dist_means"]]

    x = np.arange(len(classes))
    width = 0.38

    ax.bar(x - width / 2, hard_means, width, label="Hard Student", color="#d95f02", alpha=0.85)
    ax.bar(x + width / 2, dist_means, width, label="Distilled Student", color="#1b9e77", alpha=0.85)

    ax.set_title("Per-Class Accuracy: Hard vs Distilled Student (Dev Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(bottom=max(0, min(hard_means + dist_means) - 10), top=105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_pareto(models_data: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Plot Batch-1 CPU latency vs Dev Set accuracy Pareto frontier.

    Args:
        models_data: List of model dicts containing 'name', 'b1_ms', 'dev_mean'.
        out_path: Output image destination path.

    Returns:
        Path to saved figure.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)

    colors = {
        "Teacher (CNN)": "#2b5c8f",
        "Hard Student (MLP)": "#d95f02",
        "Distilled Student (MLP)": "#1b9e77",
    }

    for m in models_data:
        name = m["name"]
        b1 = m.get("b1_ms")
        acc = m.get("dev_mean")
        if b1 is None or acc is None:
            continue

        acc_pct = acc * 100
        color = colors.get(name, "black")
        ax.scatter(b1, acc_pct, color=color, s=90, zorder=5, label=name)

        offset_x = 1.08
        offset_y = 0.15
        if "Teacher" in name:
            offset_x = 0.55
            offset_y = -0.6
        elif "Hard" in name:
            offset_x = 1.15
            offset_y = 0.1

        ax.annotate(
            f"{name}\n({b1:.3f} ms, {acc_pct:.2f}%)",
            xy=(b1, acc_pct),
            xytext=(b1 * offset_x, acc_pct + offset_y),
            fontsize=8.5,
            fontweight="normal",
        )

    ax.set_xscale("log")
    ax.set_title("Inference Latency vs Dev Set Accuracy Pareto Frontier", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Batch-1 CPU Latency (ms) [log scale]", fontsize=10)
    ax.set_ylabel("Dev Set Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p
