"""Compile Phase 1 tables and figures from experiment JSON artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from paradigms.reporting.phase1 import (
    aggregate_main_table,
    aggregate_paired_table,
    aggregate_perclass_table,
    aggregate_sweep_stage,
)
from paradigms.reporting.plots import (
    plot_alpha,
    plot_fraction,
    plot_main_bars,
    plot_pareto,
    plot_perclass,
    plot_size,
    plot_temperature,
)
from paradigms.reporting.tables import write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 summary tables and figures from experimental artifacts."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments",
        help="Root directory containing experimental results (default: experiments).",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="reports/tables",
        help="Directory to save output tables (.md and .csv) (default: reports/tables).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="reports/figures",
        help="Directory to save output plots (.png) (default: reports/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated_tables: list[Path] = []
    generated_figures: list[Path] = []

    print("Generating Phase 1 Tables and Figures...")

    # 1. Main comparison table + Main bars + Pareto frontier
    main_res = aggregate_main_table(exp_dir)
    t_md, t_csv = write_table("phase1_main", main_res["headers"], main_res["rows"], tables_dir)
    generated_tables.extend([t_md, t_csv])
    print(f"  [Table]  {t_md} and {t_csv}")

    f_main = plot_main_bars(main_res["models_data"], figures_dir / "fig_main_bars.png")
    generated_figures.append(f_main)
    print(f"  [Figure] {f_main}")

    f_pareto = plot_pareto(main_res["models_data"], figures_dir / "fig_pareto.png")
    generated_figures.append(f_pareto)
    print(f"  [Figure] {f_pareto}")

    # 2. Paired difference table
    paired_res = aggregate_paired_table(exp_dir)
    p_md, p_csv = write_table("phase1_paired", paired_res["headers"], paired_res["rows"], tables_dir)
    generated_tables.extend([p_md, p_csv])
    print(f"  [Table]  {p_md} and {p_csv}")

    # 3. Sweep stages: temperature, alpha, fraction, size
    # 3a. Temperature
    temp_res = aggregate_sweep_stage("temperature", exp_dir)
    t_md, t_csv = write_table("phase1_temperature", temp_res["headers"], temp_res["rows"], tables_dir)
    generated_tables.extend([t_md, t_csv])
    print(f"  [Table]  {t_md} and {t_csv}")
    f_temp = plot_temperature(temp_res["plot_data"], temp_res["hard_fixed_mean"], figures_dir / "fig_temperature.png")
    generated_figures.append(f_temp)
    print(f"  [Figure] {f_temp}")

    # 3b. Alpha
    alpha_res = aggregate_sweep_stage("alpha", exp_dir)
    a_md, a_csv = write_table("phase1_alpha", alpha_res["headers"], alpha_res["rows"], tables_dir)
    generated_tables.extend([a_md, a_csv])
    print(f"  [Table]  {a_md} and {a_csv}")
    f_alpha = plot_alpha(alpha_res["plot_data"], alpha_res["hard_fixed_mean"], figures_dir / "fig_alpha.png")
    generated_figures.append(f_alpha)
    print(f"  [Figure] {f_alpha}")

    # 3c. Fraction
    frac_res = aggregate_sweep_stage("fraction", exp_dir)
    f_md, f_csv = write_table("phase1_fraction", frac_res["headers"], frac_res["rows"], tables_dir)
    generated_tables.extend([f_md, f_csv])
    print(f"  [Table]  {f_md} and {f_csv}")
    f_frac = plot_fraction(frac_res["plot_data"], figures_dir / "fig_fraction.png")
    generated_figures.append(f_frac)
    print(f"  [Figure] {f_frac}")

    # 3d. Size
    size_res = aggregate_sweep_stage("size", exp_dir)
    s_md, s_csv = write_table("phase1_size", size_res["headers"], size_res["rows"], tables_dir)
    generated_tables.extend([s_md, s_csv])
    print(f"  [Table]  {s_md} and {s_csv}")
    f_size = plot_size(size_res["plot_data"], figures_dir / "fig_size.png")
    generated_figures.append(f_size)
    print(f"  [Figure] {f_size}")

    # 4. Per-class table and figure
    perclass_res = aggregate_perclass_table(exp_dir)
    pc_md, pc_csv = write_table("phase1_perclass", perclass_res["headers"], perclass_res["rows"], tables_dir)
    generated_tables.extend([pc_md, pc_csv])
    print(f"  [Table]  {pc_md} and {pc_csv}")
    f_perclass = plot_perclass(perclass_res, figures_dir / "fig_perclass.png")
    generated_figures.append(f_perclass)
    print(f"  [Figure] {f_perclass}")

    print("\nPhase 1 Report Generation Complete!")
    print("\nGenerated Tables:")
    for t in generated_tables:
        print(f"  - {t}")
    print("\nGenerated Figures:")
    for f in generated_figures:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
