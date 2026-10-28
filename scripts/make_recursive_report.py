#!/usr/bin/env python3
"""Generate summary tables, progression figures, and findings report for the recursive loop.

Reads results strictly from serialized JSON files in experiments/recursive/ and writes outputs to
reports/recursive/tables/ and reports/recursive/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and scripts to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradigms.reporting.recursive import (
    compute_recursive_summary,
    compute_round_progression,
    generate_findings_markdown,
    generate_rounds_table,
    generate_summary_table,
    load_recursive_results,
    plot_acceptance_counts,
    plot_gate_gap,
    plot_perclass_deltas,
    plot_recursive_curves,
    plot_verifier_precision,
)
from paradigms.reporting.tables import write_table


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile recursive self-improvement loop tables, figures, and findings."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments/recursive",
        help="Path to recursive experiment results (default: experiments/recursive).",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="reports/recursive/tables",
        help="Output directory for markdown and CSV tables (default: reports/recursive/tables).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="reports/recursive/figures",
        help="Output directory for figure plots (default: reports/recursive/figures).",
    )
    return parser.parse_args(args)


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    exp_dir = Path(args.exp_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    print("\n" + "=" * 70)
    print("COMPILING RECURSIVE SELF-IMPROVEMENT REPORT")
    print(f"Reading results from: {exp_dir}")
    print(f"Tables output to:     {tables_dir}")
    print(f"Figures output to:    {figures_dir}")
    print("=" * 70 + "\n")

    # 1. Load results
    results = load_recursive_results(exp_dir)
    if not results:
        print(f"Error: No valid experiment results loaded from '{exp_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded results for {len(results)} configurations: {list(results.keys())}")

    # 2. Compute summaries
    summary = compute_recursive_summary(results, control_config="control_none")
    progression = compute_round_progression(results, max_rounds=4)

    # 3. Write tables
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_headers, summary_rows = generate_summary_table(summary)
    sum_md, sum_csv = write_table("rec_summary", summary_headers, summary_rows, tables_dir)
    print(f"Generated summary table: {sum_md} and {sum_csv}")

    rounds_headers, rounds_rows = generate_rounds_table(progression)
    rnd_md, rnd_csv = write_table("rec_rounds", rounds_headers, rounds_rows, tables_dir)
    print(f"Generated rounds table:  {rnd_md} and {rnd_csv}")

    # 4. Generate findings document
    findings_path = tables_dir / "rec_findings.md"
    generate_findings_markdown(summary, results, progression, findings_path)
    print(f"Generated findings doc:  {findings_path}")

    # 5. Generate figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    f1 = plot_recursive_curves(progression, figures_dir / "fig_rec_curves.png")
    f2 = plot_verifier_precision(results, figures_dir / "fig_rec_precision.png")
    f3 = plot_acceptance_counts(results, figures_dir / "fig_rec_accept.png")
    f4 = plot_gate_gap(progression, figures_dir / "fig_rec_gate_gap.png")
    f5 = plot_perclass_deltas(results, figures_dir / "fig_rec_perclass.png")

    print(f"Generated figures:")
    for fig_path in [f1, f2, f3, f4, f5]:
        print(f"  - {fig_path}")

    print("\nReport compilation complete.\n")


if __name__ == "__main__":
    main()
