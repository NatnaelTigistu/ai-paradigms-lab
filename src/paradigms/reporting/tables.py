"""Table serialization utilities supporting GitHub Flavored Markdown and CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Format tabular data into a GitHub Flavored Markdown table string.

    Args:
        headers: Column header strings.
        rows: 2D sequence of row cell contents.

    Returns:
        Formatted markdown table string.
    """
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]

    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
            else:
                col_widths.append(len(cell))

    # Pad headers
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(str_headers, col_widths)) + " |"
    sep_line = "| " + " | ".join("-" * max(w, 3) for w in col_widths) + " |"

    body_lines = []
    for row in str_rows:
        # Pad row to match number of columns if shorter
        padded_row = list(row) + [""] * (len(col_widths) - len(row))
        body_lines.append(
            "| " + " | ".join(cell.ljust(w) for cell, w in zip(padded_row, col_widths)) + " |"
        )

    return "\n".join([header_line, sep_line] + body_lines) + "\n"


def save_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], path: str | Path) -> None:
    """Write table data to a Markdown file.

    Args:
        headers: Column header strings.
        rows: 2D sequence of row cell contents.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = format_markdown_table(headers, rows)
    p.write_text(content, encoding="utf-8")


def save_csv_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], path: str | Path) -> None:
    """Write table data to a CSV file.

    Args:
        headers: Column header strings.
        rows: 2D sequence of row cell contents.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def write_table(
    name: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    tables_dir: str | Path,
) -> tuple[Path, Path]:
    """Write both Markdown (.md) and CSV (.csv) versions of a table.

    Args:
        name: Base file name without extension.
        headers: Column header strings.
        rows: 2D sequence of row cell contents.
        tables_dir: Directory where files will be created.

    Returns:
        Tuple of (md_path, csv_path).
    """
    td = Path(tables_dir)
    md_path = td / f"{name}.md"
    csv_path = td / f"{name}.csv"
    save_markdown_table(headers, rows, md_path)
    save_csv_table(headers, rows, csv_path)
    return md_path, csv_path
