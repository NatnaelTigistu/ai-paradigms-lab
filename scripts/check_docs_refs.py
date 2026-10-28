#!/usr/bin/env python3
"""Validate code symbol references and links across all repository documentation.

Scans every markdown file in docs/, reports/, and README.md to ensure:
1. Every `path/to/file.py::name` points to an existing file containing `def name` or `class name`.
2. Every relative link or image reference resolves to an existing file or directory on disk.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_symbol_exists(py_path: Path, symbol_name: str) -> bool:
    """Check if Python file exists and defines `def symbol_name` or `class symbol_name`."""
    if not py_path.exists() or not py_path.is_file():
        return False

    try:
        content = py_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_path))
    except Exception:
        # Fallback to regex line check if AST fails
        lines = py_path.read_text(encoding="utf-8").splitlines()
        def_pattern = re.compile(rf"^\s*(?:def|class)\s+{re.escape(symbol_name)}\b")
        return any(def_pattern.search(line) for line in lines)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol_name:
                return True
    return False


def validate_markdown_documentation(
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], list[str]]:
    """Validate all markdown files for valid symbol references and link targets.

    Returns:
        Tuple of (symbol_errors, link_errors).
    """
    symbol_errors: list[str] = []
    link_errors: list[str] = []

    # Collect markdown files
    md_files = sorted(
        list((repo_root / "docs").rglob("*.md"))
        + list((repo_root / "reports").rglob("*.md"))
        + ([repo_root / "README.md"] if (repo_root / "README.md").exists() else [])
    )

    # Patterns
    # 1. path/to/file.py::name
    ref_pattern = re.compile(r"([a-zA-Z0-9_\/.-]+\.py)::([a-zA-Z0-9_]+)")
    # 2. Markdown links and images [alt](target) or ![alt](target)
    link_pattern = re.compile(r"!?\[([^\]]*)\]\(([^)#\s]+)(?:#[^\)]*)?\)")
    # 3. HTML img src or href
    html_pattern = re.compile(r'(?:src|href)=["\']([^"\']+)["\']')

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            link_errors.append(f"{md_file.relative_to(repo_root)}: Unable to read file ({e})")
            continue

        rel_md = md_file.relative_to(repo_root)

        # Check 1: Symbol references (path/to/file.py::name)
        for py_rel, symbol_name in ref_pattern.findall(content):
            py_path = repo_root / py_rel
            if not py_path.exists():
                symbol_errors.append(
                    f"{rel_md}: Referenced file does not exist -> '{py_rel}' in '{py_rel}::{symbol_name}'"
                )
            elif not check_symbol_exists(py_path, symbol_name):
                symbol_errors.append(
                    f"{rel_md}: Symbol '{symbol_name}' not defined in '{py_rel}'"
                )

        # Check 2: Relative links and image paths
        targets: list[str] = []
        for _, target in link_pattern.findall(content):
            targets.append(target)
        for target in html_pattern.findall(content):
            targets.append(target)

        for target in targets:
            target_clean = target.strip()
            # Ignore external web links, anchors, or mailto
            if not target_clean or target_clean.startswith(("http://", "https://", "mailto:", "#")):
                continue

            # Handle file:// URIs
            if target_clean.startswith("file://"):
                target_path_str = target_clean.replace("file://", "")
                resolved_path = Path(target_path_str)
            elif target_clean.startswith("/"):
                # Absolute path from root
                resolved_path = Path(target_clean)
            else:
                # Relative link from markdown file parent
                resolved_path = (md_file.parent / target_clean).resolve()

            if not resolved_path.exists():
                link_errors.append(
                    f"{rel_md}: Link target does not exist -> '{target_clean}' (resolved to '{resolved_path}')"
                )

    return symbol_errors, link_errors


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify code symbol references and relative links in documentation."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(REPO_ROOT),
        help="Path to repository root (default: current repository).",
    )
    return parser.parse_args(cli_args)


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    root = Path(args.repo_root)

    print("\n" + "=" * 70)
    print("CHECKING DOCUMENTATION CODE REFERENCES AND RELATIVE LINKS")
    print(f"Repository Root: {root}")
    print("=" * 70 + "\n")

    symbol_errors, link_errors = validate_markdown_documentation(root)

    total_errors = len(symbol_errors) + len(link_errors)
    if total_errors == 0:
        print("SUCCESS: All symbol references (`path::name`) and relative links are valid.\n")
        sys.exit(0)

    print(f"FAILURE: Detected {total_errors} documentation validation errors:\n")
    if symbol_errors:
        print("SYMBOL REFERENCE ERRORS:")
        for err in symbol_errors:
            print(f"  - {err}")
        print()

    if link_errors:
        print("LINK / IMAGE RESOLUTION ERRORS:")
        for err in link_errors:
            print(f"  - {err}")
        print()

    sys.exit(1)


if __name__ == "__main__":
    main()
