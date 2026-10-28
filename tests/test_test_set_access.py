"""Static analysis test verifying that protected test set access is restricted.

Asserts that any invocation of `allow_test=True` across the entire codebase
is confined strictly to scripts named `final_test_*.py` or files within `tests/`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_test_set_access_violations(repo_root: Path) -> list[tuple[Path, int, str]]:
    """Scan all Python files for allow_test=True invocations outside authorized locations.

    Authorized locations:
    1. Scripts matching `final_test_*.py`
    2. Any file located under the `tests/` directory

    Returns:
        List of (file_path, line_number, matching_line) tuples representing violations.
    """
    violations: list[tuple[Path, int, str]] = []
    py_files = sorted(repo_root.rglob("*.py"))

    # Regex targeting keyword argument assignment allow_test=True
    pattern = re.compile(r"\ballow_test\s*=\s*True\b")

    for py_file in py_files:
        rel_parts = py_file.relative_to(repo_root).parts

        # Ignore virtual environments, git metadata, and caches
        if any(part in (".venv", "venv", ".git", "__pycache__", ".pytest_cache") for part in rel_parts):
            continue

        # Authorized check
        is_final_test_script = py_file.name.startswith("final_test_")
        is_tests_directory = "tests" in rel_parts

        if is_final_test_script or is_tests_directory:
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Quick pre-filter
        if "allow_test" not in content:
            continue

        # Check AST for actual Call nodes passing allow_test=True
        try:
            tree = ast.parse(content, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == "allow_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            line_no = node.lineno
                            line_text = content.splitlines()[line_no - 1].strip() if line_no <= len(content.splitlines()) else ""
                            violations.append((py_file.relative_to(repo_root), line_no, line_text))
        except SyntaxError:
            # Fallback to regex line scanning if AST parse fails
            for idx, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    # Exclude docstrings, comments, and exception error message strings
                    stripped = line.strip()
                    if not stripped.startswith("#") and "def " not in stripped and "raise " not in stripped:
                        violations.append((py_file.relative_to(repo_root), idx, stripped))

    return violations


def test_test_set_access_is_strictly_confined():
    """Verify that no unauthorized file invokes allow_test=True."""
    violations = find_test_set_access_violations(REPO_ROOT)
    assert not violations, (
        f"Protected test set access violation detected!\n"
        f"The following unauthorized files invoke allow_test=True:\n"
        + "\n".join(f"  - {path}:{line}: {text}" for path, line, text in violations)
        + "\nAccess to the official test set is restricted strictly to final_test_*.py and tests/."
    )
