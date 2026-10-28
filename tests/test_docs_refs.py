"""Test verifying documentation code references (`path::name`) and links."""

from __future__ import annotations

from pathlib import Path

from scripts.check_docs_refs import validate_markdown_documentation

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_markdown_documentation_references_and_links():
    """Verify that all markdown code symbol references and links resolve cleanly."""
    symbol_errors, link_errors = validate_markdown_documentation(REPO_ROOT)

    assert not symbol_errors, (
        f"Detected {len(symbol_errors)} invalid symbol references:\n"
        + "\n".join(f"  - {err}" for err in symbol_errors)
    )
    assert not link_errors, (
        f"Detected {len(link_errors)} broken markdown links or image paths:\n"
        + "\n".join(f"  - {err}" for err in link_errors)
    )
