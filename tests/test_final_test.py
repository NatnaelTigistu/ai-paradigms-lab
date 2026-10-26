"""Tests for Phase 1 final test set evaluation guard and refusal logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.final_test_phase1 import check_refusal, main


def test_refusal_when_output_file_exists_without_force(tmp_path: Path):
    """Verify script refuses execution with non-zero exit code when output exists."""
    fake_out = tmp_path / "phase1_test.json"
    fake_out.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        check_refusal(fake_out, force=False)

    assert exc_info.value.code != 0


def test_no_refusal_when_output_file_does_not_exist(tmp_path: Path):
    """Verify execution proceeds normally if output file does not already exist."""
    fake_out = tmp_path / "nonexistent_phase1_test.json"
    # Should not raise
    check_refusal(fake_out, force=False)


def test_no_refusal_with_force_flag_on_existing_file(tmp_path: Path):
    """Verify execution proceeds when --force is provided despite existing file."""
    fake_out = tmp_path / "phase1_test.json"
    fake_out.write_text("{}", encoding="utf-8")

    # Should not raise when force=True
    check_refusal(fake_out, force=True)


def test_main_cli_refusal(tmp_path: Path):
    """Verify main entry point aborts with non-zero code on existing file without --force."""
    fake_out = tmp_path / "phase1_test.json"
    fake_out.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["--out-file", str(fake_out)])

    assert exc_info.value.code != 0
