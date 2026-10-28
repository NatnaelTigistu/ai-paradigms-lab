"""Tests for recursive loop reporting aggregation, carry-forward, and final test refusal logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from scripts.final_test_recursive import check_refusal


def _create_mock_run(
    seed: int,
    rounds: int,
    r0_mon: float,
    final_mon: float,
    accepted_rounds: int,
    rejected_rounds: int,
    precision: float = 1.0,
    n_selected: int = 1000,
) -> dict:
    log = [
        {
            "round": 0,
            "accepted": True,
            "reason": "initial",
            "n_candidates": 0,
            "n_verified": 0,
            "n_selected": 0,
            "selected_precision": 1.0,
            "train_size": 2000,
            "cand_gate_acc": r0_mon,
            "cand_monitor_acc": r0_mon,
            "current_gate_acc": r0_mon,
            "current_monitor_acc": r0_mon,
            "current_min_class_acc": 0.5,
            "current_worst_class": 0,
            "label_noise_in_training": 0.0,
            "seconds": 5.0,
        }
    ]
    for r in range(1, rounds + 1):
        is_acc = r <= accepted_rounds
        cur_mon = final_mon if is_acc else r0_mon
        log.append(
            {
                "round": r,
                "accepted": is_acc,
                "reason": "accepted" if is_acc else "rejected",
                "n_candidates": 5000,
                "n_verified": 4000,
                "n_selected": n_selected if is_acc else 0,
                "selected_precision": precision,
                "train_size": 2000 + r * 1000,
                "cand_gate_acc": cur_mon,
                "cand_monitor_acc": cur_mon,
                "current_gate_acc": cur_mon,
                "current_monitor_acc": cur_mon,
                "current_min_class_acc": 0.5,
                "current_worst_class": 0,
                "label_noise_in_training": 0.0,
                "seconds": 2.0,
            }
        )

    return {
        "log": log,
        "stop_reason": "completed" if rounds == 4 else "patience",
        "round0_gate_acc": r0_mon,
        "round0_monitor_acc": r0_mon,
        "round0_per_class": [r0_mon] * 10,
        "final_monitor_acc": final_mon,
        "final_per_class": [final_mon] * 10,
        "n_accepted_rounds": accepted_rounds,
        "n_rejected_rounds": rejected_rounds,
        "n_pseudo_labels_total": accepted_rounds * n_selected,
        "total_verifier_calls": rounds * 5000,
        "total_seconds": 5.0 + rounds * 2.0,
        "verifier": "mock",
        "gate_enabled": True,
        "seed": seed,
    }


def test_missing_config_skipped_with_warning(tmp_path: Path):
    """Verify missing configurations emit a warning and are cleanly skipped."""
    c1_dir = tmp_path / "oracle_gate"
    c1_dir.mkdir(parents=True)
    (c1_dir / "seed0.json").write_text(
        json.dumps(_create_mock_run(0, 4, 0.80, 0.85, 2, 2)), encoding="utf-8"
    )

    with pytest.warns(UserWarning, match="skipping"):
        results = load_recursive_results(tmp_path, configs=["oracle_gate", "missing_config"])

    assert "oracle_gate" in results
    assert "missing_config" not in results
    assert len(results["oracle_gate"]) == 1


def test_carry_forward_and_aggregation(tmp_path: Path):
    """Verify carry-forward logic when a run terminates early (fewer rounds)."""
    # oracle_gate: 4 rounds, progresses from 0.80 to 0.86
    og_dir = tmp_path / "oracle_gate"
    og_dir.mkdir(parents=True)
    # control_none: 2 rounds (early stopped), stays at 0.80
    ctrl_dir = tmp_path / "control_none"
    ctrl_dir.mkdir(parents=True)

    for s in [0, 1]:
        (og_dir / f"seed{s}.json").write_text(
            json.dumps(_create_mock_run(s, 4, 0.80, 0.86 + s * 0.01, 3, 1)), encoding="utf-8"
        )
        (ctrl_dir / f"seed{s}.json").write_text(
            json.dumps(_create_mock_run(s, 2, 0.80, 0.80, 0, 2)), encoding="utf-8"
        )

    results = load_recursive_results(tmp_path, configs=["oracle_gate", "control_none"])
    progression = compute_round_progression(results, max_rounds=4)

    # control_none stopped after round 2; check rounds 3 and 4 carry forward round 2 monitor acc
    ctrl_entries = progression["control_none"]
    assert len(ctrl_entries) == 5  # rounds 0, 1, 2, 3, 4
    for r in range(5):
        assert abs(ctrl_entries[r]["monitor_acc_mean"] - 0.80) < 1e-6

    # Test summary computation and paired gains
    summary = compute_recursive_summary(results, control_config="control_none")
    assert "oracle_gate" in summary
    assert "control_none" in summary

    p_ctrl = summary["oracle_gate"]["paired_vs_control"]
    assert p_ctrl is not None
    assert p_ctrl["mean_diff"] > 0
    assert p_ctrl["wins"] == 2
    assert summary["oracle_gate"]["verdict"] == "WIN"


def test_tables_and_plots_on_mock_results(tmp_path: Path):
    """Verify markdown/CSV table generation and plot creation on mock results."""
    for cfg in ["oracle_gate", "control_none", "agreement_gate", "selfconf_gate", "selfconf_nogate"]:
        c_dir = tmp_path / "data" / cfg
        c_dir.mkdir(parents=True)
        for s in [0, 1]:
            (c_dir / f"seed{s}.json").write_text(
                json.dumps(_create_mock_run(s, 4, 0.82, 0.85, 2, 2, precision=0.95)),
                encoding="utf-8",
            )

    results = load_recursive_results(tmp_path / "data")
    summary = compute_recursive_summary(results, control_config="control_none")
    progression = compute_round_progression(results, max_rounds=4)

    sum_headers, sum_rows = generate_summary_table(summary)
    assert len(sum_headers) == 13
    assert len(sum_rows) == 5

    rnd_headers, rnd_rows = generate_rounds_table(progression)
    assert len(rnd_headers) == 6
    assert len(rnd_rows) == 25  # 5 configs * 5 rounds

    fig_dir = tmp_path / "figs"
    fig_dir.mkdir(parents=True)

    f1 = plot_recursive_curves(progression, fig_dir / "fig_rec_curves.png")
    f2 = plot_verifier_precision(results, fig_dir / "fig_rec_precision.png")
    f3 = plot_acceptance_counts(results, fig_dir / "fig_rec_accept.png")
    f4 = plot_gate_gap(progression, fig_dir / "fig_rec_gate_gap.png")
    f5 = plot_perclass_deltas(results, fig_dir / "fig_rec_perclass.png")

    for f in [f1, f2, f3, f4, f5]:
        assert f.exists()
        assert f.stat().st_size > 0

    findings_path = tmp_path / "rec_findings.md"
    generate_findings_markdown(summary, results, progression, findings_path)
    assert findings_path.exists()
    findings_text = findings_path.read_text(encoding="utf-8")
    assert "Verdict Summary vs control_none" in findings_text


def test_final_test_refusal_when_output_exists_without_force(tmp_path: Path):
    """Verify final test script refuses execution with non-zero exit when out-file exists."""
    fake_out = tmp_path / "recursive_test.json"
    fake_out.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        check_refusal(fake_out, force=False)

    assert exc_info.value.code != 0


def test_final_test_no_refusal_with_force_flag(tmp_path: Path):
    """Verify execution proceeds when --force is provided despite existing file."""
    fake_out = tmp_path / "recursive_test.json"
    fake_out.write_text("{}", encoding="utf-8")

    # Should not raise when force=True
    check_refusal(fake_out, force=True)


def test_final_test_no_refusal_when_file_absent(tmp_path: Path):
    """Verify execution proceeds normally if output file does not exist."""
    fake_out = tmp_path / "absent_recursive_test.json"
    check_refusal(fake_out, force=False)
