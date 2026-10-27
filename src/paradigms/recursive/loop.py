"""Bounded recursive self-training loop with verification, acceptance gating, and rollback."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from paradigms.data.tensor_loader import make_fast_loader
from paradigms.models import build_model
from paradigms.recursive.curate import curate
from paradigms.recursive.datasets import PseudoLabeledDataset
from paradigms.recursive.gate import decide_gate
from paradigms.recursive.generate import generate
from paradigms.recursive.state import LoopBundle, assert_no_leakage
from paradigms.recursive.verifiers import build_verifier
from paradigms.seed import set_seed
from paradigms.training.evaluate import evaluate
from paradigms.training.losses import hard_label_loss
from paradigms.training.trainer import fit


def run_loop(
    cfg: dict[str, Any] | Any,
    bundle: LoopBundle,
    *,
    seed: int,
    verifier_name: str,
    gate_enabled: bool,
    ckpt_dir: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute the bounded recursive self-training loop with gate verification and rollback.

    Args:
        cfg: Configuration dictionary or Config instance containing recursive and training keys.
        bundle: Verified LoopBundle containing base dataset, partitions, and protected subsets.
        seed: Experiment random seed.
        verifier_name: Identifier of the verification strategy ("oracle", "agreement", etc.).
        gate_enabled: Whether the acceptance gate is active.
        ckpt_dir: Directory path to persist round0.pt and final.pt checkpoints.
        verbose: If True, prints epoch progress during model fits.

    Returns:
        Summary dictionary containing log records, final metrics, and stopping criteria.
    """
    assert_no_leakage(bundle.partition)
    set_seed(seed)

    rcfg = cfg.get("recursive", cfg) if hasattr(cfg, "get") else cfg
    train_cfg_raw = cfg.get("training", {}) if hasattr(cfg, "get") else getattr(cfg, "training", {})
    train_cfg = dict(train_cfg_raw) if train_cfg_raw else {}

    model_name = rcfg.get("model", "small_cnn") if hasattr(rcfg, "get") else getattr(rcfg, "model", "small_cnn")
    model_kwargs = rcfg.get("model_kwargs", {}) if hasattr(rcfg, "get") else getattr(rcfg, "model_kwargs", {})
    rounds = int(rcfg.get("rounds", 4) if hasattr(rcfg, "get") else getattr(rcfg, "rounds", 4))
    epochs_initial = int(rcfg.get("epochs_initial", 30) if hasattr(rcfg, "get") else getattr(rcfg, "epochs_initial", 30))
    epochs_per_round = int(rcfg.get("epochs_per_round", 4) if hasattr(rcfg, "get") else getattr(rcfg, "epochs_per_round", 4))
    lr_round = float(rcfg.get("lr_round", 0.0005) if hasattr(rcfg, "get") else getattr(rcfg, "lr_round", 0.0005))
    batch_size = int(rcfg.get("batch_size", 128) if hasattr(rcfg, "get") else getattr(rcfg, "batch_size", 128))
    cap_per_class = int(rcfg.get("cap_per_class", 300) if hasattr(rcfg, "get") else getattr(rcfg, "cap_per_class", 300))
    curation_strategy = str(rcfg.get("curation", "random") if hasattr(rcfg, "get") else getattr(rcfg, "curation", "random"))

    gate_cfg = rcfg.get("gate", {}) if hasattr(rcfg, "get") else getattr(rcfg, "gate", {})
    patience = int(gate_cfg.get("patience", 2) if hasattr(gate_cfg, "get") else getattr(gate_cfg, "patience", 2))

    optimizer = str(train_cfg.get("optimizer", "adam"))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    lr_initial = float(train_cfg.get("lr", 0.001))

    round0_train_cfg = {
        "epochs": epochs_initial,
        "lr": lr_initial,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
    }

    base = bundle.base
    labeled_ds = Subset(base, bundle.partition.labeled)

    # Fast in-memory data loaders
    labeled_loader = make_fast_loader(labeled_ds, batch_size=batch_size, shuffle=True, seed=seed)
    gate_loader = make_fast_loader(bundle.gate_ds, batch_size=batch_size, shuffle=False)
    monitor_loader = make_fast_loader(bundle.monitor_ds, batch_size=batch_size, shuffle=False)

    ckpt_path = Path(ckpt_dir) if ckpt_dir is not None else None
    if ckpt_path is not None:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    # ------------------ ROUND 0: Initial Supervised Seed ------------------
    t0_r0 = time.time()
    model = build_model(model_name, **model_kwargs)
    fit(
        model,
        labeled_loader,
        dev_loader=None,
        loss_fn=hard_label_loss,
        training_cfg=round0_train_cfg,
        device="cpu",
        verbose=verbose,
    )
    if ckpt_path is not None:
        torch.save(model.state_dict(), ckpt_path / "round0.pt")

    current = model
    base_gate = evaluate(current, gate_loader, device="cpu")
    base_mon = evaluate(current, monitor_loader, device="cpu")
    round0_per_class = list(base_gate["per_class_accuracy"])

    verifier = build_verifier(verifier_name, rcfg, bundle, seed=seed)

    pool = set(bundle.partition.pool)
    acc_idx: list[int] = []
    acc_lab: list[int] = []
    rejections = 0
    total_verifier_calls = 0

    valid_pc_r0 = [(c, acc) for c, acc in enumerate(base_gate["per_class_accuracy"]) if acc is not None]
    worst_c_r0, min_pc_r0 = min(valid_pc_r0, key=lambda x: x[1]) if valid_pc_r0 else (-1, 0.0)

    log: list[dict[str, Any]] = [
        {
            "round": 0,
            "accepted": True,
            "reason": "initial",
            "n_candidates": 0,
            "n_verified": 0,
            "n_selected": 0,
            "selected_precision": 1.0,
            "train_size": len(bundle.partition.labeled),
            "cand_gate_acc": base_gate["accuracy"],
            "cand_monitor_acc": base_mon["accuracy"],
            "current_gate_acc": base_gate["accuracy"],
            "current_monitor_acc": base_mon["accuracy"],
            "current_min_class_acc": min_pc_r0,
            "current_worst_class": worst_c_r0,
            "label_noise_in_training": 0.0,
            "seconds": time.time() - t0_r0,
        }
    ]

    stop_reason = "completed"

    # ------------------ ROUNDS 1..N: Recursive Loop ------------------
    for r in range(1, rounds + 1):
        t0_round = time.time()
        if len(pool) == 0:
            stop_reason = "pool_empty"
            break

        pool_sorted = sorted(pool)
        n_candidates = len(pool_sorted)
        total_verifier_calls += n_candidates

        cands = generate(current, base, pool_sorted, batch_size=batch_size)
        mask = verifier.verify(cands.indices, cands.preds, cands.confs)
        n_verified = int(mask.sum())

        sel_idx_arr, sel_lab_arr = curate(
            cands.indices,
            cands.preds,
            cands.confs,
            mask,
            cap_per_class=cap_per_class,
            seed=seed * 100 + r,
            strategy=curation_strategy,
        )
        sel_idx = sel_idx_arr.tolist()
        sel_lab = sel_lab_arr.tolist()
        n_selected = len(sel_idx)

        # DIAGNOSTIC ONLY, logged, never used to decide
        if n_selected > 0:
            precision = float(np.mean(sel_lab_arr == bundle.hidden_labels[sel_idx_arr]))
        else:
            precision = 0.0

        candidate = copy.deepcopy(current)  # update a COPY, never the original

        # Accepted pseudo-labels are fixed, not refreshed across rounds, to provide monotonic
        # dataset expansion, anchor the model against catastrophic drift/forgetting, and prevent
        # cyclical prediction flip-flops.
        all_pseudo_idx = acc_idx + sel_idx
        all_pseudo_lab = acc_lab + sel_lab

        if len(all_pseudo_idx) > 0:
            train_ds = ConcatDataset([labeled_ds, PseudoLabeledDataset(base, all_pseudo_idx, all_pseudo_lab)])
        else:
            train_ds = labeled_ds

        round_train_loader = make_fast_loader(
            train_ds, batch_size=batch_size, shuffle=True, seed=seed * 100 + r
        )
        fit(
            candidate,
            round_train_loader,
            dev_loader=None,
            loss_fn=hard_label_loss,
            training_cfg={
                "epochs": epochs_per_round,
                "lr": lr_round,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
            },
            device="cpu",
            verbose=verbose,
        )

        cand_gate = evaluate(candidate, gate_loader, device="cpu")
        cand_mon = evaluate(candidate, monitor_loader, device="cpu")

        if gate_enabled:
            ok, reason = decide_gate(cand_gate, base_gate, gate_cfg)
        else:
            ok, reason = True, "gate disabled"

        if ok:
            current = candidate
            base_gate = cand_gate
            base_mon = cand_mon
            acc_idx += sel_idx
            acc_lab += sel_lab
            pool -= set(sel_idx)
            rejections = 0
        else:
            rejections += 1  # ROLLBACK: candidate discarded

        # Diagnostic label noise calculation
        if len(acc_idx) > 0:
            label_noise = float(np.mean(np.array(acc_lab) != bundle.hidden_labels[acc_idx]))
        else:
            label_noise = 0.0

        valid_pc = [(c, acc) for c, acc in enumerate(base_gate["per_class_accuracy"]) if acc is not None]
        worst_c, min_pc = min(valid_pc, key=lambda x: x[1]) if valid_pc else (-1, 0.0)

        log.append(
            {
                "round": r,
                "accepted": ok,
                "reason": reason,
                "n_candidates": n_candidates,
                "n_verified": n_verified,
                "n_selected": n_selected,
                "selected_precision": precision,
                "train_size": len(train_ds),
                "cand_gate_acc": cand_gate["accuracy"],
                "cand_monitor_acc": cand_mon["accuracy"],
                "current_gate_acc": base_gate["accuracy"],
                "current_monitor_acc": base_mon["accuracy"],
                "current_min_class_acc": min_pc,
                "current_worst_class": worst_c,
                "label_noise_in_training": label_noise,
                "seconds": time.time() - t0_round,
            }
        )

        if gate_enabled and rejections >= patience:
            stop_reason = "patience"
            break

    if ckpt_path is not None:
        torch.save(current.state_dict(), ckpt_path / "final.pt")

    n_accepted_rounds = sum(1 for row in log[1:] if row["accepted"])
    n_rejected_rounds = sum(1 for row in log[1:] if not row["accepted"])
    total_seconds = sum(row["seconds"] for row in log)

    return {
        "log": log,
        "stop_reason": stop_reason,
        "round0_gate_acc": log[0]["current_gate_acc"],
        "round0_monitor_acc": log[0]["current_monitor_acc"],
        "round0_per_class": round0_per_class,
        "final_monitor_acc": base_mon["accuracy"],
        "final_per_class": base_mon["per_class_accuracy"],
        "n_accepted_rounds": n_accepted_rounds,
        "n_rejected_rounds": n_rejected_rounds,
        "n_pseudo_labels_total": len(acc_idx),
        "total_verifier_calls": total_verifier_calls,
        "total_seconds": total_seconds,
        "verifier": verifier_name,
        "gate_enabled": gate_enabled,
        "seed": seed,
    }
