"""Recursive self-improvement paradigms, datasets, candidate generation, curation, verifiers, gate, and loop."""

from paradigms.recursive.curate import curate
from paradigms.recursive.datasets import PseudoLabeledDataset
from paradigms.recursive.gate import decide_gate
from paradigms.recursive.generate import Candidates, generate
from paradigms.recursive.loop import run_loop
from paradigms.recursive.state import (
    LoopBundle,
    LoopPartition,
    assert_no_leakage,
    make_bundle,
    make_partition,
)
from paradigms.recursive.verifiers import (
    AgreementVerifier,
    NoVerifier,
    OracleVerifier,
    SelfConfidenceVerifier,
    Verifier,
    build_verifier,
)

__all__ = [
    "PseudoLabeledDataset",
    "LoopPartition",
    "make_partition",
    "assert_no_leakage",
    "LoopBundle",
    "make_bundle",
    "Candidates",
    "generate",
    "curate",
    "Verifier",
    "OracleVerifier",
    "AgreementVerifier",
    "SelfConfidenceVerifier",
    "NoVerifier",
    "build_verifier",
    "decide_gate",
    "run_loop",
]
