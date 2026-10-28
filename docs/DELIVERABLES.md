# Assignment Deliverables Manifest

This document maps each required assignment deliverable to its corresponding repository artifacts and current execution status.

---

## Deliverables Status Table

| Assignment Deliverable | Artifact Path(s) | Status | Description |
| :--- | :--- | :--- | :--- |
| **Working code with README** | `README.md`<br>`src/paradigms/`<br>`scripts/` | **DONE** | Fully documented modular codebase with verified reproduction commands. |
| **Reproducible dataset and fixed seeds** | `docs/split_manifest.json`<br>`scripts/verify_splits.py`<br>`docs/REPRODUCIBILITY.md` | **DONE** | Deterministic Fashion-MNIST split protocol with SHA-256 partition index hashes. |
| **Separate train/dev/final-test data** | `src/paradigms/data/splits.py`<br>`tests/test_test_set_access.py` | **DONE** | 5-way disjoint partition (`train`, `dev`, `gate`, `monitor`, `test`) guarded by static test-set access analysis. |
| **Results table and charts** | `reports/RESULTS.md`<br>`reports/recursive/figures/`<br>`reports/v2/figures/` | **DONE** | Consolidated empirical tables across all phases with embedded visualization plots. |
| **Failure analysis** | `reports/FAILURE_ANALYSIS.md` | **DONE** | Detailed technical analysis covering v1 deficits, 100% data marginal gains, recursive loop collapse, and process failures. |
| **1-2 page explanation** | `reports/REPORT.md` | **DONE** | Concise executive summary (783 words body text) explaining problems, methods, fairness rules, results, and limitations. |
| **8-12 slide presentation or demo** | `reports/PRESENTATION_OUTLINE.md` | **DONE** (Outline) | 10-slide technical presentation outline with speaker notes and a 3-minute live execution demo script. Slides `.pptx` file: **NOT INCLUDED** (outline provided as specified). |
| **Phase 1 v2 Official Test Set Evaluation** | `experiments/v2/final_test.json` | **NOT RUN** | Protected test set evaluation for Phase 1 v2 was strictly locked and withheld to preserve test set hygiene. |
