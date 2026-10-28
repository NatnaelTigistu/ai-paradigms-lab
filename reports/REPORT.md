Draft written by an agent; author review needed.

# Executive Report: Knowledge Distillation and Recursive Self-Improvement

## 1. Problem and Approach
Modern machine learning models face two practical challenges. First, high-capacity neural networks are too slow and memory-intensive for edge or CPU deployment. Second, labeled data is scarce, while unlabeled data is abundant. 

In this lab, we studied two learning paradigms on Fashion-MNIST:
1. **Knowledge Distillation (Phase 1)**: Compressing a high-capacity Convolutional Neural Network (CNN, 421,642 parameters) into a compact Multi-Layer Perceptron (MLP, 25,450 parameters) using soft teacher probability targets.
2. **Recursive Self-Improvement (Phase 2)**: Bootstrapping a compact CNN (105,866 parameters) by iteratively labeling unlabeled pool samples, filtering candidates through independent verifiers, and gating dataset expansion with rollback.

We followed strict scientific fairness rules throughout:
- Both hard and distilled models received identical hyperparameter search budgets.
- The v2 distillation study evaluated 5 fresh test seeds (`[10, 11, 12, 13, 14]`) never seen during tuning.
- The official 10,000-sample test set was accessed strictly through dedicated one-shot scripts (`scripts/final_test_phase1.py` and `scripts/final_test_recursive.py`). Phase 1 v2 test evaluation was NOT RUN.
- An earlier v1 test evaluation (`tag phase1-final`) occurred prematurely on under-trained 8-epoch models with un-tuned temperatures (`docs/phase1_v2/WHY_RETRAIN.md`). We corrected this in v2 with full convergence (up to 400 epochs) and coordinate descent tuning.

---

## 2. Methods
All experiments ran in a CPU-only environment with deterministic seeding.

### Data Splits and Partitions
The official Fashion-MNIST dataset was split into disjoint subsets (`scripts/verify_splits.py::verify_and_generate_manifest`, verified in `docs/split_manifest.json`):
- `train`: 50,000 samples.
- `dev`: 10,000 samples (split into 4,997 `gate` and 5,003 `monitor` samples).
- `test`: 10,000 samples (official test set).

For Phase 2, the 50,000 training samples were partitioned into:
- Initial labeled seed: 2,005 stratified samples.
- Unlabeled pool: 20,000 samples.
- Disjointness was enforced by `src/paradigms/recursive/state.py::assert_no_leakage`.

### Knowledge Distillation (Phase 1)
We trained `src/paradigms/models/cnn_teacher.py::TeacherCNN` and distilled into `src/paradigms/models/mlp_student.py::StudentMLP`. In v2 (`scripts/run_v2_experiments.py`), we evaluated three label fractions ($f \in \{0.02, 0.10, 1.0\}$) across three regimes:
- **Hard Student**: Trained on ground-truth subset labels using cross-entropy.
- **Distilled Student**: Trained on the same subset with Kullback-Leibler loss (`src/paradigms/distillation/cached.py::masked_distillation_loss`).
- **Distilled Transfer**: Trained on subset labels plus dark knowledge over all 50,000 unlabelled samples using cached teacher logits (`src/paradigms/distillation/cached.py::cache_teacher_logits`).

### Recursive Loop (Phase 2)
In Phase 2 (`src/paradigms/recursive/loop.py::run_loop`), `src/paradigms/models/small_cnn.py::SmallCNN` underwent 4 recursive rounds. In each round:
1. The model predicted labels on the unlabeled pool (`src/paradigms/recursive/generate.py::generate`).
2. Predictions were filtered by a verifier (`src/paradigms/recursive/verifiers.py::build_verifier`).
3. Candidates were capped at 300 samples per class (`src/paradigms/recursive/curate.py::curate`).
4. An acceptance gate checked validation gain ($\ge +0.001$) and guarded against per-class drops ($> 0.05$) (`src/paradigms/recursive/gate.py::decide_gate`).
5. Failing candidates were rolled back to the previous checkpoint.

We tested four verifiers: `OracleVerifier` (simulated upper bound), `AgreementVerifier` (independent Random Forest), `SelfConfidenceVerifier` (confidence $\ge 0.90$), and `NoVerifier` (compute control).

---

## 3. Results Summary
Full tables and charts are consolidated in `reports/RESULTS.md`.

### Distillation Key Findings
- **Low-Data Advantage**: At 2% labels (1,000 samples), distillation provided a +1.61% ± 0.19% gain over the hard student (82.48% vs 80.87%, 5/5 wins, `experiments/v2/final/summary.json`). Distilled transfer reached 87.02% (+6.15% gain).
- **Diminishing Returns**: At 100% labels (50,000 samples), distillation improved accuracy by only +0.20% ± 0.17% (88.39% vs 88.20%, 4/5 wins, `experiments/v2/final/summary.json`). When ground-truth labels are abundant, soft dark knowledge adds little marginal value.

### Recursive Loop Key Findings
- **Oracle Ceiling**: With perfect verification, `oracle_gate` gained +2.19% ± 1.07% on the official test set (83.89% to 86.08%, 3/3 wins, `experiments/final/recursive_test.json`).
- **The Pseudo-Labeling Trap**: `selfconf_nogate` achieved a +1.08% ± 0.54% test gain (83.89% to 84.97%, `experiments/final/recursive_test.json`), but accumulated 8.66% label noise (`reports/recursive/tables/rec_findings.md`). This caused Class 4 (Coat) accuracy to drop by -11.36% on average.
- **The Gate as a Shield**: `selfconf_gate` caught this degradation early, rejecting failing rounds and limiting noise to 2.17% (`reports/recursive/tables/rec_findings.md`).
- **Verifier Bottleneck**: `agreement_gate` achieved 0 accepted rounds. The Random Forest operated on raw pixels, lacking the spatial invariance needed to supply new information to the CNN.

---

## 4. Limitations
1. **CPU Speed Constraints**: Compute budgets limited model sizes to 32 hidden units (MLP) and 32 filters (CNN).
2. **Oracle Artificiality**: The Oracle verifier reads hidden labels. It acts as an idealized reference ceiling, not a real-world algorithm.
3. **Flat Pixel Verification**: The Random Forest verifier operated on flattened 784-dimensional pixel arrays rather than spatial convolutional embeddings.
4. **Binary Gate Hardness**: The gate's fixed 5% class drop threshold rejected marginal improvements in noisy classes.

---

## 5. Next Steps
1. **Phase 3 Synthesis**: Distill recursively improved teachers into compact students using temperature-scaled soft loss.
2. **Latent Embedding Verification**: Replace raw-pixel Random Forest verifiers with support vector classifiers trained on penultimate CNN embeddings.
3. **Soft Pseudo-Labeling**: Train recursive candidates with soft distillation loss rather than hard cross-entropy to absorb pseudo-label uncertainty safely.
