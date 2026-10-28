# Failure Analysis: Empirical Deficits, Root Causes, and Process Retrospective

This document analyzes what did not work across the distillation and recursive self-improvement experiments. Every explanation is either grounded in empirical numbers cited directly from result files (Rule R2) or explicitly labeled as `HYPOTHESIS (not tested)`.

---

## 1. Phase 1 v1 Distillation: Hard Student Outperformed Distilled Student

### What Happened
In Phase 1 v1, the hard student baseline matched or beat the distilled student across all 5 evaluation seeds:
- Hard Student dev accuracy: **87.34% ± 0.21%** (`reports/tables/phase1_main.md`).
- Distilled Student dev accuracy: **86.85% ± 0.12%** (`reports/tables/phase1_main.md`).
- Paired difference (Distilled - Hard): **-0.49% ± 0.13%** (0 / 5 wins, `reports/tables/phase1_paired.md`).
- Official test set evaluation: Hard was 86.45% ± 0.06% vs Distilled 86.64% ± 0.25% (marginal +0.20% gain, `experiments/final/phase1_test.json`).

### Likely Causes
1. **Severe Under-Training (8 Epochs)**: Models were trained for only 8 epochs (`configs/base.yaml`). A compact MLP requires 30 to 400 epochs to converge (`docs/phase1_v2/WHY_RETRAIN.md`). At 8 epochs, the student was under-fitted and could not resolve soft teacher targets.
2. **Sub-optimal Temperature Choice ($T=4.0$)**: In v1, temperature was fixed to $T=4.0$. In v2 hyperparameter sweeps at 100% data (`experiments/v2/best_hparams.json`), $T=1.0$ was optimal. A high temperature ($T=4.0$) excessively flattened logits, diluting correct-class gradients when full ground-truth labels were already available.
3. **Incomplete Tuning Sweeps**: v1 sweeps tuned single parameters on single seeds without joint optimization of weight decay, learning rate, and epochs.

### What We Changed
In Phase 1 v2, we redesigned the study (`scripts/run_v2_experiments.py`):
- Extended training to convergence (30 to 400 effective epochs depending on data fraction).
- Applied coordinate descent hyperparameter tuning independently for hard and distilled models (`experiments/v2/best_hparams.json`).
- Pre-computed teacher logits in memory (`src/paradigms/distillation/cached.py::cache_teacher_logits`), providing a 5.2x CPU speedup (`docs/COMPREHENSIVE_GUIDE.md`).

---

## 2. Phase 1 v2 at 100% Labels: Statistically Marginal Gain

### What Happened
At 100% labels (50,000 samples), the distilled student achieved:
- Hard Student dev accuracy: **88.20% ± 0.12%** (`experiments/v2/final/summary.json`).
- Distilled Student dev accuracy: **88.39% ± 0.16%** (`experiments/v2/final/summary.json`).
- Paired difference: **+0.20% ± 0.17%** (population std) / **0.19%** (sample std) with 4 / 5 wins.
- Fixed Rule Verdict: Marked as **WIN** because mean paired diff ($0.00198$) exceeded population std ($0.00171$) and wins $\ge 4/5$.

### Statistical Significance Test
We computed a two-tailed paired t-test across the 5 paired seed differences (`experiments/v2/final/f1.0/`):
- Seed differences: `[-0.0002, +0.0019, +0.0033, +0.0044, +0.0005]`
- $t$-statistic: **2.3215**
- $p$-value: **0.0810**

Because $p = 0.0810 > 0.05$, the improvement is **not statistically significant** at the standard 5% significance level. The gain of 0.20 percentage points is marginal.

### Likely Cause
When 50,000 clean ground-truth labels are available, cross-entropy gradients provide strong supervisory signals. The extra dark knowledge from teacher probabilities provides negligible marginal regularization.

---

## 3. Diminishing Benefit of Distillation Across Label Regimes

### What Happened
The empirical advantage of Knowledge Distillation shrank monotonically as labeled training data increased (`experiments/v2/final/summary.json`):
- **2% Labels (1,000 samples)**: Paired gain = **+1.61% ± 0.19%** (82.48% vs 80.87%, 5/5 wins).
- **10% Labels (5,000 samples)**: Paired gain = **+0.98% ± 0.22%** (85.99% vs 85.01%, 5/5 wins).
- **100% Labels (50,000 samples)**: Paired gain = **+0.20% ± 0.17%** (88.39% vs 88.20%, 4/5 wins).

```
+-------------------------------------------------------------+
|    Distillation Paired Gain vs Ground-Truth Data Fraction   |
+-------------------------------------------------------------+
|  Label Regime  | Labeled Samples | Paired Accuracy Gain (%) |
|----------------|-----------------|--------------------------|
|     f = 0.02   |      1,000      |  +1.61%                  |
|     f = 0.10   |      5,000      |  +0.98%                  |
|     f = 1.00   |     50,000      |  +0.20% (marginal)       |
+-------------------------------------------------------------+
```

### Likely Cause
In low-data regimes, the model suffers from severe sample variance and overfits hard labels. Soft teacher probabilities act as an effective prior over class similarities. As sample size grows 50-fold, the empirical data distribution approximates the true data manifold, diminishing the need for teacher smoothing.

---

## 4. The Distilled Transfer Variant: Not a Same-Data Comparison

### What Happened
In `experiments/v2/final/summary.json`, Distilled Transfer achieved large gains:
- At 2% data: **87.02% ± 0.11%** (+6.15% over hard student).
- At 10% data: **87.13% ± 0.05%** (+2.11% over hard student).

### Why This Is Not a Same-Data Comparison
The Distilled Transfer model is trained on 1,000 (or 5,000) labeled samples plus 49,000 (or 45,000) unlabelled samples paired with teacher logits (`src/paradigms/distillation/cached.py::masked_distillation_loss`).
However, the Teacher CNN was pre-trained on **all 50,000 ground-truth labels** (`experiments/teacher/seed*.json`). 

Therefore, the student indirectly accesses information derived from the full 50,000-sample label set through the teacher's predictions. The hard student only saw 1,000 or 5,000 labels. This is semi-supervised representation transfer with an oracle teacher, not a fair same-data benchmark.

---

## 5. Phase 2 Recursive Loop Failures: Configs with TIE / LOSS

### Summary of Failed Configurations
Three out of five recursive configurations failed to improve over the `control_none` baseline (`experiments/final/recursive_test.json` and `reports/recursive/tables/rec_summary.md`):

| Configuration | Test Gain | Wins / Seeds | Verdict vs `control_none` | Accepted Rounds |
| :--- | :--- | :--- | :--- | :--- |
| `agreement_gate` | **+0.00% ± 0.00%** | 0 / 3 | **TIE** | **0.0 ± 0.0** |
| `selfconf_gate` | **+0.08% ± 0.12%** | 1 / 3 | **TIE** | **0.7 ± 0.9** |
| `control_none` | **+0.00% ± 0.00%** | 0 / 3 | **TIE (Baseline)** | **0.0 ± 0.0** |

`agreement_gate` and `control_none` **improved nothing** on either monitor or test sets.

---

### Failure Mode 5A: Agreement Verifier Bottleneck (`agreement_gate`)
- **What Happened**: In all 3 seeds, the gate rejected rounds 1 and 2, triggering early stopping via patience (`n_rejected_rounds = 2.0`, `n_pseudo_labels = 0`, `rec_summary.md`).
- **Verifier Precision**: Random Forest precision was **95.16% ± 0.60%** on selected samples (`rec_findings.md`).
- **Likely Cause**: The Random Forest operated on flattened 784-dimensional raw pixels (`src/paradigms/recursive/verifiers.py::AgreementVerifier.fit`) trained on only 2,005 samples. Without convolutional spatial invariance, the Random Forest agreed with the CNN only on prototypical, easy samples. Candidates trained on these samples achieved tiny gains (+0.02% to +0.08%) that fell below the gate's `min_improvement=0.001` threshold or slightly perturbed minority classes.

---

### Failure Mode 5B: The Gating Paradox in `selfconf_gate`
- **What Happened**: `selfconf_gate` accepted only 1 round in seed 1, and 0 rounds in seeds 0 and 2 (`rec_summary.md`). Final test gain was +0.08% ± 0.12% (statistical TIE).
- **Likely Cause**: Self-confidence verification admits label noise (precision = 93.09%, `rec_findings.md`). When the candidate model was retrained on noisy pseudo-labels, at least one minority class suffered an accuracy drop exceeding `max_class_drop=0.05` on the gate set (`src/paradigms/recursive/gate.py::decide_gate`). The gate correctly rejected these candidates to prevent collapse, but because the generator could not improve its own precision, the loop deadlocked.

---

### Failure Mode 5C: Unchecked Semantic Drift in `selfconf_nogate`
- **What Happened**: `selfconf_nogate` registered a +1.08% ± 0.54% test gain (Verdict: WIN, `experiments/final/recursive_test.json`), but suffered severe structural damage:
  1. **Compounding Label Noise**: Selected precision fell each round: 93.83% (R1) $\rightarrow$ 92.46% (R2) $\rightarrow$ 90.47% (R3) $\rightarrow$ 88.55% (R4). Final training label noise reached **8.66% ± 0.26%** (`rec_findings.md`).
  2. **Catastrophic Class Collapse**: Worst-class accuracy change was **-11.36% ± 5.84%** across seeds (`rec_summary.md`). On seed 0, Class 4 (Coat) collapsed by **-19.00%** (`experiments/recursive/selfconf_nogate/seed0.json`).
  3. **Diminishing Returns**: Round 1 gain was +0.63% ± 1.00%, while rounds 2-4 combined produced only +0.30% ± 0.49% (`rec_findings.md`).
- **Likely Cause**: Without gate enforcement, the model self-reinforced its own high-confidence errors on ambiguous garment categories (Coats, Shirts, Pullovers), sacrificing difficult semantic boundaries.

---

## 6. Process Failures and Retrospective

### Process Failure 1: Premature Final Test Evaluation in Phase 1 v1
- **What Happened**: The official protected test set was evaluated in commit `e52f5a4` (`tag phase1-final`, `experiments/final/phase1_test.json`) before checking convergence.
- **Why It Was a Process Failure**: Good scientific hygiene requires freezing test evaluation until all hyperparameter exploration is complete. Evaluating the test set on an under-trained 8-epoch model wasted an official evaluation.
- **What We Changed**: In Phase 1 v2, official test access remained strictly locked (`scripts/final_test_phase1_v2.py` was NOT RUN). In Phase 2, the test set was touched strictly once after all code, tables, and reports were committed (`scripts/final_test_recursive.py`).

### Process Failure 2: Hardware-Driven Compute Constraints
- **What Happened**: In Phase 2, execution was restricted to 4 rounds and 20,000 pool samples.
- **Why It Occurred**: CPU execution without GPU acceleration mandated keeping total experiment run times under 120 minutes. Across 15 runs, total compute was 609.3 seconds (10.15 minutes).
- **Hypothesis (not tested)**: Increasing rounds to 10 or enlarging the pool to all 48,000 remaining samples might enable `oracle_gate` to achieve gains exceeding +3.5%, but would exacerbate noise collapse in `selfconf_nogate`.
