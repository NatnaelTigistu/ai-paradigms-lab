# Presentation Outline: Knowledge Distillation & Recursive Self-Improvement

This document provides a slide-by-slide outline for a 20-minute technical presentation, accompanied by a 3-minute live demonstration script.

---

## Slide Deck Outline (10 Slides, ~20 Minutes)

### Slide 1: Title & The Core Dilemma
- **Title**: Beyond Static Supervision: Knowledge Distillation and Bounded Recursive Self-Improvement
- **Bullet Points**:
  - Deploying high-capacity neural networks to latency-critical edge environments is constrained by parameter size and compute budgets.
  - Annotation is expensive, but naive self-training on unlabeled data triggers runaway confirmation bias.
  - We evaluate two complementary paradigms on Fashion-MNIST: Knowledge Distillation (Phase 1) and Bounded Recursive Self-Training (Phase 2).
  - All experiments are executed under strict CPU-only constraints with fixed seeds and zero data leakage.
- **Figure to Show**: `reports/figures/fig_main_bars.png`
- **Speaker Notes**:
  "Welcome everyone. Today we examine two fundamental approaches for maximizing model efficiency: transferring knowledge from large to small models, and recursively bootstrapping models from unlabeled data without human intervention."

---

### Slide 2: Experimental Hygiene & The 5-Way Partition Protocol
- **Title**: Rigorous Partitioning: Guarding Against Leakage and Confirmation
- **Bullet Points**:
  - 50,000 training images partitioned into 2,005 labeled seed samples and 20,000 unlabeled pool candidates.
  - Development pool split into 4,997 Gate samples (decision-making) and 5,003 Monitor samples (passive tracking).
  - The official 10,000-sample test set is locked behind one-shot evaluation guards (`tests/test_test_set_access.py::test_test_set_access_is_strictly_confined`).
  - Disjointness verified deterministically via SHA-256 manifest hashing (`scripts/verify_splits.py::verify_and_generate_manifest`).
- **Figure to Show**: Architecture diagram or manifest table from `reports/RESULTS.md`
- **Speaker Notes**:
  "Scientific integrity requires that models never evaluate on data they used for decisions. We enforce a 5-way partition where gate decisions and monitor diagnostics are kept completely disjoint from training."

---

### Slide 3: Phase 1 v1: Why Naive Distillation Failed
- **Title**: The Baseline Pitfall: When Hard-Label Training Beats Distillation
- **Bullet Points**:
  - Initial 8-epoch training resulted in hard student (87.34%) outperforming distilled student (86.85%, -0.49% diff, 0/5 wins).
  - Premature official test evaluation (`experiments/final/phase1_test.json`) confirmed the deficit (+0.20% marginal diff).
  - Root cause: Models were severely under-trained at 8 epochs, and fixed temperature $T=4.0$ diluted gradients at 100% data.
  - Key lesson: Compression requires verified convergence and independent hyperparameter tuning.
- **Figure to Show**: `reports/figures/fig_main_bars.png`
- **Speaker Notes**:
  "In our first phase, the hard-label student beat the distilled student across every single seed. Under-training and an arbitrary temperature setting completely erased the theoretical benefits of dark knowledge."

---

### Slide 4: Phase 1 v2: Label-Efficient Distillation Across Regimes
- **Title**: The Label-Efficient Breakthrough: When Dark Knowledge Dominates
- **Bullet Points**:
  - At 2% labels (1,000 samples), distillation delivers a **+1.61% ± 0.19% gain** (82.48% vs 80.87%, 5/5 wins).
  - Distilled Transfer across all 50k images achieves **87.02% ± 0.11% (+6.15% gain)**.
  - At 10% labels (5,000 samples), distillation provides **+0.98% ± 0.22% gain** (85.99% vs 85.01%).
  - At 100% labels (50,000 samples), benefit shrinks to **+0.20% ± 0.17%** ($p = 0.0810$, not statistically significant).
- **Figure to Show**: `reports/v2/figures/fig_v2_fraction.png`
- **Speaker Notes**:
  "When we re-architected Phase 1 with full convergence and fair tuning, distillation shone brilliantly in low-data regimes. However, as ground-truth labels increase, the marginal advantage of teacher dark knowledge drops toward zero."

---

### Slide 5: Optimization Engineering (The 5.2x CPU Speedup)
- **Title**: Systems Engineering: Achieving Feasibility on CPU Hardware
- **Bullet Points**:
  - Baseline student training required 168 minutes across 124 experimental runs, exceeding compute limits.
  - Pre-computing teacher logits in memory (`src/paradigms/distillation/cached.py::cache_teacher_logits`) eliminated 11,730 redundant forward passes per run.
  - Restricting PyTorch to single-thread execution (`torch.set_num_threads(1)`) removed thread synchronization bottlenecks.
  - Result: 30-epoch student runtime dropped from 120.8s to 18.0s, reducing total study time to 42.6 minutes.
- **Figure to Show**: `reports/v2/figures/fig_v2_tuning.png`
- **Speaker Notes**:
  "Algorithmic research is constrained by engineering reality. By caching teacher logits and eliminating CPU thread contention, we unlocked a 5.2x speedup that made rigorous coordinate descent tuning feasible."

---

### Slide 6: Phase 2: The Bounded Recursive Loop Architecture
- **Title**: Closed-Loop Self-Improvement: Generation, Verification, and Rollback
- **Bullet Points**:
  - 105,866-parameter SmallCNN trained on 2,005 labeled samples, iteratively expanding over 20,000 pool samples.
  - Class-capped curation (`cap_per_class=300`) prevents runaway easy-class dominance.
  - Monotonic dataset expansion anchors the model with fixed pseudo-labels to prevent cyclical flip-flops.
  - Dual-metric Acceptance Gate (`src/paradigms/recursive/gate.py::decide_gate`): overall gain $\ge +0.001$, no class drop $> 0.05$.
- **Figure to Show**: Data flow diagram from `docs/RECURSIVE_LOOP_GUIDE.md`
- **Speaker Notes**:
  "To safely bootstrap from unlabeled data, we implemented an automated acceptance gate. If a candidate model degrades even a single class by more than 5 percent, it is discarded and rolled back to the previous stable state."

---

### Slide 7: Recursive Trajectories: The Oracle Ceiling vs Reality
- **Title**: The Verifier Hierarchy: Clean Ceilings and Flat Baselines
- **Bullet Points**:
  - `oracle_gate`: Monotonic climb from 83.89% to **86.08% ± 0.26% test accuracy (+2.19% gain, 3/3 wins)**.
  - `control_none`: Stays flat at 83.89% (0 rounds accepted), proving gains require data expansion, not extra training.
  - `agreement_gate`: 0 accepted rounds (+0.00% gain, TIE) despite 95.16% precision from Random Forest verifier.
  - `selfconf_gate`: Rejection safeguards prevent degradation (+0.08% test gain, TIE).
- **Figure to Show**: `reports/recursive/figures/fig_rec_curves.png`
- **Speaker Notes**:
  "The Oracle configuration proves that recursive self-improvement works remarkably well when verification is perfect. But independent verifiers face steep representational challenges when operating on complex visual data."

---

### Slide 8: The Confirmation Bias Trap in Naive Self-Training
- **Title**: The Hidden Collapse: How `selfconf_nogate` Sacrificed Fine-Grained Classes
- **Bullet Points**:
  - `selfconf_nogate` achieved a deceptive **+1.08% ± 0.54% global test gain** (84.97% vs 83.89%, 3/3 wins).
  - Selected precision degraded from 93.83% in Round 1 to 88.55% in Round 4; label noise reached **8.66% ± 0.26%**.
  - **Catastrophic Class Collapse**: Class 4 (Coat) accuracy collapsed by **-11.36% ± 5.84%** (down -19.00% on seed 0).
  - Diminishing returns: Round 1 gain (+0.63%) exceeded Rounds 2–4 combined (+0.30%).
- **Figure to Show**: `reports/recursive/figures/fig_rec_perclass.png`
- **Speaker Notes**:
  "This is the central cautionary finding of our research. Naive self-training showed a global accuracy win, but internally it hallucinated labels and suffered an 11% collapse on difficult garment classes."

---

### Slide 9: Acceptance Gating & Representation Bottlenecks
- **Title**: The Safety Shield & The Representation Gap
- **Bullet Points**:
  - `selfconf_gate` successfully arrested runaway noise, keeping training label noise down to 2.17% (vs 8.66% in no-gate).
  - `agreement_gate` bottleneck: Random Forest operating on raw flat pixels lacked spatial convolutional invariance.
  - High precision (95.16%) was confined to easy, prototypical examples; marginal gains failed the gate's +0.1% threshold.
  - Gate-to-monitor gap remained tightly bounded (-0.08% in Oracle, +0.20% in No-Gate), confirming zero gate overfitting.
- **Figure to Show**: `reports/recursive/figures/fig_rec_accept.png`
- **Speaker Notes**:
  "Our acceptance gate worked as intended, halting corrupted candidates before they could degrade the system. However, an independent verifier must share spatial representations with the generator to provide useful corrective signal."

---

### Slide 10: Retrospective & Synthesis for Phase 3
- **Title**: Key Takeaways and Roadmap for Phase 3
- **Bullet Points**:
  - **Takeaway 1**: Distillation is most powerful in low-data regimes; transfer learning yields massive +6.15% gains.
  - **Takeaway 2**: Self-training without acceptance gating produces dangerous confirmation bias and semantic drift.
  - **Phase 3 Roadmap**: Synthesize both paradigms by distilling recursively improved teachers into compact students.
  - **Key Adjustment**: Replace hard pseudo-labels with soft temperature-scaled KL targets to absorb prediction uncertainty.
- **Figure to Show**: `reports/v2/figures/fig_v2_fraction.png`
- **Speaker Notes**:
  "In Phase 3, we will combine these paradigms: using recursive self-improvement to build a superior teacher, and knowledge distillation to transfer its dark knowledge safely into efficient edge models. Thank you."

---

## 3-Minute Live Demonstration Script

### Setup (Before Starting)
Ensure virtual environment is active in the repository root:
```bash
source .venv/bin/activate
```

---

### Minute 1: Verify Partitions & Non-Leakage (0:00 - 1:00)
**Speaker**: *"First, let's verify that all dataset partitions are strictly disjoint and mathematically reproducible."*

**Command to execute**:
```bash
python scripts/verify_splits.py
```
**Expected Output**:
- Displays dataset split sizes (50k train, 10k dev, 10k test).
- Prints deterministic SHA-256 partition hashes.
- Confirms disjointness assertions across all seeds.

---

### Minute 2: Validate Documentation References & Integrity (1:00 - 2:00)
**Speaker**: *"Next, we run our static documentation validator to confirm that every code symbol and relative link in our reports exists on disk."*

**Command to execute**:
```bash
python scripts/check_docs_refs.py
```
**Expected Output**:
- Scans all markdown files in `docs/` and `reports/`.
- Confirms: `SUCCESS: All symbol references (path::name) and relative links are valid.`

---

### Minute 3: Generate Consolidated Deliverables & Inspect Results (2:00 - 3:00)
**Speaker**: *"Finally, we compile our unified results document from disk artifacts and inspect official test numbers."*

**Commands to execute**:
```bash
python scripts/make_deliverables.py
head -n 45 reports/RESULTS.md
```
**Expected Output**:
- Generates `reports/RESULTS.md`.
- Displays consolidated tables: manifest partitions, Phase 1 v2 label-efficient results (+1.61% gain at 2% data), and Phase 2 recursive loop test outcomes (`oracle_gate` +2.19% gain).
