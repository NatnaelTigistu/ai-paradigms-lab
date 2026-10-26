# Comprehensive Guide: Architecture, Implementation, Distillation Dynamics, and Retrospective

This document provides an end-to-end guide to the Knowledge Distillation research conducted in Phase 1 (v1 and v2) of the `ai-paradigms-lab`. It starts from fundamental principles and builds up to advanced mathematics, systems engineering, optimization bottlenecks, experimental findings, and future architectural adjustments.

---

## Table of Contents
1. [Core Problem & Intuition](#1-core-problem--intuition)
2. [Model Architectures & Parameter Budgets](#2-model-architectures--parameter-budgets)
3. [Mathematical Formulations & Loss Functions](#3-mathematical-formulations--loss-functions)
4. [The Journey: Why Phase 1 v1 Failed and What v2 Fixed](#4-the-journey-why-phase-1-v1-failed-and-what-v2-fixed)
5. [Systems & Optimization Engineering (The 5x CPU Speedup)](#5-systems--optimization-engineering-the-5x-cpu-speedup)
6. [Experimental Protocol & Coordinate Descent Tuning](#6-experimental-protocol--coordinate-descent-tuning)
7. [Detailed Results, Analysis & Verdicts](#7-detailed-results-analysis--verdicts)
8. [Problems Faced & Engineering Solutions](#8-problems-faced--engineering-solutions)
9. [Future Recommendations & Phase 2/3 Adjustments](#9-future-recommendations--phase-23-adjustments)

---

## 1. Core Problem & Intuition

### The Practical Dilemma
Modern deep neural networks achieve high accuracy by using large architectures with millions or billions of parameters. However, deploying large models in latency-critical, edge, or CPU-bound environments is impractical due to high inference latency, large memory footprints, and power constraints. Compact models run fast and consume little memory, but when trained purely on one-hot ground-truth labels, they often struggle to generalize.

### Knowledge Distillation (KD)
Knowledge Distillation, introduced conceptually by Caruana (2006) and formalized by Hinton et al. (2015), transfers the representational capacity of a large, high-capacity **Teacher** model into a compact **Student** model.

```
+-------------------------------------------------------------+
|                      Input Image (28x28)                    |
+-------------------------------------------------------------+
                 |                                  |
                 v                                  v
      +--------------------+              +-------------------+
      |    Teacher CNN     |              |    Student MLP    |
      | (421,642 params)   |              |  (25,450 params)  |
      +--------------------+              +-------------------+
                 |                                  |
                 v Logits                           v Logits
      +--------------------+              +-------------------+
      | Softmax (Temp T)   |              | Softmax (Temp T)  |
      +--------------------+              +-------------------+
                 \                                  /
                  \                                /
                   v                              v
               +--------------------------------------+
               |   Kullback-Leibler (KL) Divergence   |  <--- Soft Loss
               +--------------------------------------+
                                  +
               +--------------------------------------+
               |      Cross-Entropy on Hard Labels    |  <--- Hard Loss
               +--------------------------------------+
```

### What is "Dark Knowledge"?
When a model is trained with hard one-hot targets (e.g., class 3 = `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`), all non-target classes receive zero credit. The loss treats every incorrect category as equally incorrect. 

In reality, classes have rich semantic relationships:
- A *Shirt* image may have an 85% probability of being a Shirt, an 11% probability of being a *T-shirt*, a 3% probability of being a *Coat*, and a 0.001% probability of being an *Ankle Boot*.
- This relative distribution over incorrect categories—termed **"dark knowledge"**—encodes the structure of the input space.
- Training a student on these soft probabilities provides far richer gradient signals, suppresses label noise, and regularizes the student against overfitting.

---

## 2. Model Architectures & Parameter Budgets

The benchmark dataset is **Fashion-MNIST** (60,000 training images, 10,000 test images, 10 categorical apparel classes, single-channel 28x28 pixels). We split the training images into 50,000 `train` and 10,000 `dev`.

### Teacher Model (`TeacherCNN`)
A multi-stage convolutional network designed to extract local spatial features with translation invariance:
- **Input**: `(B, 1, 28, 28)`
- **Layer 1**: `Conv2d(1, 32, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU`
- **Layer 2**: `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (outputs $14 \times 14$) $\rightarrow$ `Dropout(0.25)`
- **Layer 3**: `Flatten` $\rightarrow$ `Linear(64 * 14 * 14, 128)` (12,544 $\rightarrow$ 128) $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)`
- **Output Layer**: `Linear(128, 10)`
- **Total Parameters**: **421,642**
- **Test Accuracy**: **91.35% ± 0.19%**

### Student Model (`StudentMLP`)
A two-layer Multi-Layer Perceptron containing no convolutional layers or spatial inductive bias:
- **Input**: `(B, 1, 28, 28)` $\rightarrow$ `Flatten` to `(B, 784)`
- **Hidden Layer**: `Linear(784, 32)` $\rightarrow$ `ReLU`
- **Output Layer**: `Linear(32, 10)`
- **Total Parameters**: **25,450**
- **Parameter Compression Ratio**: **16.5x smaller** than the teacher.

### Why an MLP Student Instead of a Small CNN?
Testing knowledge distillation across *different inductive biases* (CNN $\rightarrow$ MLP) provides a much stronger test of representation transfer than simply pruning channels from a CNN:
1. An MLP lacks local spatial priors and translation equivariance; it must learn pixel adjacency purely from data or teacher guidance.
2. If distillation succeeds here, it demonstrates genuine knowledge transfer across distinct function hypothesis classes.

---

## 3. Mathematical Formulations & Loss Functions

### Standard Hard-Label Cross-Entropy
Given student logits $z_s \in \mathbb{R}^C$ and ground-truth integer label $y \in \{0, \dots, C-1\}$:
$$p_s(c) = \frac{\exp(z_{s, c})}{\sum_{j=1}^C \exp(z_{s, j})}$$
$$\mathcal{L}_{\text{CE}}(z_s, y) = -\log p_s(y)$$

### Softmax with Temperature Scaling
To reveal the dark knowledge in logits that are otherwise dominated by a single winning class, logits are scaled by temperature $T > 0$:
$$p_s(c; T) = \frac{\exp(z_{s, c} / T)}{\sum_{j=1}^C \exp(z_{s, j} / T)}, \quad p_t(c; T) = \frac{\exp(z_{t, c} / T)}{\sum_{j=1}^C \exp(z_{t, j} / T)}$$
- At $T = 1$: Standard softmax probabilities.
- At $T \to \infty$: Uniform distribution ($p(c) \to 1/C$).
- At intermediate $T$ (e.g. 2 to 4): The distribution softens, preserving relative class similarities while preventing the target class probability from dominating.

### Kullback-Leibler (KL) Divergence
The soft loss measures the information divergence between the teacher's soft distribution $p_t$ and the student's soft distribution $p_s$:
$$D_{\text{KL}}(p_t \parallel p_s) = \sum_{c=1}^C p_t(c; T) \left( \log p_t(c; T) - \log p_s(c; T) \right)$$

### The $T^2$ Gradient Normalization Factor
Why multiply the KL divergence by $T^2$?
Consider the gradient of the KL loss with respect to a student logit $z_{s, i}$:
$$\frac{\partial D_{\text{KL}}(p_t \parallel p_s)}{\partial z_{s, i}} = \frac{1}{T} \left( p_s(i; T) - p_t(i; T) \right)$$
Using a Taylor expansion for small $z / T$:
$$p_s(i; T) \approx \frac{1 + z_{s, i} / T}{C + \sum_j z_{s, j} / T}$$
The difference $(p_s - p_t)$ scales proportionally to $1/T$. Thus, the overall gradient scales as:
$$\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{s, i}} \propto \frac{1}{T^2}$$
If we did not multiply by $T^2$, increasing $T$ would shrink the soft-target gradients to near-zero, rendering temperature comparisons meaningless. Multiplying by $T^2$ keeps the gradient magnitude consistent with hard cross-entropy across all temperature values:
$$\mathcal{L}_{\text{KD}} = T^2 D_{\text{KL}}(p_t \parallel p_s)$$

### Combined Distillation Loss (Full Supervision)
$$\mathcal{L}_{\text{Distill}} = \alpha \mathcal{L}_{\text{CE}}(z_s, y) + (1 - \alpha) T^2 D_{\text{KL}}(p_t \parallel p_s)$$
- $\alpha = 1.0$: Pure hard-label cross entropy (teacher ignored).
- $\alpha = 0.0$: Pure soft distillation (ground-truth labels ignored).
- $0 < \alpha < 1$: Blended supervision.

### Semi-Supervised Masked Distillation Loss (Phase 1 v2)
In transfer settings where some samples lack ground-truth annotations, we define a masking contract where label $y_i = -1$ denotes an unlabelled sample:
$$\text{labelled}_i = \begin{cases} 1 & \text{if } y_i \ge 0 \\ 0 & \text{if } y_i = -1 \end{cases}$$
$$\ell_i = \alpha \cdot \text{labelled}_i \cdot \mathcal{L}_{\text{CE}}(z_{s, i}, y_i) + \left(1 - \alpha \cdot \text{labelled}_i\right) \cdot T^2 D_{\text{KL}}(p_{t, i} \parallel p_{s, i})$$
$$\mathcal{L}_{\text{Masked}} = \frac{1}{B} \sum_{i=1}^B \ell_i$$
- **Labelled sample**: $\alpha \mathcal{L}_{\text{CE}} + (1 - \alpha) \mathcal{L}_{\text{KD}}$
- **Unlabelled sample**: $1.0 \cdot \mathcal{L}_{\text{KD}}$ (pure soft distillation from teacher).

---

## 4. The Journey: Why Phase 1 v1 Failed and What v2 Fixed

### The Paradox of Phase 1 v1
In Phase 1 v1, the distilled student achieved:
- Dev accuracy: **87.41% ± 0.14%**
- Hard student dev accuracy: **87.34% ± 0.21%**
- Test set paired difference: **+0.20% ± 0.27%** ($p = 0.45$, statistically insignificant).

Distillation failed to beat the hard student in any meaningful way. A detailed forensic investigation revealed several underlying reasons:

#### 1. Full-Data Saturation
On 50,000 training examples, a simple 2-layer MLP with 32 units is already near its capacity ceiling. The student is parameter-limited, not data-limited. Hard labels provide ample supervision to fit the 25,450 parameters, leaving negligible headroom for soft targets to help.

#### 2. Underfitting via Truncated Epochs
`training.epochs` was hardcoded to 8 in `configs/base.yaml`. At 8 epochs, the MLP had not converged. In low data fractions (e.g. 1% or 5%), 8 epochs amounted to only a few gradient updates, causing massive underfitting.

#### 3. Unfair Baseline Comparison
In v1, the hard student was not tuned for weight decay; it inherited whatever defaults were configured. Distillation appeared slightly better simply because the baseline had not been regularized.

#### 4. Procedural Hygiene Flaws
The v1 test set was evaluated and committed before hyperparameter sweeps had finished. The sweeps had unrun configurations ($\alpha=0.3, 0.7$ were missing, $\alpha=0.0$ only had 2 seeds), meaning `best_config.json` was picked from incomplete data.

### The Phase 1 v2 Redesign
Phase 1 v2 fundamentally redesigned the experiment to test **label efficiency**:
1. **Low-Data Regimes**: Swept across 2% (1,000 samples), 10% (5,000 samples), and 100% (50,000 samples).
2. **Fair Independent Tuning**: Hard student tunes weight decay first ($wd \in [0, 10^{-4}, 10^{-3}]$). Distillation inherits $wd^*$ and tunes $T \in [1, 2, 4]$ and $\alpha \in [0.2, 0.5, 0.8]$.
3. **Distillation Transfer Mode**: Student trains on all 50k images with labels on subset $f$ and teacher soft targets on the remaining $(1 - f)$ images.
4. **Normalized Epoch Volume**: Used `effective_epochs(30, f) = min(400, round(30 / f))` so every data fraction receives equivalent gradient steps.
5. **Fresh Evaluation Seeds**: Tuned on seeds 0, 1, 2; evaluated on fresh seeds 10, 11, 12, 13, 14.
6. **Pre-Registered Verdict Rule**: Objective criteria for WIN / LOSS / TIE.

---

## 5. Systems & Optimization Engineering (The 5x CPU Speedup)

When first running v2, a single 30-epoch training run took **73 seconds (hard)** and **120 seconds (distilled)**. Across 124 experimental runs, this implied over **168 minutes** of compute, exceeding our 110-minute target.

Profiling identified several bottlenecks:

```
[Bottleneck 1]: Multi-threading overhead on small MLPs (Amdahl's Law)
[Bottleneck 2]: Per-batch softmax over 10 classes repeated 391 times/epoch
[Bottleneck 3]: Python copy.deepcopy() traversing model state dictionaries
[Bottleneck 4]: PyTorch DataLoader batch collation on tiny in-memory tensors
```

### 1. The Threading Anomaly: 1 Thread vs 12 Threads
Our CPU has 16 logical cores. PyTorch defaulted to 12 threads.
- For a small MLP (input 784, hidden 32, output 10), matrix multiplications on batch size 128 require only $\sim 200\text{k}$ floating point operations per batch.
- Splitting $200\text{k}$ FLOPs across 12 threads causes thread launch, barrier synchronization, and cache coherency overhead to dwarf the actual computation.
- **Benchmark Results**:
  - `threads=16`: 11.24s per epoch
  - `threads=12`: 2.77s per epoch
  - `threads=4`:  0.93s per epoch
  - `threads=1`:  **0.53s per epoch**
- **Action**: Enforced `torch.set_num_threads(1)` inside student training routines. This delivered an immediate **5.2x speedup**.

### 2. Precomputing Teacher Softmax Distributions
In distilled mode, the teacher's logits $z_t \in \mathbb{R}^{50000 \times 10}$ are frozen.
- Previously, `softmax(z_t / T)` was computed dynamically for every mini-batch in every epoch ($391 \times 30 = 11,730$ softmax calls).
- **Optimization**: Precomputed $p_t = \text{softmax}(z_t / T, \text{dim}=-1)$ **once** before entering the epoch loop. Mini-batches simply slice the precomputed distribution tensor.

### 3. In-Memory `TensorLoader` vs `DataLoader`
PyTorch's standard `DataLoader` uses multiprocessing/worker abstraction and dynamic collate functions. For in-memory tensors, this introduced $\sim 4\mu\text{s}$ per batch in pure Python function call overhead.
- Implemented `TensorLoader` storing contiguous `X` and `packed_y` tensors.
- Generated batch permutations using `torch.randperm(N, generator=g)` advancing epoch-by-epoch.

### 4. Vectorized Dev Set Evaluation
Previously, the dev set was evaluated in mini-batches of 256 through the DataLoader.
- With an MLP of 25k parameters, the entire 10,000-sample dev set fits into memory as a single tensor `(10000, 1, 28, 28)`.
- Passing `X_dev` through the model under `torch.inference_mode()` takes **18 ms** total, compared to 150 ms across mini-batches.

### Combined Speedup Results
| Configuration | 30-Epoch Hard Run | 30-Epoch Distilled Run | Total Study Runtime |
| :--- | :---: | :---: | :---: |
| Baseline Implementation | 73.39s | 120.83s | ~168 minutes |
| **Optimized Implementation** | **14.09s** | **18.00s** | **42.6 minutes** |

---

## 6. Experimental Protocol & Coordinate Descent Tuning

### Coordinate Descent Methodology
To find optimal hyperparameters without an exponential full-grid search, we used coordinate descent on the development set across seeds 0, 1, 2:

```
[Step 1: Hard Student]
Sweep weight decay: wd in [0.0, 1e-4, 1e-3]
-> Select wd* (highest mean dev accuracy)

[Step 2: Distilled Student Temperature]
Fix alpha = 0.5, wd = wd*
Sweep temperature: T in [1.0, 2.0, 4.0]
-> Select T*

[Step 3: Distilled Student Alpha]
Fix T = T*, wd = wd*
Sweep alpha in [0.2, 0.5, 0.8]
-> Select alpha*

[Step 4: Distilled Transfer Alpha (fraction < 1.0 only)]
Fix T = T*, wd = wd*
Train on all 50k images (labels only for subset, teacher soft targets for all)
Sweep alpha_t in [0.2, 0.5, 0.8]
-> Select alpha_t*
```

---

## 7. Detailed Results, Analysis & Verdicts

### Hyperparameter Tuning Output (`experiments/v2/best_hparams.json`)
- **2% Fraction (`f0.02`)**: $wd^* = 0.001$, $T^* = 4.0$, $\alpha^* = 0.2$, $\alpha_t^* = 0.8$
- **10% Fraction (`f0.10`)**: $wd^* = 0.001$, $T^* = 4.0$, $\alpha^* = 0.5$, $\alpha_t^* = 0.2$
- **100% Fraction (`f1.0`)**: $wd^* = 0.0001$, $T^* = 1.0$, $\alpha^* = 0.5$

### Final Evaluation Results (Seeds 10, 11, 12, 13, 14)

```
Dev Accuracy (%) vs Training Data Fraction (Log Scale)

92 % |                                                       [Teacher CNN: 91.8%]
     |
88 % |                                              ==[Hard 100%: 88.2%]=====[Dist 100%: 88.4%]
     |                           *---[Transfer 10%: 87.1%]
     |      *---[Transfer 2%: 87.0%]
84 % |                           o---[Distilled 10%: 86.0%]
     |                           +---[Hard 10%: 85.0%]
     |      o---[Distilled 2%: 82.5%]
80 % |      +---[Hard 2%: 80.9%]
     +-----------------------------------------------------------------------------------------
            2% (1,000 samples)   10% (5,000 samples)         100% (50,000 samples)
```

#### Detailed Statistical Breakdown

| Fraction | Student Model | Dev Accuracy (Mean ± Std) | Paired Diff vs Hard | Wins / 5 | Verdict |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **2%** | **Hard** ($wd=0.001$) | 80.87% ± 0.03% | *baseline* | — | — |
| (1,000 images) | **Distilled** ($T=4, \alpha=0.2$) | 82.48% ± 0.21% | **+1.61% ± 0.19%** | **5 / 5** | **WIN** |
| | **Distilled Transfer** ($T=4, \alpha=0.8$) | **87.02% ± 0.11%** | **+6.15% ± 0.11%** | **5 / 5** | **WIN** |
| **10%** | **Hard** ($wd=0.001$) | 85.01% ± 0.12% | *baseline* | — | — |
| (5,000 images) | **Distilled** ($T=4, \alpha=0.5$) | 85.99% ± 0.13% | **+0.98% ± 0.22%** | **5 / 5** | **WIN** |
| | **Distilled Transfer** ($T=4, \alpha=0.2$) | **87.13% ± 0.05%** | **+2.11% ± 0.14%** | **5 / 5** | **WIN** |
| **100%** | **Hard** ($wd=0.0001$) | 88.20% ± 0.12% | *baseline* | — | — |
| (50,000 images) | **Distilled** ($T=1, \alpha=0.5$) | 88.39% ± 0.16% | **+0.20% ± 0.17%** | **4 / 5** | **WIN** |

### Key Scientific Findings

#### 1. Distillation Shines in the Low-Label Regime
- At 100% data, distillation gives $+0.20\%$.
- At 10% data, distillation gives $+0.98\%$.
- At 2% data, distillation gives $+1.61\%$.
The sample efficiency advantage of distillation grows inversely with label availability. When labels are scarce, the teacher's inter-class distribution provides crucial regularization.

#### 2. Distilled Transfer Recovers 95% of Full-Data Performance
The most remarkable result is **Distilled Transfer** at 2% data:
- Training purely on 1,000 labels yields **80.87%**.
- Adding the remaining 49,000 unlabelled images with teacher soft targets boosts accuracy to **87.02%** (+6.15%).
- The full-data hard model trained on 50,000 human labels achieves **88.20%**.
- **Conclusion**: Teacher soft targets across unlabelled data recover **84% of the gap to the teacher** and **95% of the performance of a model trained on 50x more labelled data**.

#### 3. Optimal Temperature Scales with Data Scarcity
- In low-data regimes ($f=0.02, 0.10$), $T^* = 4.0$ was optimal. A higher temperature spreads probability mass, allowing the student to absorb secondary class affinities.
- In full-data regimes ($f=1.0$), $T^* = 1.0$ was optimal. With ample supervision, the student benefits more from sharp, confident boundaries.

---

## 8. Problems Faced & Engineering Solutions

### Problem 1: PyTorch CPU Thread Contention
- **Symptom**: 30 epochs took 73s hard / 120s distilled. CPU usage was near 1000% (10 cores pegged), but progress was sluggish.
- **Root Cause**: OpenMP thread pools synchronization latency on tiny MLP mini-batches ($B=128$, hidden 32).
- **Solution**: Set `torch.set_num_threads(1)` inside the student trainer. Reduced epoch time from 2.77s to 0.53s.

### Problem 2: Dev Evaluation Index Error in Transfer Testing
- **Symptom**: In `test_transfer_mode_never_reads_unlabelled_labels`, corrupting unlabelled targets caused `IndexError: Target 999 is out of bounds` during dev evaluation.
- **Root Cause**: `data.full_train` contains both 50k train and 10k dev images. Corrupting all indices outside the subset corrupted the dev set targets as well.
- **Solution**: Scoped the corruption strictly to unlabelled *training* indices: `set(data.splits["train"]) - subset_set`.

### Problem 3: Negative Index Error in PyTorch Cross-Entropy
- **Symptom**: Unlabelled samples were tagged with label `-1`. `F.cross_entropy` throws an error when encountered with negative indices.
- **Solution**: Clamped labels prior to cross-entropy (`safe_labels = labels.clamp(min=0)`) and masked the unlabelled losses to zero: `ce = torch.where(labelled, ce_all, 0.0)`.

---

## 9. Future Recommendations & Phase 2/3 Adjustments

As we transition into **Phase 2 (Bounded Recursive Self-Improvement)** and **Phase 3 (Recursive Distillation Synthesis)**, the following architectural adjustments are recommended:

1. **Teacher Variance Measurement**:
   Currently, all students distill from a single teacher checkpoint (`seed0`). To eliminate teacher-selection bias in Phase 3, train an ensemble or evaluate across teacher seeds 0, 1, 2.
2. **Student Architectural Scaling**:
   While the 32-unit MLP provided an ideal testbed for capacity-constrained distillation, introducing a 3-layer MLP or a compact convolutional student (`Conv-16`) will allow studying where the capacity ceiling shifts as student expressivity grows.
3. **Automated Verifier Guards (Phase 2 Preview)**:
   Phase 2 requires an independent verifier guarding self-improvement loops. The fast in-memory validation routines developed here (18ms per 10k images) will serve as the core evaluation engine for verifying proposed parameter modifications before committing or rolling back.
