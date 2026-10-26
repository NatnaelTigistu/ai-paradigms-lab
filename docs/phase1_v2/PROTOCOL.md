# Phase 1 v2 Experimental Protocol

## 1. Experimental Cohorts and Seeds
- **Tuning Seeds**: `0, 1, 2` (used exclusively for coordinate descent hyperparameter tuning on the dev set).
- **Final Evaluation Seeds**: `10, 11, 12, 13, 14` (five fresh seeds evaluated with frozen hyperparameters).
- **Data Subset Seed**: `42` (fixed across all training seeds to ensure deterministic, nested sample partitions).

## 2. Hyperparameter Search Grids
Evaluated independently for each data fraction in `[0.02, 0.10, 1.0]`:
1. **Hard Student Weight Decay**: $wd \in [0.0, 10^{-4}, 10^{-3}]$ (Adam optimizer, $\text{lr}=0.001$). Selects $wd^*$.
2. **Distilled Temperature**: $T \in [1, 2, 4]$ (fixed $\alpha=0.5, wd=wd^*$). Selects $T^*$.
3. **Distilled Alpha**: $\alpha \in [0.2, 0.5, 0.8]$ (fixed $T=T^*, wd=wd^*$). Selects $\alpha^*$.
4. **Transfer Alpha** (fractions $0.02$ and $0.10$ only): $\alpha_t \in [0.2, 0.5, 0.8]$ (fixed $T=T^*, wd=wd^*$). Selects $\alpha_t^*$.

## 3. Student Training Modes & Exact Inputs
| Mode | Input Images | Target Labels | Teacher Soft Targets | Epoch Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Hard Student** | Subset ($f \times 50\text{k}$) | Ground truth class for subset | None | $\text{effective\_epochs}(30, f)$ |
| **Distilled Student** | Subset ($f \times 50\text{k}$) | Ground truth class for subset | Teacher logits scaled by $T$ | $\text{effective\_epochs}(30, f)$ |
| **Distilled Transfer** | All 50,000 images | Subset labels ($f \times 50\text{k}$), `-1` elsewhere | Teacher logits on all 50k images | $30$ base epochs |

> [!NOTE]
> The Teacher model is pre-trained once (seed 0 CNN) on all 50,000 labelled training images. In transfer mode, the student observes unlabelled images paired with teacher soft predictions, testing representation transfer.

## 4. Selection and Decision Rules
- **Model Selection**: Within each run, the best model weights are checkpointed at the epoch achieving the highest accuracy on the development set (ties favor the earlier epoch).
- **Hyperparameter Selection**: The candidate achieving the highest mean dev accuracy across tuning seeds 0, 1, 2 is selected.
- **Verdict Decision Rule (Fixed)**:
  - **WIN**: Paired mean difference $> \text{std}$ and $\text{wins} \ge 4 / 5$.
  - **LOSS**: Paired mean difference $< -\text{std}$ and $\text{wins} \le 1 / 5$.
  - **TIE**: Otherwise.

## 5. Limitations
1. **Single Fixed Teacher**: Uses a single pre-trained teacher checkpoint (seed 0). Teacher variance is not measured.
2. **Dev Set Selection**: All hyperparameter decisions are based on the development set. Test data is strictly locked.
3. **Tuning Seed Sample Size**: 3 seeds provide modest sample size for hyperparameter ranking, but are validated on 5 fresh seeds.
4. **Single Dataset**: Conducted exclusively on Fashion-MNIST grayscale images.

## 6. Deviations for Compute
- **None**: Thanks to CPU optimizations (single-thread execution eliminating thread synchronization overhead, teacher softmax precomputation, in-memory `TensorLoader`, and vectorized dev evaluation), total execution across all 84 tuning runs and 40 final evaluation runs was reduced from ~168 minutes to ~33 minutes.
- As a result, the **original protocol was executed completely unchanged** without dropping any seeds, fractions, or epoch counts.