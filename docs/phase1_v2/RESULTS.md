# Phase 1 v2 Experimental Results: Label-Efficient Distillation Study

## 1. Reference Baselines (Dev Set)
- **Teacher CNN (Seed 0, Full 50k Data)**: 91.82%
- **Phase 1 v1 Hard Student (MLP, Full 50k Data, 8 Epochs)**: 87.34%

## 2. Final Evaluation Tables across Fractions (Fresh Seeds 10, 11, 12, 13, 14)

### Fraction 2% Training Data (`f0.02`)
- **Optimal Hyperparameters**: $wd^*=0.001$, $T^*=4.0$, $\alpha^*=0.2$, $\alpha_t^*=0.8$

| Mode | Hyperparameters | Dev Accuracy (Mean ± Std) | Paired Diff vs Hard | Wins / 5 | Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Hard Student** | $wd=0.001$ | 80.87% ± 0.03% | baseline | - | - |
| **Distilled Student** | $wd=0.001, T=4.0, \alpha=0.2$ | 82.48% ± 0.21% | +1.61% ± 0.19% | 5/5 | **WIN** |
| **Distilled Transfer** | $wd=0.001, T=4.0, \alpha=0.8$ | 87.02% ± 0.11% | +6.15% ± 0.11% | 5/5 | **WIN** |

<!-- TODO(human): Add domain interpretation of results for this fraction -->

### Fraction 10% Training Data (`f0.1`)
- **Optimal Hyperparameters**: $wd^*=0.001$, $T^*=4.0$, $\alpha^*=0.5$, $\alpha_t^*=0.2$

| Mode | Hyperparameters | Dev Accuracy (Mean ± Std) | Paired Diff vs Hard | Wins / 5 | Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Hard Student** | $wd=0.001$ | 85.01% ± 0.12% | baseline | - | - |
| **Distilled Student** | $wd=0.001, T=4.0, \alpha=0.5$ | 85.99% ± 0.13% | +0.98% ± 0.22% | 5/5 | **WIN** |
| **Distilled Transfer** | $wd=0.001, T=4.0, \alpha=0.2$ | 87.13% ± 0.05% | +2.11% ± 0.14% | 5/5 | **WIN** |

<!-- TODO(human): Add domain interpretation of results for this fraction -->

### Fraction 100% Training Data (`f1.0`)
- **Optimal Hyperparameters**: $wd^*=0.0001$, $T^*=1.0$, $\alpha^*=0.5$

| Mode | Hyperparameters | Dev Accuracy (Mean ± Std) | Paired Diff vs Hard | Wins / 5 | Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Hard Student** | $wd=0.0001$ | 88.20% ± 0.12% | baseline | - | - |
| **Distilled Student** | $wd=0.0001, T=1.0, \alpha=0.5$ | 88.39% ± 0.16% | +0.20% ± 0.17% | 4/5 | **WIN** |

<!-- TODO(human): Add domain interpretation of results for this fraction -->

## 3. Key Observations

<!-- TODO(human): Add overarching summary of label efficiency findings and transfer behavior -->
