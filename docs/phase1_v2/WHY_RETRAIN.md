# Why Retrain: Analysis of Phase 1 v1 Limitations and Rationale for v2

## 1. What Phase 1 v1 Showed (Extracted from Stored Artifacts)

### Main Model Comparison (Dev Set)
| Model | Dev Accuracy (Mean ± Std) | Seeds | Parameters |
| :--- | :--- | :---: | :---: |
| Teacher (CNN) | 91.82% ± 0.23% | 5 | 421,642 |
| Hard Student (MLP) | 87.34% ± 0.21% | 5 | 25,450 |
| Distilled Student (MLP, T=1) | 87.41% ± 0.14% | 5 | 25,450 |

### Temperature Sweep (fixed alpha=0.5, hidden=32, fraction=1.0)
| Temperature | Dev Accuracy (Mean ± Std) | Seeds |
| :---: | :--- | :---: |
| T=1 | 87.41% ± 0.14% | 5 |
| T=2 | 87.20% ± 0.13% | 5 |
| T=4 | 86.85% ± 0.12% | 5 |
| T=8 | 86.50% ± 0.14% | 5 |
| T=16 | 86.18% ± 0.12% | 5 |

### Alpha Sweep Audit (fixed T=1, hidden=32, fraction=1.0)
| Alpha Setting | Status | Seed Count | Dev Accuracy (Mean ± Std) |
| :---: | :--- | :---: | :--- |
| alpha=0.0 | PARTIAL (Incomplete) | 2 | 87.06% ± 0.01% |
| alpha=0.3 | MISSING (Never Run) | 0 | MISSING |
| alpha=0.5 | COMPLETE | 5 | 87.41% ± 0.14% |
| alpha=0.7 | MISSING (Never Run) | 0 | MISSING |
| alpha=1.0 | COMPLETE | 5 | 87.34% ± 0.21% |

### Training Duration and Hyperparameter Configuration
- **Training Epochs in `configs/base.yaml`**: `training.epochs = 8`.
- **Teacher Run (`experiments/teacher/seed0.json`)**: `config.training.epochs = 8`.
- **Student Run (`experiments/student_hard/h32_f1.0/seed0.json`)**: `config.training.epochs = 8`.
- **Premature Test Evaluation**: The v1 test evaluation was executed and committed before the alpha, fraction, and size sweeps had completed.
- **Incomplete `best_config.json`**: Selected `T=1, alpha=0.5` without evaluating `alpha=0.3` or `alpha=0.7`, and with only 2 seeds completed for `alpha=0.0`.

## 2. What Phase 1 v2 Changes and Why

1. **Focus on Label-Efficiency**: In full-data Fashion-MNIST (50,000 training images), a standard MLP already nears its capacity limit on hard labels alone (87.34%), leaving little room for standard distillation to show statistically significant gains (+0.07% on dev, p=0.45 on test). In low-label regimes (e.g. 2% or 10% labelled data), representation transfer and teacher guidance can provide substantial, measurable value.
2. **Fair, Independent Tuning for Both Paradigms**: The hard-label baseline student is granted its own weight decay hyperparameter search ($wd \in [0, 10^{-4}, 10^{-3}]$) to ensure it is not penalized by a suboptimal optimizer configuration. Distillation inherits this optimal weight decay and separately tunes temperature $T \in [1, 2, 4]$ and blending factor $\alpha \in [0.2, 0.5, 0.8]$.
3. **Distillation Transfer Mode**: Introduces an explicit transfer setting where the student trains on all 50,000 images, receiving labels only for the chosen subset ($f$) and pure teacher soft targets on the remaining $(1 - f)$ images. This tests whether teacher representations can propagate across unlabelled data.
4. **Sufficient and Normalized Epoch Budgets**: Uses `effective_epochs` (30 base epochs, scaled up to 400 epochs for small fractions) to eliminate the underfitting caused by v1's 8-epoch cap.
5. **Strict Train/Dev/Test Separation**: Hyperparameters are tuned strictly on the development set across seeds 0, 1, 2. The final comparison is performed on five completely fresh seeds (10, 11, 12, 13, 14), with frozen settings.
6. **Pre-Registered Verdict Decision Rule**: Adopts an objective statistical rule ($\text{WIN}$ if mean diff > std and wins $\ge 4/5$) to eliminate subjective post-hoc cherry-picking.
