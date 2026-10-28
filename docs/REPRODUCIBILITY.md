# Reproducibility Guide & Experimental Audit

This document details the exact environment, deterministic random seeds, dataset partition manifests, and security audit outputs required to independently reproduce all findings of the `ai-paradigms-lab`.

---

## 1. Dataset Source & Preprocessing

- **Dataset**: Fashion-MNIST (Xiao et al., 2017).
- **Source Package**: `torchvision.datasets.FashionMNIST(root="./data", download=True)`.
- **Image Dimensions**: $28 \times 28$ grayscale pixels ($1 \times 28 \times 28$).
- **Preprocessing & Normalization**:
  $$x_{\text{norm}} = \frac{x / 255.0 - \mu}{\sigma}, \quad \mu = 0.2860, \quad \sigma = 0.3530$$
  Normalized tensors are cached contiguously in memory via `src/paradigms/data/tensor_loader.py::TensorLoader`.

---

## 2. Dataset Partition Manifest

All splits are generated deterministically by `scripts/verify_splits.py` using `split_seed=42`. Partitions are strictly disjoint and verified by SHA-256 index hashing (`docs/split_manifest.json`):

| Split Name | Sample Count | Class Stratification | Scope / Permitted Usage | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| `train` | 50,000 | 5,000 per class | Training base for Phase 1 teacher and student | `2aa19d8c872458d0...` |
| `dev` | 10,000 | 1,000 per class | Model selection and hyperparameter tuning | `21d3012417c9b2fc...` |
| `gate` | 4,997 | 500 or 499 per class | Phase 2 candidate acceptance decisions (within dev) | `cf6ae10381946b94...` |
| `monitor` | 5,003 | 500 or 501 per class | Phase 2 passive generalization tracking (within dev) | `287f908038d45c77...` |
| `test` | 10,000 | 1,000 per class | Protected official test set (one-shot access only) | `a74675bfed3834ac...` |

### Phase 2 Disjoint Partitions (Per Seed)

| Seed | Initial Labeled Count | Labeled SHA-256 | Unlabeled Pool Count | Pool SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| `seed0` | 2,005 | `ddafb45e...` | 20,000 | `b94cb7f3...` |
| `seed1` | 2,005 | `2dfe4384...` | 20,000 | `4a0e14da...` |
| `seed2` | 2,005 | `a3a6a2b3...` | 20,000 | `d9a759ae...` |

---

## 3. Random Seeds Inventory

All random seeds are fixed and isolated across phases:

| Phase / Component | Seeds List | Source Configuration / Script | Purpose |
| :--- | :--- | :--- | :--- |
| **Split Generation** | `42` | `configs/base.yaml` | Fixes train/dev/gate/monitor partition boundaries |
| **Phase 1 v1 Teacher** | `[0, 1, 2, 3, 4]` | `configs/base.yaml` | Multi-seed teacher ensemble initialization |
| **Phase 1 v1 Student** | `[0, 1, 2, 3, 4]` | `configs/base.yaml` | Matched hard vs distilled student comparison |
| **Phase 1 v2 Final Eval** | `[10, 11, 12, 13, 14]` | `scripts/run_v2_experiments.py` | Fresh unseen seeds for final label-efficient evaluation |
| **Phase 2 Recursive Loop** | `[0, 1, 2]` | `configs/recursive.yaml` | Multi-seed evaluation across 5 recursive loop configs |

---

## 4. Determinism & Hardware Specifications

1. **CPU-Only Execution**: All experiments execute strictly on CPU hardware (`device="cpu"`). This eliminates non-deterministic atomic reductions common to CUDA GPU kernels.
2. **Single-Threaded PyTorch**: In student training, `torch.set_num_threads(1)` is explicitly enforced. Beyond providing a 5.2x speedup on CPU, single-thread execution guarantees deterministic sequential batch accumulation without multi-core thread barrier variations.
3. **Seeding Protocol**: Every training loop invokes:
   ```python
   torch.manual_seed(seed)
   np.random.seed(seed)
   ```
4. **Floating-Point Note**: While random number generators and indices are 100% bit-exact across platforms, slight variations at the 7th decimal place of loss values can occur across differing CPU architectures (e.g. Intel AVX2 vs AMD AVX-512) due to SIMD compiler floating-point associativity optimizations.

---

## 5. Software Environment (`requirements.lock.txt`)

All experiments were executed on Python 3.12 (Linux x86_64). The pinned package versions are:

| Package | Version | Role |
| :--- | :--- | :--- |
| `torch` | `2.14.1+cpu` | Neural network computation and CPU autograd |
| `torchvision` | `0.29.1+cpu` | Fashion-MNIST dataset loader |
| `numpy` | `2.5.2` | Vectorized numerical operations and data partitioning |
| `scipy` | `1.14.1` | Statistical significance testing (paired t-test) |
| `pandas` | `3.1.0` | Table data serialization |
| `matplotlib` | `3.11.2` | Headless plot rendering (Agg backend) |
| `pyyaml` | `6.0.3` | YAML configuration loading |
| `tqdm` | `4.70.1` | Progress bar visualization |
| `scikit-learn` | `1.9.1` | Random Forest verifier and stratification |
| `pytest` | `9.1.1` | Automated test suite execution |

---

## 6. Static Test Set Access Audit

To verify that the official test partition is protected from data snooping, `tests/test_test_set_access.py` performs static AST analysis across all Python files in the repository.

### Audit Result
```text
$ pytest tests/test_test_set_access.py -v
============================= test session starts ==============================
collected 1 item

tests/test_test_set_access.py::test_test_set_access_is_strictly_confined PASSED [100%]
============================== 1 passed in 0.77s ===============================
```

### Audit Findings
- Invocations of `allow_test=True` are strictly confined to:
  1. `scripts/final_test_phase1.py` (Phase 1 v1 test evaluation)
  2. `scripts/final_test_phase1_v2.py` (Phase 1 v2 test script, NOT RUN)
  3. `scripts/final_test_recursive.py` (Phase 2 recursive test evaluation)
  4. `tests/test_data.py` (Unit tests verifying the PermissionError guard)
- Zero training, sweep, tuning, or reporting scripts access the protected test partition.
