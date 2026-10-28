# Recursive Self-Improvement Loop Findings

This document summarizes empirical measurements recorded across all five recursive loop configurations evaluated on Fashion-MNIST (seeds 0, 1, 2). All metrics are derived directly from serialized experiment logs without editorializing or value judgments.

---

### 1. Gain of oracle_gate / agreement_gate over control_none
- `oracle_gate`: +2.07% ± 1.17% (wins: 3/3 seeds)
- `agreement_gate`: +0.00% ± 0.00% (wins: 0/3 seeds)

### 2. Pseudo-Label Precision of Each Verifier
  - `oracle_gate`: 1.0000 ± 0.0000 (100.00%)
  - `agreement_gate`: 0.9516 ± 0.0060 (95.16%)
  - `selfconf_gate`: 0.9309 ± 0.0111 (93.09%)
  - `selfconf_nogate`: 0.9133 ± 0.0206 (91.33%)
  - `control_none`: N/A (0 selected pseudo-labels)

### 3. Label Noise Accumulated in Training Set
- `selfconf_nogate`: +8.66% ± 0.26%
- `selfconf_gate`: +2.17% ± 3.07%

### 4. Rounds Rejected by Gate and Computation Time
  - `oracle_gate`: 4 rejected rounds total across seeds, 40.76s total (13.59s per seed)
  - `agreement_gate`: 6 rejected rounds total across seeds, 46.00s total (15.33s per seed)
  - `selfconf_gate`: 6 rejected rounds total across seeds, 45.51s total (15.17s per seed)
  - `control_none`: 6 rejected rounds total across seeds, 22.73s total (7.58s per seed)

### 5. Gate-vs-Monitor Gap in Final Round (Current Model)
  - `oracle_gate`: -0.08%
  - `agreement_gate`: -0.13%
  - `selfconf_gate`: +0.04%
  - `selfconf_nogate`: +0.20%
  - `control_none`: -0.13%

### 6. Monitor Accuracy Gain in Round 1 vs Later Rounds
  - `oracle_gate`: Round 1 gain = +1.05% ± 1.18%; Rounds 2-4 combined gain = +1.01% ± 0.17%
  - `selfconf_nogate`: Round 1 gain = +0.63% ± 1.00%; Rounds 2-4 combined gain = +0.30% ± 0.49%
  - `selfconf_gate`: Round 1 gain = +0.10% ± 0.14%; Rounds 2-4 combined gain = +0.01% ± 0.01%

### 7. Worst-Class Accuracy Change vs Round 0 (Minimum Class Delta)
  - `oracle_gate`: -4.58% ± 1.45%
  - `agreement_gate`: -3.42% ± 1.67%
  - `selfconf_gate`: -4.89% ± 1.51%
  - `selfconf_nogate`: -11.36% ± 5.84%
  - `control_none`: -3.42% ± 1.67%

### 8. Verifier Calls per Accepted Pseudo-Label
  - `oracle_gate`: 8.75 calls per label (70,000 calls / 8,000 labels)
  - `agreement_gate`: undefined (40,000 calls / 0 labels)
  - `selfconf_gate`: 25.45 calls per label (49,498 calls / 1,945 labels)
  - `selfconf_nogate`: 5.29 calls per label (62,253 calls / 11,761 labels)
  - `control_none`: undefined (40,000 calls / 0 labels)

---

## Verdict Summary vs control_none

**Fixed Verdict Rule**:
- **WIN**: Paired mean difference > std and wins >= 2/3 (or >= 4/5 with 5 seeds).
- **LOSS**: Paired mean difference < -std.
- **TIE**: Otherwise.

| Configuration | Paired Diff vs control_none (%) | Std Diff (%) | Wins / Seeds | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `oracle_gate` | +2.07% | 1.17% | 3 / 3 | **WIN** |
| `agreement_gate` | +0.00% | 0.00% | 0 / 3 | **TIE** |
| `selfconf_gate` | +0.11% | 0.15% | 1 / 3 | **TIE** |
| `selfconf_nogate` | +0.93% | 0.59% | 3 / 3 | **WIN** |
| `control_none` | +0.00% | 0.00% | 0 / 3 | **TIE** |
