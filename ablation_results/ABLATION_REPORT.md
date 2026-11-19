# Biomimetic oxidation: ablation results

Split mode: `scaffold` | CV: 5-fold x 5 repeats | conformers: 5

Primary metric: out-of-fold PR-AUC from repeated CV on the training pool, mean ± SD across repeats. The held-out columns are a sanity check, not the comparison: the test set is too small for a single PR-AUC to be stable.

## Arms

- `0_prior` — 0  class prior (dummy)
- `1_hot_logreg` — 1  HOT_SMARTS + logistic
- `2a_ecfp4_logreg` — 2a ECFP4 + logistic
- `2b_ecfp4_rf` — 2b ECFP4 + random forest
- `3_ecfp4_desc_rf` — 3  ECFP4 + RDKit + Percepta + RF
- `4_all_rf` — 4  arm 3 + HOT_SMARTS + RF
- `5_gnn_dimenetpp` — 5  single-task DimeNet++ + bond graph
- `6_multitask_dimenetpp` — 6  multitask DimeNet++

## CV PR-AUC by assay

| assay | prevalence | 0_prior | 1_hot_logreg | 2a_ecfp4_logreg | 2b_ecfp4_rf | 3_ecfp4_desc_rf | 4_all_rf | 5_gnn_dimenetpp | 6_multitask_dimenetpp | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| percent | 0.35 | 0.401 ± 0.016 | 0.604 ± 0.032 | 0.504 ± 0.035 | 0.545 ± 0.044 | 0.622 ± 0.040 | 0.673 ± 0.035 | 0.515 ± 0.049 | — | 4_all_rf |
| Ar_ox | 0.07 | 0.087 ± 0.016 | 0.159 ± 0.026 | 0.427 ± 0.151 | 0.327 ± 0.050 | 0.247 ± 0.083 | 0.274 ± 0.154 | 0.095 ± 0.024 | 0.124 ± 0.064 | 2a_ecfp4_logreg |
| Bn_ox | 0.08 | 0.080 ± 0.003 | 0.196 ± 0.026 | 0.207 ± 0.029 | 0.185 ± 0.047 | 0.191 ± 0.032 | 0.283 ± 0.066 | 0.117 ± 0.076 | 0.139 ± 0.039 | 4_all_rf |
| deH_ox | 0.25 | 0.278 ± 0.019 | 0.275 ± 0.024 | 0.398 ± 0.030 | 0.303 ± 0.019 | 0.295 ± 0.015 | 0.301 ± 0.022 | 0.302 ± 0.026 | 0.279 ± 0.027 | 2a_ecfp4_logreg |
| N_dealk | 0.38 | 0.362 ± 0.016 | 0.718 ± 0.046 | 0.490 ± 0.053 | 0.606 ± 0.043 | 0.633 ± 0.041 | 0.677 ± 0.044 | 0.498 ± 0.042 | 0.397 ± 0.068 | 1_hot_logreg |
| N_dearyl | 0.06 | 0.094 ± 0.069 | 0.182 ± 0.027 | 0.192 ± 0.036 | 0.215 ± 0.050 | 0.234 ± 0.082 | 0.249 ± 0.115 | 0.118 ± 0.078 | 0.092 ± 0.044 | 4_all_rf |
| O_dealk | 0.13 | 0.136 ± 0.015 | 0.465 ± 0.064 | 0.328 ± 0.054 | 0.480 ± 0.080 | 0.434 ± 0.148 | 0.372 ± 0.065 | 0.193 ± 0.014 | 0.172 ± 0.049 | 2b_ecfp4_rf |
| O_dearyl | 0.04 | 0.046 ± 0.014 | 0.045 ± 0.015 | 0.039 ± 0.012 | 0.141 ± 0.086 | 0.134 ± 0.089 | 0.128 ± 0.081 | 0.094 ± 0.055 | 0.139 ± 0.020 | 2b_ecfp4_rf |
| OH_ox | 0.28 | 0.285 ± 0.022 | 0.279 ± 0.025 | 0.378 ± 0.035 | 0.329 ± 0.040 | 0.326 ± 0.028 | 0.336 ± 0.018 | 0.345 ± 0.039 | 0.325 ± 0.053 | 2a_ecfp4_logreg |
| R_ox | 0.23 | 0.233 ± 0.018 | 0.344 ± 0.044 | 0.266 ± 0.048 | 0.294 ± 0.035 | 0.315 ± 0.047 | 0.308 ± 0.067 | 0.309 ± 0.031 | 0.281 ± 0.065 | 1_hot_logreg |
| S_ox | 0.02 | 0.020 ± 0.000 | 0.562 ± 0.042 | 0.275 ± 0.066 | 0.417 ± 0.102 | 0.355 ± 0.152 | 0.581 ± 0.237 | 0.055 ± 0.011 | 0.033 ± 0.007 | 4_all_rf |
| unk_ox | 0.24 | 0.227 ± 0.009 | 0.422 ± 0.073 | 0.256 ± 0.033 | 0.251 ± 0.020 | 0.325 ± 0.020 | 0.349 ± 0.041 | 0.351 ± 0.093 | 0.291 ± 0.031 | 1_hot_logreg |

## Paired comparisons across assays

| comparison | n | median Δ PR-AUC | A better | B better | test | p |
|---|---|---|---|---|---|---|
| 5_gnn_dimenetpp vs best_baseline_per_assay | 12 | -0.145 | 0 | 12 | wilcoxon_signed_rank | 0.0005 |
| 6_multitask_dimenetpp vs best_baseline_per_assay | 11 | -0.145 | 0 | 11 | wilcoxon_signed_rank | 0.0010 |
| 4_all_rf vs 1_hot_logreg | 12 | 0.042 | 8 | 4 | wilcoxon_signed_rank | 0.3394 |
| 4_all_rf vs 2a_ecfp4_logreg | 12 | 0.067 | 9 | 3 | wilcoxon_signed_rank | 0.1099 |
| 4_all_rf vs 0_prior | 12 | 0.171 | 12 | 0 | wilcoxon_signed_rank | 0.0005 |
| 2b_ecfp4_rf vs 1_hot_logreg | 12 | 0.002 | 6 | 6 | wilcoxon_signed_rank | 0.7334 |
| 2b_ecfp4_rf vs 2a_ecfp4_logreg | 12 | 0.025 | 7 | 5 | wilcoxon_signed_rank | 0.2661 |
| 2b_ecfp4_rf vs 0_prior | 12 | 0.113 | 12 | 0 | wilcoxon_signed_rank | 0.0005 |
| 6_multitask_dimenetpp vs 5_gnn_dimenetpp | 11 | -0.021 | 3 | 8 | wilcoxon_signed_rank | 0.2783 |

Δ is arm A minus arm B, so a positive median favours A. With 12 assays these tests have little power (the smallest attainable two-sided p is around 0.000488); treat them as a direction indicator, not proof.

## Multitask vs single-task, per task

Reported per task on purpose: an improved average can hide individual tasks getting worse.

| task | single-task | multitask | Δ | verdict |
|---|---|---|---|---|
| Ar_ox | 0.095 | 0.124 | 0.029 | positive transfer |
| Bn_ox | 0.117 | 0.139 | 0.021 | within noise |
| deH_ox | 0.302 | 0.279 | -0.023 | within noise |
| N_dealk | 0.498 | 0.397 | -0.101 | negative transfer |
| N_dearyl | 0.118 | 0.092 | -0.026 | within noise |
| O_dealk | 0.193 | 0.172 | -0.020 | negative transfer |
| O_dearyl | 0.094 | 0.139 | 0.044 | within noise |
| OH_ox | 0.345 | 0.325 | -0.021 | within noise |
| R_ox | 0.309 | 0.281 | -0.028 | within noise |
| S_ox | 0.055 | 0.033 | -0.021 | negative transfer |
| unk_ox | 0.351 | 0.291 | -0.060 | within noise |

Positive transfer on 1 task(s), negative transfer on 3 task(s), the rest within noise.