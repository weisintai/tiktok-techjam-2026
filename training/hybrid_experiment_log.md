# Hybrid Robustness Experiments

This log records isolated experiments on `hybrid-robustness-experiment`, based
on `weisintai/main` commit `02d8b34`. Candidates are selected on public,
paraphrase-stress, and synthetic-validation gates. The synthetic test remains
untouched unless a candidate passes those development gates.

## Baseline

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official public, 200 | 0.995 | 0.985167 | 2.970 | 0.953650 |
| Paraphrase stress, 200 | 0.995 | 0.978667 | 2.975 | 0.951600 |
| Synthetic validation, 400 | 0.990 | 0.977583 | 3.240 | 0.943475 |
| Untouched synthetic test, 400 | 0.980 | 0.970792 | 3.1525 | 0.938188 |

All 29 baseline tests passed under Python 3.13 with NumPy 2.5.2.

## Boundary-Only Information Gain

Shwe's entropy-and-coverage question scorer was activated only after the
released Boundary signal: `I don't have a preference ... use your judgment`.
All other sessions retained main's broad `other` question. The candidate added
no runtime dependency and passed 30 tests, including isolation coverage.

| Evaluation | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Public TechnicalScore | 0.953650 | 0.953150 | -0.000500 |
| Stress TechnicalScore | 0.951600 | 0.951200 | -0.000400 |
| Validation TechnicalScore | 0.943475 | 0.941775 | -0.001700 |
| Public Boundary MTTC | 3.80 | 4.30 | +0.50 |
| Validation Boundary MRR | 0.975000 | 0.968333 | -0.006667 |
| Validation Boundary MTTC | 3.72 | 3.96 | +0.24 |

Decision: rejected and removed from runtime. Narrow attribute questions delayed
useful broad disclosure under the simulator and did not improve candidate
ranking. The untouched synthetic test was not evaluated.
