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

## Evidence-Grounded Deterministic Delta

A deterministic fallback was added behind main's existing confidence check. It
activates only when the established official and paraphrase parsers are not
confident. The fallback extracts explicit apparel categories and facets plus
add, replace, remove, exclude, no-preference, unresolved-budget, and
show-options-first operations. All state changes still pass through the existing
`StructuredTurn` application boundary. No LLM, network call, or runtime
dependency was added.

Free-form extraction results over the frozen 200-case corpus:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Raw delta micro-F1 | 0.0583 | 0.7046 | +0.6463 |
| Applied-state micro-F1 | 0.5689 | 0.8388 | +0.2699 |
| Applied-state precision | 0.6946 | 0.8411 | +0.1465 |
| Applied-state recall | 0.4817 | 0.8366 | +0.3549 |
| False-addition rate | 0.9545 | 0.3172 | -0.6373 |
| Sibling preservation | 1.0000 | 0.9900 | -0.0100 |

The frozen 40-case free-form test split reached applied-state F1 `0.8782`, above
development F1 `0.7945`. This does not establish broad natural-language
generalization, but it reduces the risk that the aggregate gain comes only from
the training families.

End-to-end gates:

| Evaluation | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Public TechnicalScore | 0.953650 | 0.953650 | 0.000000 |
| Stress TechnicalScore | 0.951600 | 0.951600 | 0.000000 |
| Validation TechnicalScore | 0.943475 | 0.943475 | 0.000000 |
| Untouched test TechnicalScore | 0.938188 | 0.938188 | 0.000000 |

Decision: accepted as a robustness improvement. Established simulator-shaped
messages remain on the original fast path, while unfamiliar messages gain a
deterministic structured fallback. The next evaluation should add independently
written free-form cases because the current corpus contains templated families
from one writer.
