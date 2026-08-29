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

## Independent Free-Form Holdout

A second 60-case corpus was authored only after the deterministic fallback was
implemented. It contains 20 inspectable development cases and 40 frozen test
cases across 21 conversational families. The scorer was extended to include
no-preference, unresolved-slot, and show-options-first state. The test split was
run once without individual failure output.

| Extractor | Split | Exact state | State F1 | False additions |
|---|---|---:|---:|---:|
| Main legacy rules | Development | 0.0500 | 0.4923 | 0.9000 |
| Hybrid fallback | Development | 0.9000 | 0.9804 | 0.1081 |
| Main legacy rules | Frozen test | 0.1000 | 0.5234 | 0.9500 |
| Hybrid fallback | Frozen test | 0.3750 | 0.8345 | 0.2466 |

This reveals a measurable free-form robustness gap relative to the original
seed test state F1 of `0.8782`. The corpus remains single-author evidence, so
future confidence should come from examples independently supplied by multiple
team members rather than further tuning on the frozen test cases.

Against main's reproduced legacy rule path, hybrid gains `+0.3111` absolute
state F1 on the frozen test and reduces false additions by `0.7034`. This is an
extraction-only A/B; the official end-to-end gates remain unchanged.

## Catalog-Derived Lexicon

The fallback's product vocabulary now extends itself from the frozen catalog
during the existing in-memory index build. Only repeated short phrases are
admitted: leaf categories and typed intent-card facets with frequency of at
least three. Generic hierarchy nodes, malformed values, boilerplate, and nested
short matches are filtered. The resulting lexicon contains 535 categories and
1,145 facets; conversational operators remain hand-audited rules.

| Evaluation | Before | Candidate | Delta |
|---|---:|---:|---:|
| Public TechnicalScore | 0.953650 | 0.953650 | 0.000000 |
| Stress TechnicalScore | 0.951600 | 0.951600 | 0.000000 |
| Reported model tokens | 0 | 0 | 0 |

Decision: accepted provisionally as fixed-catalog vocabulary coverage. It does
not establish an official score gain; its purpose is reducing private-set risk
from clean but unfamiliar catalog terminology.

The catalog audit rejected the initial broad facet admission because intent-card
metadata includes mislabeled values such as department labels in style and
manufacturer flags in color. Slot-specific evidence filters now retain only
high-confidence forms. Unmatched fallback text is kept as bounded soft retrieval
context rather than promoted to a hard slot. Development state F1 improved from
`0.8649` to `0.9804`; the uninspected frozen-test aggregate improved from
`0.8207` to `0.8345`. Public and stress TechnicalScores remained unchanged.

## Gated Ranking and Recall Experiments

Failure tracing motivated three isolated experiments: field-aware reranking of
the top 100 candidates, SQLite FTS5 trigram recall, and confidence-based early
Top-K widening. All are reproducible through `training.evaluate_ranking_variants`
and remain disabled by default.

| Candidate | Public TechnicalScore | Stress TechnicalScore | Model dev | Frozen model test | Decision |
|---|---:|---:|---:|---:|---|
| Baseline | 0.953650 | 0.951600 | 0.103250 | 0.065334 | Keep |
| Ungated field reranker | 0.952600 | 0.953050 | 0.113250 | Not run | Reject |
| Free-form-gated field reranker | 0.953650 | 0.951600 | 0.113250 | 0.065334 | Experimental only |
| Trigram retrieval | 0.952500 | 0.950575 | 0.103916 | Not run | Reject |
| Confidence Top-K widening, 20-session smoke | 0.881000 | Not run | Not run | Not run | Reject |

The free-form gate uses the bounded residual query that is populated only when
the established parser is not confident. This completely isolates the official
and paraphrase fast paths. It improved development by `0.010000`, but produced
no frozen-test improvement, so the gain is not sufficient for promotion.

Trigram retrieval had a high one-time indexing cost and reduced MRR on both
public and stress evaluations. Early Top-K widening reduced 20-session smoke
MRR from `0.950000` to `0.816667` while improving MTTC by only `0.35` turns.
These results reinforce that candidate confidence is not calibrated well enough
to trade rank precision for earlier exposure.

Decision: no default behavior changed. The next reranking attempt needs a new
signal that generalizes across transcript writers; further weighting of the same
lexical and exact-card evidence is unlikely to close the measured gap.
