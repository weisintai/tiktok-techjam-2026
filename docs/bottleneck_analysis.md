# Pipeline Bottleneck Analysis

Last updated: 28 August 2026

This analysis uses `results_systematic_intent_bounded.json` and a replay of all
200 public sessions that recorded target rank after fused retrieval and final
reranking.

## Score Gap

Current result after confidence-based early stopping:

| Metric | Current | Maximum | TechnicalScore gap |
|---|---:|---:|---:|
| Hit@10 | 0.995 | 1.000 | 0.0025 |
| MRR | 0.905875 | 1.000 | 0.0282 |
| Efficiency | 0.8195 | 1.000 | 0.0361 |

Efficiency/MTTC is the largest remaining public-score bottleneck, followed by
rank-one precision. Recall has little remaining headroom.

## Phase 1: Intent Understanding

Public simulator templates are extracted effectively, but free-form testing
found incorrect mode routing, incomplete preference extraction, and confusion
between removal and exclusion. `intent.py` also relies on a manually enumerated
ontology. Local LLM experiments did not solve this: Llama 3.2 3B reached 0.3623
intent micro-F1 and Qwen3 1.7B reached 0.4272.

Gap-closing work:

- Generate product taxonomy and facet vocabularies from the catalog.
- Extract evidence spans first, then ground them deterministically.
- Use one state-update path so excluded terms cannot be reintroduced by a legacy
  extractor.
- Distinguish accumulate, replace, remove, exclude, and no-preference operations.
- Keep LLM extraction experimental until it exceeds the deterministic parser on
  a larger normalized benchmark.

## Phase 2: Context and State Evolution

Official override handling is strong, with override MRR 0.912222, but free-form
`forget running shoes; need boots` can re-extract the forgotten product from the
same message. Parsed mode is not yet consistently applied to the dialogue router.

Gap-closing work:

- Parse negative/removal spans before positive spans.
- Apply intent deltas atomically against the previous state.
- Preserve unrelated slots during slot-level replacement.
- Add confidence and provenance per slot: explicit, inferred, profile, or stale.
- Decay inferred/profile values but never decay explicit hard constraints.

## Phase 3: Candidate Discovery

At the terminal turn, fused retrieval target ranks were:

| Retrieval position | Sessions |
|---|---:|
| Rank 1 | 109 |
| Rank 2-10 | 67 |
| Rank 11-40 | 17 |
| Rank 41-350 | 7 |

Retrieval is not the global bottleneck. Boundary is the exception: its median
retrieval rank is 24 and three of ten targets are below rank 40. Buying has two
targets below rank 40, Browsing has none, and Intent Override has two.

Gap-closing work:

- Keep the current 350-candidate retrieval pool.
- Add a Boundary fallback route using category plus quality and profile terms
  only after the user declines a preference.
- Use catalog-derived product-family grounding for free-form requests.
- Diagnose the seven below-rank-40 targets individually before adding another
  global retrieval route.

## Phase 4: Matching and Ranking

The final rank distribution is strong but not perfect:

| Scenario | Rank 1 | Rank 1-3 | Misses |
|---|---:|---:|---:|
| Buying (80) | 72 | 75 | 1 |
| Browsing (80) | 70 | 77 | 0 |
| Intent Override (30) | 26 | 28 | 0 |
| Boundary (10) | 4 | 6 | 0 |

Several non-rank-one targets were retrieval rank 1 but were demoted by the
reranker. This occurs in Browsing and Buying and indicates calibration or feature
interaction errors rather than retrieval failure.

Gap-closing work:

- Train a listwise or rank-weighted model that optimizes MRR directly.
- Weight negatives by damage, emphasizing products above the purchased target.
- Add interactions such as product-family match multiplied by hard-constraint
  completeness.
- Add Exact/Substitute/Complement/Irrelevant classes before final ordering.
- Calibrate rank-one confidence from margin, route agreement, constraint
  completeness, and rank stability.

## Phase 5: Dialogue Policy and Conversion Timing

The current fixed policy withholds Buying and Browsing recommendations through
turn two. As a result, 78 of 80 Buying sessions convert at turn three and all 80
Browsing sessions convert at turn three, even though many are already easy.

| Scenario | Terminal-turn distribution |
|---|---|
| Buying | Turn 2: 1, Turn 3: 78, miss: 1 |
| Browsing | Turn 3: 80 |
| Intent Override | Turn 4: 12, Turn 5: 18 |
| Boundary | Turn 3: 8, Turn 4: 1, Turn 5: 1 |

This was the largest public TechnicalScore bottleneck under fixed stopping.

Gap-closing work:

- Replace fixed deferral with a calibrated recommend-versus-ask policy.
- Recommend early when rank one is stable, complete, contradiction-free, and
  separated by a strong margin.
- Estimate question value as expected reciprocal-rank gain minus turn cost.
- Use counterfactual evaluator replay to label the best action at each state.
- Train an interpretable decision tree over ranking and state features.

The first counterfactual replay measured an oracle TechnicalScore improvement of
`0.0306`. A conservative policy now recommends during turns one or two only
when the learned-score margin is at least `2.0` and the same product is within
the top 20 of fused retrieval. Five scenario-grouped folds all improved over
fixed stopping. The official evaluator preserved Hit@10 and MRR while reducing
MTTC from `3.29` to `2.805`, increasing TechnicalScore from `0.923462` to
`0.933163`.

One additional average turn costs 0.02 TechnicalScore. It is worthwhile only
when expected MRR improves by more than approximately 0.0667.

## Prioritized Roadmap

1. Listwise/rank-weighted reranker for the 27 non-rank-one or missed sessions.
2. Expand stopping calibration on non-public paraphrase sessions.
3. Boundary-specific retrieval and no-preference policy.
4. Catalog-derived ontology and unified state updates for private/free-form
   robustness.
5. Cross-encoder reranking only after the lightweight policy/ranking work
   plateaus.
