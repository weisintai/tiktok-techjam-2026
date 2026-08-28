# Experiment Log

Last updated: 28 August 2026

This log records tested hypotheses, including regressions. Scores are from the
200-session public evaluator unless stated otherwise. Result JSON files remain
local experiment artifacts and are not all intended for final submission.

## Score Progression

| Experiment | TechnicalScore | Hit@10 | MRR | MTTC | Decision |
|---|---:|---:|---:|---:|---|
| Organizer weak BM25 starter | 0.106710 | 0.125 | 0.068034 | 9.81 | Replaced |
| Initial information-gain agent | 0.520333 | - | - | - | Improved further |
| Override state fix | 0.574477 | - | - | - | Kept |
| Phrase retrieval routes | 0.606716 | - | - | - | Kept |
| Full-token phrase routes | 0.663715 | - | - | - | Kept |
| Early `other` clarification | 0.787426 | - | - | - | Kept |
| Two `other` questions | 0.784533 | - | - | - | Rejected as global policy |
| Override-aware `other` cap | 0.784633 | - | - | - | Superseded |
| Intent router and cap | 0.807585 | - | - | - | Kept |
| Candidate pool 800 | 0.802800 | - | - | - | Rejected; restored 350 |
| IDF-aware ranking | 0.810552 | - | - | - | Kept |
| Product quality prior | 0.824517 | - | - | - | Kept at 0.015 |
| Quality prior 0.030 | 0.819595 | - | - | - | Rejected |
| Relevance-order selection variant | 0.807789 | - | - | - | Rejected |
| Phrase coverage scoring | 0.829043 | - | - | - | Kept |
| Structured facet route | 0.829636 | 0.985 | 0.534121 | 2.155 | Kept |
| Facet route weight 0.70 | 0.823866 | - | - | - | Rejected; restored 1.05 |
| Selective/boilerplate phrase weighting | 0.829242 | 0.985 | 0.533808 | 2.17 | Rejected |
| Handwritten route-specific reranking | 0.822508 | 0.980 | 0.520694 | 2.185 | Rejected |
| Deterministic baseline | 0.829536 | 0.985 | 0.534121 | 2.16 | Kept deterministic ordering |
| Pairwise top-40 reranker | 0.884374 | 0.995 | 0.671579 | 1.73 | Kept |
| Defer Browsing turn 1 | 0.894899 | 0.995 | 0.724998 | 2.005 | Superseded |
| Defer Buying and Browsing turn 1 | 0.910310 | 0.995 | 0.798367 | 2.335 | Superseded |
| Defer Buying and Browsing through turn 2 | 0.918193 | 0.995 | 0.878645 | 3.145 | Kept |
| Also defer first post-override result | **0.924963** | **0.995** | **0.910875** | **3.29** | Superseded |
| Strict systematic compatibility over full pool | 0.858888 | 0.925 | 0.843625 | 3.835 | Rejected; over-filtered |
| Bounded systematic intent matching | **0.923462** | **0.995** | **0.905875** | **3.29** | Current robust default |
| Confidence-based early stopping | **0.933163** | **0.995** | **0.905875** | **2.805** | Current best |

Some early result files did not capture every aggregate metric in the working
notes, so missing cells are intentionally shown as `-` rather than reconstructed.

## Pairwise Reranker Validation

The dependency-free trainer learned from purchased-product versus hard-negative
pairs among the current candidates. All turns from a session stay in one fold.

| Fold | Baseline replay MRR | Learned MRR |
|---|---:|---:|
| 0 | 0.567073 | 0.690030 |
| 1 | 0.567391 | 0.609613 |
| 2 | 0.573433 | 0.700655 |
| 3 | 0.500446 | 0.658125 |
| 4 | 0.547847 | 0.643571 |
| Mean/full comparison | 0.551238 | 0.660399 mean validation |

The untouched full evaluator subsequently improved from MRR `0.534121` to
`0.671579`, validating that the changed rankings still worked when clarification
trajectories were allowed to change.

## Local LLM Intent Extraction

`llama3.2:latest` (3B, Ollama) was evaluated on eight free-form intent messages
covering buying, browsing, negation, correction, no-preference, budget, and
override behavior. Generation used temperature zero and a constrained JSON
schema.

| Metric | Result |
|---|---:|
| Valid JSON | 8/8 |
| Micro precision | 0.2717 |
| Micro recall | 0.5435 |
| Micro F1 | 0.3623 |
| Mean latency | 2.658 seconds |
| Maximum latency | 3.068 seconds |

The model frequently changed `accumulate` into `replace`, invented audiences and
product types, duplicated constraints across fields, and emitted negation
keywords as excluded values. It was rejected for integration. Schema-constrained
generation solved output syntax but not semantic correctness.

`qwen3:1.7b` was then evaluated with thinking disabled, current-state context,
an evidence-backed delta schema, and verbatim evidence requirements.

| Metric | Result |
|---|---:|
| Valid JSON | 8/8 |
| Micro precision | 0.3860 |
| Micro recall | 0.4783 |
| Micro F1 | 0.4272 |
| Mean latency | 2.435 seconds |
| Maximum latency | 3.864 seconds |

It improved over Llama 3.2 but still frequently emitted `replace`, copied prior
state into the delta, hallucinated defaults, and confused required with preferred
attributes. Evidence validation can remove unsupported values, but cannot repair
the incorrect operation and preference semantics. It was not integrated.

## Findings

### Confidence-based stopping

A 2,000-state counterfactual replay estimated `0.0306` TechnicalScore of oracle
headroom in recommendation timing. The integrated policy exposes early results
only when the reranker margin is at least `2.0` and the top result ranks within
the first 20 fused-retrieval candidates. All five grouped validation folds
improved over fixed stopping. On the official public evaluator, this reduced
MTTC by `0.485` with no change to Hit@10 or MRR.

### What consistently helped

- Structured state accumulation and explicit override erasure.
- Separate lexical, phrase, latest-message, and facet retrieval routes.
- IDF-weighted exact constraints.
- Broad `other` clarification under the official simulator.
- Pairwise learning from the current ranker's hard negatives.
- Delaying recommendation exposure until clarification improves rank.
- Explicit deterministic ordering for all set- and tie-dependent operations.

### What did not help

- Global dense retrieval and semantic fusion reduced the score in earlier tests.
- Enlarging the candidate pool from 350 to 800 added noise.
- Increasing the quality prior promoted popular but less exact products.
- Handwritten per-route ranking weights did not generalize across scenarios.
- Boilerplate suppression slightly reduced MRR.
- A global second `other` policy hurt override behavior until routing was added.

### Interpretation

The system's main bottleneck changed over time:

1. Early versions lacked recall and state correctness.
2. After multi-route retrieval, Hit@10 reached approximately 0.985 and ranking
   became the bottleneck.
3. Pairwise reranking moved more retrieved targets toward rank one.
4. Recommendation gating prevented low-rank early hits from ending sessions
   before useful clarification arrived.

At the current checkpoint, private-set generalization is a larger concern than
public recall. New experiments should use grouped validation and should avoid
hardcoding public ASINs or simulator messages.

## Main-Branch Robustness Harnesses

The deterministic paraphrase and ASIN-separated synthetic harnesses from
`weisintai/main` commit `02d8b34` were run against this branch. Only the imported
agent and its constructor call were adapted; message transformations, synthetic
split assignment, session generation, and official evaluator scoring were
unchanged.

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official public, 200 | 0.995 | 0.905875 | 2.805 | 0.933163 |
| Deterministic paraphrase stress, 200 | 0.640 | 0.350141 | 5.40 | 0.537042 |
| Unseen-ASIN validation, 400 | 0.970 | 0.786016 | 3.465 | 0.871505 |
| Untouched synthetic test, 400 | 0.9375 | 0.775077 | 3.705 | 0.847173 |

The unseen-ASIN results show that retrieval generalizes beyond public targets,
although ranking and override behavior degrade. The much larger paraphrase drop
isolates intent normalization as the dominant robustness bottleneck. Main's
normalized constraint variants and exact catalog-card index directly address
that failure mode.

## Next Experiments

- Confidence-based gating instead of fixed turn-based withholding.
- Nested cross-validation for reranker regularization and feature ablation.
- A separate free-form paraphrase validation set to test template dependence.
- Route-specific learned models only after enough non-public validation exists.
- Calibration of score margins into expected MRR gain versus turn cost.
- Evidence-grounded paraphrase normalization using catalog-derived constraint
  variants.
