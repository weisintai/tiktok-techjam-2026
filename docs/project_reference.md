# Shopping Copilot Project Reference

Last updated: 28 August 2026

This is the canonical working reference for Track 4. It combines the organizer
brief supplied to the team with the architecture and status of the current
implementation. The machine-readable API contract and official evaluator remain
the source of truth when this document and executable behavior differ.

## Track Objective

Build a next-generation conversational shopping agent over the frozen Amazon
Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog. The agent must distinguish
high-intent buying from open-ended browsing, maintain evolving constraints over
multiple turns, retrieve and rank the purchased product, and guide users toward
conversion efficiently.

The competition kit contains:

- 50,000 frozen products.
- 200 public development sessions.
- 800 private final-evaluation sessions using separate users and targets.
- A weak Python BM25 starter agent.
- A deterministic local evaluator and machine-readable Agent contract.

Original resources:

- Participant repository: <https://github.com/TechJam2026/techjam-conversational-search>
- Participant kit: <https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit>
- Amazon Reviews 2023: <https://amazon-reviews-2023.github.io/>

Workshop information supplied by the organizer:

- Technical workshop and Q&A: 28 August, 4:00-4:45 PM.
- Recording expected by 29 August, 12:00 PM.

## Required Capabilities

### Intent routing and retrieval

- Route high-intent Buying queries toward hard filtering and exact constraints.
- Route exploratory Browsing queries toward broader, diverse retrieval.
- Use an in-memory `multi-route retrieval -> semantic ranking` pipeline.
- Combine keyword, category, structured facet, and optional vector signals.

### Multi-turn dialogue

- Accumulate constraints without losing unrelated preferences.
- Detect overrides and erase or rewrite superseded state.
- Stop broad retrieval when the candidate pool is too ambiguous.
- Ask structured clarification questions that reduce uncertainty.

### Dynamic context programming

- Distill dialogue history into compact session state.
- Treat the aggregate user profile as weak personalization, below explicit intent.
- Adapt retrieval, ranking, and clarification strategy at runtime.

### Evaluation

- Coverage: Hit Rate@10.
- Precision: MRR and Top-K hit rate.
- Efficiency: MTTC, with misses assigned turn 11.

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

The maximum session length is 10 turns. Going beyond it produces no valid
conversion. Only exact catalog `parent_asin` values count.

## Scope and Constraints

In scope:

- Buying/Browsing intent detection.
- Dynamic retrieval weights and truncation.
- Slot accumulation, decay, negation, and override handling.
- Runtime-adaptive memory and local scoring.
- Prompt or local-model ranking experiments.

Out of scope:

- UI/UX work for scoring.
- Full-parameter foundation-model training.
- External industrial vector databases.
- Multimodal processing.
- Catalog mutation or synthetic ASIN insertion.

Allowed assumptions:

- Input strings are pre-cleaned.
- Catalog metadata, prices, and category trees are static.
- Sessions are isolated and do not require concurrency handling.

## Current Architecture

The current implementation is in `starter/agent.py` and uses only the Python
standard library during evaluation.

### Conversation state

Each session tracks:

- Messages and distilled category, hard, and soft phrases.
- Structured material, color, size, style, use-case, and budget slots.
- Asked attributes and explicit no-preference responses.
- Intent mode: unknown, Buying, Browsing, possible override, or override.
- Number of broad `other` clarification questions already asked.
- Typed product families, audiences, excluded terms, and maximum budget.

`starter/intent.py` parses each message into an `IntentDelta` with explicit
accumulate or replace semantics. It systematically extracts product type,
audience, material, color, size, style, use case, features, exclusions, budget,
and required versus preferred phrases.

An override clears superseded constraints, shown-state assumptions, and earlier
clarification state while retaining the initial product category where possible.

### Catalog index

Startup preprocessing creates:

- An in-memory SQLite FTS5 index with weighted product fields.
- Typed product facets and inverted facet lookups.
- Product-level normalized text.
- Catalog term document-frequency/IDF values.
- A small rating and review-count quality prior.

The catalog remains read-only.

### Retrieval

The agent retrieves from several routes:

- Full conversation plus weak profile terms.
- Hard-constraint phrases.
- Category phrases.
- Latest user message.
- Combined category, hard, and soft constraints.
- Structured facet lookup.

The route rankings are combined using weighted reciprocal-rank-style fusion and
truncated to 350 candidates.

### Ranking

Stage one uses deterministic constraint-aware scoring over retrieval score,
slot matches and misses, IDF coverage, phrase coverage, and product quality.

Stage two reranks the top 40 with a dependency-free pairwise logistic model.
It uses 14 features covering retrieval position, slots, hard/soft/category/query
coverage, phrase coverage, quality, and constraint completeness. The trainer is
`experiments/train_pairwise_reranker.py`.

The product-matching layer then inspects the top 40 for typed product-family and
audience compatibility plus explicit constraint completeness. At most three
strongly compatible candidates are promoted. This bounded promotion improves
free-form robustness without allowing parser uncertainty to erase retrieval
recall.

### Clarification and recommendation gating

The agent first uses broad `other` questions because the official simulator can
reveal any undisclosed constraint through that attribute. It then uses entropy,
candidate coverage, and attribute priorities to choose a narrower facet.

The current score-optimized policy normally withholds Buying and Browsing
recommendations during turns one and two, but recommends early when the learned
ranker margin is at least `2.0` and the selected result is also within the top 20
of fused retrieval. It also collects one fresh clarification after an intent
override. This balances the reciprocal-rank value of another clarification
against its MTTC cost.

### Reproducibility

Set iteration, feature truncation, fusion ties, and information-gain sampling use
explicit deterministic ordering. Runs with `PYTHONHASHSEED=1` and `777` produce
identical metrics.

## Current Verified Result

Public evaluator result from `results/shwe_experiment_public.json`:

| Metric | Result |
|---|---:|
| TechnicalScore | **0.933163** |
| Hit Rate@10 | **0.995** |
| MRR | **0.905875** |
| MTTC | **2.805** |
| Efficiency | **0.8195** |
| Model/API calls | **0** |

Scenario results:

| Scenario | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 1.0000 | 0.574167 | 3.30 |
| Browsing | 1.0000 | 0.922292 | 2.4875 |
| Buying | 0.9875 | 0.928542 | 2.3875 |
| Intent override | 1.0000 | 0.912222 | 4.60 |

## Judging Rubric

| Criterion | Weight | Evidence to emphasize |
|---|---:|---|
| Technical Execution | 35% | Stateful architecture, deterministic offline execution, multi-route retrieval, learned reranker, evaluator results |
| Innovation and Problem Insight | 20% | Intent routing, information-gain clarification, conversion-aware recommendation gating |
| Impact and Relevance | 20% | Better discovery and exact-product ranking with fewer irrelevant recommendations |
| Feasibility and Practicality | 15% | In-memory index, no paid API, no external vector database, zero inference tokens |
| Presentation and Communication | 10% | Clear problem-to-solution narrative, reproducible demo, honest limitations |

## Deliverables

### Devpost project description

Explain the problem, solution, development tools, APIs, libraries, datasets, and
assets. The current default agent uses Python, SQLite FTS5, and standard-library
local scoring; it makes no external API calls during evaluation.

### Public repository

Include structured and commented code, setup instructions, reproduction steps,
limitations, future improvements, and team contributions.

### Demo video

Show an end-to-end inference or evaluator run, result analysis, and at least one
multi-turn override flow. Upload it publicly to YouTube and link it from Devpost.

## Caveats

- Public sessions are development data. Repeated tuning can overfit evaluator
  templates and may not transfer fully to the 800 private sessions.
- The deterministic extractor is optimized for official message templates and
  is weaker on genuinely free-form language.
- The pairwise model was trained on public-session hard negatives. Session-grouped
  five-fold validation reduced, but did not remove, generalization risk.
- `other` is unusually effective because of the public simulator contract. A
  real customer-facing system should generate a natural, specific question.
- Boundary has only 10 public samples, so its metric is noisy and should not be
  tuned directly without broader validation.
- No separate stress dataset or stress evaluator is present in this workspace;
  a current stress TechnicalScore cannot be verified locally.

## Reproduction

```bash
python3 -m unittest
PYTHONHASHSEED=1 python3 -m evaluator.local_evaluator \
  --output results/results_pairwise_defer_override.json
```

Retrain the pairwise experiment with:

```bash
PYTHONHASHSEED=1 python3 experiments/train_pairwise_reranker.py
```

See `docs/experiment_log.md` for the complete experiment history.
See `docs/bottleneck_analysis.md` for the latest phase-by-phase gap analysis and
prioritized roadmap.
