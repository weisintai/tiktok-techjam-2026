# How the Shopping Agent Works

This document explains the runtime behavior of the implementation in
`starter/agent.py` and `starter/intent.py`.

## System Flow

Each call to `Agent.respond()` follows this pipeline:

```text
User message
    -> parse intent and update session state
    -> retrieve candidates through six routes
    -> fuse route rankings into 350 candidates
    -> apply deterministic constraint-aware scoring
    -> rerank the top 40 with a learned pairwise model
    -> promote strongly compatible product types and audiences
    -> estimate result confidence
    -> choose a clarification question
    -> recommend now or defer until more evidence arrives
```

All catalog processing and inference run locally in memory. The default pipeline
makes no LLM or network calls during evaluation.

## 1. Startup and Catalog Indexing

Creating an `Agent` reads the frozen JSONL catalog once and builds:

- An in-memory SQLite FTS5 index over title, category, features, details, store,
  and description.
- Typed facets for product type, audience, material, color, size, style, use
  case, brand, budget, and general features.
- Inverted facet lookups from `(attribute, value)` to matching ASINs.
- Normalized product text for phrase matching.
- IDF values that give rare, discriminative terms more weight.
- A small quality prior based on rating and review count.

The catalog is never modified. Indexing is repeated when a new `Agent` process
starts and then reused by every session in that process.

## 2. Session State

`Agent.reset(session_id, user_profile)` creates isolated state for a conversation.
The state stores:

- All retained messages.
- Intent mode: `unknown`, `buying`, `browsing`, `possible_override`, or
  `override`.
- Category, hard-constraint, and soft-preference phrases.
- Structured slots such as color, material, size, style, and use case.
- Product types, audiences, exclusions, and maximum budget.
- Attributes already asked and attributes for which the user has no preference.
- Weak profile terms, retrieval scores, and recommendation confidence.

The profile is deliberately weaker than explicit session requirements. It helps
break ties but should not override a stated constraint.

## 3. Intent Parsing and State Updates

`parse_intent()` converts each message into an `IntentDelta`. It extracts:

- Buying versus browsing evidence.
- Product families such as `running_shoes`, `jackets`, or `bags`.
- Audience such as women, men, children, or unisex.
- Typed attribute values.
- Required and preferred phrases.
- Negative constraints such as `no leather`.
- Numeric maximum budgets.
- Whether the message accumulates evidence or replaces an earlier request.

Normal turns accumulate evidence. An override containing terms such as
`actually`, `instead`, `ignore`, or `forget` starts a new recommendation epoch:
superseded slots and clarification history are cleared, while the initial
product category is retained when possible.

Example:

```text
Turn 1: "I'm looking for women's running shoes. A key requirement is: wide fit."

mode          = buying
product_types = {running_shoes, shoes}
audiences     = {women}
size          = {wide}
hard phrases  = [wide fit]

Turn 2: "What matters is: waterproof; suitable for trail running."

feature       += {waterproof}
use_case      += {trail, running}
soft phrases  += [waterproof, suitable for trail running]
```

## 4. Multi-Route Candidate Retrieval

The agent runs six complementary retrieval routes on every turn:

| Route | Purpose | Weight |
|---|---|---:|
| Conversation and profile | Broad accumulated context | 0.85 |
| Hard phrases | Exact required properties | 1.25 |
| Category phrases | Product-family recall | 0.95 |
| Latest message | React quickly to new evidence | 1.10 |
| Combined constraints | Match the complete request | 1.35 |
| Structured facets | Exact typed attribute matches | 1.05 |

FTS fields are also weighted: titles have the highest lexical weight, followed
by categories, features, details/store, and description.

The route lists are fused with weighted reciprocal-rank fusion:

```text
fused_score(product) += route_weight / (60 + route_rank)
```

The top 350 fused products proceed to ranking. This broad pool is responsible
for recall; later stages are responsible for precision.

## 5. Ranking and Compatibility

Ranking has three stages.

### Deterministic scoring

Every candidate receives a score from fused retrieval position, exact slot
matches, hard/soft/category/query IDF coverage, phrase coverage, and the quality
prior. Missing explicit material, color, or size matches receive a penalty.

### Pairwise reranking

The top 40 candidates are rescored with a dependency-free logistic model trained
from purchased products versus hard negatives. Its 14 features measure retrieval
position, slot matches and misses, phrase and IDF coverage, quality, and hard or
soft constraint completeness. Fixed learned weights in `PAIRWISE_WEIGHTS` make
evaluation deterministic; `experiments/train_pairwise_reranker.py` reproduces
the training procedure.

### Bounded compatibility promotion

The agent checks the reranked top 40 for product-family and audience agreement,
exclusion conflicts, and exact constraint matches. At most three candidates with
both a matching product type and matching audience are promoted. The limit
prevents a parser mistake from replacing the entire lexical ranking.

## 6. Clarification Strategy

The agent can return recommendations and ask a question in the same response.
It initially asks the broad `other` question because the official simulator can
use it to disclose any remaining constraint. Broad questions are capped at two,
or one for possible-override and override sessions.

After that, it scores unresolved attributes using:

```text
information gain = entropy(candidate facet values)
                 * candidate coverage
                 * attribute priority
```

Already answered, already asked, and explicit no-preference attributes are
excluded. Questions with insufficient candidate coverage or information gain
are skipped.

## 7. Recommendation Timing

Returning a recommendation that contains the purchased product ends an evaluator
session. Recommending too early can therefore lock in a poor reciprocal rank;
waiting too long increases MTTC.

The agent normally defers Buying and Browsing recommendations during turns one
and two while a useful clarification is available. It recommends early only if
both confidence conditions hold:

- The learned score margin between the first and second result is at least `2.0`.
- The selected result is also within the top 20 of fused retrieval, represented
  by a retrieval-rank reciprocal of at least `0.05`.

After an intent override, the first broad clarification is deferred so the new
request has enough evidence. Counterfactual replay selected these thresholds;
they reduced public MTTC from `3.29` to `2.805` without changing MRR or Hit@10.

## 8. Response Contract

`respond()` returns the organizer's required structure:

```python
{
    "message": "Here are strong matches so far. Do you have a material preference?",
    "ask_attribute": "material",
    "recommendations": [
        {"parent_asin": "B000..."},
        {"parent_asin": "B001..."},
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
```

An empty `recommendations` list means the agent has intentionally deferred its
answer pending clarification.

## 9. Running and Inspecting It

Run the official local evaluation:

```bash
PYTHONHASHSEED=1 PYTHONPATH=. python3 -m evaluator.local_evaluator \
  --output results/public_evaluation.json
```

Run a manual conversation:

```bash
PYTHONPATH=. python3 scripts/manual_chat.py
```

Run the counterfactual stopping analysis:

```bash
PYTHONHASHSEED=1 PYTHONPATH=. python3 experiments/analyze_stopping_policy.py
```

The latest verified public result is TechnicalScore `0.933163`, Hit@10 `0.995`,
MRR `0.905875`, and MTTC `2.805`.

## 10. Known Limitations

- The product and slot ontology is manually enumerated and strongest on the
  organizer's Clothing, Shoes and Jewelry catalog.
- Template-aware extraction is substantially stronger than genuinely free-form
  intent extraction.
- Override handling currently resets the full constraint set rather than
  performing every possible slot-level correction.
- Budget is extracted into state, but numeric catalog-price enforcement is not
  yet fully integrated into final ranking.
- Pairwise weights and stopping thresholds were developed on the public set;
  private-set generalization remains the primary risk.
- Boundary sessions remain the weakest scenario because users provide little
  discriminative evidence.

See `docs/experiment_log.md` for accepted and rejected experiments and
`docs/bottleneck_analysis.md` for the prioritized improvement roadmap.
