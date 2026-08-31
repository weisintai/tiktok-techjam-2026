# Architecture

Shopping Copilot's default submission is a deterministic, offline Python agent.
It reads the supplied 50,000-product catalog once, builds an in-memory SQLite
FTS5 index and catalog-derived facet indexes, and keeps each conversation's
state in memory. It makes no external model or network calls.

## Default pipeline

```text
                           Customer Message
                                  |
                +-----------------+-----------------+
                |                                   |
                v                                   v
        Recognized Message                  Unfamiliar Free Form
                |                                   |
                v                                   v
       Deterministic Parser          Catalog-Grounded Deterministic
                                               Fallback
                +-----------------+-----------------+
                                  |
                                  v
                         Typed Multi-Turn State
                    (Add | Replace | Remove | Exclude)
                                  |
               +------------------+------------------+
               |                  |                  |
               v                  v                  v
         Weighted BM25       Exact-Facet       Category-Phrase
         Text Retrieval        Lookup            Resolution
               +------------------+------------------+
                                  |
                                  v
                         Candidate Pool Fusion
                                  |
                                  v
                      Constraint-Aware Ranking
               (Exclusions | Exact Evidence | Category Fit)
                                  |
                                  v
                Bounded Popularity + Lexical Tie-Breaking
                                  |
                                  v
                      Recommendations + Clarification
```

## Conversation state

The agent stores category, material, color, size, style, budget, use case,
features, exclusions and previously shown products. Updates cross a typed
boundary that distinguishes four operations:

- **Add:** preserve prior preferences and add new evidence.
- **Replace:** update only the affected slot, such as changing black to blue.
- **Remove:** forget a previously stated preference without negating it.
- **Exclude:** retain a negative requirement such as "no leather."

A category switch updates the active category. Override handling selectively
replaces affected slots while preserving unrelated confirmed requirements.

## Retrieval and ranking

Candidate generation unions three default signals:

1. weighted SQLite FTS5 BM25 over normalized product text;
2. exact lookup over catalog-derived facets; and
3. phrase-level resolution against informative catalog-category paths.

Ranking first penalizes exclusions, then considers exact constraint coverage
and category evidence. Review volume is blended with lexical order only as a
bounded popularity prior inside otherwise ambiguous evidence tiers; it cannot
outrank a product that satisfies more explicit buying constraints.

## Clarification and output

The agent returns ranked product identifiers through the official
`Agent.respond()` contract and can issue a broad clarification to collect more
preferences. It tracks shown products to avoid repeating recommendations.

## Optional experiments

The repository retains several components for reproducibility, but none is
enabled in the default scored path:

- MiniLM dense retrieval;
- cross-encoder reranking;
- an experimental Buying/Browsing router;
- a scikit-learn top-50 reranker for unfamiliar free-form requests; and
- a local Qwen structured-extraction model through llama.cpp.

The optional model output is schema-, confidence- and evidence-gated before it
can change confirmed state.

## Demo architecture

The Next.js frontend sends requests to `/api/copilot`. Its Node.js route keeps a
local Python bridge process alive and exchanges newline-delimited JSON with
`frontend/backend/copilot_server.py`. The bridge calls the same production
agent used by the evaluator, so displayed recommendations are live catalog
results rather than fixtures.
