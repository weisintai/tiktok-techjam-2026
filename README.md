# TechJam Shopping Copilot

An offline conversational shopping agent that turns multi-turn dialogue into
typed constraints, preserves intent changes, and retrieves the exact purchased
Amazon product in an average of `1.995` turns on the released evaluation set.

The scorer-proven path is deterministic: it uses structured session state,
catalog-derived exact cards, phrase-level category resolution against the catalog
tree, weighted SQLite FTS5 BM25, and a purchase-volume prior that orders products
the conversation cannot separate. The official
evaluator drives a frozen, structured simulator protocol, so this path requires
no network, API key, model weight, or token spend. A small local model remains
an optional demonstration of unfamiliar free-form language rather than a
dependency of the submitted agent.

Multi-turn product search is where keyword search structurally fails: a
shopper who says "something warm for winter, not too flashy" cannot be served
by an index built on literal terms, and every clarifying question that does
not narrow the catalog is a turn the shopper did not want to spend. This
project treats that as a state-tracking problem, not a bigger-model problem:
the win came from making the conversation's stated constraints impossible to
lose or misapply, not from adding an LLM to the critical path.

## How this maps to the judging rubric

| Criterion | Weight | Where the evidence is |
|---|---:|---|
| Technical Execution | 35% | The [Results](#results) table and [Engineering evidence](#engineering-evidence) log every accepted *and* rejected experiment with its measured score, so the architecture reflects deliberate decisions rather than untested guesses. All numbers reproduce from a clean checkout (see [Reproduce the verified checks](#reproduce-the-verified-checks)); 49/49 unit tests pass. |
| Innovation & Problem Insight | 20% | [What produced the gain](#what-produced-the-gain) and [Why the prior is defensible](#why-the-prior-is-defensible-and-where-it-was-previously-rejected) show the core insight: two mechanisms that only ever act on evidence the shopper already gave or a catalog fact available for any product, applied in strict priority order so neither can override the other. |
| Impact & Relevance | 20% | The paragraph above states the real-world failure mode this solves. [Limitations](#limitations) is explicit about where the approach's assumptions stop holding, which is itself the boundary of its practical applicability. |
| Feasibility & Practicality | 15% | The scorer-proven path needs zero network access, zero API keys, and zero token spend (see [Technology and cost](#technology-and-cost)); it runs from a fresh `uv` environment with only the frozen catalog added, and a working [demo console](#demo) sits on top of the same production `Agent`, not a mock. |
| Presentation & Communication | 10% | Judged live at the final event; see [`docs/team_handoff.md`](docs/team_handoff.md) for the demo script and anticipated Q&A. |

## Results

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore | Previous |
|---|---:|---:|---:|---:|---:|
| Official public set, 200 sessions | `1.000` | `1.000000` | `1.995` | **`0.98010`** | `0.95415` |
| Deterministic paraphrase stress set | `1.000` | `0.997500` | `2.305` | **`0.97315`** | `0.95210` |
| Unseen-ASIN synthetic validation, 400 sessions | `1.000` | `1.000000` | `2.5725` | `0.96855` | `0.944175` |
| Untouched synthetic test, 400 sessions | `1.000` | `0.991500` | `2.560` | `0.96625` | `0.938888` |

The released BM25 starter achieved Hit@10 `0.125`, MRR `0.068034`, and MTTC
`9.81`. This agent reaches Hit@10 `1.000`, MRR `1.000000`, and MTTC `1.995`
without changing the official evaluator, catalog, labels, or protocol. Every
released session now converges, and every converged session converges at rank
one, so the entire remaining score gap is turns.

Per scenario the released set converges in `1.5125` turns for buying, `1.7875`
for browsing, `2.500` for boundary, and `3.6667` for intent override. The
override figure is within `0.067` turns of its structural floor: the evaluator
refuses to count a hit before the override lands (`evaluator/local_evaluator.py:238`),
and the override lands on turn `3` or `4`, giving a floor of `3.600` on this set.

### What produced the gain

Two mechanisms, both reading only catalog fields that are available for any
product at serving time.

**Phrase-level category resolution.** The shopper names a shelf ("Women
Sweaters"), and the previous pipeline dissolved that phrase into BM25 terms
that also match unrelated products. The catalog stores a breadcrumb path whose
upper levels are identical for every row, so `_category_labels` indexes the
informative tail at three widths and matches the stated category as a whole
phrase. Products on the named shelf are then ordered ahead of products that
merely share vocabulary with it. This is evidence the shopper actually stated,
so it is unconditional.

**A blended purchase-volume prior.** Once the stated constraints are exhausted,
hundreds of products can satisfy the intent card identically; at the turn where
the target is not yet rank one, the median satisfying block holds `168` to `487`
products. Nothing in the conversation separates them, because the card is
derived from features those products share verbatim. Review volume is the
available proxy for sales volume and the label is a purchased product, so the
prior is blended against lexical rank rather than replacing it:

```text
tier score = popularity_weight · log1p(reviews) − log1p(bm25_rank)
```

It is applied inside a tier that already agrees on satisfied constraints and
category, so it can reorder products the shopper cannot distinguish and can
never promote one that fails something the shopper said.

### Why the prior is defensible, and where it was previously rejected

The earlier revision rejected this prior because it lost `-0.0237` on the
synthetic test split. That gate is invalid for this specific question: the
synthetic generator emits four sessions for every catalog product, so its
targets are uniform by construction and no purchase prior can help there. The
released sessions are drawn from real purchase records, which makes the skew
structural rather than incidental — the median public target sits at the
**`0.9945` percentile** of the catalog by review count, with `6846` reviews
against a catalog median of `12`, and `63%` of targets fall in the catalog's
most-reviewed **one percent**. The private sessions come from the same
generator against the same catalog, so the same skew applies.

The prior no longer needs that argument to clear the synthetic gates, because
the two mechanisms are complementary rather than additive:

| Configuration | Public | Synthetic validation | Synthetic test |
|---|---:|---:|---:|
| Neither | `0.95415` | `0.94417` | `0.93889` |
| Category resolution only | `0.95935` | `0.95835` | `0.94870` |
| Purchase prior only | `0.96415` | `0.94203` | `0.93088` |
| Both | `0.97890` | `0.96855` | `0.96740` |

Alone, the prior does what the earlier revision measured: it helps the real
distribution and hurts the uniform one. Once the category tier pins the correct
shelf first, the prior only ever orders products that are already on it, and it
gains on all three splits including the two where its target distribution is
uniform. Ordering, not the signal, was the problem.

`popularity_weight` is a fitted parameter, so it was fitted twice. Tuned
independently on the disjoint odd and even halves of the public set, both halves
peak at `5.0` and both stay within `0.0015` across the range `2.4` to `8.0` —
a ridge rather than a spike. Half A improves `0.9628` to `0.9808` and half B
improves `0.9455` to `0.9770`.

Rating and price were tested as alternative or additional priors and carry no
signal: ordering the correct category shelf by review count alone puts the
target first in `70` of `200` sessions, and adding average rating (`68`) or
subtracting log price (`66`) makes it worse.

Override preference retention and `--recombine-constraints` are also on by
default; `--no-override-retain-hard` and `--no-recombine-constraints` restore
the earlier behaviour.

## How it works

```text
Shopper message
  ├─ released simulator wording ──> deterministic typed parser
  └─ unfamiliar free-form text ──> optional local model
                                      │
                                      └─ timeout + confidence + evidence gate
                                                       │
                                                       v
Typed session state: category, positive slots, negative slots, overrides
                              │
                              v
Exact catalog-card matches + phrase-level category resolution
      + weighted in-memory SQLite FTS5 BM25
                              │
                              v
Constraint tier -> category tier -> purchase prior blended with lexical rank
                              │
                              v
Unseen filtering + ambiguity-aware Top-K
                              │
                              v
Recommendations and a candidate-informed clarification question
```

The session state accumulates confirmed information while replacing only the
slot named by an override. “No leather” becomes a negative material constraint,
“forget leather” removes a prior material constraint, and “blue instead of
black” rewrites colour without erasing unrelated requirements.

The default ranker resolves candidates in strict evidence order: never violate
a stated negative, then satisfy the most stated constraints, then match the
stated category as a phrase, and only then fall back to the purchase prior
blended with lexical rank. Everything the shopper said is consumed before any
prior is consulted, so the prior can only ever order products the conversation
has left indistinguishable. The agent returns Top-1 while confidence is high,
widens under large exact-card ties, and uses all ten allowed positions on the
final turn. Already shown products
are filtered so each turn explores new candidates.

An explicit Buying/Browsing/Uncertain router is available behind
`--experimental-router`. Buying uses exact constraints and BM25; Browsing adds
dense retrieval and diversity; Uncertain fuses both routes. It remains off by
default because it did not beat the deterministic ranker end to end.

## Quick start

Requirements: Python `3.10+` (`3.13` is used in the verified setup), `uv`, and
the official frozen catalog.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r solution/requirements.txt
```

Download `catalog.jsonl.gz` from the
[official participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit),
decompress it, and place the 50,000-row file at:

```text
data/catalog.jsonl
```

Run the public evaluation:

```bash
.venv/bin/python run_solution.py --output results.json
```

Expected TechnicalScore: `0.98010` with zero reported model tokens.

## Demo

`frontend/` is a Next.js console for interactively driving the same
production `Agent` used for scoring — it is a UI layer over `solution/agent.py`,
not a separate reimplementation. A Node API route spawns
`frontend/backend/copilot_server.py`, which loads the real catalog and agent
once and answers turns over stdio, so the demo and the scorer never diverge.
Complete the [Quick start](#quick-start) steps first so `.venv` and
`data/catalog.jsonl` exist, then:

```bash
cd frontend
npm install
npm run dev
```

Set the `PYTHON` environment variable if the backend should use a different
interpreter than `../.venv/bin/python`.

The console shows the shopper-facing message, the structured `ask_attribute`,
ranked recommendations with catalog metadata, and the agent's internal typed
state (category, slots, negative constraints, inferred intent) for each turn,
so a reviewer can see the state machine described in
[How it works](#how-it-works) update live rather than trusting it from prose.

## Reproduce the verified checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_solution.py --output results.json
.venv/bin/python stress_eval.py --output stress_results.json
.venv/bin/python -m training.evaluate_extraction
```

Expected results:

- Tests: `49/49`
- Public TechnicalScore: `0.98010`
- Stress TechnicalScore: `0.97315`
- Rule-only free-form extraction micro-F1: `0.0583` across 200 seed cases
- Model calls on released simulator templates: `0`

The submission ZIP was also verified from a fresh temporary directory with
only `data/catalog.jsonl` added; it reproduced the same tests and scores.

## Optional experiments

The deterministic default only requires NumPy. Install heavier research
dependencies explicitly:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-experiments.txt
```

This enables dense retrieval, the experimental intent router, cross-encoder
diagnostics, and the scikit-learn training scripts. These paths may download
model weights and are not required for official scoring.

For the optional local structured extractor:

```bash
uv pip install --python .venv/bin/python -r solution/requirements-llm.txt
.venv/bin/python run_solution.py \
  --extractor-gguf /absolute/path/to/Qwen3-0.6B-Q8_0.gguf \
  --extraction-timeout 3
```

On an 18 GB Apple M3 Pro using Metal, Qwen3-0.6B Q8_0 loaded in `0.54s` and
used about `1.04 GB` additional peak process RSS. Its extraction benchmark is
single-author research data, and no tested prompt met the promotion gate after
the duplicate hardcoded grounding parser was removed. DSPy and OpenAI are used
only offline to generate prompt candidates; Qwen executes and scores those
candidates locally, and neither service is needed for official evaluation.

## Engineering evidence

We kept experiments only when they passed public, stress, and ASIN-separated
synthetic gates.

| Experiment | Outcome | Decision |
|---|---|---|
| Typed state, exact cards, aliases, negative constraints | Public `0.95365`; stress `0.95160` | Accepted |
| Ambiguity-gated Top-K policy | Improved validation and test TechnicalScore | Accepted |
| Clause-order metamorphic test | 189/189 state-equivalent; score delta `0.0` | Accepted as regression harness |
| Global dense retrieval and cross-encoder reranking | Lower end-to-end score | Rejected |
| Dense Buying/Browsing router | Public `0.95355` | Experimental flag only |
| Profile, title-specificity, popularity tie-breaks | Reduced public and/or stress score | Rejected |
| Override output widening | Improved MTTC but reduced combined score | Rejected |
| Free-form-gated top-100 reranker | Public/stress unchanged; model dev improved; frozen model test neutral | Experimental flag only |
| Trigram recall and confidence Top-K | Reduced public/stress MRR or smoke TechnicalScore | Rejected |
| Catalog-trained top-50 reranker | Public/stress unchanged; product-disjoint and frozen free-form tests improved | Validated optional candidate |
| Top-K widening, 25 counterfactual policies | Every variant below baseline | Rejected |
| Override preference retention | `+0.0005` to `+0.0007` on all four gates; Hit@10 and MRR unchanged | Accepted, on by default |
| Turn-gated strict popularity tie-break | Public `+0.0016`, stress `+0.0024`, synthetic test `-0.0237` | Superseded by the blended form |
| Ungated strict popularity tie-break | Public `0.95300`; MTTC gains lost to MRR | Superseded by the blended form |
| Average-rating tie-break | `68` of `200` shelf-first placements against `70` for review volume alone | Rejected, no signal |
| Price tie-break | `66` of `200` shelf-first placements against `70` for review volume alone | Rejected, no signal |
| Front-loading specific opening questions | Every single-attribute opening ask scored below the generic ask (best `0.9787`) | Rejected |
| Phrase-level category resolution | `+0.0052` public, `+0.0142` validation, `+0.0098` test | Accepted, on by default |
| Blended purchase prior on top of it | `+0.0196` public, `+0.0102` validation, `+0.0187` test | Accepted, on by default |
| Category term above the constraint tier | Public `0.9743`, and Hit@10 falls back to `0.995` | Rejected; the tier form is better |
| `"; "` constraint recombination | Public `+0.0012` to MRR `1.000`; synthetic within noise; repairs a real parser fault | Accepted, on by default |

### Why the Top-K policy is fixed

The evaluator ends a session at the target's first appearance
(`evaluator/local_evaluator.py:252`), so each session scores
`0.50·hit + 0.30·(1/rank) + 0.20·(11−turn)/10` on that turn and gets no second
attempt. Slipping from rank 1 to rank 2 costs `0.15`, while saving a turn
returns `0.02`, so a wider list must save seven turns to break even. Replaying
the public set under 25 alternative policies puts every one of them below the
`0.95365` reference. The emit-one-until-turn-7 policy is a consequence of that
arithmetic rather than a tuned parameter.

### Why the opening question stays generic

The simulator discloses at most two undisclosed card values per turn and serves
them in card order, so a generic ask always returns two values while a specific
ask returns only the values carrying that label — sometimes none, which wastes
the turn entirely. Front-loading the specific feature strings looked attractive
because they are more discriminative than the material and colour values served
first: asking for them would leave `87` of `200` sessions with a uniquely
determined product after turn two, against `66` for the generic ask. Measured
end to end, every single-attribute opening ask lost, because dropping the
material and colour values also drops the BM25 terms they contribute. The
`--ask-plan` flag keeps that experiment reproducible.

Detailed diagnostics and ablation reports live in `training/`; raw evaluator
outputs are retained under `artifacts/evaluations/` and are not needed at
runtime.

## Limitations

- Some intent cards describe hundreds of metadata-identical products without
  disclosing the title phrase that distinguishes the purchased item. The
  purchase prior that orders them is a prior, not evidence from the
  conversation: it assumes the labelled purchase is drawn with probability
  related to sales volume, which is a property of how the sessions were sampled
  rather than anything the shopper stated. That assumption is well supported on
  the released set and structural for a generator drawing on real purchase
  records, but it cannot be checked against the private sessions. The prior is
  consulted only after every stated constraint and the stated category, so a
  shopper who states enough never reaches it, and `--no-popularity-tiebreak`
  removes it entirely at a cost of `-0.0248` on the released set.
- `popularity_weight = 5.0` is fitted on the released public set. The split-half
  check above is evidence that the value transfers, not proof; the plateau is
  wide enough that any value between `2.4` and `8.0` behaves the same.
- `public_0020` used to be the single miss and now converges. Its card holds a
  value that itself contains `"; "`, the separator the shopper uses to join
  requirements; `--recombine-constraints`, now on by default, repairs the parse
  by preferring the longest span that is a real catalog value.
- The released simulator is deterministic. The stress and metamorphic harnesses
  test paraphrases and clause order, but they are not a substitute for a large
  independently authored conversation set.
- Dense retrieval and the explicit router are implemented but remain
  experimental because they did not beat the deterministic ranker end to end.
- The optional local extractor has not met its safety and accuracy promotion
  gates. The 200-case corpus is single-author seed data and must not be
  presented as independent human validation.

Score headroom is now entirely efficiency: Hit@10 and MRR are both at `1.000`
on the released set, leaving `0.0199`. That figure is not all reachable. The
intent-override floor alone accounts for `0.0078` of it, because the evaluator
will not count a hit before the override lands, and the remaining scenarios
converge in `1.51` to `2.50` turns against a floor of one. A realistic ceiling
is around `0.988`.

The remaining competition work is to earn turn-one convergence more often,
which means narrowing the candidate set before the shopper has said much:
resolve the stated category to a deeper node of the tree than the two segments
the simulator states, trace the dense-router regression, and calibrate selective
BM25/vector fusion for the browsing turns where the shelf is large. Top-K
changes still do not need testing; the scoring arithmetic above rules them out,
and it is now doubly binding because every hit is already at rank one. More
prompt optimization is only worthwhile for a post-hackathon human-facing product
or if organizers confirm free-form hidden messages.

## Repository layout

```text
solution/agent.py                 production agent and retrieval pipeline
solution/extraction.py            optional local structured extraction
run_solution.py                   official public evaluation entry point
stress_eval.py                    deterministic paraphrase stress harness
frontend/                         Next.js demo console over the production agent
artifacts/models/                 optional learned reranker artifact
training/                         synthetic splits, traces, ablations and reports
artifacts/evaluations/            archived raw experiment outputs
evaluator/                        unmodified official evaluator
data/public_set.jsonl             official 200-session development set
tests/                            parser, state, evaluator and training checks
docs/                             challenge contract and team handoff
```

## Technology and cost

- Python, SQLite FTS5, NumPy, and `uv`
- Optional: sentence-transformers, scikit-learn, llama.cpp, Qwen3 GGUF
- Demo only: Next.js, React, and Tailwind CSS (`frontend/`), unused by the
  scored path
- Dataset: frozen 50,000-product Amazon Reviews 2023
  `Clothing_Shoes_and_Jewelry` catalog supplied by TechJam
- Scorer-proven path: fully offline, no external API, no credentials, zero model
  tokens, and zero marginal inference cost

## Team

Team names and contribution ownership are intentionally left for the team to
confirm before Devpost submission. The proposed allocation and demo checklist
are in [`docs/team_handoff.md`](docs/team_handoff.md).

## Attribution

See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for dataset attribution and
[`docs/competition_specification.md`](docs/competition_specification.md) for the
official evaluation contract.
