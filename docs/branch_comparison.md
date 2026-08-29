# Main and Shwe Branch Comparison

This document compares `weisintai/main` at commit `02d8b34` with
`shwe-experiment` at commit `5532516`. It identifies shared ideas, meaningful
implementation differences, successful and unsuccessful experiments, and the
best approaches to carry forward.

## Executive Conclusion

Main is the stronger submission baseline. It leads on the official public set,
deterministic paraphrase stress set, and both ASIN-separated synthetic splits.
Its normalized exact-card retrieval and ambiguity-aware output policy are the
most consistently validated techniques in the repository.

Shwe is valuable as an experimental branch. It demonstrates that multi-route
retrieval, information-gain clarification, pairwise reranking, compatibility
promotion, and confidence-based stopping can improve specific stages. However,
its complete pipeline should not replace main because its manually enumerated,
template-sensitive intent layer causes a severe paraphrase failure.

The recommended direction is to preserve main's production path and selectively
ablate Shwe components against it.

## Aligned Results

The same evaluator protocol, paraphrase transformation, synthetic generator,
split seed, and session limits were used for both branches. Only the imported
agent and constructor were adapted.

| Evaluation | Main TechnicalScore | Shwe TechnicalScore | Gap |
|---|---:|---:|---:|
| Official public, 200 sessions | 0.953650 | 0.933163 | -0.020487 |
| Deterministic paraphrase stress, 200 sessions | 0.951600 | 0.537042 | -0.414558 |
| Unseen-ASIN validation, 400 sessions | 0.943475 | 0.871505 | -0.071970 |
| Untouched synthetic test, 400 sessions | 0.938188 | 0.847173 | -0.091015 |

Detailed Shwe metrics:

| Evaluation | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official public | 0.995 | 0.905875 | 2.805 | 0.933163 |
| Paraphrase stress | 0.640 | 0.350141 | 5.400 | 0.537042 |
| Synthetic validation | 0.970 | 0.786016 | 3.465 | 0.871505 |
| Synthetic test | 0.9375 | 0.775077 | 3.705 | 0.847173 |

The synthetic results show that Shwe retrieves many unseen products, but ranks
them less precisely than main. The much larger stress drop shows that Shwe's
primary failure occurs before retrieval: semantically equivalent paraphrases are
not converted into equivalent state.

## Shared Approaches

Both branches use several sound foundations:

- A deterministic, zero-model-call default path.
- In-memory SQLite FTS5 retrieval over the read-only 50,000-product catalog.
- Structured conversational state accumulated across turns.
- Explicit handling for positive and negative constraints.
- Weighted lexical fields rather than uniform keyword matching.
- Alias or normalization logic for common product language.
- Clarification questions based on missing or useful product attributes.
- Recommendation timing designed around the MRR-versus-MTTC tradeoff.
- Optional dense, cross-encoder, router, or local-LLM experiments kept out of the
  default path when they failed end-to-end evaluation.
- Deterministic tie ordering and reproducible evaluation.

These shared decisions are well supported and should remain in the final system.

## Architectural Differences

| Area | Main | Shwe | Assessment |
|---|---|---|---|
| Intent representation | Category plus normalized constraints and typed slots | `IntentDelta`, phrases, product types, audiences, and manually defined slots | Main is simpler and more robust on the tested language distribution |
| Normalization | Generates multiple normalized variants for each constraint | Small static alias table and token vocabularies | Main wins decisively on paraphrase stress |
| Catalog grounding | Builds exact intent cards and an inverted exact-card index | Builds broad token facets and facet lookups | Main gives stronger evaluator-aligned evidence; Shwe gives broader recall |
| Retrieval | Exact-card candidates plus weighted BM25; optional selective semantic routes | Six FTS/facet routes fused into 350 candidates | Shwe is more heterogeneous, but added complexity does not improve aligned scores |
| Initial ranking | Constraint-first exact evidence followed by lexical rank | Handwritten constraint score over all 350 candidates | Main is easier to calibrate and explain |
| Learned ranking | No learned reranker in the proven default | Pairwise logistic reranker over the top 40 | Shwe improves its own MRR, but still trails main on unseen ASINs |
| Compatibility | Constraint and slot-aware ordering | Bounded promotion by product family and audience | Shwe helps free-form product-type errors, but depends on parser accuracy |
| Overrides | Slot-level replacement, removal, and exclusion | Broad recommendation-epoch reset with category retention | Main preserves unrelated confirmed constraints more reliably |
| Product history | Filters products already shown in the session | Does not use shown-product filtering in the current path | Main explores more catalog candidates across turns |
| Top-K policy | Top-1 when confident, wider under exact-card ambiguity, Top-10 on final turn | Usually defers early results; releases early on score-margin and retrieval-agreement thresholds | Main has better MRR and synthetic generalization |
| Clarification | Candidate-informed missing attribute questions | Broad `other` questions followed by entropy-based information gain | Shwe is more principled, but broad `other` is simulator-specific |
| Intent router | Buying/Browsing/Uncertain router available but disabled by default | Mode labels influence question caps and stopping | Neither branch has proven a routed default superior to the simpler path |

## What Went Well in Main

### Exact catalog cards

Main uses the same catalog-derived intent-card abstraction that drives the
released evaluator. Exact-card matches provide high-precision evidence before
BM25 tie-breaking. This is the largest architectural reason for its high MRR.

### Constraint variants

Normalized variants let messages such as `polyester` and `synthetic textile`
reach equivalent constraints. The near-identical public and stress scores show
that this normalization survives the deterministic paraphrase harness.

### Ambiguity-gated Top-K

Main does not use a fixed output size. It emits Top-1 when the evidence is clear,
widens when many products share the same exact card, and uses the full allowance
on the final turn. This directly manages MRR, Hit@10, and MTTC together.

### Seen-product filtering

Products already exposed to the evaluator are removed on later turns. This turns
additional conversation rounds into exploration instead of repeating the same
ranking.

### Promotion discipline

Main kept only changes that survived public, stress, validation, and untouched
test gates. Dense retrieval, cross-encoder ranking, the explicit router, profile
tie-breaks, and override widening remained experimental or were rejected when
their end-to-end score declined.

## What Did Not Go Well in Main

### Arbitrary free-form extraction remains weak

The deterministic stress harness uses a finite transformation set. Main's
rule-only raw intent-delta micro-F1 was reported as `0.0583`, so the strong stress
score should not be interpreted as broad natural-language understanding.

### Exact-card alignment is evaluator-specific

Exact cards are highly effective under the released simulator because both are
derived from the same catalog metadata. If private prompts disclose different
attributes or use genuinely independent language, that advantage may shrink.

### Metadata-identical products remain ambiguous

Some products share the same visible intent card. When the conversation exposes
no distinguishing evidence, any ordering among those products is a prior rather
than a logically justified match.

### Heavier semantic methods did not earn their cost

Global dense retrieval and cross-encoder reranking reduced end-to-end quality.
The Buying/Browsing router also scored slightly below the proven default. These
components increase operational complexity without a validated score gain.

## What Went Well in Shwe

### Multi-route retrieval achieved high public recall

Shwe combines conversation, hard phrase, category, latest-message, combined
constraint, and structured-facet routes. Public Hit@10 reached `0.995`, showing
that the broad candidate stage rarely loses the target under official wording.

### Pairwise reranking improved Shwe's baseline

Training against hard negatives among current candidates increased public MRR
substantially over Shwe's earlier handwritten ranker. It was dependency-free at
runtime and trained with session-grouped folds.

### Confidence-based stopping reduced MTTC safely

Counterfactual replay identified score margin and fused-retrieval agreement as
useful stopping signals. The accepted policy reduced public MTTC from `3.29` to
`2.805` without changing public MRR or Hit@10, increasing TechnicalScore from
`0.923462` to `0.933163`.

### Information-gain clarification is a useful research direction

Entropy and candidate-facet coverage provide a defensible way to choose among
unresolved attributes. This is more general than a fixed question sequence and
is worth testing on top of main's stronger state and retrieval foundation.

### Bounded compatibility promotion limits damage

Shwe promotes at most three products that match both product type and audience.
The bound is important: it adds systematic compatibility without letting one
uncertain parser decision replace the whole retrieval ranking.

## What Did Not Go Well in Shwe

### Template-sensitive intent parsing

Shwe recognizes official phrases such as `A key requirement is` and
`What matters is` much better than equivalent free-form language. Its stress
TechnicalScore falls to `0.537042`, with Hit@10 only `0.640`. This is the
branch's largest gap.

### Multiple extraction paths can disagree

`starter/intent.py`, template phrase extraction, and `_message_slots()` all add
state. A term removed or excluded by one path can be reintroduced by another.
The final system should have one atomic evidence-to-delta-to-state pipeline.

### Manually enumerated ontology limits product breadth

Product types, colors, materials, styles, and use cases are static code lists.
The 50,000-product catalog contains terminology outside those lists, making
private and free-form behavior less predictable.

### Broad override reset loses valid context

An override clears most accumulated constraints instead of rewriting only the
affected slot. This is safe for the released full-override template but weak for
requests such as `blue instead of black, but still waterproof`.

### Learned ranking does not fully generalize

Shwe's synthetic test MRR is `0.775077`, versus main's `0.970792`. Some public
targets that retrieval placed first were demoted by learned feature interactions.
The pairwise model should not be moved into main without an ablation showing a
gain on all four gates.

### Boundary and override scenarios remain weak

Boundary users disclose little evidence, while overrides require correct state
editing and renewed retrieval. These scenarios account for much of Shwe's
synthetic degradation and longer MTTC.

## Approaches to Keep

The final candidate should retain these proven techniques:

1. Main's normalized exact-card index plus weighted BM25.
2. Main's atomic slot-level accumulate, replace, remove, and exclude semantics.
3. Main's seen-product filtering and ambiguity-gated Top-K policy.
4. Deterministic execution with zero model calls on official templates.
5. Shwe's counterfactual policy replay as an offline analysis tool.
6. Shwe's information-gain question scoring as an optional candidate policy.
7. Bounded compatibility checks, provided they use catalog-grounded types.
8. Four-gate promotion: public, paraphrase stress, synthetic validation, and
   untouched synthetic test.

## Approaches to Avoid or Isolate

- Do not use a global dense route merely because it is semantically appealing.
- Do not add a cross-encoder without a measured end-to-end gain.
- Do not enable the explicit Buying/Browsing router by default yet.
- Do not use popularity, profile, or title-specificity tie-breaks without user
  evidence.
- Do not expand a manually coded ontology category by category.
- Do not allow an LLM delta to mutate state without evidence validation.
- Do not tune only on the 200 public sessions.
- Do not combine multiple successful local experiments at once; interactions
  can erase their individual gains.

## Recommended Convergence Plan

### Phase 1: Freeze main as the control

Preserve main's current default agent and reproduce all four benchmark rows.
Every experiment must report its delta against that fixed control.

### Phase 2: Unify intent updates

Keep main's state semantics, then add evidence-span extraction and catalog-derived
normalization. Test arbitrary independently written prompts in addition to the
deterministic stress transformations. A local LLM may propose a delta only when
rules are uncertain, and unsupported evidence must be rejected.

### Phase 3: Test Shwe clarification independently

Port only information-gain attribute selection. Do not port Shwe retrieval or
reranking in the same experiment. Promote it only if TechnicalScore does not
decline on any aligned gate and boundary/override metrics remain stable.

### Phase 4: Test ranking additions independently

Evaluate bounded compatibility first because it is interpretable. Evaluate the
pairwise reranker separately with ASIN-grouped training and untouched-test
selection. Optimize reciprocal-rank damage, not generic pair accuracy.

### Phase 5: Calibrate output policy last

Once state and ranking are frozen, replay recommend-now versus ask decisions.
Compare main's ambiguity-gated Top-K with Shwe's confidence stopping and a hybrid
policy. Select thresholds on synthetic validation, then evaluate the frozen
policy once on synthetic test.

## Promotion Checklist

A change is ready for the default path only when:

- Public TechnicalScore is not materially lower.
- Paraphrase stress does not regress.
- Synthetic validation improves for a stated reason.
- The frozen change also improves or preserves untouched synthetic test.
- Buying, Browsing, Boundary, and Intent Override metrics are inspected
  separately.
- Free-form state tests cover accumulation, slot override, removal, exclusion,
  no-preference, and category switch.
- Runtime dependencies, latency, memory, and model calls remain acceptable.
- The experiment changes one major variable and records rejected outcomes.

This process narrows the solution toward techniques that improve the complete
shopping workflow, rather than techniques that look promising in isolation.
