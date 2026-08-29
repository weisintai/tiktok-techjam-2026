# Literature Review: Conversational Product Search

Reviewed 29 August 2026. This document compares the hybrid robustness branch
with primary research and records competition-specific decisions.

## Architectural comparison

| Research pattern | Evidence | Current implementation | Decision |
|---|---|---|---|
| Structured dialogue memory plus a closed retrieve/clarify loop | [ProductAgent (EMNLP Industry 2025)](https://aclanthology.org/2025.emnlp-industry.25/) | Typed slots, negative constraints, overrides, no-preference and unresolved state feed retrieval every turn | Keep |
| Text context alongside imperfect product attributes | [ConvSearch (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.280/) | Exact facets are hard evidence; bounded residual free text is soft BM25/dense evidence | Keep; this directly supports the soft-query design |
| Constrained schema outputs for robust state tracking | [SPLAT (ACL 2023)](https://aclanthology.org/2023.acl-long.6/) | `StructuredTurn` constrains intent and operations to seven slots while catalog phrases supply values | Keep |
| Fine-grained positive and negative aspect feedback | [Conversational Product Search Based on Negative Feedback](https://arxiv.org/abs/1909.02071) | Explicit add/remove/exclude and slot replacement are supported | Extend recommendation-referential feedback only after blind examples exist |
| Candidate-grounded strategic clarification | [ProductAgent](https://aclanthology.org/2025.emnlp-industry.25/), [Expected Value of Perfect Information](https://aclanthology.org/P18-1255/) | Candidate facet entropy and coverage are available; a narrower information-gain policy was tested and reduced TechnicalScore | Keep the proven broad policy; test gating, not global activation |
| Check question presuppositions before information gain | [Asking More Informative Questions for Grounded Retrieval](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-naacl.276.pdf) | Low-coverage facets are suppressed, but answerability is only approximated by candidate coverage | Add an answerability gate only in a controlled clarification ablation |
| Usage intent independent of a rigid product ontology | [Usage-centric Intent Understanding (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.14/) | `use_case` is typed when known; unknown scenario language survives as soft query text | Keep; expand evaluation rather than hardcoding more use cases |
| Consultation history can reveal motivation beyond the final query | [MAPS (ACL 2025)](https://aclanthology.org/2025.acl-long.152/) | Up to three fallback turns are retained as soft retrieval context; explicit state remains authoritative | Keep bounded memory to control stale-context noise |
| Hybrid symbolic and dense retrieval with semantic reranking | [ProductAgent](https://aclanthology.org/2025.emnlp-industry.25/) | BM25, exact facets, optional dense retrieval, RRF and optional cross-encoder are implemented | Do not enable globally: prior local ablations reduced score; retest browsing-only |

## Where this agent is strong

1. The state representation is explicit, inspectable and supports addition,
   replacement, removal, exclusion and category override.
2. Exact constraints and negative constraints are separated from soft text,
   avoiding the common failure where uncertain language becomes a hard filter.
3. The fixed catalog contributes vocabulary without mutating the catalog or
   treating noisy metadata as trusted ontology.
4. Retrieval combines weighted lexical evidence, exact facet candidates and
   optional semantic routes in memory.
5. Every promoted change is gated by Hit@10, MRR, MTTC and TechnicalScore; this
   has correctly rejected plausible research ideas that do not fit the official
   simulator or ten-turn objective.

## Remaining gaps

### 1. Independent language coverage

The largest evidence gap is evaluation data, not another model. The current
free-form holdout is temporally separated but single-author. Research datasets
use crowd workers, transferred utterances or simulators to create linguistic
variation. Recruit multiple team members to write blind multi-turn sessions
without seeing parser vocabulary, then freeze ASIN-separated development and
test partitions.

### 2. Recommendation-referential feedback

The parser handles `no leather` but not reliably `the second one is too bulky`
or `more like the first, but lighter`. Research on negative and comparative
feedback models these as aspect evidence from displayed items. Supporting this
requires storing the displayed product-to-facet snapshot and resolving product
references before updating slots. Do not infer it from keywords alone.

### 3. Browsing semantics

Open-ended scenario matching benefits most from dense retrieval, while buying
benefits from exact constraints. The code supports this distinction, but the
proven default remains lexical because global dense retrieval reduced score.
The next retrieval experiment should activate dense candidates only when the
router reports browsing and no hard facet has high confidence, then fuse with
BM25 using RRF and preserve exact constraints during reranking.

### 4. Clarification answerability

Entropy alone can ask a high-information question that the user cannot answer.
Before asking, estimate answerability from explicit unresolved slots, prior
language, and candidate facet coverage. Clarify only when expected rank gain is
large enough to offset one MTTC turn; otherwise recommend immediately.

## Prioritized experiments

1. **Blind human end-to-end set:** highest value and required for defensible
   robustness claims.
2. **Browsing-only dense route:** compare against the frozen lexical default on
   public, stress, synthetic validation and blind free-form sessions.
3. **Reference-aware feedback:** add displayed-item facet snapshots and test
   `first/second/these` plus comparative language on a separate branch.
4. **Answerability-gated clarification:** require predicted candidate reduction
   and unresolved-slot evidence; retain only if TechnicalScore and MTTC improve.
5. **Small model extractor:** lowest priority. Previous local-model experiments
   did not beat deterministic extraction reliably enough to justify latency.

## Techniques not recommended now

- Global dense retrieval or global cross-encoder reranking without a passing
  ablation.
- A fully generative dialogue-state tracker, which adds latency and unsupported
  state mutation risk for a fixed, small schema.
- Reinforcement learning on the 200 public sessions, which would overfit the
  released simulator and weaken private-set evidence.
- More manually hardcoded catalog words. Prefer filtered catalog evidence,
  soft retrieval context and independently measured error families.

## Experiment results

Experiments were run after this review against the catalog-grounded robustness
baseline (`0.953650` public, `0.951600` stress).

| Candidate | Public TechnicalScore | Stress TechnicalScore | Decision |
|---|---:|---:|---|
| Router only, lexical retrieval | 0.953650 | Not required | Neutral control |
| Router plus browsing-only MiniLM dense retrieval | 0.953550 | Not run | Rejected at public gate |
| Ungated adaptive questions | 0.946150 | Not run | Rejected at public gate |
| Answerability-gated questions | 0.953650 | 0.951600 combined below | Keep experimental |
| Reference feedback plus gated questions | 0.953650 | 0.951600 | Keep experimental |

Browsing-only dense retrieval preserved Hit@10 and MRR but increased browsing
MTTC from `2.7500` to `2.7625`. Ungated adaptive questions reduced MRR from
`0.985167` to `0.980167` and increased MTTC from `2.97` to `3.27`.

Reference resolution was separately checked on 200 catalog-grounded facet
references and 89 whole-product similarity references across ordinal and
numeric forms. Exact resolution was `1.0000` with zero leaked slots. These are
generated boundary tests, not independent purchase-ranking evidence.

Blind evaluation tooling is provided by `prepare_blind_sessions.py` and
`evaluate_blind_transcripts.py`. No blind TechnicalScore is reported until
independent writers complete the generated packets.

### Local-model writer probe

As a substitute for unavailable subagent spawning, three pre-existing local
Ollama models independently filled 20 disjoint assignments each: Qwen3 1.7B,
Gemma3 1B and Llama 3.2 1B. They received only scenario instructions and
product briefs. The resulting 60-session corpus is frozen by
`model_blind_packets.sha256` and explicitly labeled model-authored.

| Split | Default TechnicalScore | Reference + gated questions |
|---|---:|---:|
| Development, 30 | 0.103250 | 0.103250 |
| Frozen test, 30 | 0.065334 | 0.065334 |

This is an adversarial robustness probe, not a private-score estimate. The
writers paraphrased raw catalog briefs, while the official simulator reveals
intent-card constraints in response to agent questions. Fixed transcripts also
cannot answer the agent's actual clarification wording. The result shows that
exact-ASIN ranking from unconstrained natural descriptions is a major gap, and
that the two experimental dialogue features do not address that retrieval gap.
