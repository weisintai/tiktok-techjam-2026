# Team handoff

## Current recommendation

Submit the deterministic agent as the official default. It is the only path
that passes every promotion gate, runs offline, requires no credentials, and
reproduces TechnicalScore `0.95365` from the clean checkpoint.

Keep the Buying/Browsing router and Qwen extractor as demoable experiments.
Presenting them honestly is stronger than enabling paths that reduced the
measured end-to-end score.

Do not assign more competition time to DSPy prompt optimization unless the
organizers confirm that private sessions contain arbitrary free-form language.
The official contract describes scenario-driven simulator sessions with unseen
users and products, so the remaining score work belongs in retrieval, ranking,
Top-K policy, and regression verification.

## Remaining engineering ownership

- **Retrieval:** Trace the current dense-router regression and test selective
  BM25/vector fusion on ASIN-separated validation, because candidate recall
  directly controls private Hit@10.
- **Ranking:** Diagnose target-rank losses, then test constraint-first semantic
  reranking and Top-K replay, because moving an already-retrieved target upward
  directly controls MRR.
- **Evaluation and integration:** Guard the `0.95365` public and `0.95160`
  stress baselines, run the frozen synthetic test only after validation wins,
  and verify latency, memory, offline execution, and submission contents.
- **Prompt/model work:** Keep the existing Qwen and DSPy artifacts reproducible
  for the demo, but do not continue optimizing them for the official scorer.

## Decisions to confirm together

- [ ] Final project name and one-sentence tagline
- [ ] Team member names and contribution ownership
- [ ] Who records and uploads the public YouTube demo
- [ ] Whether the demo machine will include the optional Qwen GGUF model
- [ ] Whether to demonstrate `--experimental-router` live or only explain its ablation
- [ ] Final GitHub repository URL and Devpost URL

## Suggested contribution table

| Member | Proposed ownership                     | Confirmed contribution   |
| ------ | -------------------------------------- | ------------------------ |
| _Name_ | Agent architecture and retrieval       | _Fill before submission_ |
| _Name_ | Evaluation, stress tests and ablations | _Fill before submission_ |
| _Name_ | Demo, Devpost story and presentation   | _Fill before submission_ |

Delete unused rows rather than assigning work that did not happen.

## Suggested three-minute demo

1. **Problem and result — 20 seconds.** Show the starter-to-final comparison:
   Hit@10 `0.125 → 0.995`, MRR `0.068034 → 0.985167`, MTTC `9.81 → 2.97`.
2. **Buying flow — 40 seconds.** Show a concrete category plus hard constraint,
   typed state accumulation, Top-1 recommendations, and a clarification.
3. **Intent override — 40 seconds.** Demonstrate “blue instead of black” and
   show that unrelated material or budget slots survive.
4. **Browsing and routing — 35 seconds.** Explain the explicit router, then show
   why the proven default remains deterministic after the dense route regressed.
5. **Safety and practicality — 30 seconds.** Show zero official model calls,
   offline execution, timeout fallback, and evidence quarantine for an invented
   model field.
6. **Close — 15 seconds.** State the user benefit: fewer turns without losing
   hard constraints or silently rewriting intent.

## Judge-facing story

The memorable point is: **we used models where they added evidence and removed
them where they only added complexity.** The result is a stateful conversational
agent that behaves like a modern copilot while remaining cheap, offline, and
measurably stronger than the starter.

Useful proof points:

- Public TechnicalScore `0.95365`; stress `0.95160`
- Zero model calls and zero token cost on released simulator templates
- 189/189 state-equivalent clause mutations produced identical outputs
- Dense, cross-encoder, profile, popularity and widening experiments were
  rejected when they failed end-to-end gates
- The final ZIP reproduced from a fresh `uv` environment with only the frozen
  catalog added

## Likely Q&A

**Why no LLM in the official path?** The evaluator uses structured,
scenario-driven simulated messages, and the deterministic path handles every
released template without model calls. Private sessions use unseen users and
products, but the contract does not say they switch to arbitrary human-written
turns. The optional local model demonstrates the broader product direction
behind confidence, timeout, and evidence gates.

**Is the Buying/Browsing router real?** Yes. It is implemented behind a flag,
but the dense browsing route reduced public TechnicalScore by `0.00010`, so it
was not promoted. This is an evidence-based production decision, not an absent
feature.

**What limits the remaining score?** A few intent cards leave hundreds of
metadata-identical products tied while withholding the title phrase that would
identify the purchase. Popularity and title priors moved those examples but
hurt the wider evaluation.

**How would this generalize beyond the hackathon?** Build a larger independently
authored dialogue set, calibrate routing and Top-K policies on that holdout, and
retain the deterministic constraint/state layer as the safety boundary.

## Final checklist

- [ ] Add team names and contributions to the root README
- [ ] Record one complete multi-turn session
- [ ] Upload the video publicly and add its link to Devpost
- [ ] Run the clean ZIP verification one final time
- [ ] Confirm the catalog and model weights are absent from the submitted ZIP
- [ ] Confirm no `.env`, credentials, generated embeddings, or private data exist
- [ ] Tag the exact submitted revision and save its ZIP SHA-256
