# Model-authored transcript failure analysis

## Purpose

This analysis separates failures in the fixed model-authored transcript corpus into
retrieval, ranking, and evidence failures. It is diagnostic evidence, not an estimate
of the organizer's private score: the transcript writers generated adversarial,
free-form conversations without simulating the official evaluator's interaction
policy.

Reproduce it with:

```bash
python -m training.analyze_transcript_failures \
  --cases training/model_blind_packets/writer_*.jsonl \
  --split development \
  --output artifacts/evaluations/model_blind_failure_analysis_development.json

python -m training.analyze_transcript_failures \
  --cases training/model_blind_packets/writer_*.jsonl \
  --split test \
  --output artifacts/evaluations/model_blind_failure_analysis_test.json
```

## Results

| Best outcome across turns | Development (30) | Frozen test (30) |
| --- | ---: | ---: |
| Target emitted | 4 | 2 |
| Target in fused pool, below output cutoff | 17 | 17 |
| Target absent from BM25 despite lexical evidence | 2 | 5 |
| Transcript has insufficient target lexical evidence | 7 | 6 |

The fused-rank distribution adds an important qualification:

| Best fused rank | Development | Frozen test |
| --- | ---: | ---: |
| 1-3 | 4 | 2 |
| 4-10 | 4 | 4 |
| 11-100 | 6 | 8 |
| Over 100 | 7 | 5 |
| Not in fused pool | 9 | 11 |

No exact-document or identical-card near-duplicate tie was observed. The analyzer
checks this explicitly, but similarity beyond exact normalized equality remains a
future diagnostic.

## Interpretation

The largest actionable gap is ranking, not intent extraction alone. On frozen test,
17 targets enter the fused candidate pool but only four are close enough that a
larger Top-K policy could expose them. Thirteen sit below rank 10, so broadly relaxing
the output cutoff would damage MRR and MTTC without recovering most failures.

Eleven targets never enter the fused pool. Six of those conversations share less
than 15% of their query terms with the target document, indicating omitted or highly
indirect evidence. Five contain useful lexical evidence but BM25 still misses the
target, making candidate generation the secondary engineering target.

## Next experiments

1. Improve reranking only for the top 100 candidates, using field-aware compatibility
   and confidence gating. Evaluate MRR and TechnicalScore before retaining it.
2. Add a complementary retrieval route for the five lexical candidate misses, such
   as character n-grams or catalog-derived synonym expansion. Keep it in memory and
   fuse conservatively.
3. Increase output from one to a larger list only when the target-rank proxy indicates
   a weak top margin. A global Top-K increase is not supported by these results.
4. Treat insufficient-evidence cases as clarification-policy problems. Ask for a
   discriminative catalog facet rather than attempting to infer an exact ASIN from
   evidence the transcript never supplied.

The immediate implementation priority is therefore a gated top-candidate reranking
experiment, followed by candidate-recall work. Intent extraction remains important,
but its independent F1 improvement does not by itself close these ranking failures.
