# Catalog-trained top-50 reranker

## Design

The experiment leaves candidate generation and the proven simulator fast path
unchanged. On unfamiliar free-form turns only, a histogram gradient-boosting
classifier reorders the existing top 50 candidates. A missing, incompatible, or
invalid artifact silently restores the deterministic ranking.

Training uses catalog-derived intent cards. Whole ASINs are assigned by stable
SHA256 to train, validation, or test, and all 200 public target ASINs are
quarantined before example generation. Each positive is paired with the first
12 non-target products from the real retrieval result, producing hard rather
than random negatives.

The 15 runtime-observable features are baseline and BM25 reciprocal rank, exact
constraint and character coverage, negative conflicts, query-document coverage
in both directions, category overlap, rarity-weighted facet coverage, and six
typed-slot match indicators. The model never receives target ASIN, sample ID,
scenario label, future turns, or hidden intent-card values at runtime.

## Results

| Evaluation | Baseline | Learned reranker | Delta |
|---|---:|---:|---:|
| Product-disjoint validation MRR, 300 ASINs | 0.899089 | 0.908790 | +0.009701 |
| Product-disjoint frozen test MRR, 300 ASINs | 0.935378 | 0.941609 | +0.006231 |
| Official public TechnicalScore, 200 sessions | 0.953650 | 0.953650 | 0.000000 |
| Paraphrase stress TechnicalScore, 200 sessions | 0.951600 | 0.951600 | 0.000000 |
| Model-authored development TechnicalScore, 30 sessions | 0.103250 | 0.167250 | +0.064000 |
| Model-authored frozen test TechnicalScore, 30 sessions | 0.065334 | 0.098667 | +0.033333 |

Candidate recall at 50 was `0.986667` on both catalog-derived validation and
test. The reranker cannot repair the remaining candidate misses.

## Interpretation

This is the first tested ranking signal to improve both product-disjoint and
independently authored frozen evaluations without reducing the established
public or stress scores. It remains optional because loading the 223 KB joblib
artifact requires scikit-learn, whereas the default submission only requires
NumPy. The model-authored corpus is small and adversarial, so its absolute score
must not be presented as an estimate of the organizer's private score.

Reproduce training with:

```bash
.venv/bin/python -m training.train_learned_reranker \
  --train-products 800 \
  --validation-products 300
```

The checked-in artifact is `artifacts/models/catalog_reranker.joblib`.
Evaluate it end to end with:

```bash
.venv/bin/python run_solution.py \
  --learned-reranker artifacts/models/catalog_reranker.joblib
```

## Clarification invariant

The official local evaluator ignores the semantic content of the response
message when generating the next user turn. `ask_attribute="other"` reveals up
to two undisclosed constraints of any type; a named attribute reveals only
undisclosed constraints classified under that attribute. Therefore richer
question wording cannot affect TechnicalScore. A named attribute is useful only
if it skips broad but low-value disclosures and directly requests a remaining
discriminative facet. The proven default remains `other`.
