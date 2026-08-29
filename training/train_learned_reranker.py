from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluator.local_evaluator import coarse_category, intent_card, load_jsonl
from solution.agent import Agent, _constraint_variants
from training.generate_sessions import split_for_asin


def load_products(path: str | Path) -> dict[str, dict]:
    products = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                product = json.loads(line)
                products[str(product["parent_asin"])] = product
    return products


def query_for(product: dict) -> tuple[str, list[str], str]:
    card = intent_card(product)
    categories = [str(value) for value in product.get("categories") or []]
    category = coarse_category(categories)
    constraints = [
        str(value) for value in [*card["hard_constraints"], *card["soft_preferences"]]
    ]
    query = f"I need {category} suitable for " + ", ".join(constraints)
    return category, constraints, query


def candidate_features(
    agent: Agent, category: str, constraints: list[str], query: str
) -> tuple[list[str], list[list[float]]]:
    ranked, _ = agent.rank_with_diagnostics(category, constraints, soft_query=query)
    head = ranked[:50]
    groups = [_constraint_variants(value) for value in constraints]
    expanded = [variant for group in groups for variant in group]
    retrieval_query = " ".join([category, *constraints, *expanded, query]).strip()
    bm25_rank = {
        asin: rank for rank, (asin, _) in enumerate(agent._bm25_scored(retrieval_query), 1)
    }
    exact_counts = {
        asin: sum(bool(group & agent.cards.get(asin, set())) for group in groups)
        for asin in head
    }
    features = [
        agent._learned_features(
            asin, rank, retrieval_query, category, groups, [], bm25_rank, exact_counts
        )
        for rank, asin in enumerate(head, 1)
    ]
    return head, features


def collect(
    agent: Agent,
    products: dict[str, dict],
    split: str,
    quarantine: set[str],
    limit: int,
    negatives: int,
) -> tuple[list[list[float]], list[int], list[tuple[str, list[str], list[list[float]]]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    queries = []
    count = 0
    for asin, product in products.items():
        if asin in quarantine or split_for_asin(asin) != split:
            continue
        category, constraints, query = query_for(product)
        candidates, features = candidate_features(agent, category, constraints, query)
        if asin not in candidates:
            queries.append((asin, candidates, features))
        else:
            target_index = candidates.index(asin)
            selected = [target_index, *[i for i in range(len(candidates)) if i != target_index][:negatives]]
            rows.extend(features[index] for index in selected)
            labels.extend(int(index == target_index) for index in selected)
            queries.append((asin, candidates, features))
        count += 1
        if count >= limit:
            break
    return rows, labels, queries


def ranking_metrics(model: object, queries: list[tuple[str, list[str], list[list[float]]]]) -> dict:
    reciprocal_ranks = []
    baseline_ranks = []
    hits = 0
    for target, candidates, features in queries:
        if target not in candidates:
            reciprocal_ranks.append(0.0)
            baseline_ranks.append(0.0)
            continue
        hits += 1
        baseline_rank = candidates.index(target) + 1
        baseline_ranks.append(1.0 / baseline_rank)
        scores = model.predict_proba(features)[:, 1]
        reranked = [
            asin for _, _, asin in sorted(
                zip(scores, range(len(candidates)), candidates),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
        ]
        reciprocal_ranks.append(1.0 / (reranked.index(target) + 1))
    total = max(1, len(queries))
    return {
        "queries": len(queries),
        "candidate_recall_at_50": round(hits / total, 6),
        "baseline_mrr": round(float(np.mean(baseline_ranks)), 6),
        "reranked_mrr": round(float(np.mean(reciprocal_ranks)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ASIN-disjoint top-50 reranker")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/models/catalog_reranker.joblib")
    parser.add_argument("--report", default="artifacts/evaluations/catalog_reranker.json")
    parser.add_argument("--train-products", type=int, default=1500)
    parser.add_argument("--validation-products", type=int, default=500)
    parser.add_argument("--negatives", type=int, default=12)
    args = parser.parse_args()

    products = load_products(args.catalog)
    quarantine = {
        str(row["ground_truth"]["parent_asin"]) for row in load_jsonl(args.public_set)
    }
    agent = Agent(args.catalog)
    x_train, y_train, _ = collect(
        agent, products, "train", quarantine, args.train_products, args.negatives
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=20260829,
    )
    weights = np.where(np.asarray(y_train) == 1, float(args.negatives), 1.0)
    model.fit(x_train, y_train, sample_weight=weights)
    _, _, validation_queries = collect(
        agent, products, "validation", quarantine, args.validation_products, args.negatives
    )
    report = {
        "schema_version": 1,
        "public_targets_quarantined": len(quarantine),
        "train_products": args.train_products,
        "validation_products": len(validation_queries),
        "training_rows": len(y_train),
        "positive_rows": sum(y_train),
        "validation": ranking_metrics(model, validation_queries),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "model": model, "report": report}, output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
