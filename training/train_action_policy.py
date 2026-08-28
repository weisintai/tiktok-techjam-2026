from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from evaluator.local_evaluator import catalog_index, coarse_category, intent_card
from solution.agent import Agent
from training.generate_sessions import split_for_asin


DIAGNOSTIC_FEATURES = (
    "turn", "constraint_count", "variant_count", "candidate_count",
    "best_exact_count", "exact_tie_count", "complete_match_count",
    "bm25_result_count", "bm25_relative_gap", "negative_constraint_count",
)


def rank_bucket(rank: int | None) -> int:
    if rank == 1:
        return 0
    if rank is not None and rank <= 3:
        return 1
    if rank is not None and rank <= 10:
        return 2
    return 3


def feature_row(turn: int, diagnostics: dict[str, float]) -> list[float]:
    values = {"turn": float(turn), **diagnostics}
    return [float(values[name]) for name in DIAGNOSTIC_FEATURES]


def product_states(product: dict) -> list[tuple[int, list[str]]]:
    card = intent_card(product)
    constraints = list(dict.fromkeys([
        *[str(value) for value in card["hard_constraints"]],
        *[str(value) for value in card["soft_preferences"]],
    ]))
    states: list[tuple[int, list[str]]] = [(1, [])]
    for index in range(1, min(4, len(constraints)) + 1):
        states.append((min(index + 1, 4), constraints[:index]))
    return states


def collect(
    agent: Agent,
    products: dict[str, dict],
    categories: dict[str, list[str]],
    split: str,
    limit_products: int,
    seed: int,
    excluded_asins: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    asins: list[str] = []
    selected = sorted(
        asin for asin in products
        if split_for_asin(asin, seed) == split and asin not in (excluded_asins or set())
    )[:limit_products]
    for asin in selected:
        category = coarse_category(categories.get(asin, []))
        for turn, constraints in product_states(products[asin]):
            ranked, diagnostics = agent.rank_with_diagnostics(category, constraints)
            try:
                rank = ranked.index(asin) + 1
            except ValueError:
                rank = None
            rows.append(feature_row(turn, diagnostics))
            labels.append(rank_bucket(rank))
            asins.append(asin)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels), asins


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Top-K confidence policy")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="training/action_policy.joblib")
    parser.add_argument("--report", default="training/action_policy_report.json")
    parser.add_argument("--train-products", type=int, default=2500)
    parser.add_argument("--validation-products", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--exclude-dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    _, categories, products = catalog_index(args.catalog)
    with Path(args.exclude_dataset).open(encoding="utf-8") as handle:
        excluded_asins = {
            str(json.loads(line)["ground_truth"]["parent_asin"])
            for line in handle if line.strip()
        }
    agent = Agent(args.catalog)
    x_train, y_train, train_asins = collect(
        agent, products, categories, "train", args.train_products, args.seed, excluded_asins
    )
    x_validation, y_validation, validation_asins = collect(
        agent, products, categories, "validation", args.validation_products, args.seed, excluded_asins
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.07, max_iter=160, max_leaf_nodes=15,
        l2_regularization=0.5, random_state=args.seed,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    report = {
        "features": list(DIAGNOSTIC_FEATURES),
        "classes": {"0": "rank_1", "1": "rank_2_to_3", "2": "rank_4_to_10", "3": "miss"},
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "train_products": len(set(train_asins)),
        "validation_products": len(set(validation_asins)),
        "asin_overlap": len(set(train_asins) & set(validation_asins)),
        "excluded_public_targets": len(excluded_asins),
        "public_target_overlap": len((set(train_asins) | set(validation_asins)) & excluded_asins),
        "classification": classification_report(
            y_validation, predictions, labels=[0, 1, 2, 3], output_dict=True, zero_division=0
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": DIAGNOSTIC_FEATURES}, output)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
