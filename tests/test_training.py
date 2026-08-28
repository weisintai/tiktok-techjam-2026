from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.generate_sessions import SCENARIOS, generate, split_for_asin
from training.evaluate_mutations import reorder_constraints
from training.evaluate_extraction import evaluate_prediction, load_cases
from training.optimize_topk import replay, schedule_policy
from training.trace_outliers import _outrank_reason, _output_limit
from solution.extraction import StructuredTurn


class SessionGenerationTest(unittest.TestCase):
    def test_freeform_corpus_has_frozen_non_overlapping_families(self) -> None:
        cases = load_cases("training/freeform_extraction_cases.jsonl")
        self.assertEqual(len(cases), 200)
        self.assertEqual(Counter(case["split"] for case in cases), {
            "train": 120,
            "development": 40,
            "test": 40,
        })
        family_splits: dict[str, set[str]] = {}
        for case in cases:
            family_splits.setdefault(case["family_id"], set()).add(case["split"])
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))

    def test_applied_state_metric_accepts_equivalent_override_operations(self) -> None:
        case = {
            "state": {
                "category": "shoes",
                "slots": {"color": ["color: black"], "feature": ["waterproof"]},
                "inferred_intent": "buying",
            },
            "expected": {
                "intent": "buying",
                "override": True,
                "add": {"color": ["color: blue"]},
                "replace_slots": ["color"],
            },
        }
        predicted = StructuredTurn(
            intent="buying",
            override=True,
            add={"color": ["color: blue"]},
            remove={"color": ["color: black"]},
            confidence=0.9,
        )
        outcome = evaluate_prediction(case, predicted)
        self.assertFalse(outcome["state_missing"])
        self.assertFalse(outcome["state_extra"])
        self.assertEqual(outcome["preserved_correct"], outcome["preserved_total"])

    def test_constraint_order_mutation_is_narrow_and_deterministic(self) -> None:
        message = "For that, what matters is: cotton; color: blue."
        self.assertEqual(
            reorder_constraints(message),
            "For that, what matters is: color: blue; cotton.",
        )
        self.assertEqual(reorder_constraints("For that, what matters is: cotton."), "For that, what matters is: cotton.")

    def test_split_is_stable_and_asin_scoped(self) -> None:
        first = split_for_asin("ABC", 123)
        self.assertEqual(first, split_for_asin("ABC", 123))
        self.assertIn(first, {"train", "validation", "test"})

    def test_generator_emits_four_scenarios_without_leakage(self) -> None:
        products = [
            {
                "parent_asin": f"ASIN{index:03d}",
                "title": f"Product {index}",
                "categories": ["Clothing", "Shirts"],
                "features": ["cotton", "color: blue"],
                "average_rating": 4.2,
            }
            for index in range(30)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(product) + "\n" for product in products), encoding="utf-8"
            )
            quarantined = {products[0]["parent_asin"]}
            metadata = generate(
                catalog, root / "generated", seed=7, quarantine_asins=quarantined
            )
            asin_splits: dict[str, str] = {}
            counts: Counter[tuple[str, str]] = Counter()
            for split in ("train", "validation", "test", "quarantine"):
                path = root / "generated" / f"{split}.jsonl.gz"
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        sample = json.loads(line)
                        asin = sample["ground_truth"]["parent_asin"]
                        self.assertEqual(asin_splits.setdefault(asin, split), split)
                        counts[(asin, sample["scenario_type"])] += 1
            self.assertEqual(sum(metadata["products"].values()), len(products))
            self.assertEqual(metadata["products"]["quarantine"], 1)
            self.assertEqual(asin_splits[products[0]["parent_asin"]], "quarantine")
            for product in products:
                asin = product["parent_asin"]
                self.assertEqual({scenario for key, scenario in counts if key == asin}, set(SCENARIOS))
                self.assertTrue(all(counts[(asin, scenario)] == 1 for scenario in SCENARIOS))

    def test_policy_replay_respects_seen_products(self) -> None:
        trace = {
            "target": "TARGET",
            "turns": [
                {
                    "turn": 1,
                    "eligible": True,
                    "reset_seen": False,
                    "ranked": ["OTHER", "TARGET"],
                    "diagnostics": {},
                },
                {
                    "turn": 2,
                    "eligible": True,
                    "reset_seen": False,
                    "ranked": ["OTHER", "TARGET"],
                    "diagnostics": {},
                },
            ],
        }
        outcome = replay(trace, schedule_policy((1,) * 10))
        self.assertEqual(outcome["first_hit_turn"], 2)
        self.assertEqual(outcome["best_rank"], 1)

    def test_outlier_trace_explains_the_production_tie_break(self) -> None:
        target = {
            "negative_matches": 0,
            "exact_card_matches": 4,
            "exact_card_characters": 42,
            "bm25_rank": 63,
        }
        competitor = {**target, "bm25_rank": 5}
        self.assertEqual(
            _outrank_reason(competitor, target),
            "higher BM25 rank after the exact-card tie",
        )
        self.assertEqual(_output_limit(7, 27), 3)
        self.assertEqual(_output_limit(7, 186), 5)


if __name__ == "__main__":
    unittest.main()
