from __future__ import annotations

import unittest

from starter.agent import Agent
from starter.intent import parse_intent


class IntentParserTest(unittest.TestCase):
    def test_parses_typed_shopping_intent(self) -> None:
        parsed = parse_intent(
            "I'm looking for women's running shoes. A key requirement is: wide fit."
        )
        self.assertEqual(parsed.mode, "buying")
        self.assertEqual(parsed.product_types, {"running_shoes", "shoes"})
        self.assertEqual(parsed.audiences, {"women"})
        self.assertEqual(parsed.slots["size"], {"wide"})
        self.assertIn("wide fit", parsed.required_phrases)

    def test_parses_preferences_and_features(self) -> None:
        parsed = parse_intent(
            "For that, what matters is: waterproof; suitable for trail running."
        )
        self.assertEqual(parsed.slots["feature"], {"waterproof"})
        self.assertEqual(parsed.slots["use_case"], {"running", "trail"})
        self.assertEqual(
            parsed.preferred_phrases,
            ["waterproof", "suitable for trail running"],
        )

    def test_exclusion_is_not_a_positive_slot(self) -> None:
        parsed = parse_intent(
            "Actually, no leather. What I need is: black waterproof boots under $100."
        )
        self.assertEqual(parsed.operation, "replace")
        self.assertEqual(parsed.excluded_terms, {"leather"})
        self.assertNotIn("material", parsed.slots)
        self.assertEqual(parsed.budget_max, 100.0)


class ProductCompatibilityTest(unittest.TestCase):
    def test_matching_type_and_audience_outrank_contradictions(self) -> None:
        agent = Agent.__new__(Agent)
        agent._facets = {
            "women-shoe": {
                "product_type": {"shoes", "running_shoes"},
                "audience": {"women"},
                "__all": {"shoe", "running", "wide", "waterproof"},
            },
            "mens-shoe": {
                "product_type": {"shoes", "running_shoes"},
                "audience": {"men"},
                "__all": {"shoe", "running"},
            },
            "womens-hat": {
                "product_type": {"hats"},
                "audience": {"women"},
                "__all": {"hat", "running"},
            },
        }
        state = {
            "product_types": {"shoes", "running_shoes"},
            "audiences": {"women"},
            "excluded_terms": set(),
            "slots": {"size": {"wide"}, "feature": {"waterproof"}},
        }
        keys = {
            asin: agent._compatibility_key(state, asin)
            for asin in agent._facets
        }
        self.assertGreater(keys["women-shoe"], keys["mens-shoe"])
        self.assertGreater(keys["women-shoe"], keys["womens-hat"])
        self.assertEqual(keys["women-shoe"][-1], 2)


if __name__ == "__main__":
    unittest.main()
