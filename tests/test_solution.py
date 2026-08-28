from __future__ import annotations

import unittest
from time import sleep

from solution.agent import Agent, OVERRIDE_RE, _constraint_variants, _quarantine_structured_turn
from solution.extraction import StructuredTurn, TimeoutExtractor, _first_json_object
from stress_eval import transform_message


class SolutionParsingTest(unittest.TestCase):
    def test_plain_language_demo_flow_returns_products_and_updates_slots(self) -> None:
        agent = Agent("data/catalog.jsonl")
        agent.reset("demo", {})

        first = agent.respond("demo", "I need black running shoes under $80", 1, 10)
        second = agent.respond("demo", "No leather, and blue instead of black", 2, 10)

        self.assertTrue(first["recommendations"])
        self.assertTrue(second["recommendations"])
        self.assertIn("color: blue", agent.sessions["demo"]["constraints"])
        self.assertNotIn("color: black", agent.sessions["demo"]["constraints"])
        self.assertIn("leather", agent.sessions["demo"]["negative_constraints"])

    def test_late_extraction_is_discarded(self) -> None:
        class SlowExtractor:
            def extract(self, message: str, state: dict) -> StructuredTurn:
                sleep(0.05)
                return StructuredTurn(intent="buying", confidence=1.0)

        turn = TimeoutExtractor(SlowExtractor(), 0.001).extract("shoes", {})
        self.assertEqual(turn, StructuredTurn())

    def test_optional_model_operations_require_message_and_state_evidence(self) -> None:
        state = {
            "category": "shoes",
            "slots": {"color": ["color: black"], "material": ["cotton"]},
        }
        turn = _quarantine_structured_turn(
            "Actually, blue instead of black; keep everything else.",
            state,
            StructuredTurn.from_payload({
                "intent": "buying",
                "override": True,
                "category": "boots",
                "add": {"color": ["color: blue"], "material": ["wool"]},
                "remove": {"material": ["cotton"]},
                "replace_slots": ["color", "material"],
                "negative": {"material": ["leather"]},
                "confidence": 0.95,
            }),
        )
        self.assertEqual(turn.add, {"color": ["color: blue"]})
        self.assertEqual(turn.replace_slots, ("color",))
        self.assertEqual(turn.category, "")
        self.assertEqual(turn.remove, {})
        self.assertEqual(turn.negative, {})

        removal = _quarantine_structured_turn(
            "Forget leather; no wool.",
            {"slots": {"material": ["leather"]}},
            StructuredTurn.from_payload({
                "intent": "buying",
                "override": True,
                "remove": {"material": ["leather", "cotton"]},
                "negative": {"material": ["wool", "silk"]},
                "confidence": 0.95,
            }),
        )
        self.assertEqual(removal.remove, {"material": ["leather"]})
        self.assertEqual(removal.negative, {"material": ["wool"]})
        self.assertTrue(removal.override)

        positive_mention = _quarantine_structured_turn(
            "Avoid red, but wool is fine.",
            {"slots": {"material": ["wool"]}},
            StructuredTurn.from_payload({
                "intent": "buying",
                "remove": {"material": ["wool"]},
                "negative": {"material": ["wool"]},
                "confidence": 0.95,
            }),
        )
        self.assertEqual(positive_mention.remove, {})
        self.assertEqual(positive_mention.negative, {})

        excluded = _quarantine_structured_turn(
            "No leather.",
            {"slots": {}},
            StructuredTurn.from_payload({
                "intent": "buying",
                "add": {"material": ["leather"]},
                "negative": {"material": ["leather"]},
                "confidence": 0.95,
            }),
        )
        self.assertEqual(excluded.add, {})
        self.assertEqual(excluded.negative, {"material": ["leather"]})

        forgotten = _quarantine_structured_turn(
            "Forget leather.",
            {"slots": {"material": ["leather"]}},
            StructuredTurn.from_payload({
                "intent": "buying",
                "override": True,
                "add": {"material": ["leather"]},
                "remove": {"material": ["leather"]},
                "confidence": 0.95,
            }),
        )
        self.assertEqual(forgotten.add, {})
        self.assertEqual(forgotten.remove, {"material": ["leather"]})

        unsupported_intent = _quarantine_structured_turn(
            "Tell me what you think.",
            {"slots": {}},
            StructuredTurn(intent="buying", confidence=0.95),
        )
        self.assertEqual(unsupported_intent.intent, "unknown")

        exploratory_intent = _quarantine_structured_turn(
            "I'm just exploring and open to ideas.",
            {"slots": {}},
            StructuredTurn(intent="unknown", confidence=0.95),
        )
        self.assertEqual(exploratory_intent.intent, "browsing")

        soft_exploration = _quarantine_structured_turn(
            "I have no material preference; just show versatile everyday options.",
            {"slots": {}},
            StructuredTurn.from_payload({
                "intent": "buying",
                "add": {"style": ["versatile"], "use_case": ["everyday"]},
                "confidence": 0.95,
            }),
        )
        self.assertEqual(soft_exploration.intent, "browsing")

    def test_experimental_router_uses_hard_and_exploratory_signals(self) -> None:
        self.assertEqual(
            Agent._route_intent("I need shoes under $80", {
                "category": "shoes", "constraints": ["budget under $80"],
                "negative_constraints": [], "slots": {}, "inferred_intent": "unknown",
            }, 0.0),
            "buying",
        )
        self.assertEqual(
            Agent._route_intent("I'm open to something new", {
                "category": "", "constraints": [], "negative_constraints": [],
                "slots": {}, "inferred_intent": "browsing",
            }, 0.9),
            "browsing",
        )
        fused = Agent._fuse_routes(["buy", "shared"], ["browse", "shared"], 0.5)
        self.assertEqual(fused[0], "shared")

    def test_structured_payload_validation(self) -> None:
        turn = StructuredTurn.from_payload({
            "intent": "BUYING",
            "override": True,
            "category": "  running shoes  ",
            "add": {"color": ["color: blue"], "made_up": ["ignored"]},
            "replace_slots": ["color", "made_up"],
            "negative": {"material": "leather"},
            "confidence": 1.4,
        })
        self.assertEqual(turn.intent, "buying")
        self.assertEqual(turn.category, "running shoes")
        self.assertEqual(turn.add, {"color": ["color: blue"]})
        self.assertEqual(turn.replace_slots, ("color",))
        self.assertEqual(turn.negative, {"material": ["leather"]})
        self.assertEqual(turn.confidence, 1.0)

    def test_json_extraction_ignores_model_chatter(self) -> None:
        self.assertEqual(
            _first_json_object('```json\n{"intent":"browsing","confidence":0.8}\n```'),
            {"intent": "browsing", "confidence": 0.8},
        )

    def test_structured_turn_replaces_only_named_slot(self) -> None:
        state = {
            "slots": {"color": ["color: black"], "material": ["cotton"]},
            "constraints": ["color: black", "cotton"],
            "negative_constraints": [],
            "inferred_intent": "unknown",
        }
        Agent._apply_structured_turn(state, StructuredTurn.from_payload({
            "intent": "buying",
            "override": True,
            "add": {"color": ["color: blue"]},
            "replace_slots": ["color"],
            "confidence": 0.9,
        }))
        self.assertEqual(state["slots"]["color"], ["color: blue"])
        self.assertEqual(state["slots"]["material"], ["cotton"])
        self.assertEqual(state["inferred_intent"], "buying")

    def test_released_override_initial_turn_stays_on_fast_path(self) -> None:
        messages = [
            ("I'm looking for Women's Shoes. I prefer a lightweight design.", []),
            ("I'm looking for Women's Shoes, but I'm still exploring.", []),
            ("For that, what matters is: cotton; color: black.", ["cotton", "color: black"]),
            ("I don't have a preference for material; please use your judgment.", []),
            ("I don't have an additional preference for style.", []),
            ("Those options are not quite right yet. Ask me about one specific attribute.", []),
        ]
        self.assertTrue(all(Agent._rules_are_confident(message, constraints) for message, constraints in messages))

    def test_official_constraint_format(self) -> None:
        self.assertEqual(
            Agent._extract_constraints("For that, what matters is: cotton; color: black."),
            ["cotton", "color: black"],
        )

    def test_paraphrased_constraint_format(self) -> None:
        self.assertEqual(
            Agent._extract_constraints(
                "Another thing I care about is synthetic textile and also cool-toned."
            ),
            ["synthetic textile", "cool-toned"],
        )

    def test_override_variants(self) -> None:
        self.assertRegex(
            "Actually, ignore my earlier preference. What I need is: leather.", OVERRIDE_RE
        )
        self.assertRegex(
            "Change of plan—drop what I said before and prioritize animal-hide material instead.",
            OVERRIDE_RE,
        )

    def test_stress_transform_removes_verbatim_value(self) -> None:
        transformed = transform_message(
            "I'm looking for Shoes. A key requirement is: polyester."
        )
        self.assertNotIn("polyester", transformed.casefold())
        self.assertIn("synthetic textile", transformed.casefold())

    def test_facet_normalization_retains_ambiguous_alternatives(self) -> None:
        variants = _constraint_variants("durable synthetic textile")
        self.assertIn("nylon", variants)
        self.assertNotIn("polyester", variants)

    def test_facet_normalization_can_restore_multiple_terms(self) -> None:
        variants = _constraint_variants("95% synthetic textile, 5% stretchy elastane")
        self.assertIn("95% polyester, 5% spandex", variants)

    def test_explicit_negative_constraint_parsing(self) -> None:
        self.assertEqual(
            Agent._extract_negative_constraints("I want something without leather; avoid red."),
            ["leather", "red"],
        )

    def test_no_preference_is_not_a_negative_constraint(self) -> None:
        self.assertEqual(
            Agent._extract_negative_constraints(
                "I don't have a preference for material; please use your judgment."
            ),
            [],
        )

    def test_override_replaces_only_conflicting_slot(self) -> None:
        state = {
            "constraints": ["color: black", "cotton", "waterproof"],
            "slots": {},
        }
        Agent._merge_constraints(state, ["color: white"], override=True)
        self.assertEqual(state["slots"]["color"], ["color: white"])
        self.assertIn("cotton", state["constraints"])
        self.assertIn("waterproof", state["constraints"])

    def test_equivalent_override_does_not_erase_sibling_constraints(self) -> None:
        state = {
            "constraints": ["leather", "cotton"],
            "slots": {},
        }
        Agent._merge_constraints(state, ["animal-hide material"], override=True)
        self.assertIn("leather", state["constraints"])
        self.assertIn("cotton", state["constraints"])

    def test_natural_override_extracts_rewritten_facets(self) -> None:
        message = "Actually, make them casual white sneakers."
        self.assertRegex(message, OVERRIDE_RE)
        self.assertEqual(
            Agent._extract_override_constraints(message),
            ["color: white", "casual"],
        )
        self.assertEqual(Agent._extract_override_category(message), "sneakers")


if __name__ == "__main__":
    unittest.main()
