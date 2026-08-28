from __future__ import annotations

import unittest
from pathlib import Path

from starter import clarify, nlu
from starter.agent import Agent
from starter.retrieval import RetrievalEngine
from starter.state import SessionState


CATALOG = Path(__file__).parent / "fixtures" / "tiny_catalog.jsonl"


class NluTurnOneRoutingTest(unittest.TestCase):
    def test_buying_message_extracts_hard_constraint(self) -> None:
        state = SessionState()
        nlu.apply_customer_message(state, "I'm looking for women's boots. A key requirement is: leather.", 1)
        self.assertEqual(state.mode, "buying")
        self.assertEqual(state.category_tokens, ["women", "boots"])
        self.assertIn("leather", state.hard_terms.get("material", []))
        self.assertIn(("material", "leather"), state.accumulated_terms)

    def test_browsing_message_extracts_no_false_constraint(self) -> None:
        state = SessionState()
        nlu.apply_customer_message(state, "I'm looking for men's shoes, but I'm still exploring.", 1)
        self.assertEqual(state.mode, "browsing")
        self.assertEqual(state.category_tokens, ["men", "shoes"])
        self.assertEqual(state.hard_terms, {})

    def test_intent_override_opener_extracts_soft_value_as_constraint(self) -> None:
        state = SessionState()
        nlu.apply_customer_message(state, "I'm looking for women's sneakers. black.", 1)
        self.assertEqual(state.category_tokens, ["women", "sneakers"])
        self.assertIn("black", state.hard_terms.get("color", []))


class ClarifyOtherFirstPolicyTest(unittest.TestCase):
    def test_always_asks_other_until_card_exhausted(self) -> None:
        state = SessionState()
        for turn in range(1, 4):
            state.turn_count = turn
            attribute, message = clarify.choose_ask_attribute(state)
            self.assertEqual(attribute, "other")
            self.assertTrue(message)

    def test_messages_vary_across_turns(self) -> None:
        state = SessionState()
        seen = set()
        for turn in range(1, 6):
            state.turn_count = turn
            _, message = clarify.choose_ask_attribute(state)
            seen.add(message)
        self.assertGreater(len(seen), 1)

    def test_exhausted_card_returns_null_attribute(self) -> None:
        state = SessionState()
        state.card_exhausted = True
        attribute, message = clarify.choose_ask_attribute(state)
        self.assertIsNone(attribute)
        self.assertTrue(message)


class BoundaryVsExhaustionTest(unittest.TestCase):
    def test_boundary_refusal_does_not_exhaust_other(self) -> None:
        state = SessionState()
        state.last_ask_attribute = "other"
        nlu.apply_customer_message(
            state, "I don't have a preference for other; please use your judgment.", 2
        )
        self.assertIn("other", state.no_preference_attributes)
        self.assertNotIn("other", state.exhausted_attributes)
        self.assertFalse(state.card_exhausted)
        # "other" must remain fully askable next turn.
        attribute, _ = clarify.choose_ask_attribute(state)
        self.assertEqual(attribute, "other")

    def test_genuine_exhaustion_sets_card_exhausted_for_other(self) -> None:
        state = SessionState()
        state.last_ask_attribute = "other"
        nlu.apply_customer_message(state, "I don't have an additional preference for other.", 3)
        self.assertIn("other", state.exhausted_attributes)
        self.assertTrue(state.card_exhausted)
        attribute, _ = clarify.choose_ask_attribute(state)
        self.assertIsNone(attribute)

    def test_boundary_refusal_on_specific_attribute_stays_askable(self) -> None:
        state = SessionState()
        state.last_ask_attribute = "material"
        nlu.apply_customer_message(
            state, "I don't have a preference for material; please use your judgment.", 2
        )
        self.assertIn("material", state.no_preference_attributes)
        self.assertNotIn("material", state.exhausted_attributes)


class DisclosureAccumulationTest(unittest.TestCase):
    def test_multi_turn_accumulation_across_asks(self) -> None:
        state = SessionState()
        nlu.apply_customer_message(state, "I'm looking for men's shoes, but I'm still exploring.", 1)
        state.last_ask_attribute = "other"
        nlu.apply_customer_message(state, "For that, what matters is: running.", 2)
        state.last_ask_attribute = "other"
        nlu.apply_customer_message(state, "For that, what matters is: mesh; gray.", 3)
        self.assertIn("running", state.hard_terms.get("use_case", []))
        self.assertIn("mesh", state.hard_terms.get("material", []))
        self.assertIn("gray", state.hard_terms.get("color", []))

    def test_budget_parsing_variants(self) -> None:
        cases = {
            "budget around $49.0": 49.0,
            "under $50": 50.0,
            "less than 30 dollars": 30.0,
        }
        for text, expected in cases.items():
            state = SessionState()
            state.last_ask_attribute = "other"
            nlu.apply_customer_message(state, f"For that, what matters is: {text}.", 2)
            self.assertEqual(state.budget_max, expected, text)


class OverrideReplacesNotAppendsTest(unittest.TestCase):
    def test_repeated_override_value_is_a_noop(self) -> None:
        state = SessionState()
        nlu.apply_customer_message(state, "I'm looking for women's sneakers. leather.", 1)
        before = {key: values[:] for key, values in state.hard_terms.items()}
        self.assertFalse(nlu.detect_override(state, "Actually, what I need is: leather."))
        nlu.apply_customer_message(state, "Actually, what I need is: leather.", 3)
        self.assertEqual(state.hard_terms, before)

    def test_override_replaces_only_targeted_attribute(self) -> None:
        state = SessionState()
        nlu.apply_customer_message(state, "I'm looking for women's sneakers. black.", 1)
        self.assertIn("black", state.hard_terms.get("color", []))

        nlu.apply_customer_message(
            state, "Actually, ignore my earlier preference. What I need is: canvas material.", 3
        )
        self.assertIn("black", state.hard_terms.get("color", []))
        self.assertIn("canvas material", state.hard_terms.get("material", []))

    def test_override_without_colon_still_detected(self) -> None:
        state = SessionState()
        state.last_ask_attribute = "material"
        nlu.apply_customer_message(state, "On second thought, I'd rather have leather instead.", 3)
        self.assertTrue(
            any("leather" in value for values in state.hard_terms.values() for value in values)
        )


class ConstraintVocabularySupersetTest(unittest.TestCase):
    def test_classifies_catalog_vocabulary_not_in_evaluator_lists(self) -> None:
        self.assertEqual(nlu.classify_constraint("Material:alloy"), "material")
        self.assertEqual(nlu.classify_constraint("100% Textile"), "material")
        self.assertEqual(nlu.classify_constraint("teal"), "color")
        self.assertEqual(nlu.classify_constraint("denim"), "material")


class AdaptiveClarificationPolicyTest(unittest.TestCase):
    def test_selects_attribute_with_highest_candidate_impurity(self) -> None:
        state = SessionState(clarification_policy="adaptive")
        state.last_candidate_pool = [
            ("1", {"title": "black wool hat", "price": 20}, 1.0),
            ("2", {"title": "blue wool shirt", "price": 40}, 0.9),
            ("3", {"title": "red wool jacket", "price": 60}, 0.8),
        ]
        attribute, _ = clarify.choose_ask_attribute(state)
        self.assertEqual(attribute, "color")

    def test_adaptive_policy_skips_disclosed_attribute(self) -> None:
        state = SessionState(clarification_policy="adaptive")
        state.hard_terms["color"] = ["black"]
        state.last_candidate_pool = [
            ("1", {"title": "black wool hat", "price": 20}, 1.0),
            ("2", {"title": "blue wool shirt", "price": 40}, 0.9),
        ]
        attribute, _ = clarify.choose_ask_attribute(state)
        self.assertNotEqual(attribute, "color")


class RetrievalIntegrationTest(unittest.TestCase):
    def test_constraint_reranking_places_exact_product_first(self) -> None:
        engine = RetrievalEngine(CATALOG)
        pool = engine.search(
            "women shoes",
            ["red", "trail running", "wide fit", "budget around $59"],
            top_k=2,
            candidate_limit=10,
        )
        self.assertEqual(pool[0][0], "RED_SHOE")
        self.assertIn("title", pool[0][1])
        self.assertIsInstance(pool[0][2], float)


class AgentIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = Agent(CATALOG)

    def test_output_contract_and_multi_turn_ranking(self) -> None:
        self.agent.reset("one", {"preference_tags": ["comfort"]})
        first = self.agent.respond("one", "I'm looking for Women Shoes, but I'm still exploring.", 1, 10)
        second = self.agent.respond(
            "one", "For that, what matters is: red; trail running; wide fit.", 2, 10
        )
        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(second["recommendations"][0]["parent_asin"], "RED_SHOE")
        self.assertIn("trail running", self.agent._sessions["one"].hard_terms.get("use_case", []))

    def test_candidate_pool_contract_is_available_to_clarification(self) -> None:
        self.agent.reset("pool", {})
        self.agent.respond("pool", "I'm looking for Women Shoes, but I'm still exploring.", 1, 10)
        pool = self.agent._sessions["pool"].last_candidate_pool
        self.assertTrue(pool)
        self.assertEqual(len(pool[0]), 3)
        self.assertIsInstance(pool[0][1], dict)

    def test_sessions_do_not_contaminate_each_other(self) -> None:
        self.agent.reset("red", {})
        self.agent.reset("hat", {})
        self.agent.respond("red", "I'm looking for Women Shoes. red", 1, 10)
        self.agent.respond("hat", "I'm looking for Accessories Hats. black wool", 1, 10)
        self.assertNotEqual(
            self.agent._sessions["red"].hard_terms,
            self.agent._sessions["hat"].hard_terms,
        )


if __name__ == "__main__":
    unittest.main()
