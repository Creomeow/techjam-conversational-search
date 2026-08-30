"""Verbose single-session trace for debugging clarification-policy differences."""
from __future__ import annotations

import argparse
import uuid

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent


def trace(sample_id: str, policy: str, catalog: str, dataset: str) -> None:
    samples = load_jsonl(dataset)
    sample = next(item for item in samples if item["sample_id"] == sample_id)
    catalog_ids, categories, products = catalog_index(catalog)
    agent = Agent(catalog, clarification_policy=policy)
    session_id = f"trace_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    print("scenario:", sample["scenario_type"], "target:", target)
    print("intent_card:", effective_intent_card)
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- turn {turn} ---")
        print("customer:", user_message)
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        hit = target in ranked
        rank = ranked.index(target) + 1 if hit else None
        print("agent ask_attribute:", response.get("ask_attribute"), "| message:", response.get("message"))
        print("hit:", hit, "rank:", rank, "| top recs:", ranked[:5])
        diag = agent.get_diagnostics(session_id)
        print("route:", diag.get("route"), "buyer_state:", diag.get("buyer_state"), "confidence:", diag.get("confidence"), "pool_size:", diag.get("candidate_pool_size"))
        print("slots:", diag.get("active_slots"), "declined:", diag.get("declined_attributes"), "asked:", diag.get("asked_attributes"))
        if hit:
            print(">>> HIT")
            break
        if turn == MAX_TURNS:
            break
        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id")
    parser.add_argument("--policy", default="adaptive", choices=["adaptive", "other_first"])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()
    trace(args.sample_id, args.policy, args.catalog, args.dataset)
