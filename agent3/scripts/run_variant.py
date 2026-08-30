"""Run the local evaluator with a chosen clarification_policy, for A/B comparison.

Usage: python3 scripts/run_variant.py --policy other_first --output results_other_first.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam local evaluator (policy variant runner)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--policy", default="adaptive", choices=["adaptive", "other_first"])
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, clarification_policy=args.policy)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
