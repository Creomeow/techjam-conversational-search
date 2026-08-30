from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402


def _metrics(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"sessions", "reported_token_usage"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the default agent with the opt-in confidence-gated variant"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Samples to evaluate (0 = all)")
    parser.add_argument("--confidence-gap", type=float, default=0.18)
    parser.add_argument("--output", default=".tmp-tests/confidence-gating-ab.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    baseline = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    gated = evaluate(
        Agent(args.catalog, confidence_gating=True, confidence_gap=args.confidence_gap),
        samples,
        catalog_ids,
        categories,
        products,
    )
    payload = {
        "configuration": {
            "dataset": args.dataset,
            "samples": len(samples),
            "confidence_gap": args.confidence_gap,
            "gated_turns": [1, 2],
            "gated_modes": ["buying"],
        },
        "baseline": _metrics(baseline),
        "confidence_gated": _metrics(gated),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
