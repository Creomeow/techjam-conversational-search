from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
        description="Sweep score-gap thresholds for the opt-in Buying confidence gate"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Samples to evaluate (0 = all)")
    parser.add_argument(
        "--gaps",
        nargs="+",
        type=float,
        default=[0.00, 0.12, 0.18, 0.24],
        help="Thresholds to evaluate, in score points",
    )
    parser.add_argument("--output", default=".tmp-tests/confidence-gap-sweep.json")
    args = parser.parse_args()

    if any(not math.isfinite(gap) or gap < 0 for gap in args.gaps):
        raise SystemExit("confidence gaps must be finite and non-negative")

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    baseline_started = time.perf_counter()
    baseline = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    baseline_metrics = _metrics(baseline)
    baseline_metrics["runtime_seconds"] = round(time.perf_counter() - baseline_started, 3)

    gated: dict[str, dict] = {}
    for gap in args.gaps:
        started = time.perf_counter()
        result = evaluate(
            Agent(args.catalog, confidence_gating=True, confidence_gap=gap),
            samples,
            catalog_ids,
            categories,
            products,
        )
        metrics = _metrics(result)
        metrics["runtime_seconds"] = round(time.perf_counter() - started, 3)
        gated[str(gap)] = metrics

    payload = {
        "configuration": {
            "dataset": args.dataset,
            "samples": len(samples),
            "gaps": args.gaps,
        },
        "baseline": baseline_metrics,
        "gated": gated,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
