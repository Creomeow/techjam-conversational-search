"""Measure cold-start and warm response latency for the offline agent."""

from __future__ import annotations

import json
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure offline agent cold and warm latency")
    parser.add_argument("--confidence-gating", action="store_true")
    parser.add_argument("--confidence-gap", type=float, default=0.18)
    args = parser.parse_args()

    catalog = Path("data/catalog.jsonl")
    started = time.perf_counter()
    agent = Agent(
        catalog,
        confidence_gating=args.confidence_gating,
        confidence_gap=args.confidence_gap,
    )
    index_seconds = time.perf_counter() - started

    agent.reset("latency", {})
    messages = [
        "I'm looking for a black leather belt.",
        "It should be suitable for everyday wear.",
        "Please show me the closest matches.",
        "I prefer something durable.",
    ]
    timings: list[float] = []
    for turn, message in enumerate(messages, start=1):
        started = time.perf_counter()
        agent.respond("latency", message, turn, 10)
        timings.append(time.perf_counter() - started)

    print(json.dumps({
        "catalog": str(catalog),
        "confidence_gating": args.confidence_gating,
        "confidence_gap": args.confidence_gap,
        "index_seconds": round(index_seconds, 6),
        "cold_respond_seconds": round(timings[0], 6),
        "warm_respond_seconds": [round(value, 6) for value in timings[1:]],
        "warm_mean_seconds": round(statistics.mean(timings[1:]), 6),
        "warm_p95_seconds": round(max(timings[1:]), 6),
    }, indent=2))


if __name__ == "__main__":
    main()
