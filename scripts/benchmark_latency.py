"""Measure cold-start and warm response latency for the offline agent."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.agent import Agent


def main() -> None:
    catalog = Path("data/catalog.jsonl")
    started = time.perf_counter()
    agent = Agent(catalog)
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
        "index_seconds": round(index_seconds, 6),
        "cold_respond_seconds": round(timings[0], 6),
        "warm_respond_seconds": [round(value, 6) for value in timings[1:]],
        "warm_mean_seconds": round(statistics.mean(timings[1:]), 6),
        "warm_p95_seconds": round(max(timings[1:]), 6),
    }, indent=2))


if __name__ == "__main__":
    main()
