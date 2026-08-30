from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, metric_summary  # noqa: E402
from starter.agent import Agent  # noqa: E402

DEFAULT_BATCH_SIZE = 100

# Independent of target product — these just vary the (anonymized) buyer profile.
_PROFILE_POOL: list[dict] = [
    {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 5.0,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort", "durability"],
        "summary": "Prior purchases emphasize fit, comfort, durability; ratings are usually positive.",
    },
    {
        "purchase_frequency": "1-2 prior purchases",
        "average_prior_rating": 3.0,
        "rating_style": "mixed",
        "preference_tags": ["style", "price"],
        "summary": "Prior purchases emphasize style, price; ratings are mixed.",
    },
    {
        "purchase_frequency": "5+ prior purchases",
        "average_prior_rating": 1.0,
        "rating_style": "critical",
        "preference_tags": ["material", "performance", "durability"],
        "summary": "Prior purchases emphasize material, performance, durability; ratings are critical.",
    },
    {
        "purchase_frequency": "first purchase",
        "average_prior_rating": None,
        "rating_style": "no history",
        "preference_tags": [],
        "summary": "No prior purchase history available.",
    },
]


def build_samples(parent_asins: list[str], scenario_type: str) -> list[dict]:
    return [
        {
            "sample_id": f"synth_{scenario_type}_{i:06d}",
            "scenario_type": scenario_type,
            "user_profile": _PROFILE_POOL[i % len(_PROFILE_POOL)],
            "ground_truth": {"parent_asin": asin},
        }
        for i, asin in enumerate(parent_asins)
    ]


def parse_args(default_output: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic large-scale evaluator run")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="products per scenario type (0 = full catalog)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default=default_output)
    parser.add_argument(
        "--confidence-gating",
        action="store_true",
        help="Enable the opt-in score-gap gate for early Buying recommendations",
    )
    parser.add_argument(
        "--confidence-gap",
        type=float,
        default=0.18,
        help="Minimum top-two score gap required by the confidence gate",
    )
    return parser.parse_args()


def _running_hit_rate(sessions: list[dict]) -> float:
    return sum(1 for s in sessions if s["hit"]) / len(sessions) if sessions else 0.0


def _score_block(sessions: list[dict]) -> dict:
    """metric_summary() plus Efficiency and TechnicalScore, per docs/competition_specification.md."""
    summary = metric_summary(sessions)
    mttc = summary["mttc"]
    efficiency = 0.0 if mttc is None else max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    technical_score = 0.50 * summary["hit_rate_at_10"] + 0.30 * summary["mrr"] + 0.20 * efficiency
    return {
        **summary,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
    }


def _checkpoint(output_path: Path, sessions: list[dict], final: bool) -> None:
    grouped: dict[str, list[dict]] = {}
    for session in sessions:
        grouped.setdefault(session["scenario_type"], []).append(session)
    overall = _score_block(sessions)
    payload = {
        "final": final,
        **overall,
        "scenario_metrics": {name: _score_block(rows) for name, rows in sorted(grouped.items())},
        "sessions": sessions,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(scenario_types: tuple[str, ...], default_output: str) -> None:
    args = parse_args(default_output)
    catalog_ids, categories, products = catalog_index(args.catalog)
    all_asins = sorted(catalog_ids)
    random.Random(args.seed).shuffle(all_asins)
    if args.limit:
        all_asins = all_asins[: args.limit]

    total = len(all_asins) * len(scenario_types)
    print(
        f"Catalog size: {len(catalog_ids)} | products per scenario: {len(all_asins)} | "
        f"scenarios: {scenario_types} | total sessions: {total}"
    )

    agent = Agent(
        args.catalog,
        confidence_gating=args.confidence_gating,
        confidence_gap=args.confidence_gap,
    )
    output_path = Path(args.output)
    all_sessions: list[dict] = []
    started = time.time()
    done = 0

    for scenario_type in scenario_types:
        samples = build_samples(all_asins, scenario_type)
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            result = evaluate(agent, batch, catalog_ids, categories, products)
            all_sessions.extend(result["sessions"])
            done += len(batch)
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0.0
            eta_min = (total - done) / rate / 60 if rate else float("inf")
            print(
                f"[{scenario_type}] {done}/{total} | {elapsed / 60:.1f}m elapsed | "
                f"ETA {eta_min:.1f}m | running HR@10={_running_hit_rate(all_sessions):.4f}"
            )
            _checkpoint(output_path, all_sessions, final=False)

    _checkpoint(output_path, all_sessions, final=True)
    print(f"Done in {(time.time() - started) / 60:.1f}m. Wrote {output_path}")
