from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate  # noqa: E402
from scripts.synthetic_common import build_samples, _score_block  # noqa: E402
from starter.agent import Agent  # noqa: E402


def run_variant(
    catalog_path: str,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    confidence_gating: bool,
    confidence_gap: float,
    batch_size: int,
) -> dict:
    label = "confidence-gated" if confidence_gating else "baseline"
    agent = Agent(
        catalog_path,
        confidence_gating=confidence_gating,
        confidence_gap=confidence_gap,
    )
    sessions: list[dict] = []
    started = time.perf_counter()
    for start in range(0, len(samples), batch_size):
        batch_number = start // batch_size + 1
        total_batches = (len(samples) + batch_size - 1) // batch_size
        result = evaluate(
            agent,
            samples[start : start + batch_size],
            catalog_ids,
            categories,
            products,
        )
        sessions.extend(result["sessions"])
        elapsed = time.perf_counter() - started
        rate = len(sessions) / elapsed if elapsed else 0.0
        remaining = (len(samples) - len(sessions)) / rate if rate else 0.0
        print(
            f"[{label}] batch {batch_number}/{total_batches} | "
            f"{len(sessions):,}/{len(samples):,} sessions | "
            f"{elapsed / 60:.1f}m elapsed | "
            f"{rate:.2f} sessions/s | ETA {remaining / 60:.1f}m",
            flush=True,
        )
    elapsed = time.perf_counter() - started

    grouped: dict[str, list[dict]] = {}
    for session in sessions:
        grouped.setdefault(session["scenario_type"], []).append(session)
    output = {
        **_score_block(sessions),
        "scenario_metrics": {
            name: _score_block(rows) for name, rows in sorted(grouped.items())
        },
        "runtime_seconds": round(elapsed, 3),
    }
    print(
        f"[{label}] complete | HR@10={output['hit_rate_at_10']:.6f} | "
        f"MRR={output['mrr']:.6f} | MTTC={output['mttc']:.6f} | "
        f"TechnicalScore={output['recommended_technical_score']:.6f} | "
        f"runtime={output['runtime_seconds']:.1f}s",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare confidence gating on the same synthetic Buying/Override sessions"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Products per scenario type; 10000 means 20000 sessions total",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--confidence-gap", type=float, default=0.18)
    parser.add_argument("--output", default="results_buying_override_v4_comparison.json")
    args = parser.parse_args()

    if args.limit < 0 or args.batch_size <= 0:
        raise SystemExit("limit must be non-negative and batch-size must be positive")

    catalog_ids, categories, products = catalog_index(args.catalog)
    asins = sorted(catalog_ids)
    random.Random(args.seed).shuffle(asins)
    if args.limit:
        asins = asins[: args.limit]
    samples = build_samples(asins, "buying") + build_samples(asins, "intent_override")

    print(
        f"Catalog: {len(catalog_ids):,} products | "
        f"Selected: {len(asins):,} products per scenario | "
        f"Sessions per variant: {len(samples):,} | "
        f"Scenarios: buying, intent_override | seed={args.seed} | "
        f"confidence_gap={args.confidence_gap}",
        flush=True,
    )
    print("Starting baseline (confidence gating disabled)...", flush=True)
    baseline = run_variant(
        args.catalog, samples, catalog_ids, categories, products,
        False, args.confidence_gap, args.batch_size,
    )
    print("Starting confidence-gated variant...", flush=True)
    gated = run_variant(
        args.catalog, samples, catalog_ids, categories, products,
        True, args.confidence_gap, args.batch_size,
    )

    payload = {
        "configuration": {
            "catalog": args.catalog,
            "products_per_scenario": len(asins),
            "total_sessions_per_variant": len(samples),
            "scenario_types": ["buying", "intent_override"],
            "seed": args.seed,
            "batch_size": args.batch_size,
            "confidence_gap": args.confidence_gap,
        },
        "baseline": baseline,
        "confidence_gated": gated,
        "delta_gated_minus_baseline": {
            key: round(gated[key] - baseline[key], 6)
            for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {output_path}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
