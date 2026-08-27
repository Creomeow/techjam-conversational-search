from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCENARIO_WEIGHTS = (
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scenario_counts(total: int) -> dict[str, int]:
    counts = {name: int(total * weight) for name, weight in SCENARIO_WEIGHTS}
    remainder = total - sum(counts.values())
    order = [name for name, _ in SCENARIO_WEIGHTS]
    for index in range(remainder):
        counts[order[index % len(order)]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic catalog-held-out sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--output", default="data/synthetic_set.jsonl")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    public = load_jsonl(Path(args.public))
    excluded = {str(item["ground_truth"]["parent_asin"]) for item in public}
    profiles = [dict(item["user_profile"]) for item in public]
    catalog_ids: list[str] = []
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin not in excluded:
                catalog_ids.append(parent_asin)
    if args.count > len(catalog_ids):
        raise SystemExit(f"requested {args.count} sessions but only {len(catalog_ids)} held-out products exist")

    rng = random.Random(args.seed)
    targets = rng.sample(catalog_ids, args.count)
    scenarios = [name for name, count in scenario_counts(args.count).items() for _ in range(count)]
    rng.shuffle(scenarios)
    rows = []
    for index, (target, scenario) in enumerate(zip(targets, scenarios), start=1):
        rows.append({
            "sample_id": f"synthetic_{args.seed}_{index:05d}",
            "scenario_type": scenario,
            "user_profile": dict(rng.choice(profiles)),
            "ground_truth": {"parent_asin": target},
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "sample_count": len(rows),
        "excluded_public_targets": len(excluded),
        "scenario_counts": scenario_counts(args.count),
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
