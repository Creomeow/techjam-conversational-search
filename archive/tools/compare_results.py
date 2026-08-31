from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")


def load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: object) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluator result files")
    parser.add_argument("results", nargs="+", help="result JSON files; first file is the reference")
    args = parser.parse_args()
    loaded = [(path, load(path)) for path in args.results]
    reference = loaded[0][1]

    print("overall")
    print("file\t" + "\t".join(METRICS))
    for path, result in loaded:
        print(Path(path).name + "\t" + "\t".join(fmt(result.get(metric)) for metric in METRICS))

    if len(loaded) > 1:
        print("\ndelta_vs_first (positive is better; MTTC sign is inverted)")
        print("file\t" + "\t".join(METRICS))
        for path, result in loaded[1:]:
            values = []
            for metric in METRICS:
                current = result.get(metric)
                base = reference.get(metric)
                if current is None or base is None:
                    values.append("-")
                else:
                    delta = float(current) - float(base)
                    if metric == "mttc":
                        delta = -delta
                    values.append(f"{delta:+.6f}")
            print(Path(path).name + "\t" + "\t".join(values))

    scenarios = sorted({name for _, result in loaded for name in result.get("scenario_metrics", {})})
    for scenario in scenarios:
        print(f"\n{scenario}")
        print("file\thit_rate_at_10\tmrr\tmttc")
        for path, result in loaded:
            metrics = result.get("scenario_metrics", {}).get(scenario, {})
            print(Path(path).name + "\t" + "\t".join(
                fmt(metrics.get(metric)) for metric in ("hit_rate_at_10", "mrr", "mttc")
            ))


if __name__ == "__main__":
    main()
