from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public and deterministic synthetic evaluator gates")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--synthetic-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir or f"runs/{stamp}")
    run_dir.mkdir(parents=True, exist_ok=False)
    synthetic = run_dir / "synthetic.jsonl"
    public_result = run_dir / "public-results.json"
    synthetic_result = run_dir / "synthetic-results.json"

    python = sys.executable
    run([
        python, "tools/generate_synthetic.py",
        "--catalog", args.catalog,
        "--public", args.public,
        "--output", str(synthetic),
        "--count", str(args.synthetic_count),
        "--seed", str(args.seed),
    ])
    run([
        python, "-m", "evaluator.local_evaluator",
        "--catalog", args.catalog,
        "--dataset", args.public,
        "--output", str(public_result),
    ])
    run([
        python, "-m", "evaluator.local_evaluator",
        "--catalog", args.catalog,
        "--dataset", str(synthetic),
        "--output", str(synthetic_result),
    ])
    print(f"\nCompleted run: {run_dir}")
    print(f"Public result: {public_result}")
    print(f"Synthetic result: {synthetic_result}")


if __name__ == "__main__":
    main()
