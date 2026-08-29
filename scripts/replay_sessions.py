from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)
from scripts.synthetic_common import build_samples  # noqa: E402
from starter.agent import Agent  # noqa: E402


SYNTHETIC_ID = re.compile(
    r"^synth_(buying|browsing|intent_override|boundary)_(\d{6})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay misses or low-rank sessions through the current agent."
    )
    parser.add_argument("--results", help="Evaluator JSON used to select sessions.")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print one diagnosis line per session instead of the full transcript.",
    )
    parser.add_argument(
        "--rank-threshold",
        type=int,
        default=5,
        help="Select hits worse than this rank when sample IDs are not provided.",
    )
    return parser.parse_args()


def selected_ids(args: argparse.Namespace) -> list[str]:
    if args.sample_id:
        return list(dict.fromkeys(args.sample_id))
    if not args.results:
        raise SystemExit("Provide --results or at least one --sample-id.")
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("sessions", [])
        if not row.get("hit") or int(row.get("best_rank") or 0) > args.rank_threshold
    ]
    rows.sort(
        key=lambda row: (
            bool(row.get("hit")),
            -(int(row.get("best_rank") or TOP_K + 1)),
            str(row.get("sample_id", "")),
        )
    )
    return [str(row["sample_id"]) for row in rows[: args.limit]]


def load_public_samples(path: str | Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            samples[str(sample["sample_id"])] = sample
    return samples


def resolve_sample(
    sample_id: str,
    shuffled_asins: list[str],
    public_samples: dict[str, dict],
) -> dict:
    match = SYNTHETIC_ID.match(sample_id)
    if match:
        scenario, index_text = match.groups()
        index = int(index_text)
        if index >= len(shuffled_asins):
            raise ValueError(f"Synthetic index out of range: {sample_id}")
        return build_samples(shuffled_asins, scenario)[index]
    try:
        return public_samples[sample_id]
    except KeyError as error:
        raise ValueError(f"Unknown sample ID: {sample_id}") from error


def replay(
    agent: Agent,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    verbose: bool = True,
) -> str:
    sample_id = str(sample["sample_id"])
    scenario = str(sample["scenario_type"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}

    session_id = f"replay_{sample_id}"
    agent.reset(session_id, sample.get("user_profile") or {})
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = scenario != "intent_override"
    user_message = initial_message(
        effective_sample, coarse_category(categories.get(target, [])), disclosed
    )

    title = str(products[target].get("title") or "")
    emit = print if verbose else lambda *args, **kwargs: None
    emit(f"\n=== {sample_id} | {scenario} ===")
    emit(f"Target: {target} | {title}")
    emit(f"Intent card: {json.dumps(card, ensure_ascii=False)}")
    if scenario == "intent_override":
        emit(f"Override: {json.dumps(behavior.get('override'), ensure_ascii=False)}")

    pool_ranks: list[int] = []
    submitted_target = False
    final_rank: int | None = None
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        rank = ranked.index(target) + 1 if target in ranked else None
        state = agent._sessions[session_id]
        pool_ids = [parent_asin for parent_asin, _, _ in state.last_candidate_pool]
        pool_rank = pool_ids.index(target) + 1 if target in pool_ids else None
        if pool_rank is not None:
            pool_ranks.append(pool_rank)
        submitted_target = submitted_target or rank is not None

        emit(f"Turn {turn} user: {user_message}")
        emit(
            f"  ask={response.get('ask_attribute')!r} submitted_rank={rank} "
            f"pool_rank={pool_rank} pool_size={len(pool_ids)}"
        )
        emit(
            "  hard="
            + json.dumps(state.hard_terms, ensure_ascii=False)
            + " soft="
            + json.dumps(state.soft_terms, ensure_ascii=False)
        )

        if override_applied and rank is not None:
            final_rank = rank
            emit(f"  HIT at turn {turn}, rank {rank}")
            break
        if turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )

    if final_rank is not None and final_rank > 5:
        diagnosis = "ranking: target was submitted, but below rank 5"
    elif final_rank is not None:
        diagnosis = "successful top-5 retrieval"
    elif not pool_ranks:
        diagnosis = "candidate recall: target never entered the 240-item pool"
    elif not submitted_target:
        diagnosis = "window/pagination: target entered the pool but was never submitted"
    else:
        diagnosis = "intent timing: target was submitted before it became score-eligible"
    emit(f"Diagnosis: {diagnosis}")
    return diagnosis


def main() -> None:
    args = parse_args()
    sample_ids = selected_ids(args)
    if not sample_ids:
        raise SystemExit("No misses or low-rank sessions matched the selection.")

    catalog_ids, categories, products = catalog_index(args.catalog)
    shuffled_asins = sorted(catalog_ids)
    random.Random(args.seed).shuffle(shuffled_asins)
    public_samples = load_public_samples(args.dataset)
    agent = Agent(args.catalog)

    diagnosis_counts: dict[str, int] = {}
    for sample_id in sample_ids:
        sample = resolve_sample(sample_id, shuffled_asins, public_samples)
        diagnosis = replay(
            agent,
            sample,
            catalog_ids,
            categories,
            products,
            verbose=not args.compact,
        )
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
        if args.compact:
            print(f"{sample_id}: {diagnosis}")
    if args.compact:
        print("\nDiagnosis counts:")
        for diagnosis, count in sorted(diagnosis_counts.items()):
            print(f"  {count:>4}  {diagnosis}")


if __name__ == "__main__":
    main()
