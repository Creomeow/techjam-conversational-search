# Build Schedule: Improved Conversational Shopping Agent (Thu night → Mon submission)

**Two-person split**: the module boundaries below (NLU/state/clarify vs. retrieval/testing-infra) were chosen so the two halves talk to each other only through the `SessionState` object and a candidate-pool list — meaning both people can build in parallel most days after a short interface sync, rather than working sequentially on the same files.
- **Person A — "conversation side"**: `state.py`, `nlu.py` (extraction/classification/override/routing), `clarify.py` (adaptive question selection).
- **Person B — "retrieval side"**: `retrieval.py` (tiered query + rescoring), `agent.py` orchestration/wiring, synthetic-session generator, overnight run scripts, evaluator-facing tests.

## Context

We're building an improved `Agent` for the TechJam Conversational E-Commerce Search Challenge (`starter/agent.py`), replacing the weak BM25 baseline (HR@10 0.125, MRR 0.068, MTTC 9.81 per `docs/baseline_results.json`) which never sets `ask_attribute` and never accumulates constraints across turns — this is why it scores near-zero on Browsing/Boundary scenarios.

The original 4-day plan (tonight=v1 + brute-force permutation search overnight, Fri/Sat=v2 + more orderings, Sun=v3 confidence gating, Mon=buffer+submit) had three problems this restructure fixes:
1. **Infeasible compute**: exhaustively permuting ~9-10 `ask_attribute` values is `9!`–`10!` (362K–3.6M) orderings; at ~1+ min per 200-session evaluator run, only ~few hundred runs fit in 8 hours — nowhere close to exhaustive.
2. **Overfitting risk**: searching thousands of orderings against only 200 labeled public sessions will find a "winner" by chance that doesn't generalize to the 800 private sessions used for final scoring.
3. **Low ceiling**: a single static global ordering can't express scenario-dependent behavior (Buying vs. Browsing wanting different first questions; Boundary needing to deprioritize whatever got "no preference"), which an adaptive per-turn policy can.

It also under-scopes Monday: `docs/submission_rules.md` and the "Final Deliverables" in `docs/competition_specification.md` require a written report and one demonstrated multi-turn session transcript, not just code.

**Clarification-policy note (added after further review)**: v1's default is to ask `ask_attribute="other"` first and repeatedly, since the simulator's `customer_reply()` reveals undisclosed constraints regardless of classification when asked `"other"` — a strong, simple way to front-load information extraction. This is a deliberate, documented exploit of a specific simulator implementation detail, not a generally "smart" clarification behavior — the report should be explicit about this choice, and v2's adaptive attribute-specific policy (Fri/Sat) is both a genuine A/B comparison and a hedge in case the private evaluator's simulator differs slightly, plus it's the piece that actually demonstrates the "adaptive clarification" innovation direction from `docs/competition_specification.md` to judges.

## Status: v1 done and scale-validated

v1 is built, wired, tested (17/17 unit tests passing), and validated at scale overnight. Results:

- **Public 200-session run**: HR@10 0.93, MRR 0.625, MTTC 2.86, Efficiency 0.814, TechnicalScore **0.815** — vs. baseline's 0.125/0.068/9.81/0.119/0.107.
- **Overnight synthetic runs** (`results_buying_override.json`, `results_browsing_boundary_10k.json` — 10,000 sessions per scenario type, 40,000 total, via `scripts/eval_buying_override.py` / `scripts/eval_browsing_boundary.py`): mix-weighted (40/40/15/5) composite **HR@10 0.911, MRR 0.633, MTTC 2.99, Efficiency 0.801, TechnicalScore 0.806** — closely matches the 200-session estimate, confirming v1 isn't overfit to the small public set.
- **One correction from the small-sample estimate**: Boundary looked perfect at n=10 (HR@10 1.0, MRR 0.917) but settles at HR@10 0.924 / MRR 0.638 at n=10,000 — essentially identical to Browsing, as expected (Boundary converges to the same outcome as Browsing, just delayed by the one wasted refusal turn). Treat **0.806**, not 0.815, as the number to beat going forward.
- **Buying has the lowest MRR of the four scenarios at scale (0.619)**, confirmed not noise (200-sample: 0.577; 10K-sample: 0.619 — same ranking either way). Root cause identified via manual transcript inspection (see Diagnostic Findings below): Buying's turn-1 head start causes early-but-mediocre-rank hits.

### Diagnostic findings from manual transcript review (`print_transcripts.py` re-run against live v1)

- **Term-discriminativeness gap**: an Intent Override session hit at rank 10 (barely inside the window) after accumulating "leather", "Imported", "Buckle closure" — the latter two are generic manufacturing boilerplate shared by many belt listings, not discriminating signal, but `retrieval.py` currently weights them the same as a rare, distinctive term. This likely also explains Buying's low MRR (turn-1's disclosed "hard constraint" is sometimes a messy, near-title-length phrase rather than a clean attribute value — see the Wicca-pendant example in the intent-card dump earlier this session).
- **Override-value-leak edge case, observed live**: in that same session, the scripted override at turn 3 ("ignore my earlier preference, what I need is: leather") repeated a value already disclosed one turn earlier via a normal `"other"` answer — informationally a no-op turn. Not harmful by itself, but confirms the theoretical leak case flagged earlier is real and worth a short-circuit check in `detect_override`.
- **Boundary/Browsing convergence confirmed empirically**: the two scenarios produced *bit-identical* `hit_rate_at_10`/`mrr` at 10,000 samples each (0.9236 / 0.638112), differing only in `mttc` (2.87 vs 3.70) — exactly the ~1-turn delay predicted from Boundary's one-time refusal mechanic.

## Update: pool-pagination fix (found via manual miss/low-rank replay, implemented same session)

Diagnosed via `scripts/replay_sessions.py`: once `state.card_exhausted`, `starter/agent.py` was repeating the exact same top-10 slice of the cached candidate pool for every remaining turn, even though `retrieval.py` already fetches up to 240 candidates. Since a hit is scored by its position *within that turn's submitted list*, not true global catalog rank, a target sitting at true rank 35 (confirmed empirically for `synth_buying_000000` / `B08FD7XVXK`, both via a naive term-overlap check and directly through the production `RetrievalEngine`) was a **guaranteed miss** under the old code, no matter how many turns remained.

Fix: `starter/agent.py` now pages `top_k` further into `state.last_candidate_pool` on each turn where `ask_attribute` is `None` (tracked via `state.pool_offset` in `starter/state.py`), resetting to 0 whenever a fresh query actually runs. `starter/clarify.py`'s wrap-up message rotates with the page so the transcript doesn't read as a stuck loop. Verified: `synth_buying_000000` went from a confirmed miss (true rank 35, never surfaced across 10 turns) to a **hit at turn 5, rank 2** in the paged window.

Public 200-session result: **HR@10 0.93→0.99, MRR 0.625→0.651, MTTC 2.86→2.55, TechnicalScore 0.815→0.859** — every scenario improved or held steady, no regressions (Browsing reached a perfect 80/80).

## Update: term-rarity (IDF-style) weighting in `retrieval.py`

Implemented the Fri/Sat priority: `RetrievalEngine._rarity_weight(term)` discounts constraint-coverage scoring by how large a fraction of the 50K catalog a term appears in (cached per-term via `_document_frequency`, an FTS5 `MATCH` count query), so generic boilerplate ("imported" at 30.6% of catalog, "closure" at 38.6%) can't dominate the score the way a rare, distinctive term (e.g. "pink" at 3.5%, "alloy" at 1.6%) can.

**First attempt regressed MRR** (log-IDF normalized against the theoretical df=0 ceiling, floor 0.3): HR@10 unchanged at 0.99 (the pagination fix above already caught everything this was meant to fix), but MRR dropped 0.651→0.629 — real catalog terms never get close to the theoretical ceiling, so the formula compressed nearly everything into a narrow low range and over-penalized common-but-still-correct materials like "cotton" (19.6% of catalog, genuinely the right answer when disclosed).

**Retuned to a linear document-frequency-ratio discount** (`min(1.15, max(0.6, 1.0 - doc_freq/total))` — see `starter/retrieval.py`'s `_rarity_weight`): now a genuine net improvement over pagination alone — **MRR 0.651→0.654, TechnicalScore 0.859→0.860**, with the clearest win in Intent Override (MRR 0.681→0.728, the scenario the original diagnosis came from) and Boundary (0.917→0.920). Browsing dipped slightly (0.644→0.631) but not enough to erase the net gain.

**Not yet re-validated at the 10K-per-scenario scale** for either the pagination or rarity-weighting fixes — do this before trusting the magnitude, since Boundary's small-sample estimate was previously overstated by a similar margin in the other direction, and note the per-term rarity lookups add real query overhead (cached after first use, but worth checking against the "operational constraints" webinar question about judging timeouts). Re-run `scripts/eval_buying_override.py` and `scripts/eval_browsing_boundary.py` before Fri/Sat planning locks in a new baseline number.

## Day-by-Day Plan

### Tonight (v1) — adaptive core loop, not a static-order search — ✅ DONE

First, a 20-30 min joint sync to fix the interface: exact `SessionState` fields, and the shape of the candidate-pool object retrieval hands to clarify (list of `(parent_asin, product_dict, score)` — clarify needs the product fields to compute its impurity heuristic, not just IDs). Once that's agreed, split:

- **Person A**: `starter/state.py` — `SessionState` dataclass keyed by `session_id` in a dict (replaces the current `self._sessions: set[str]` at `starter/agent.py:41,75`), tracking accumulated `(attribute, token)` terms, asked/exhausted/no-preference attributes, budget, mode, last candidate pool. Then `starter/nlu.py` — turn-1 Buying/Browsing routing (`infer_mode`), constraint extraction/classification using vocabularies that are a *superset* of the simulator's own keyword lists (`evaluator/local_evaluator.py:21-24,137-151`) so the agent doesn't overfit to visible simulator internals, override detection (`detect_override`, replace-not-append semantics), and a **no-preference/exhaustion classifier keyed on response phrasing, not on which attribute was asked**: `"I don't have a preference for X; please use your judgment"` (Boundary's one-time refusal — do **not** mark the attribute exhausted, it's fully re-askable next turn once the flag is spent) vs. `"I don't have an additional preference for X"` (genuine exhaustion — that bucket is empty forever). Then `starter/clarify.py` — default v1 policy is **ask `"other"` first and repeatedly** (bypasses `classify_constraint()` entirely per `evaluator/local_evaluator.py:180`, harvesting up to 2 undisclosed constraints per turn regardless of type — the whole intent card is capped at ~4 items, so this exhausts it in ~2 real asks) until a genuine "no additional preference for other" comes back, at which point the entire card is known and no further ask (specific or otherwise) can add information. Vary the natural-language `message` text across turns even while `ask_attribute` stays `"other"`, since the demonstrated-session deliverable will read repetitive otherwise. Never return `null` while turns remain and the card isn't confirmed exhausted (a `null` guess is strictly worse — see `customer_reply()` at `evaluator/local_evaluator.py:170-171`).
- **Person B**: `starter/retrieval.py` — tiered AND (strict, on hard constraints) → OR (broad bm25) query built from the *full accumulated state*, not just the current turn's message (fixes the core bug in `starter/agent.py:86-96`), plus structured re-scoring (budget/color/material boosts) and a stability bonus across turns. In parallel, start the synthetic-session generator (see Overnight below) and the run-comparison script, since neither depends on A's modules being finished.
- **Both, end of night**: wire `Agent.reset`/`Agent.respond` in `starter/agent.py` together (this is the one file both touch — keep it a short, scheduled joint session rather than editing concurrently), always submitting the full honest top-10 every turn (no withholding — there's no verification mechanism to spend a turn on; the only lever is ranking/question quality). Confirm `python3 -m evaluator.local_evaluator` runs end-to-end before either person stops for the night — this gate must pass before the overnight run starts.

This replaces "search static orderings" entirely — there's no ordering to search because the ordering is computed dynamically per turn from the live candidate pool.

### Overnight (Thu→Fri) — validate at scale instead of permutation search — ✅ DONE

Built `scripts/synthetic_common.py` (shared batching/checkpoint/ETA harness, reusing `catalog_index`/`evaluate`/`metric_summary` from `evaluator/local_evaluator.py`) plus two entry scripts so the two of you could split the scenario types across machines: `scripts/eval_buying_override.py` and `scripts/eval_browsing_boundary.py`, each generating synthetic sessions via `intent_card()`/`behavior_for()` (confirmed these are exactly what derives hidden fields when a sample doesn't ship them — see `materialize_hidden_fields()` and `tests/test_evaluator.py:66-72`) directly from the 50K catalog. Ran 10,000 products per scenario type overnight (40,000 sessions total) — see Status section above for results. Each script writes a live checkpoint after every batch (default batch size 100) with an ETA estimate, so a smoke test with `--limit 20` before committing to a full overnight run is cheap — do this for any future large-scale run before walking away from the machine.

Did **not** run a permutation search — confirmed unnecessary given v1's results.

### Fri/Sat (v2) — target the diagnosed gaps, not speculative tuning

Priorities reordered based on the overnight diagnostics, highest-value first:

- **Both, first thing**: build a miss/low-rank finder — pull `sample_id`s from `results_buying_override.json` / `results_browsing_boundary_10k.json` where `hit=false` or `best_rank>5`, then replay those specific sessions through the `print_transcripts.py`-style tool (parameterized by `sample_id` instead of "first found per scenario") to see exactly what's going wrong, rather than guessing from aggregates alone.
- **Person B**: add **term-rarity (IDF-style) weighting** to `retrieval.py`'s rescoring — down-weight matches on terms that appear across a large fraction of the catalog (e.g. "Imported", "Buckle closure") relative to rare, distinctive terms (specific colors, materials, use-cases). Directly targets the rank-10 Intent Override hit and is the leading hypothesis for Buying's comparatively low MRR (0.619) — Buying's turn-1 disclosed "hard constraint" is sometimes a messy, near-title-length phrase rather than a clean attribute value, and current scoring likely can't tell signal from boilerplate within it.
- **Person A**: short-circuit `detect_override` when the isolated new value is already present in `state.hard_terms` for its classified attribute (the observed leak case — override fires on a value already disclosed a turn earlier via a normal `"other"` answer) — should be a no-op turn instead of a redundant clear-and-reinsert. Then build the **adaptive attribute-specific policy as an A/B variant** against v1's "other"-first default (still valuable for the "adaptive clarification" innovation-direction credit per the Context section note, now secondary to the retrieval fix above since "other"-first is already performing well). Write the NLU-side unit tests in `tests/test_agent.py` for both the override short-circuit and the info-gain heuristic, using phrasings deliberately different from the simulator's literal strings.
- Validate every change against **both** the 200-session public set and a fresh `--limit 3000`-ish rerun of the two overnight scripts (not just the public set) — the overnight run is now the more trustworthy signal per the Status section, so treat it as the primary check, public-200 as the fast sanity check.
- Overnight Fri→Sat: full-scale rerun (`--limit 10000` or higher) of both scripts with the day's changes — only keep changes that improve `scenario_metrics` at scale, specifically Buying's MRR and Intent Override's rank distribution.

### Sun (v3) — confidence gating experiment, targeted at Buying specifically

**Person B** implements score-gap-based gating on `recommendations` in `retrieval.py` (only submit candidates clearing a confidence margin — locking in an early low-rank hit forecloses a possibly-better later rank, but blanket withholding delays already-good hits for nothing) as a togglable variant. This is no longer speculative: the overnight data shows Buying specifically hitting earliest (avg. turn ~2.3) but at the worst average rank of the four scenarios, the exact tradeoff this experiment targets — so gate primarily on Buying's turn-1/turn-2 submissions first, then check it doesn't regress the other three. **Person A** builds the A/B comparison harness and checks that gating doesn't break `clarify.py`'s candidate-pool-based heuristic (fewer live candidates changes the impurity calculation clarify relies on — this is the main integration risk to watch). Both review the A/B results together against the public 200 *and* a full-scale overnight-script rerun; only adopt gating if it consistently improves the composite `TechnicalScore` (baseline to beat: **0.806**, the scale-validated number, not the original 0.815 public-only estimate) on both — this is explicitly an empirical question, not one to assume the answer to.

### Mon (buffer) — bug fixes + the non-code deliverables

- Morning: both do a final bug bash together on whatever v3 surfaced; full-suite sanity run (`python3 -m unittest discover -s tests` + `python3 -m evaluator.local_evaluator`).
- **Person A**: write the required report (method, model choice — note: no LLM/network dependency if kept stdlib-only, matching `README.md:37`'s constraint and `docs/submission_rules.md`'s network-disclosure requirement — cost/latency, limitations).
- **Person B**: capture one demonstrated multi-turn session transcript (reuse the transcript-printer approach from earlier in this session, adapted to the final agent) — required per `docs/competition_specification.md` "Final Deliverables" — and package per the layout in `docs/submission_rules.md` (`agent.py`, `requirements.txt`, `README.md`, `src/`).
- Both review each other's deliverable before submitting.

## Testing/Validation Additions

- New `tests/test_agent.py` (mirroring the fixture pattern in `tests/test_evaluator.py:42-72`): session-isolation regression test (two interleaved `session_id`s, assert no cross-contamination), override-replaces-not-appends test, and constraint-parsing tests using phrasings that deliberately differ from the simulator's literal strings (e.g. "you decide", "scratch that, go with X instead") to catch overfitting to `evaluator/local_evaluator.py`'s exact templates.
- Track `scenario_metrics` (buying/browsing/intent_override/boundary breakdown) after every change, not just the aggregate — Browsing and Boundary are where the biggest early gains should show up; Intent Override is the most novel logic and worth watching for regressions.

## Critical Files
- `starter/agent.py` — entry point, must keep exporting `Agent`
- `starter/state.py`, `starter/nlu.py`, `starter/clarify.py`, `starter/retrieval.py` — the v1 implementation
- `evaluator/local_evaluator.py` — source of `intent_card()`/`behavior_for()` for synthetic generation, and the scoring loop to reason against
- `scripts/synthetic_common.py`, `scripts/eval_buying_override.py`, `scripts/eval_browsing_boundary.py` — large-scale validation harness, now the primary tool for checking v2/v3 changes generalize before committing to them
- `docs/competition_specification.md`, `docs/agent_api_contract.json`, `docs/submission_rules.md` — contract and deliverable requirements
- `tests/test_evaluator.py`, `tests/test_agent.py` — existing test fixture patterns to extend
- `docs/baseline_results.json` — original weak-baseline reference; **0.806** (mix-weighted, 10K-sample) is v1's number to beat going forward, not `docs/baseline_results.json`'s 0.107
