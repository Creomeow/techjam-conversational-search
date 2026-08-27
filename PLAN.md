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

## Day-by-Day Plan

### Tonight (v1) — adaptive core loop, not a static-order search

First, a 20-30 min joint sync to fix the interface: exact `SessionState` fields, and the shape of the candidate-pool object retrieval hands to clarify (list of `(parent_asin, product_dict, score)` — clarify needs the product fields to compute its impurity heuristic, not just IDs). Once that's agreed, split:

- **Person A**: `starter/state.py` — `SessionState` dataclass keyed by `session_id` in a dict (replaces the current `self._sessions: set[str]` at `starter/agent.py:41,75`), tracking accumulated `(attribute, token)` terms, asked/exhausted/no-preference attributes, budget, mode, last candidate pool. Then `starter/nlu.py` — turn-1 Buying/Browsing routing (`infer_mode`), constraint extraction/classification using vocabularies that are a *superset* of the simulator's own keyword lists (`evaluator/local_evaluator.py:21-24,137-151`) so the agent doesn't overfit to visible simulator internals, no-preference/exhaustion detection, override detection (`detect_override`, replace-not-append semantics). Then `starter/clarify.py` — **adaptive** `choose_ask_attribute(state, candidate_pool)`: skip exhausted/no-preference attributes, and among what's left, prefer whichever attribute is least homogeneous in the current candidate pool (an information-gain proxy) rather than a fixed global order. Never return `null` while turns remain and attributes are un-exhausted (a `null` guess is strictly worse — see `customer_reply()` at `evaluator/local_evaluator.py:170-171`).
- **Person B**: `starter/retrieval.py` — tiered AND (strict, on hard constraints) → OR (broad bm25) query built from the *full accumulated state*, not just the current turn's message (fixes the core bug in `starter/agent.py:86-96`), plus structured re-scoring (budget/color/material boosts) and a stability bonus across turns. In parallel, start the synthetic-session generator (see Overnight below) and the run-comparison script, since neither depends on A's modules being finished.
- **Both, end of night**: wire `Agent.reset`/`Agent.respond` in `starter/agent.py` together (this is the one file both touch — keep it a short, scheduled joint session rather than editing concurrently), always submitting the full honest top-10 every turn (no withholding — there's no verification mechanism to spend a turn on; the only lever is ranking/question quality). Confirm `python3 -m evaluator.local_evaluator` runs end-to-end before either person stops for the night — this gate must pass before the overnight run starts.

This replaces "search static orderings" entirely — there's no ordering to search because the ordering is computed dynamically per turn from the live candidate pool.

### Overnight (Thu→Fri) — validate at scale instead of permutation search

Person B kicks this off once the joint wiring gate above passes (so it should be scripted/ready before that moment, not built from scratch after):
1. Run `python3 -m evaluator.local_evaluator` against the 200 public sessions; check `scenario_metrics` against baseline.
2. Run a synthetic-session generator reusing `intent_card()` and `behavior_for()` from `evaluator/local_evaluator.py` (confirmed these are exactly what derives hidden fields when a sample doesn't ship them — see `materialize_hidden_fields()` and `tests/test_evaluator.py:66-72`) to sample a few thousand `(parent_asin, scenario_type)` pairs directly from the 50K catalog, matching the official 40/40/15/5 mix. Run the full evaluate loop against this larger set overnight for statistical power beyond 200 samples.

Do **not** run a permutation search tonight. If there's leftover overnight compute, spend it re-running step 2 with a handful of `clarify.py` heuristic variants (e.g., weight choices in the info-gain proxy) — not thousands.

### Fri/Sat (v2) — refine retrieval + clarification, validate with a held-out split

- **Person A**: refine the info-gain heuristic in `clarify.py`; add scenario-conditional first-question logic (once `mode` is known) instead of one universal order. Write the NLU-side unit tests in `tests/test_agent.py` — constraint-parsing and override tests using phrasings deliberately different from the simulator's literal strings (e.g. "you decide", "scratch that, go with X instead") to catch overfitting to `evaluator/local_evaluator.py`'s exact templates.
- **Person B**: retune retrieval field weights in the bm25 call, AND-relaxation backoff steps, candidate pool size before re-scoring. Own the **150/50 split** of the 200 public sessions (tune only against the 150, both people validate changes against the held-out 50 before accepting them — this is the guard against the overfitting risk flagged earlier) and the three-way overnight comparison script (150-search-set vs. 50-held-out vs. synthetic set). Write the session-isolation regression test in `tests/test_agent.py` (interleave two `session_id`s, assert no cross-contamination).
- Overnight Fri→Sat: re-run held-out 50 + synthetic set with the day's changes from both people together (single combined build, not two separate branches) — only keep changes that improve the held-out and synthetic numbers, not just the search set.

### Sun (v3) — confidence gating experiment

**Person B** implements score-gap-based gating on `recommendations` in `retrieval.py` (only submit candidates clearing a confidence margin — locking in an early low-rank hit forecloses a possibly-better later rank, but blanket withholding delays already-good hits for nothing) as a togglable variant. **Person A** builds the A/B comparison harness and checks that gating doesn't break `clarify.py`'s candidate-pool-based heuristic (fewer live candidates changes the impurity calculation clarify relies on — this is the main integration risk to watch). Both review the A/B results together against the held-out 50 *and* the synthetic set; only adopt gating if it consistently improves the composite `TechnicalScore = 0.50×HR@10 + 0.30×MRR + 0.20×Efficiency` on both — this is explicitly an empirical question, not one to assume the answer to.

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
- `evaluator/local_evaluator.py` — source of `intent_card()`/`behavior_for()` for synthetic generation, and the scoring loop to reason against
- `docs/competition_specification.md`, `docs/agent_api_contract.json`, `docs/submission_rules.md` — contract and deliverable requirements
- `tests/test_evaluator.py` — existing test fixture pattern to mirror
- `docs/baseline_results.json` — reference numbers to beat
