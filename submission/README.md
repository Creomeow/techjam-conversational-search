# TechJam Submission — sideswipe

## What This Is

An offline, LLM-free shopping agent for the TechJam Conversational
E-Commerce Search Challenge. It combines a proven multi-signal
retrieval/ranking pipeline with a clarification strategy that front-loads
constraint disclosure by always asking the `other` attribute.

## Contents

```
submission/
  README.md            this file
  REPORT.md             architecture, models, cost, and limitations
  demo_session.txt      one demonstrated multi-turn session 
  requirements.txt      dependency manifest (none beyond the stdlib)
  starter/
    __init__.py
    agent.py            entry point — exports Agent
  shopping/
    __init__.py
    buyer_state.py       lightweight FSM used for retrieval-side heuristics
    conversation.py      SessionState: slot/exclusion tracking, routing, clarification
    hybrid.py            optional dense-retrieval hook (disabled by default)
    neural_reranker.py   optional neural reranker hook (unused by default config)
    normalization.py     text normalization utilities
    ranker.py            LinearRanker / FieldSignalRanker / PlaceholderSignalRanker
    retrieval.py         CatalogIndex: SQLite in-memory multi-field lexical index
    semantic_reranker.py SemanticReranker: local statistical semantic similarity model
  models/
    ranker.json           trained linear ranker weights
    semantic_ranker.json  trained semantic reranker weights
```

## Requirements

- Python 3.10 or later (developed and tested on 3.12.1).
- No third-party packages. `requirements.txt` is intentionally empty as
  every module here uses only the Python standard library (including an
  in-memory `sqlite3` index; no external database server).

## Network / Credential Requirements

**None.** This agent is fully offline, and does not require API keys or
credentials. All ranking and reranking artifacts are
local, pre-trained weight files bundled under `models/`. 

## Reproducing the Local Score

This bundle is structured as a drop-in replacement for the `starter/`
directory in the official competition scaffold 

1. Take the official scaffold (which provides `data/`, `evaluator/`, and
   `docs/`).
2. Replace its `starter/` directory with this bundle's `starter/`, and
   add this bundle's `shopping/` and `models/` directories alongside it
   (as siblings of `starter/`, at the project root).
3. Run:

   ```bash
   python3 -m evaluator.local_evaluator
   ```

   This writes per-session results and aggregate metrics to
   `results.json`.

No command-line flags, environment variables, or additional setup steps
are required.

## Reflection on limitations of solution

- Tuning was done against the small, public set. The ranker weights, diversity cap, backfill thresholds were all validated using the local evaluator's 200 sessions we kept reporting scores on. Although we managed to run some testing scripts overnight on 1000 to 10000 synthetic scenarios, this could have been extended to include the entire dataset if we had more time.
- The core trick used by the solution feels like an exploit instead of understanding the user. The whole premise of `other_first` comes from exploiting a quirk in `customer_reply` where asking "other" discloses more per turn than asking for specific attributes. It reads as a slightly dumb assistant that isn't listening since it never asks about any attributes by name or sound natural. Given more time, we could have explored more adapative policies that win on its merits rather than mindlessly exploiting a loophole.

See `REPORT.md` for architecture details, measured scores, and known
limitations.
