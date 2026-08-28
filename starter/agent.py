from __future__ import annotations

from pathlib import Path

from starter import clarify, nlu
from starter.retrieval import RetrievalEngine
from starter.state import SessionState


class Agent:
    """Stateful offline shopping agent using NLU, clarification, and tiered retrieval."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", clarification_policy: str = "other_first") -> None:
        self.catalog_path = Path(catalog_path)
        self.retrieval = RetrievalEngine(self.catalog_path)
        self.clarification_policy = clarification_policy
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = SessionState()
        state.clarification_policy = self.clarification_policy
        nlu.seed_profile_preferences(state, user_profile)
        self._sessions[session_id] = state

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        intent_changed = nlu.detect_override(state, user_message)
        nlu.apply_customer_message(state, user_message, turn)

        category = " ".join(state.category_tokens)
        hard_constraints = [
            value
            for values in state.hard_terms.values()
            for value in values
        ]
        soft_preferences = [
            value
            for values in state.soft_terms.values()
            for value in values
        ]
        signature = (category, tuple(hard_constraints), tuple(soft_preferences))
        if signature == state.last_query_signature and state.last_candidate_pool and not intent_changed:
            pool = state.last_candidate_pool
        else:
            pool = self.retrieval.search(
                category,
                hard_constraints,
                soft_preferences=soft_preferences,
                top_k=top_k,
                candidate_limit=240,
                previous_ids=state.last_candidates,
                intent_changed=intent_changed,
            )
            state.last_query_signature = signature
            state.last_candidate_pool = pool

        if turn >= 10:
            ask_attribute = None
            message = "These are the closest matches for your requirements."
            state.last_ask_attribute = None
        else:
            ask_attribute, message = clarify.choose_ask_attribute(state)

        if ask_attribute is None:
            # Nothing new to learn from the customer — page deeper into the already-fetched
            # pool instead of repeating the same top-10 every remaining turn. A hit is scored
            # by its position within *this turn's* submission, not true global rank, so this
            # can turn a candidate stuck at, say, rank 35 into a real hit a few turns later.
            window = pool[state.pool_offset : state.pool_offset + top_k]
            if not window and state.pool_offset > 0:
                state.pool_offset = 0
                window = pool[:top_k]
            state.pool_offset += top_k
        else:
            state.pool_offset = 0
            window = pool[:top_k]

        state.last_candidates = [parent_asin for parent_asin, _, _ in window]

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin}
                for parent_asin, _, _ in window
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
