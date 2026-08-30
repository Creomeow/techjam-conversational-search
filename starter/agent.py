from __future__ import annotations

from pathlib import Path
from typing import Iterable

from starter import clarify, nlu
from starter.retrieval import RetrievalEngine
from starter.state import SessionState


class Agent:
    """Stateful offline shopping agent using NLU, clarification, and tiered retrieval."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        clarification_policy: str = "other_first",
        *,
        confidence_gating: bool = False,
        confidence_gap: float = 0.18,
        confidence_gating_turns: Iterable[int] = (1, 2),
        confidence_gating_modes: Iterable[str] = ("buying",),
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retrieval = RetrievalEngine(
            self.catalog_path,
            confidence_gating=confidence_gating,
            confidence_gap=confidence_gap,
            confidence_gating_turns=confidence_gating_turns,
            confidence_gating_modes=confidence_gating_modes,
        )
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
        pool_reused = (
            signature == state.last_query_signature
            and bool(state.last_candidate_pool)
            and not intent_changed
        )
        if pool_reused:
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
            state.pool_offset = 0

        if turn >= 10:
            ask_attribute = None
            message = "These are the closest matches for your requirements."
            state.last_ask_attribute = None
        else:
            ask_attribute, message = clarify.choose_ask_attribute(state)

        # A fresh query starts at the top. While clarification is still active, keep a reused
        # pool at the top as well: Intent Override can make those candidates score-eligible
        # without changing the query signature. Once clarification ends, continue from the
        # next unseen window instead of repeating ranks 1-10 at the exhaustion transition.
        if ask_attribute is not None and pool_reused:
            state.pool_offset = 0
        window = pool[state.pool_offset : state.pool_offset + top_k]
        if not window and state.pool_offset > 0:
            state.pool_offset = 0
            window = pool[:top_k]
        state.pool_offset += top_k

        recommendations = self.retrieval.recommendation_window(
            pool,
            top_k,
            turn=turn,
            mode=state.mode,
            offset=state.pool_offset - top_k,
        )
        state.last_candidates = [parent_asin for parent_asin, _, _ in recommendations]

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin}
                for parent_asin, _, _ in recommendations
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
