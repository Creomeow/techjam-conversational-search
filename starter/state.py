from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Per-session conversation memory. One instance per session_id, held in Agent."""

    turn_count: int = 0
    mode: str | None = None  # "buying" or "browsing", set at turn 1
    # Clarification strategy is an A/B knob.  Keep the simulator-validated v1
    # behavior as the default; the adaptive policy is opt-in for experiments.
    clarification_policy: str = "other_first"

    category_tokens: list[str] = field(default_factory=list)

    # attribute -> raw disclosed strings, in disclosure order
    hard_terms: dict[str, list[str]] = field(default_factory=dict)
    # attribute -> low-weight signals (e.g. seeded from user_profile.preference_tags)
    soft_terms: dict[str, list[str]] = field(default_factory=dict)

    # (attribute, token) -> weight; the actual retrieval fuel
    accumulated_terms: dict[tuple[str, str], float] = field(default_factory=dict)

    budget_max: float | None = None

    asked_attributes: set[str] = field(default_factory=set)
    no_preference_attributes: set[str] = field(default_factory=set)  # one-off refusal, NOT exhaustion
    exhausted_attributes: set[str] = field(default_factory=set)      # genuinely empty bucket

    card_exhausted: bool = False  # true once "other" itself comes back empty

    last_ask_attribute: str | None = None
    last_candidates: list[str] = field(default_factory=list)
    # Retrieval-to-clarification integration contract: (parent_asin, product fields, score).
    last_candidate_pool: list[tuple[str, dict, float]] = field(default_factory=list)
    last_query_signature: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
    # Once the card is exhausted (ask_attribute=None), page deeper into the cached pool each
    # turn instead of repeating the same top-10 — a hit is scored by its position within
    # whatever's submitted that turn, not by true global rank, so paging can convert a
    # rank-35 candidate into a real hit a few turns later instead of a guaranteed miss.
    pool_offset: int = 0

    def add_term(self, attribute: str, token: str, weight: float = 1.0) -> None:
        key = (attribute, token)
        self.accumulated_terms[key] = max(self.accumulated_terms.get(key, 0.0), weight)

    def clear_attribute_terms(self, attribute: str) -> None:
        for key in [k for k in self.accumulated_terms if k[0] == attribute]:
            del self.accumulated_terms[key]
        self.hard_terms.pop(attribute, None)
