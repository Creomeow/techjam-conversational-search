"""Small, observable buyer-state machine for dialogue-policy experiments.

The machine is deliberately separate from product constraints and ranking.  It
is used in shadow mode first, so adding telemetry cannot change responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


BUYER_STATES = (
    "exploring",
    "specifying",
    "narrowing",
    "repairing",
    "overriding",
    "ready",
)


@dataclass
class BuyerStateMachine:
    """Finite state machine driven by explicit conversation events."""

    state: str = "exploring"
    state_confidence: float = 0.5
    last_transition_turn: int = 0
    observations: int = 0
    transitions: list[dict[str, object]] = field(default_factory=list)

    def transition(
        self,
        event: str,
        *,
        turn: int,
        constraint_count: int = 0,
        confidence: float = 0.0,
        candidate_pool_size: int = 0,
    ) -> str:
        previous = self.state
        self.observations += 1
        if event == "override_detected":
            next_state = "overriding"
        elif event == "dissatisfaction_detected":
            next_state = "repairing"
        elif event == "constraint_added":
            next_state = "narrowing" if constraint_count >= 2 else "specifying"
        elif event == "category_added":
            # A category-only first turn is intent evidence, but not a
            # searchable attribute. Keep the FSM in specifying without
            # manufacturing a feature slot in the parser.
            next_state = "specifying"
        elif event == "results_observed":
            if previous in {"overriding", "repairing"}:
                next_state = previous
            elif constraint_count >= 2 and confidence >= 0.55 and candidate_pool_size <= 30:
                next_state = "ready"
            elif constraint_count >= 2:
                next_state = "narrowing"
            elif constraint_count:
                next_state = "specifying"
            else:
                next_state = "exploring"
        elif event == "no_constraint":
            next_state = "exploring" if constraint_count == 0 else previous
        else:
            raise ValueError(f"unknown buyer-state event: {event}")

        if next_state not in BUYER_STATES:
            raise ValueError(f"unknown buyer state: {next_state}")
        self.state = next_state
        evidence_confidence = self._evidence_confidence(
            next_state,
            constraint_count=constraint_count,
            retrieval_confidence=confidence,
            candidate_pool_size=candidate_pool_size,
        )
        # Hysteresis keeps a single noisy retrieval observation from making
        # the state telemetry oscillate wildly. Explicit user events remain
        # decisive and receive full weight.
        event_weight = 1.0 if event in {
            "override_detected", "dissatisfaction_detected", "constraint_added",
            "category_added",
        } else 0.45
        self.state_confidence = round(
            event_weight * evidence_confidence
            + (1.0 - event_weight) * self.state_confidence,
            6,
        )
        if previous != next_state or event in {"override_detected", "dissatisfaction_detected"}:
            self.last_transition_turn = int(turn)
            self.transitions.append(
                {
                    "turn": int(turn),
                    "from": previous,
                    "to": next_state,
                    "event": event,
                    "confidence": self.state_confidence,
                    "constraint_count": int(constraint_count),
                    "candidate_pool_size": int(candidate_pool_size),
                }
            )
        return next_state

    @staticmethod
    def _evidence_confidence(
        state: str,
        *,
        constraint_count: int,
        retrieval_confidence: float,
        candidate_pool_size: int,
    ) -> float:
        if state in {"overriding", "repairing"}:
            return 1.0
        constraint_signal = min(1.0, constraint_count / 3.0)
        retrieval_signal = min(1.0, max(0.0, retrieval_confidence) / 0.12)
        pool_signal = 0.0 if candidate_pool_size <= 0 else max(
            0.0, min(1.0, (120.0 - candidate_pool_size) / 110.0)
        )
        if state == "exploring":
            return max(0.55, 1.0 - constraint_signal)
        if state == "specifying":
            return max(0.55, 0.7 * constraint_signal + 0.3 * pool_signal)
        if state == "narrowing":
            return max(0.55, 0.55 * constraint_signal + 0.25 * pool_signal + 0.2 * retrieval_signal)
        if state == "ready":
            return max(0.55, 0.4 * constraint_signal + 0.3 * pool_signal + 0.3 * retrieval_signal)
        return 0.5

    def suggested_action(self) -> str:
        return {
            "exploring": "elicit",
            "specifying": "clarify",
            "narrowing": "discriminate",
            "repairing": "repair",
            "overriding": "confirm_replacement",
            "ready": "recommend",
        }[self.state]

    def diagnostics(self) -> dict[str, object]:
        return {
            "state": self.state,
            "state_confidence": self.state_confidence,
            "suggested_action": self.suggested_action(),
            "last_transition_turn": self.last_transition_turn,
            "observations": self.observations,
            "transitions": list(self.transitions),
        }
