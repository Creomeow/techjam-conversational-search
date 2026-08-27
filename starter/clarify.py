from __future__ import annotations

from .state import SessionState

# v1 policy: ask "other" first and repeatedly. customer_reply() in evaluator/local_evaluator.py
# discloses undisclosed constraints regardless of classification when asked "other", so this
# front-loads information extraction faster and more reliably than guessing specific attributes.
# v2 (Fri/Sat) A/B-tests an adaptive attribute-specific policy against this default — see PLAN.md.
_OTHER_PROMPTS = [
    "Is there anything else about it that matters to you — material, color, budget, anything?",
    "Anything else I should know to help find the right one?",
    "What else is important to you here?",
    "Any other details or preferences I should factor in?",
    "Is there anything more you'd like me to consider?",
]

_EXHAUSTED_MESSAGE = "Got it — let me pull together the best options based on everything you've shared."


def choose_ask_attribute(state: SessionState) -> tuple[str | None, str]:
    if state.card_exhausted:
        state.last_ask_attribute = None
        return None, _EXHAUSTED_MESSAGE
    prompt = _OTHER_PROMPTS[state.turn_count % len(_OTHER_PROMPTS)]
    state.asked_attributes.add("other")
    state.last_ask_attribute = "other"
    return "other", prompt
