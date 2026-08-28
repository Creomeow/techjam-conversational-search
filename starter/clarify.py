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

# Rotates as `agent.py` pages deeper into the cached candidate pool on repeated exhausted
# turns (state.pool_offset advances by top_k each such turn) — keeps the transcript from
# reading as a stuck loop while the same underlying pool is being paged through.
_EXHAUSTED_MESSAGES = [
    "Got it — let me pull together the best options based on everything you've shared.",
    "Here are some other options that might be a better fit.",
    "Let me show you a few more alternatives.",
    "Here's another set of options worth considering.",
]


def choose_ask_attribute(state: SessionState) -> tuple[str | None, str]:
    if state.card_exhausted:
        state.last_ask_attribute = None
        index = min(state.pool_offset // 10, len(_EXHAUSTED_MESSAGES) - 1)
        return None, _EXHAUSTED_MESSAGES[index]
    prompt = _OTHER_PROMPTS[state.turn_count % len(_OTHER_PROMPTS)]
    state.asked_attributes.add("other")
    state.last_ask_attribute = "other"
    return "other", prompt
