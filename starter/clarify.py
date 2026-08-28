from __future__ import annotations

import math
import re

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

_ADAPTIVE_ATTRIBUTES = ("material", "color", "size", "style", "budget", "use_case", "feature", "brand")
_ADAPTIVE_PROMPTS = {
    "material": "Do you have a material in mind, such as leather, cotton, or something else?",
    "color": "Is there a color you would especially like?",
    "size": "What size or fit should I look for?",
    "style": "Is there a particular style or design you prefer?",
    "budget": "What price range would work best for you?",
    "use_case": "What will you mainly use it for?",
    "feature": "Is there a feature that matters most to you?",
    "brand": "Do you have a preferred brand?",
}


def _product_text(product: dict) -> str:
    parts: list[str] = []
    for value in product.values():
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def _attribute_signature(attribute: str, product: dict) -> str:
    text = _product_text(product)
    if attribute == "budget":
        price = product.get("price")
        if isinstance(price, (int, float)):
            return f"{int(float(price) // 25)}"
        return "unknown"
    patterns = {
        "material": r"cotton|polyester|nylon|leather|wool|silk|fabric|canvas|denim|mesh|suede|alloy|rubber",
        "color": r"black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|navy|beige",
        "size": r"size\s*[a-z0-9]+|wide|narrow|small|medium|large|petite|plus",
        "style": r"style|casual|formal|pattern|print|design|fit|sleeve|collar",
        "use_case": r"hiking|running|gym|winter|outdoor|work|casual|formal|travel|athletic|beach|wedding",
        "feature": r"feature|breathable|waterproof|warm|comfort|durable|closure|pocket",
        "brand": r"store|brand",
    }
    match = re.search(patterns.get(attribute, r"$^"), text)
    return match.group(0) if match else "none"


def attribute_information_gain(state: SessionState, attribute: str) -> float:
    """Estimate how much a question partitions the current candidate pool."""
    products = [product for _, product, _ in state.last_candidate_pool]
    if len(products) < 2:
        return 0.0
    counts: dict[str, int] = {}
    for product in products:
        key = _attribute_signature(attribute, product)
        counts[key] = counts.get(key, 0) + 1
    total = len(products)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _choose_adaptive(state: SessionState) -> tuple[str | None, str]:
    if state.card_exhausted:
        state.last_ask_attribute = None
        index = min(state.pool_offset // 10, len(_EXHAUSTED_MESSAGES) - 1)
        return None, _EXHAUSTED_MESSAGES[index]

    available = [
        attribute for attribute in _ADAPTIVE_ATTRIBUTES
        if attribute not in state.asked_attributes
        and attribute not in state.no_preference_attributes
        and attribute not in state.exhausted_attributes
        and attribute not in state.hard_terms
    ]
    ranked = sorted(
        enumerate(available),
        key=lambda item: (-attribute_information_gain(state, item[1]), item[0]),
    )
    attribute = ranked[0][1] if ranked and attribute_information_gain(state, ranked[0][1]) > 0 else "other"
    state.asked_attributes.add(attribute)
    state.last_ask_attribute = attribute
    if attribute == "other":
        return attribute, _OTHER_PROMPTS[state.turn_count % len(_OTHER_PROMPTS)]
    return attribute, _ADAPTIVE_PROMPTS[attribute]


def choose_ask_attribute(state: SessionState, policy: str | None = None) -> tuple[str | None, str]:
    if (policy or state.clarification_policy) == "adaptive":
        return _choose_adaptive(state)
    if state.card_exhausted:
        state.last_ask_attribute = None
        index = min(state.pool_offset // 10, len(_EXHAUSTED_MESSAGES) - 1)
        return None, _EXHAUSTED_MESSAGES[index]
    prompt = _OTHER_PROMPTS[state.turn_count % len(_OTHER_PROMPTS)]
    state.asked_attributes.add("other")
    state.last_ask_attribute = "other"
    return "other", prompt
