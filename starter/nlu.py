from __future__ import annotations

import re

from .state import SessionState

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "your", "looking",
    "need", "prefer", "preference", "additional",
}

# Deliberately a superset of evaluator/local_evaluator.py's own MATERIAL_RE/COLOR_RE/MATERIALS
# and classify_constraint() keyword lists — the simulator's lists are simulator internals, not
# ground truth, and matching only those risks missing real catalog vocabulary.
MATERIAL_WORDS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
    "denim", "linen", "suede", "cashmere", "canvas", "mesh", "velvet", "satin", "fleece",
    "corduroy", "chiffon", "lace", "knit", "textile", "alloy", "brass", "copper", "titanium",
    "platinum", "rhodium", "stainless steel", "sterling silver", "gold plated", "rubber",
    "vinyl", "acrylic", "resin", "plastic", "wood", "bamboo", "vegan leather", "faux leather",
}
COLOR_WORDS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple",
    "yellow", "orange", "navy", "beige", "maroon", "teal", "tan", "olive", "cream", "gold",
    "silver", "burgundy", "ivory", "charcoal", "khaki", "rose gold",
}
SIZE_WORDS = {
    "size", "sizing", "width", "wide", "narrow", "petite", "plus", "small", "medium",
    "large", "xl", "xs", "regular", "tall", "short",
}
STYLE_WORDS = {
    "style", "fit", "sleeve", "neckline", "department", "collar", "cut", "silhouette",
    "pattern", "print", "design",
}
USE_CASE_WORDS = {
    "hiking", "running", "gym", "winter", "outdoor", "work", "casual", "formal", "travel",
    "everyday", "athletic", "beach", "wedding", "sport", "workout", "party",
}

BUDGET_RE = re.compile(
    r"\$\s?(\d+(?:\.\d+)?)|under\s+\$?\s?(\d+(?:\.\d+)?)|less than\s+\$?\s?(\d+(?:\.\d+)?)|"
    r"budget[^0-9]{0,15}(\d+(?:\.\d+)?)",
    re.I,
)

VAGUE_MARKERS = re.compile(
    r"\bexplor\w*\b|\bnot sure\b|\bjust looking\b|\bbrowsing\b|\bany (?:suggestion|option)s?\b|"
    r"\bopen to\b|\bnot (?:sure|certain|committed)\b|\bhelp me find\b|\bstill (?:deciding|looking)\b|"
    r"\bno specific\b",
    re.I,
)

LEAD_IN_RE = re.compile(r"^(i am|i'm)\s+looking for\s+|^looking for\s+|^i need\s+|^i want\s+", re.I)
CLAUSE_BREAK_RE = re.compile(r"[.!?]|,?\s*\bbut\b", re.I)

OVERRIDE_MARKERS_RE = re.compile(
    r"\bactually\b|\binstead\b|\bignore (my|that|the) (earlier|previous|last)\b|"
    r"\bscratch that\b|\bchange(d)? my mind\b|\bno longer\b|\brather than\b|"
    r"\bon second thought\b|\bforget what i said\b",
    re.I,
)

# Boundary's one-time "use your judgment" refusal — NOT exhaustion, the attribute stays askable.
NO_PREFERENCE_JUDGMENT_RE = re.compile(
    r"use your (judgment|discretion)|your call|up to you|either (is|way) (fine|works|ok)|"
    r"doesn't matter to me|not particular",
    re.I,
)
# Genuine "nothing left in this bucket" — blocks re-asking that specific attribute.
ADDITIONAL_PREFERENCE_EXHAUSTED_RE = re.compile(
    r"don't have (?:an )?additional preference|nothing (else|more) (to add|comes to mind)|"
    r"that's (all|everything) i (know|have)|no further preference",
    re.I,
)
NULL_NUDGE_RE = re.compile(r"ask me about (one|a) specific attribute|not quite right yet", re.I)


def extract_terms(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOPWORDS]


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|usd)\s*\d", lowered) or re.search(
        r"\b(under|less than|around)\b.{0,10}\d", lowered
    ):
        return "budget"
    if any(word in lowered for word in MATERIAL_WORDS):
        return "material"
    if any(word in lowered for word in COLOR_WORDS):
        return "color"
    if any(word in lowered for word in SIZE_WORDS):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    if any(word in lowered for word in USE_CASE_WORDS):
        return "use_case"
    return "feature"


def extract_category_clause(message: str) -> str:
    text = LEAD_IN_RE.sub("", message.strip())
    match = CLAUSE_BREAK_RE.search(text)
    return text[: match.start()] if match else text


def infer_mode(user_message: str) -> str:
    if VAGUE_MARKERS.search(user_message):
        return "browsing"
    if BUDGET_RE.search(user_message):
        return "buying"
    remainder = _remainder_after_category(user_message)
    if remainder and classify_constraint(remainder) != "feature":
        return "buying"
    return "browsing"


def _remainder_after_category(message: str) -> str:
    clause = extract_category_clause(message)
    idx = message.lower().find(clause.lower()) if clause else -1
    if idx == -1:
        return message.strip()
    return message[idx + len(clause):].strip(" .,")


def _parse_budget(text: str) -> float | None:
    match = BUDGET_RE.search(text)
    if not match:
        return None
    for group in match.groups():
        if group:
            try:
                return float(group)
            except ValueError:
                continue
    return None


def _split_disclosure_chunks(message: str) -> list[str]:
    colon_match = re.search(r":\s*(.+)$", message.strip())
    body = colon_match.group(1) if colon_match else message.strip()
    body = body.rstrip(" .")
    parts = re.split(r";|(?<!\d),(?!\d)", body)
    return [p.strip() for p in parts if p.strip()]


def _record_constraint(state: SessionState, attribute: str, raw_value: str, weight: float = 1.0) -> None:
    state.hard_terms.setdefault(attribute, [])
    if raw_value not in state.hard_terms[attribute]:
        state.hard_terms[attribute].append(raw_value)
    for token in extract_terms(raw_value):
        state.add_term(attribute, token, weight)


def _apply_disclosure(state: SessionState, message: str, attribute: str | None) -> None:
    for chunk in _split_disclosure_chunks(message):
        chunk = chunk.strip(" .;")
        if not chunk:
            continue
        bucket = classify_constraint(chunk)
        target_attr = attribute if attribute and attribute != "other" else bucket
        _record_constraint(state, target_attr, chunk)
        if bucket != target_attr:
            # Dual-tag: keep the term usable for retrieval even if our attribute guess
            # (or the simulator's) disagrees with the requested bucket.
            for token in extract_terms(chunk):
                state.add_term(bucket, token)
        budget = _parse_budget(chunk)
        if budget is not None:
            state.budget_max = budget


def _isolate_override_value(message: str) -> str | None:
    for pattern in (
        r":\s*(.+)$",
        r"\binstead\b[,:]?\s*(.+)$",
        r"\bwhat i need is\s*(.+)$",
        r"\bi (?:actually )?(?:want|need|prefer)\s+(.+)$",
    ):
        match = re.search(pattern, message, re.I)
        if match:
            candidate = match.group(1).strip(" .")
            if candidate:
                return candidate
    stripped = OVERRIDE_MARKERS_RE.sub("", message).strip(" ,.")
    return stripped or None


def _most_recent_attribute(state: SessionState) -> str | None:
    if state.last_ask_attribute and state.last_ask_attribute != "other":
        return state.last_ask_attribute
    for attr in reversed(list(state.hard_terms)):
        return attr
    return None


def _apply_override(state: SessionState, message: str) -> None:
    new_value = _isolate_override_value(message)
    if not new_value:
        target_attr = _most_recent_attribute(state)
        if target_attr:
            state.clear_attribute_terms(target_attr)
        return
    attribute = classify_constraint(new_value)
    state.clear_attribute_terms(attribute)
    _record_constraint(state, attribute, new_value)
    state.exhausted_attributes.discard(attribute)
    state.no_preference_attributes.discard(attribute)


def apply_customer_message(state: SessionState, message: str, turn: int) -> None:
    """Mutates state in place from the latest customer turn. Call once per respond()."""
    state.turn_count = turn
    if turn == 1:
        state.mode = infer_mode(message)
        state.category_tokens = extract_terms(extract_category_clause(message))

    if OVERRIDE_MARKERS_RE.search(message):
        _apply_override(state, message)
        return

    attribute = state.last_ask_attribute
    if NO_PREFERENCE_JUDGMENT_RE.search(message):
        if attribute:
            state.no_preference_attributes.add(attribute)
        return
    if ADDITIONAL_PREFERENCE_EXHAUSTED_RE.search(message):
        if attribute:
            state.exhausted_attributes.add(attribute)
            if attribute == "other":
                state.card_exhausted = True
        return
    if NULL_NUDGE_RE.search(message):
        return

    if turn == 1:
        remainder = _remainder_after_category(message)
        if not remainder or VAGUE_MARKERS.search(remainder):
            return
        _apply_disclosure(state, remainder, attribute=None)
        return

    _apply_disclosure(state, message, attribute)


def seed_profile_preferences(state: SessionState, user_profile: dict) -> None:
    """Safe personalization: low-weight signal from the anonymized aggregate profile."""
    for tag in (user_profile or {}).get("preference_tags") or []:
        tag = str(tag).strip()
        if not tag:
            continue
        state.soft_terms.setdefault("feature", [])
        if tag not in state.soft_terms["feature"]:
            state.soft_terms["feature"].append(tag)
        for token in extract_terms(tag):
            state.add_term("feature", token, weight=0.5)
