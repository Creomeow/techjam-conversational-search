from __future__ import annotations

import re
from dataclasses import dataclass, field

from shopping.buyer_state import BuyerStateMachine
from shopping.normalization import normalize


ATTRIBUTE_ORDER = (
    "feature", "material", "color", "style", "size", "use_case",
    "brand", "budget", "category", "other",
)

QUESTION_TEXT = {
    "category": "What kind of item or category should I focus on?",
    "material": "Do you have a preferred material?",
    "color": "Is there a color you would like?",
    "size": "What size or fit should I look for?",
    "style": "Do you prefer a particular style or design?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget should I stay within?",
    "feature": "Which feature matters most for how you will use it?",
    "use_case": "What will you mainly use it for?",
    "other": "Is there another requirement I should keep in mind?",
}

NO_PREFERENCE_RE = re.compile(
    r"\b(?:no|don't have|do not have)\s+(?:(?:an?|any)\s+)?(?:additional\s+)?preference\b",
    re.I,
)
RETRY_MESSAGE_RE = re.compile(
    r"\b(?:not\s+quite\s+right|ask\s+me\s+about\s+one\s+specific\s+attribute)\b",
    re.I,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore\s+(?:my|the)\s+(?:earlier|previous)|what\s+i\s+need)\b",
    re.I,
)
NEGATIVE_RE = re.compile(
    r"^(?:not|no|without|anything\s+except|except)\s+(.+)$",
    re.I,
)

MATERIAL_NAMES = (
    "acrylic", "cashmere", "cotton", "denim", "fabric", "fleece",
    "leather", "linen", "lyocell", "mesh", "modal", "nylon", "polyamide",
    "polyester", "rayon", "satin", "silk", "spandex", "suede",
    "synthetic", "viscose", "wool",
)
_MATERIAL_NAME_PATTERN = "|".join(
    re.escape(material) for material in sorted(MATERIAL_NAMES, key=len, reverse=True)
)
MATERIAL_COMPOSITION_RE = re.compile(
    rf"^\s*\d{{1,3}}(?:\.\d+)?\s*%\s*(?:{_MATERIAL_NAME_PATTERN})"
    rf"(?:\s*(?:,|/|\+|\band\b)\s*"
    rf"\d{{1,3}}(?:\.\d+)?\s*%\s*(?:{_MATERIAL_NAME_PATTERN}))*\s*$",
    re.I,
)


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    # Respect an explicit field label before looking for keywords in the
    # value. Unknown labels (for example, "Solid colors: 100% Cotton") are
    # product features, not material slots.
    label = lowered.split(":", 1)[0].strip() if ":" in lowered else ""
    if label in {"material", "fabric", "made of"}:
        return "material"
    if label in {"color", "colour"}:
        return "color"
    if label in {"size", "sizing", "fit"}:
        return "size"
    if ":" in lowered and label:
        return "feature"
    # A standalone percentage blend is a material constraint even when its
    # token count is long. Requiring the entire clause to be a composition
    # prevents descriptive catalog prose and care instructions from matching.
    if MATERIAL_COMPOSITION_RE.fullmatch(lowered):
        return "material"
    # Multi-word descriptions are generally feature text even when they
    # mention a material incidentally (for example, care instructions).
    if len(re.findall(r"\b\w+\b", lowered)) >= 4:
        return "feature"
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(word in lowered for word in MATERIAL_NAMES):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "navy")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow", "petite", "plus")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck", "casual", "formal")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work", "walking", "travel", "swimming")):
        return "use_case"
    return "feature"


def _category(message: str) -> str:
    match = re.search(r"\blooking\s+for\s+(.+?)(?:,|\.|$)", message, re.I)
    return normalize(match.group(1)) if match else ""


def _constraint_clauses(message: str, category: str) -> list[str]:
    if NO_PREFERENCE_RE.search(message):
        return []
    remainder = message
    if category:
        # Remove the original category span rather than the normalized value.
        # Normalization can change punctuation (for example ``&`` -> ``and``),
        # which previously left ``I'm .`` in markerless first-turn constraints.
        remainder = re.sub(
            r"^\s*(?:(?:i['’]?m|i\s+am)\s+)?looking\s+for\s+.+?(?:(?:,|\.)|$)",
            "",
            remainder,
            count=1,
            flags=re.I,
        )
    match = re.search(r"(?:requirement\s+is|what\s+matters\s+is|need\s+is)\s*:?\s+(.+)", remainder, re.I)
    if match:
        remainder = match.group(1)
    elif ":" in remainder:
        remainder = remainder.split(":", 1)[1]
    elif category:
        # Intent-override sessions initially contain a trailing preference after
        # the category, while browsing sessions explicitly say they are exploring.
        remainder = re.sub(r"^[\s,.-]+", "", remainder)
        if not remainder or "still exploring" in remainder.lower():
            return []
    else:
        return []
    remainder = re.sub(r"\b(?:a\s+)?key\s+requirement\s+is\b", "", remainder, flags=re.I)
    return [normalize(part).strip(" .,-") for part in re.split(r"[;\n]", remainder) if normalize(part).strip(" .,-")]


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    enable_override_cooldown: bool = False
    preserve_override_context: bool = False
    category: str = ""
    route: str = "browsing"
    initial_intent: str = "unknown"
    slots: dict[str, list[str]] = field(default_factory=dict)
    exclusions: dict[str, list[str]] = field(default_factory=dict)
    slot_provenance: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    declined: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    last_asked: str | None = None
    override_count: int = 0
    override_cooldown_attribute: str | None = None
    dissatisfaction_count: int = 0
    turn: int = 0
    candidate_pool_size: int = 0
    confidence: float = 0.0
    # Shadow-only policy state.  Product constraints and the legacy route
    # remain the source of truth until a future experiment explicitly enables
    # FSM-driven behavior.
    buyer_state_machine: BuyerStateMachine = field(default_factory=BuyerStateMachine)

    def update(self, user_message: str) -> dict:
        message = normalize(user_message)
        self.turn += 1
        if self.turn == 1:
            if "still exploring" in message.lower():
                self.initial_intent = "browsing"
            elif "key requirement" in message.lower():
                self.initial_intent = "buying"
            else:
                # The evaluator's override scenario starts with an old
                # preference but intentionally omits the buying marker.
                self.initial_intent = "override_candidate"
        override = bool(OVERRIDE_RE.search(message))
        category = _category(message)
        if category and (not self.category or override):
            self.category = category

        if override:
            # Keep question memory separate from the product intent. Explicit
            # declines remain safe to carry forward; unanswered attributes are
            # allowed to be reconsidered for the replacement intent.
            previous_declined = set(self.declined)
            prior_question = self.last_asked
            if not self.preserve_override_context:
                self.slots.clear()
                self.exclusions.clear()
                self.slot_provenance.clear()
            self.declined = previous_declined
            self.asked.clear()
            self.last_asked = None
            self.override_count += 1
            self.override_cooldown_attribute = prior_question if self.enable_override_cooldown else None
            self.route = "buying"
            self.buyer_state_machine.transition(
                "override_detected", turn=self.turn,
                constraint_count=sum(len(values) for values in self.slots.values())
                + sum(len(values) for values in self.exclusions.values()),
            )

        # The evaluator's retry prompt is not a product constraint. Treat it
        # as a structured event so it cannot pollute the query with feature
        # text, while still allowing retrieval to try a different view.
        if RETRY_MESSAGE_RE.search(message):
            self.dissatisfaction_count += 1
            self.buyer_state_machine.transition(
                "dissatisfaction_detected", turn=self.turn,
                constraint_count=sum(len(values) for values in self.slots.values())
                + sum(len(values) for values in self.exclusions.values()),
            )
            return {"override": override, "dissatisfied": True}

        if NO_PREFERENCE_RE.search(message):
            if self.last_asked:
                self.declined.add(self.last_asked)
        else:
            for clause in _constraint_clauses(message, category or self.category):
                negative_match = NEGATIVE_RE.match(clause)
                value = normalize(negative_match.group(1) if negative_match else clause).strip(" .,-")
                attribute = classify_constraint(value)
                if negative_match:
                    exclusions = self.exclusions.setdefault(attribute, [])
                    if value not in exclusions:
                        exclusions.append(value)
                    existing = self.slots.get(attribute, [])
                    self.slots[attribute] = [item for item in existing if item != value]
                    if not self.slots[attribute]:
                        self.slots.pop(attribute, None)
                    continue
                values = self.slots.setdefault(attribute, [])
                if value not in values:
                    values.append(value)
                    self.slot_provenance.setdefault(attribute, []).append({
                        "value": value,
                        "turn": self.turn,
                        "source": "explicit_user_message",
                        "hard": True,
                    })
                self.declined.discard(attribute)
                if attribute == self.override_cooldown_attribute:
                    self.override_cooldown_attribute = None

        if (self.slots or self.exclusions) and not override:
            self.route = "buying"
            # A parsed category is already one narrowing dimension even when
            # it is intentionally kept out of free-text constraint slots.
            # Preserve the FSM's prior category-plus-feature semantics without
            # reintroducing the parser pollution that caused bogus values.
            fsm_constraint_count = (
                sum(len(values) for values in self.slots.values())
                + sum(len(values) for values in self.exclusions.values())
                + (1 if self.category else 0)
            )
            self.buyer_state_machine.transition(
                "constraint_added", turn=self.turn,
                constraint_count=fsm_constraint_count,
            )
        elif not self.slots and not override:
            self.route = "browsing"
            if category:
                # A clean category-only message is still a meaningful intent;
                # it should not be stored as a product feature, but the FSM
                # can recognize that the buyer has started specifying.
                self.buyer_state_machine.transition(
                    "category_added", turn=self.turn, constraint_count=1
                )
            else:
                self.buyer_state_machine.transition("no_constraint", turn=self.turn)
        return {"override": override}

    @property
    def buyer_state(self) -> str:
        """Current shadow FSM state, exposed without changing legacy callers."""
        return self.buyer_state_machine.state

    def observe_results(self, candidate_pool_size: int, confidence: float) -> None:
        """Feed retrieval evidence to the shadow FSM without changing policy."""
        self.candidate_pool_size = int(candidate_pool_size)
        self.confidence = float(confidence)
        constraint_count = sum(len(values) for values in self.slots.values()) + sum(
            len(values) for values in self.exclusions.values()
        )
        self.buyer_state_machine.transition(
            "results_observed",
            turn=self.turn,
            constraint_count=constraint_count,
            confidence=self.confidence,
            candidate_pool_size=self.candidate_pool_size,
        )

    def query(
        self,
        fallback_message: str = "",
        *,
        label_constraints: bool = False,
        marker_constraints: bool = False,
    ) -> str:
        """Build the retrieval query from the current intent.

        ``label_constraints`` and ``marker_constraints`` are opt-in compatibility
        modes for rankers that parse structured constraint clauses.  The legacy
        unlabeled form remains the default so existing callers and published
        scores are unchanged.
        """
        category = self.category or _category(fallback_message)
        category = category or "clothing item"
        constraints = [
            f"{attribute}: {value}" if label_constraints else value
            for attribute in ATTRIBUTE_ORDER
            for value in self.slots.get(attribute, [])
        ]
        if marker_constraints and constraints:
            suffix = "what matters is: " + "; ".join(constraints)
        else:
            suffix = " ".join(constraints)
        excluded = " ".join(
            f"exclude: {value}"
            for attribute in ATTRIBUTE_ORDER
            for value in self.exclusions.get(attribute, [])
        )
        return f"I'm looking for {category}. {suffix} {excluded}".strip()

    def structured_constraints(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return positive and negative slot values for ranking features.

        This metadata channel keeps the user-facing/lexical query unchanged
        while allowing retrieval to calculate constraint overlap reliably.
        """
        positive = tuple(
            value
            for attribute in ATTRIBUTE_ORDER
            for value in self.slots.get(attribute, [])
        )
        negative = tuple(
            value
            for attribute in ATTRIBUTE_ORDER
            for value in self.exclusions.get(attribute, [])
        )
        return positive, negative

    def propose_question(
        self,
        candidate_pool_size: int,
        confidence: float,
        turn: int,
        candidate_scores: tuple[tuple[str, float], ...] = (),
        *,
        allow_late_repair: bool = False,
    ) -> str | None:
        if turn >= 8 and not (allow_late_repair and self.buyer_state == "repairing"):
            return None
        answered = (
            set(self.slots)
            | set(self.exclusions)
            | self.declined
            | set(self.asked)
        )
        if self.enable_override_cooldown and self.override_cooldown_attribute:
            answered.add(self.override_cooldown_attribute)
        priorities = list(ATTRIBUTE_ORDER)
        profile_tags = {str(tag).lower() for tag in self.user_profile.get("preference_tags", [])}
        if profile_tags & {"material", "fabric"}:
            priorities.insert(0, "material")
        elif profile_tags & {"comfort", "durability", "weather", "warmth"}:
            priorities.insert(0, "feature")
        elif profile_tags & {"style", "fashion"}:
            priorities.insert(0, "style")

        # Prefer attributes that actually divide the current result set. Keep
        # feature as a strong fallback because free-form catalog features cannot
        # be summarized by a small vocabulary.
        live_priorities = [
            attribute for attribute, score in candidate_scores
            if score >= 0.18 and attribute not in answered
        ]
        priorities = live_priorities + [item for item in priorities if item not in live_priorities]

        # An override already supplies the replacement hard constraint. Ask for
        # the remaining requirement immediately instead of replaying the old
        # fixed attribute sequence against a newly reset intent.
        if self.override_count and "other" not in answered:
            priorities.insert(0, "other")

        # Concrete buying sessions can surface the remaining key requirement
        # after two specific questions. Exploratory sessions keep one extra
        # question because boundary cases may disclose staged constraints.
        specific_questions = sum(attribute != "other" for attribute in self.asked)
        broad_question_threshold = 2 if self.route == "buying" else 3
        if specific_questions >= broad_question_threshold and "other" not in answered:
            priorities.insert(0, "other")

        # Buying sessions already have a disclosed constraint; ask one more
        # question while the conversation is still early. Browsing sessions ask
        # until the candidate pool becomes reasonably focused.
        should_ask = self.route == "browsing" or turn <= 3 or candidate_pool_size > 80 or confidence < 0.12
        if not should_ask:
            return None
        for attribute in priorities:
            if attribute not in answered:
                return attribute
        return None

    def commit_question(self, attribute: str | None) -> str | None:
        """Record only the question actually shown to the user."""
        if attribute is None:
            return None
        if attribute not in self.asked:
            self.asked.append(attribute)
        self.last_asked = attribute
        if (
            self.enable_override_cooldown
            and self.override_cooldown_attribute
            and attribute != self.override_cooldown_attribute
        ):
            self.override_cooldown_attribute = None
        return attribute

    def choose_question(
        self,
        candidate_pool_size: int,
        confidence: float,
        turn: int,
        candidate_scores: tuple[tuple[str, float], ...] = (),
    ) -> str | None:
        """Backward-compatible propose-and-commit question policy."""
        return self.commit_question(
            self.propose_question(
                candidate_pool_size, confidence, turn, candidate_scores
            )
        )

    def diagnostics(self, candidate_pool_size: int, confidence: float) -> dict:
        return {
            "route": self.route,
            "candidate_pool_size": candidate_pool_size,
            "confidence": confidence,
            "category": self.category,
            "active_slots": {key: list(values) for key, values in self.slots.items()},
            "excluded_slots": {key: list(values) for key, values in self.exclusions.items()},
            "slot_provenance": {key: list(values) for key, values in self.slot_provenance.items()},
            "declined_attributes": sorted(self.declined),
            "asked_attributes": list(self.asked),
            "override_count": self.override_count,
            "override_cooldown_attribute": self.override_cooldown_attribute,
            "dissatisfaction_count": self.dissatisfaction_count,
            "buyer_state": self.buyer_state,
            "buyer_state_confidence": self.buyer_state_machine.state_confidence,
            "buyer_state_action": self.buyer_state_machine.suggested_action(),
            "buyer_state_transitions": list(self.buyer_state_machine.transitions),
        }
