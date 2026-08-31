from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from shopping.retrieval import RetrievalHit


LEGACY_SIGNAL_NAMES = (
    "retrieval_score", "rank_prior", "category_overlap", "constraint_overlap",
    "title_overlap", "attribute_overlap", "feature_overlap", "brand_overlap",
)

SIGNAL_NAMES = LEGACY_SIGNAL_NAMES + (
    "query_coverage", "all_constraints_match", "missing_constraint_ratio",
    "exact_category_phrase", "price_fit", "route_consensus",
    "excluded_constraint_violation",
    "title_phrase_match",
    "brand_exact_match",
    "material_composition_match",
    "feature_phrase_match",
    "generic_match_penalty",
)


# Explicit, untrained adjustments for the legacy artifact. The title offset
# partially neutralizes its negative title weight; the remaining values cover
# signals absent from that artifact. Full-schema models bypass this wrapper.
PLACEHOLDER_SIGNAL_WEIGHTS = (
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    8.0, 4.0, -4.0, 8.0, 8.0, 2.0, -20.0,
)

FIELD_SIGNAL_PROFILES = {
    "off": (0.0, 0.0, 0.0, 0.0, 0.0),
    # Deliberately small until a full-schema model is trained. These signals
    # should refine close lexical ties, not overpower the learned artifact.
    "conservative": (0.20, 0.10, 0.20, 0.20, -0.10),
    # Targeted exact-match profile: only the explicit title/brand/material/
    # feature signals receive additional weight.  It remains opt-in until it
    # clears the full promotion floor.
    "targeted_exact": (1.50, 1.00, 1.25, 1.50, -0.30),
    "targeted_exact_light": (1.00, 0.50, 0.75, 1.00, -0.20),
    "targeted_exact_strong": (2.00, 1.50, 1.75, 2.00, -0.50),
    "targeted_exact_plus": (1.60, 1.10, 1.35, 1.60, -0.30),
}


@dataclass(frozen=True)
class LinearRanker:
    weights: tuple[float, ...]
    bias: float = 0.0
    version: str = "pairwise_sgd_v1"
    candidate_k: int = 40
    legacy_schema: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "LinearRanker | None":
        model_path = Path(path)
        if not model_path.exists():
            return None
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        payload_signals = tuple(payload.get("signal_names", ()))
        if payload_signals not in (SIGNAL_NAMES, LEGACY_SIGNAL_NAMES):
            raise ValueError(f"ranker signal schema mismatch in {model_path}")
        weights = tuple(float(value) for value in payload["weights"])
        if len(weights) != len(payload_signals):
            raise ValueError("ranker weight count does not match signal schema")
        if payload_signals == LEGACY_SIGNAL_NAMES:
            weights += (0.0,) * (len(SIGNAL_NAMES) - len(LEGACY_SIGNAL_NAMES))
        training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
        selected = training.get("selected_config") if isinstance(training.get("selected_config"), dict) else {}
        candidate_k = payload.get("candidate_k", selected.get("candidate_k", training.get("candidate_k", 40)))
        return cls(
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            version=str(payload.get("version", "unknown")),
            candidate_k=max(10, int(candidate_k)),
            legacy_schema=payload_signals == LEGACY_SIGNAL_NAMES,
        )

    def score(self, hit: RetrievalHit) -> float:
        return self.bias + sum(weight * value for weight, value in zip(self.weights, hit.signals))

    def rerank(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        ranked = sorted(hits, key=lambda hit: (-self.score(hit), hit.parent_asin))
        return [
            RetrievalHit(hit.parent_asin, round(self.score(hit), 8), hit.signals)
            for hit in ranked[:top_k]
        ]

    def payload(self) -> dict:
        return {
            "version": self.version,
            "signal_names": list(SIGNAL_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "candidate_k": self.candidate_k,
        }


@dataclass(frozen=True)
class PlaceholderSignalRanker:
    """Add documented heuristic weights only to a legacy trained ranker."""

    base: LinearRanker
    weights: tuple[float, ...] = PLACEHOLDER_SIGNAL_WEIGHTS
    field_weights: tuple[float, ...] = FIELD_SIGNAL_PROFILES["off"]
    version: str = "placeholder_signals_v2"

    @property
    def candidate_k(self) -> int:
        return self.base.candidate_k

    def score(self, hit: RetrievalHit) -> float:
        return self.base.score(hit) + sum(
            weight * value for weight, value in zip(self.weights, hit.signals)
        ) + sum(
            weight * value
            for weight, value in zip(self.field_weights, hit.signals[len(self.weights):])
        )

    def rerank(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        ranked = sorted(hits, key=lambda hit: (-self.score(hit), hit.parent_asin))
        return [
            RetrievalHit(hit.parent_asin, round(self.score(hit), 8), hit.signals)
            for hit in ranked[:top_k]
        ]


@dataclass(frozen=True)
class FieldSignalRanker:
    """Apply an opt-in field profile to an already full-schema ranker."""

    base: LinearRanker
    field_weights: tuple[float, ...]
    version: str = "field_signal_profile"

    @property
    def candidate_k(self) -> int:
        return self.base.candidate_k

    def score(self, hit: RetrievalHit) -> float:
        return self.base.score(hit) + sum(
            weight * value
            for weight, value in zip(self.field_weights, hit.signals[-len(self.field_weights):])
        )

    def rerank(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        ranked = sorted(hits, key=lambda hit: (-self.score(hit), hit.parent_asin))
        return [
            RetrievalHit(hit.parent_asin, round(self.score(hit), 8), hit.signals)
            for hit in ranked[:top_k]
        ]
