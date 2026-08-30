"""Optional sentence-transformers cross-encoder reranker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shopping.retrieval import RetrievalHit


MINIMUM_PROMOTION_SCORE = 0.891234


@dataclass
class NeuralReranker:
    model: Any
    version: str = "cross_encoder_minilm_v1"
    blend_weight: float = 1.0
    blend_mode: str = "additive"
    activation_mode: str = "all"
    selective_margin: float = 0.0
    selective_states: tuple[str, ...] = ("narrowing", "repairing")
    score_cache: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_unpromoted: bool = False,
    ) -> "NeuralReranker | None":
        model_path = Path(path)
        if not model_path.is_dir():
            return None
        try:
            from sentence_transformers import CrossEncoder

            metadata_path = model_path / "reranker_metadata.json"
            metadata = {}
            if metadata_path.exists():
                import json

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            # A neural directory can be left behind by an interrupted or
            # older training process.  Never activate it without explicit
            # full-set evidence for the current promotion floor.
            promotion_floor = float(metadata.get("promotion_floor", 0.0))
            full_metrics = metadata.get("full_metrics")
            full_score = float(full_metrics.get("recommended_technical_score", 0.0)) if isinstance(full_metrics, dict) else 0.0
            if not allow_unpromoted and (
                promotion_floor < MINIMUM_PROMOTION_SCORE
                or full_score < MINIMUM_PROMOTION_SCORE
            ):
                return None
            # The promoted artifact is self-contained.  Keep evaluation and
            # serving deterministic/offline instead of retrying the model hub
            # when optional metadata is absent or the machine has no network.
            model = CrossEncoder(str(model_path), num_labels=1, local_files_only=True)
            blend_mode = str(metadata.get("blend_mode", "additive"))
            if blend_mode not in {"additive", "normalized"}:
                return None
            activation_mode = str(metadata.get("activation_mode", "all"))
            if activation_mode not in {"all", "selective"}:
                return None
            return cls(
                model=model,
                version=str(metadata.get("version", "cross_encoder_minilm_v1")),
                blend_weight=max(0.0, float(metadata.get("blend_weight", 1.0))),
                blend_mode=blend_mode,
                activation_mode=activation_mode,
                selective_margin=max(0.0, float(metadata.get("selective_margin", 0.0))),
                selective_states=tuple(
                    str(value) for value in metadata.get(
                        "selective_states", ["narrowing", "repairing"]
                    )
                ),
            )
        except (ImportError, OSError, TypeError, ValueError):
            return None

    @staticmethod
    def product_text(document: tuple[str, ...]) -> str:
        return " ".join(str(value) for value in document if value)

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        index: Any,
        top_k: int,
    ) -> list[RetrievalHit]:
        pairs: list[list[str]] = []
        valid_hits: list[RetrievalHit] = []
        for hit in hits:
            document = index._document(hit.parent_asin)
            if document is None:
                continue
            pairs.append([query, self.product_text(document)])
            valid_hits.append(hit)
        if not valid_hits:
            return hits[:top_k]
        keys = [(pair[0], pair[1]) for pair in pairs]
        missing_keys = list(dict.fromkeys(key for key in keys if key not in self.score_cache))
        if missing_keys:
            missing_pairs = [[query_text, product_text] for query_text, product_text in missing_keys]
            missing_scores = self.model.predict(missing_pairs, show_progress_bar=False)
            self.score_cache.update(
                (key, float(score)) for key, score in zip(missing_keys, missing_scores)
            )
        scores = [self.score_cache[key] for key in keys]
        neural_scores = [float(value) for value in scores]
        if self.blend_mode == "normalized":
            base_scores = self._minmax([hit.score for hit in valid_hits])
            normalized_neural = self._minmax(neural_scores)
            weight = min(1.0, self.blend_weight)
            fused_scores = [
                (1.0 - weight) * base + weight * neural
                for base, neural in zip(base_scores, normalized_neural)
            ]
        else:
            fused_scores = [
                hit.score + self.blend_weight * score
                for hit, score in zip(valid_hits, neural_scores)
            ]
        ranked = sorted(
            zip(valid_hits, fused_scores),
            key=lambda item: (-item[1], item[0].parent_asin),
        )
        return [
            RetrievalHit(hit.parent_asin, round(score, 8), hit.signals)
            for hit, score in ranked[:top_k]
        ]

    @staticmethod
    def _minmax(values: list[float]) -> list[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if high - low <= 1e-12:
            return [0.5] * len(values)
        return [(value - low) / (high - low) for value in values]

    def should_activate(self, buyer_state: str, hits: list[RetrievalHit]) -> bool:
        """Return whether a promoted selective model should affect this turn."""
        if self.activation_mode == "all":
            return True
        if buyer_state not in self.selective_states:
            return False
        if len(hits) < 2:
            return False
        return hits[0].score - hits[1].score <= self.selective_margin
