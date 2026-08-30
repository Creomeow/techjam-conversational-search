"""Optional artifact-backed semantic-style reranker.

The runtime remains fully functional without this optional artifact.  The
training script currently uses a word/character TF-IDF pair classifier from
scikit-learn; the artifact format is deliberately isolated so a neural
cross-encoder backend can replace it later without changing Agent responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from pathlib import Path
from typing import Any

from shopping.retrieval import RetrievalHit


@dataclass
class SemanticReranker:
    vectorizer: Any
    classifier: Any
    version: str = "tfidf_semantic_reranker_v1"
    blend_weight: float = 1.0
    vocabulary: dict[str, int] | None = None
    idf: list[float] | None = None
    weights: list[float] | None = None
    bias: float = 0.0
    score_cache: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticReranker | None":
        model_path = Path(path)
        if not model_path.exists():
            return None
        try:
            if model_path.suffix.lower() == ".json":
                payload = json.loads(model_path.read_text(encoding="utf-8"))
                vocabulary = {str(key): int(value) for key, value in payload["vocabulary"].items()}
                idf = [float(value) for value in payload["idf"]]
                weights = [float(value) for value in payload["weights"]]
                if len(idf) != len(vocabulary) or len(weights) != len(vocabulary):
                    raise ValueError("semantic JSON dimensions do not match")
                return cls(
                    vectorizer=None,
                    classifier=None,
                    version=str(payload.get("version", "tfidf_semantic_reranker_v1")),
                    blend_weight=max(0.0, float(payload.get("blend_weight", 1.0))),
                    vocabulary=vocabulary,
                    idf=idf,
                    weights=weights,
                    bias=float(payload.get("bias", 0.0)),
                )
            import joblib

            payload = joblib.load(model_path)
            if not isinstance(payload, dict):
                raise ValueError("semantic artifact must be a mapping")
            vectorizer = payload["vectorizer"]
            classifier = payload["classifier"]
            if not hasattr(vectorizer, "transform") or not hasattr(classifier, "predict_proba"):
                raise ValueError("semantic artifact has incompatible components")
            return cls(
                vectorizer=vectorizer,
                classifier=classifier,
                version=str(payload.get("version", "tfidf_semantic_reranker_v1")),
                blend_weight=max(0.0, float(payload.get("blend_weight", 1.0))),
            )
        except (OSError, ImportError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def product_text(document: tuple[str, ...]) -> str:
        return " ".join(str(value) for value in document if value)

    def score_texts(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        missing = list(dict.fromkeys(text for text in texts if text not in self.score_cache))
        if missing:
            computed = self._score_texts_uncached(missing)
            self.score_cache.update(zip(missing, computed))
        return [self.score_cache[text] for text in texts]

    def _score_texts_uncached(self, texts: list[str]) -> list[float]:
        if self.vocabulary is not None and self.idf is not None and self.weights is not None:
            token_pattern = re.compile(r"(?u)\b\w\w+\b")
            scores: list[float] = []
            for text in texts:
                tokens = token_pattern.findall(text.lower())
                counts: dict[int, int] = {}
                for token in tokens:
                    index = self.vocabulary.get(token)
                    if index is not None:
                        counts[index] = counts.get(index, 0) + 1
                for left, right in zip(tokens, tokens[1:]):
                    index = self.vocabulary.get(f"{left} {right}")
                    if index is not None:
                        counts[index] = counts.get(index, 0) + 1
                norm = math.sqrt(sum((1.0 + math.log(count)) ** 2 * self.idf[index] ** 2 for index, count in counts.items()))
                margin = self.bias
                if norm:
                    margin += sum(
                        self.weights[index] * (1.0 + math.log(count)) * self.idf[index] / norm
                        for index, count in counts.items()
                    )
                scores.append(1.0 / (1.0 + math.exp(max(-40.0, min(40.0, -margin)))))
            return scores
        matrix = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(matrix)
        classes = list(getattr(self.classifier, "classes_", ()))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        return [float(row[positive_index]) for row in probabilities]

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        index: Any,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        texts: list[str] = []
        valid_hits: list[RetrievalHit] = []
        for hit in hits:
            document = index._document(hit.parent_asin)
            if document is None:
                continue
            texts.append(f"query: {query} product: {self.product_text(document)}")
            valid_hits.append(hit)
        if not valid_hits:
            return hits[:top_k]
        semantic_scores = self.score_texts(texts)
        ranked = sorted(
            zip(valid_hits, semantic_scores),
            key=lambda item: (
                -(item[0].score + self.blend_weight * item[1]),
                item[0].parent_asin,
            ),
        )
        return [
            RetrievalHit(hit.parent_asin, round(hit.score + self.blend_weight * semantic, 8), hit.signals)
            for hit, semantic in ranked[:top_k]
        ]
