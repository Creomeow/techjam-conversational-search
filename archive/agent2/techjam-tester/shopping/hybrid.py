from __future__ import annotations

from pathlib import Path


class HybridIndex:
    """Optional sparse lexical-semantic index loaded from a trusted local artifact."""

    def __init__(self, vectorizer: object, matrix: object, identifiers: list[str]) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.identifiers = identifiers

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("hybrid retrieval requires scikit-learn/joblib") from exc
        payload = joblib.load(Path(path))
        if payload.get("version") != "hybrid_tfidf_v1":
            raise ValueError("unsupported hybrid index version")
        identifiers = [str(value) for value in payload["identifiers"]]
        matrix = payload["matrix"]
        if matrix.shape[0] != len(identifiers):
            raise ValueError("hybrid matrix and identifier count do not match")
        return cls(payload["vectorizer"], matrix, identifiers)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        import numpy as np

        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        limit = max(1, min(int(limit), len(scores)))
        indices = np.argpartition(scores, -limit)[-limit:]
        ranked = sorted(indices, key=lambda index: (-float(scores[index]), self.identifiers[index]))
        return [
            (self.identifiers[index], float(scores[index]))
            for index in ranked
            if scores[index] > 0.0
        ]
