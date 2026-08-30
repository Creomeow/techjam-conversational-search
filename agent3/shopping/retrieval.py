from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from shopping.normalization import (
    COLORS,
    MATERIALS,
    SIZES,
    USE_CASES,
    CatalogFields,
    catalog_fields,
    query_parts,
    terms,
    normalize,
)
from shopping.hybrid import HybridIndex


QUESTION_VOCABULARY = {
    "material": MATERIALS,
    "color": COLORS,
    "size": SIZES,
    "use_case": USE_CASES,
    "style": {
        "casual", "formal", "classic", "modern", "vintage", "slim", "relaxed",
        "fitted", "loose", "sleeveless", "short", "long", "crew", "vneck",
    },
}


@dataclass(frozen=True)
class RetrievalHit:
    parent_asin: str
    score: float
    signals: tuple[float, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    candidate_pool_size: int
    route: str
    confidence: float
    question_scores: tuple[tuple[str, float], ...] = ()


class CatalogIndex:
    """Normalized in-memory FTS and metadata index for the frozen catalog."""

    ROUTES = (
        # title, leaf category, category path, brand, attributes, features, details, description
        (1.00, (10.0, 11.0, 6.0, 3.0, 7.0, 4.0, 2.0, 1.0)),
        (0.90, (14.0, 14.0, 8.0, 4.0, 8.0, 2.5, 1.5, 0.5)),
        (0.80, (5.0, 6.0, 4.0, 2.0, 5.0, 9.0, 4.0, 2.5)),
    )

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        hybrid_path: str | Path | None = None,
        enable_hybrid: bool = False,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._query_cache: dict[tuple[str, int, int, int, str, bool], RetrievalResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._prices: dict[str, float | None] = {}
        self._rowids: dict[str, int] = {}
        self.hybrid: HybridIndex | None = None
        if enable_hybrid:
            artifact = Path(hybrid_path) if hybrid_path is not None else (
                Path(__file__).resolve().parents[1] / "artifacts" / "hybrid_index.joblib"
            )
            if artifact.exists():
                try:
                    self.hybrid = HybridIndex.load(artifact)
                except (OSError, RuntimeError, TypeError, ValueError):
                    self.hybrid = None
        self._build()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, leaf_category, category_path, brand, attributes, "
            "features, details, description, tokenize='unicode61 remove_diacritics 2')"
        )
        cursor.execute(
            "CREATE TABLE metadata("
            "parent_asin TEXT PRIMARY KEY, price REAL, leaf_category TEXT, brand TEXT)"
        )
        fts_batch: list[tuple[str, ...]] = []
        metadata_batch: list[tuple[str, float | None, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for rowid, line in enumerate(handle, start=1):
                product = json.loads(line)
                fields = catalog_fields(product)
                self._rowids[fields.parent_asin] = rowid
                fts_batch.append(self._fts_row(fields))
                metadata_batch.append(
                    (fields.parent_asin, fields.price, fields.leaf_category, fields.brand)
                )
                self._prices[fields.parent_asin] = fields.price
                if len(fts_batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", fts_batch)
                    cursor.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?)", metadata_batch)
                    fts_batch.clear()
                    metadata_batch.clear()
        if fts_batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", fts_batch)
            cursor.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?)", metadata_batch)
        self.connection.commit()

    @staticmethod
    def _fts_row(fields: CatalogFields) -> tuple[str, ...]:
        return (
            fields.parent_asin,
            fields.title,
            fields.leaf_category,
            fields.category_path,
            fields.brand,
            fields.attributes,
            fields.features,
            fields.details,
            fields.description,
        )

    @staticmethod
    def _expression(query_terms: list[str]) -> str:
        return " OR ".join(f'"{token}"' for token in query_terms)

    def search(
        self,
        message: str,
        top_k: int = 10,
        route_count: int | None = None,
        candidate_k: int | None = None,
        cache_variant: str = "normal",
        constraint_values: tuple[str, ...] | None = None,
        excluded_values: tuple[str, ...] | None = None,
    ) -> RetrievalResult:
        route_count = max(1, min(len(self.ROUTES), route_count or len(self.ROUTES)))
        output_k = max(top_k, candidate_k or top_k)
        cache_key = (
            normalize(message), top_k, route_count, output_k,
            str(cache_variant), self.hybrid is not None,
            tuple(constraint_values or ()), tuple(excluded_values or ()),
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        all_terms, category_terms, constraint_terms = query_parts(message)
        if constraint_values is not None:
            constraint_terms = terms(
                " ".join(str(value) for value in constraint_values),
                expand=True,
                limit=32,
            )
        excluded_source = (
            " ".join(str(value) for value in excluded_values)
            if excluded_values is not None
            else " ".join(re.findall(r"\bexclude\s*:\s*([^.;]+)", message, re.I))
        )
        excluded_terms = terms(excluded_source, expand=True, limit=32)
        excluded_set = set(excluded_terms)
        if excluded_set:
            all_terms = [value for value in all_terms if value not in excluded_set and value != "exclude"]
            constraint_terms = [value for value in constraint_terms if value not in excluded_set and value != "exclude"]
        if not all_terms:
            result = RetrievalResult([], 0, f"normalized_lexical_{route_count}view", 0.0)
            self._query_cache[cache_key] = result
            return result

        expression = self._expression(all_terms)
        route_limit = max(80, output_k * 4)
        fused: dict[str, float] = {}
        documents: dict[str, tuple[str, ...]] = {}
        best_ranks: dict[str, int] = {}
        route_votes: dict[str, int] = {}
        for route_weight, weights in self.ROUTES[:route_count]:
            rows = self.connection.execute(
                "SELECT parent_asin, title, leaf_category, category_path, brand, attributes, "
                "features, details, description FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, ?, ?, ?, ?, ?, ?, ?, ?) LIMIT ?",
                (expression, *weights, route_limit),
            ).fetchall()
            for rank, row in enumerate(rows, start=1):
                parent_asin = str(row[0])
                fused[parent_asin] = fused.get(parent_asin, 0.0) + route_weight / (40.0 + rank)
                documents[parent_asin] = tuple(str(value) for value in row[1:])
                best_ranks[parent_asin] = min(best_ranks.get(parent_asin, rank), rank)
                route_votes[parent_asin] = route_votes.get(parent_asin, 0) + 1

        if self.hybrid is not None:
            for rank, (parent_asin, _) in enumerate(self.hybrid.search(message, route_limit), start=1):
                fused[parent_asin] = fused.get(parent_asin, 0.0) + 0.75 / (40.0 + rank)
                route_votes[parent_asin] = route_votes.get(parent_asin, 0) + 1
                if parent_asin not in documents:
                    document = self._document(parent_asin)
                    if document is not None:
                        documents[parent_asin] = document

        category_set = set(category_terms)
        constraint_set = set(constraint_terms)
        for parent_asin, document in documents.items():
            title, leaf, category_path, brand, attributes, features, details, description = document
            category_document = set(terms(" ".join((title, leaf, category_path)), expand=True, limit=256))
            constraint_document = set(
                terms(" ".join((title, brand, attributes, features, details, description)), expand=True, limit=512)
            )
            if category_set:
                fused[parent_asin] += 0.025 * len(category_set & category_document) / len(category_set)
            if constraint_set:
                fused[parent_asin] += 0.080 * len(constraint_set & constraint_document) / len(constraint_set)
            if excluded_set:
                fused[parent_asin] -= 0.040 * len(excluded_set & constraint_document) / len(excluded_set)

        all_set = set(all_terms)
        category_phrase, _ = self._query_phrases(message)
        constraint_phrase = self._query_phrases(message)[1]
        constraint_phrases = [normalize(part).lower() for part in re.split(r"[;\n]", constraint_phrase) if normalize(part)]
        price_mode, requested_price = self._price_request(message)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        hits: list[RetrievalHit] = []
        for parent_asin, score in ranked[:output_k]:
            title, leaf, category_path, brand, attributes, features, details, description = documents[parent_asin]
            title_set = set(terms(title, expand=True, limit=256))
            attribute_set = set(terms(attributes, expand=True, limit=256))
            feature_set = set(terms(features, expand=True, limit=512))
            brand_set = set(terms(brand, expand=True, limit=128))
            category_document = set(terms(" ".join((title, leaf, category_path)), expand=True, limit=256))
            constraint_document = set(
                terms(" ".join((title, brand, attributes, features, details, description)), expand=True, limit=512)
            )
            denominator = max(len(all_set), 1)
            matched_constraints = len(constraint_set & constraint_document)
            constraint_denominator = max(len(constraint_set), 1)
            searchable_category = normalize(" ".join((title, leaf, category_path))).lower()
            exact_category = float(bool(category_phrase and category_phrase in searchable_category))
            normalized_title = normalize(title).lower()
            title_phrase_match = float(bool(category_phrase and category_phrase in normalized_title))
            brand_exact_match = float(bool(brand and normalize(brand).lower() in constraint_phrase))
            composition_requested = bool(re.search(r"\d{1,3}(?:\.\d+)?\s*%", constraint_phrase))
            material_composition_match = float(
                composition_requested
                and bool(re.search(r"\d{1,3}(?:\.\d+)?\s*%", normalize(attributes + " " + features)))
            )
            feature_phrase_match = float(
                bool(constraint_phrases)
                and any(
                    len(terms(phrase)) >= 2
                    and normalize(phrase).lower() in normalize(features).lower()
                    for phrase in constraint_phrases
                )
            )
            generic_match_penalty = float(
                bool(all_set)
                and len(all_set & (title_set | category_document)) / len(all_set) < 0.25
                and not exact_category
            )
            price_fit = self._price_fit(self._prices.get(parent_asin), price_mode, requested_price)
            excluded_violation = len(excluded_set & constraint_document) / max(len(excluded_set), 1)
            signals = (
                float(score),
                1.0 / (1.0 + best_ranks.get(parent_asin, output_k)),
                len(category_set & category_document) / max(len(category_set), 1),
                len(constraint_set & constraint_document) / max(len(constraint_set), 1),
                len(all_set & title_set) / denominator,
                len(all_set & attribute_set) / denominator,
                len(all_set & feature_set) / denominator,
                len(all_set & brand_set) / denominator,
                len(all_set & (category_document | constraint_document)) / denominator,
                float(bool(constraint_set) and matched_constraints == len(constraint_set)),
                (len(constraint_set) - matched_constraints) / constraint_denominator if constraint_set else 0.0,
                exact_category,
                price_fit,
                route_votes.get(parent_asin, 1) / (route_count + int(self.hybrid is not None)),
                excluded_violation,
                title_phrase_match,
                brand_exact_match,
                material_composition_match,
                feature_phrase_match,
                generic_match_penalty,
            )
            hits.append(RetrievalHit(parent_asin, round(score, 8), tuple(round(value, 8) for value in signals)))
        hits = hits[:top_k] if candidate_k is None else hits
        if not hits:
            confidence = 0.0
        elif len(hits) == 1:
            confidence = 1.0
        else:
            best = hits[0].score
            confidence = max(0.0, min(1.0, (best - hits[1].score) / max(abs(best), 1e-9)))
        result = RetrievalResult(
            hits=hits,
            candidate_pool_size=len(fused),
            route=f"normalized_lexical_{route_count}view",
            confidence=round(confidence, 6),
            question_scores=self._question_scores(hits, documents),
        )
        self._query_cache[cache_key] = result
        return result

    def cache_stats(self) -> dict[str, int]:
        """Return lightweight cache telemetry for development diagnostics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": len(self._query_cache),
        }

    def clear_cache(self) -> None:
        """Invalidate retrieval results without rebuilding the catalog index."""
        self._query_cache.clear()

    def _document(self, parent_asin: str) -> tuple[str, ...] | None:
        rowid = self._rowids.get(parent_asin)
        if rowid is None:
            return None
        row = self.connection.execute(
            "SELECT title, leaf_category, category_path, brand, attributes, features, details, description "
            "FROM products WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        return tuple(str(value) for value in row) if row else None

    @staticmethod
    def _query_phrases(message: str) -> tuple[str, str]:
        category = ""
        match = re.search(r"\blooking\s+for\s+(.+?)(?:,|\.|$)", message, re.I)
        if match:
            category = normalize(match.group(1)).lower()
        constraint = normalize(message.split(":", 1)[1]).lower() if ":" in message else ""
        return category, constraint

    @staticmethod
    def _price_request(message: str) -> tuple[str | None, float | None]:
        lowered = message.lower().replace(",", "")
        match = re.search(r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)", lowered)
        markers = ("budget", "price", "under", "below", "over", "above", "$")
        if not match or not any(marker in lowered for marker in markers):
            return None, None
        value = float(match.group(1))
        if any(marker in lowered for marker in ("under", "below", "at most", "up to")):
            return "maximum", value
        if any(marker in lowered for marker in ("over", "above", "at least")):
            return "minimum", value
        return "around", value

    @staticmethod
    def _price_fit(price: float | None, mode: str | None, requested: float | None) -> float:
        if mode is None or requested is None:
            return 0.0
        if price is None:
            return 0.25
        if mode == "maximum":
            return 1.0 if price <= requested else max(0.0, 1.0 - (price - requested) / max(requested, 1.0))
        if mode == "minimum":
            return 1.0 if price >= requested else max(0.0, price / max(requested, 1.0))
        distance = abs(price - requested) / max(requested, 1.0)
        return max(0.0, 1.0 - distance)

    @staticmethod
    def _question_scores(
        hits: list[RetrievalHit], documents: dict[str, tuple[str, ...]]
    ) -> tuple[tuple[str, float], ...]:
        """Estimate which attribute most evenly partitions the live candidates."""
        if len(hits) < 2:
            return ()
        candidate_tokens = {
            hit.parent_asin: set(terms(" ".join(documents[hit.parent_asin]), limit=768))
            for hit in hits
            if hit.parent_asin in documents
        }
        scores: list[tuple[str, float]] = []
        total = len(candidate_tokens)
        if total < 2:
            return ()
        priors = {"material": 1.0, "color": 0.75, "size": 0.65, "use_case": 0.65, "style": 0.55}
        for attribute, vocabulary in QUESTION_VOCABULARY.items():
            matching = sum(bool(tokens & vocabulary) for tokens in candidate_tokens.values())
            fraction = matching / total
            # Gini impurity peaks when the answer splits the pool in half.
            score = 4.0 * fraction * (1.0 - fraction) * priors[attribute]
            if matching and matching < total:
                scores.append((attribute, round(score, 6)))
        # Product features are common and useful even when they cannot be
        # represented by a small fixed vocabulary.
        scores.append(("feature", 0.58))
        return tuple(sorted(scores, key=lambda item: (-item[1], item[0])))
