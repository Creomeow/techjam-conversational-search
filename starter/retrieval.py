from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:\$|usd\s*)(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "have", "i", "in", "is", "it", "looking", "me", "my", "of", "on",
    "or", "please", "some", "still", "that", "the", "these", "this", "those", "to",
    "want", "what", "with", "would", "you", "your", "exploring", "requirement",
    "requirements", "matters", "prefer", "preference", "options", "right", "yet",
}

# The clarification module consumes this exact shape: (id, product fields, score).
CandidatePool = list[tuple[str, dict, float]]


def terms(text: str, limit: int = 48) -> list[str]:
    """Return stable, FTS-safe terms while preserving first occurrence order."""
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _fts_token(token: str) -> str:
    # TOKEN_RE has already removed FTS operators and punctuation.
    return f'"{token}"'


def _expression(query_terms: Sequence[str], operator: str) -> str:
    return f" {operator} ".join(_fts_token(item) for item in query_terms)


class RetrievalEngine:
    """Offline tiered FTS candidate generation followed by structured reranking."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]),
                    _flatten(product.get("title")),
                    _flatten(product.get("categories")),
                    _flatten(product.get("features")),
                    _flatten(product.get("details")),
                    _flatten(product.get("store")),
                    _flatten(product.get("description")),
                    "" if product.get("price") is None else str(product.get("price")),
                    "" if product.get("average_rating") is None else str(product.get("average_rating")),
                    "" if product.get("rating_number") is None else str(product.get("rating_number")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _query(self, expression: str, limit: int) -> list[tuple]:
        if not expression:
            return []
        try:
            return self.connection.execute(
                "SELECT parent_asin, title, categories, features, details, store, description, "
                "price, average_rating, rating_number, "
                "bm25(products, 0.0, 7.0, 5.0, 3.0, 3.0, 1.2, 1.0, 0.0, 0.0, 0.0) "
                "FROM products WHERE products MATCH ? ORDER BY 11 LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _product(row: tuple) -> dict:
        def number(value: object) -> float | None:
            try:
                return float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        return {
            "parent_asin": str(row[0]),
            "title": str(row[1] or ""),
            "categories": str(row[2] or ""),
            "features": str(row[3] or ""),
            "details": str(row[4] or ""),
            "store": str(row[5] or ""),
            "description": str(row[6] or ""),
            "price": number(row[7]),
            "average_rating": number(row[8]),
            "rating_number": number(row[9]),
        }

    def search(
        self,
        current_message: str,
        constraints: Sequence[str],
        *,
        soft_preferences: Sequence[str] = (),
        top_k: int = 10,
        candidate_limit: int = 240,
        previous_ids: Iterable[str] = (),
        intent_changed: bool = False,
    ) -> CandidatePool:
        """Return ranked candidates from strict, per-constraint, and broad routes."""
        constraint_pairs = [(value, terms(value, 16)) for value in constraints if value.strip()]
        constraint_pairs = [(value, group) for value, group in constraint_pairs if group]
        constraint_terms = [group for _, group in constraint_pairs]
        soft_pairs = [(value, terms(value, 12)) for value in soft_preferences if value.strip()]
        soft_pairs = [(value, group) for value, group in soft_pairs if group]
        soft_term_groups = [group for _, group in soft_pairs]
        message_terms = terms(current_message, 24)
        hard_query_terms = list(dict.fromkeys(
            [item for group in constraint_terms for item in group]
            + message_terms
        ))
        all_terms = list(dict.fromkeys(
            hard_query_terms
            + [item for group in soft_term_groups for item in group]
        ))
        if not all_terms:
            return []

        routes: list[tuple[float, str]] = []
        # Strict route is deliberately capped: metadata sentences can be very long.
        strict = hard_query_terms[:12]
        if strict:
            routes.append((1.30, _expression(strict, "AND")))
        for group in constraint_terms[-4:]:
            if group:
                routes.append((1.15, _expression(group[:10], "AND")))
        if message_terms:
            routes.append((0.85, _expression(message_terms[:12], "AND")))
        routes.append((0.65, _expression(all_terms[:32], "OR")))

        products: dict[str, dict] = {}
        retrieval_score: defaultdict[str, float] = defaultdict(float)
        seen_expressions: set[str] = set()
        for route_weight, expression in routes:
            if expression in seen_expressions:
                continue
            seen_expressions.add(expression)
            for rank, row in enumerate(self._query(expression, candidate_limit), start=1):
                parent_asin = str(row[0])
                products.setdefault(parent_asin, self._product(row))
                # Reciprocal-rank fusion avoids comparing BM25 scales across routes.
                retrieval_score[parent_asin] += route_weight / (20.0 + rank)

        previous = set(previous_ids)
        current_tokens = set(message_terms)
        constraint_token_sets = [set(group) for group in constraint_terms if group]
        soft_token_sets = [set(group) for group in soft_term_groups if group]
        requested_budget = self._budget([current_message, *constraints])
        scored: list[tuple[str, dict, float]] = []
        for parent_asin, product in products.items():
            title_tokens = set(terms(product["title"], 80))
            category_tokens = set(terms(product["categories"], 40))
            feature_tokens = set(terms(" ".join([
                product["features"], product["details"], product["description"], product["store"]
            ]), 400))
            searchable = " ".join([
                product["title"], product["categories"], product["features"],
                product["details"], product["store"], product["description"],
            ]).lower()

            score = retrieval_score[parent_asin] * 45.0
            score += 2.8 * len(current_tokens & title_tokens) / max(1, len(current_tokens))
            score += 1.8 * len(current_tokens & category_tokens) / max(1, len(current_tokens))
            for (raw, _), token_set in zip(constraint_pairs, constraint_token_sets):
                coverage = len(token_set & (title_tokens | category_tokens | feature_tokens)) / max(1, len(token_set))
                score += 4.0 * coverage
                normalized = " ".join(terms(raw, 24))
                if normalized and normalized in " ".join(terms(searchable, 500)):
                    score += 1.5
            for token_set in soft_token_sets:
                coverage = len(token_set & (title_tokens | category_tokens | feature_tokens)) / max(1, len(token_set))
                score += 0.40 * coverage
            if requested_budget is not None and product["price"] is not None:
                delta = abs(float(product["price"]) - requested_budget) / max(10.0, requested_budget)
                score += 2.0 * math.exp(-2.5 * delta)
            if not intent_changed and parent_asin in previous:
                score += 0.12
            rating = product.get("average_rating") or 0.0
            rating_count = product.get("rating_number") or 0.0
            score += min(0.10, max(0.0, rating - 3.5) * 0.025)
            score += min(0.08, math.log1p(max(0.0, rating_count)) * 0.008)
            scored.append((parent_asin, product, score))

        scored.sort(key=lambda item: (-item[2], item[0]))
        return scored[:max(top_k, candidate_limit)]

    @staticmethod
    def _budget(values: Sequence[str]) -> float | None:
        for value in reversed(values):
            match = MONEY_RE.search(value)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return None
