from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "fabric", "linen", "denim", "fleece", "satin", "suede", "cashmere", "acrylic",
    "rubber", "canvas", "mesh",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver", "tan", "multicolor",
}
SIZES = {
    "size", "sizing", "width", "wide", "narrow", "petite", "small", "medium", "large",
    "xl", "xxl",
}
USE_CASES = {
    "running", "walking", "hiking", "training", "gym", "workout", "winter", "outdoor",
    "work", "travel", "wedding", "party", "casual", "formal", "sport", "sports",
    "swimming", "sleep", "costume",
}
ATTRIBUTE_TERMS = MATERIALS | COLORS | SIZES | USE_CASES

STOPWORDS = {
    "a", "about", "additional", "an", "and", "are", "as", "at", "be", "but",
    "by", "closest", "do", "for", "from", "have", "here", "i", "in", "is", "it",
    "key", "looking", "matches", "me", "my", "need", "not", "of", "on", "options",
    "or", "please", "quite", "requirement", "right", "some", "that", "the", "these",
    "this", "those", "to", "want", "what", "with", "would", "yet", "you",
}

SYNONYMS = {
    "tee": ("tshirt", "shirt"),
    "tees": ("tshirt", "shirts"),
    "tshirts": ("tshirt", "shirts"),
    "sneaker": ("shoe",),
    "sneakers": ("shoes",),
    "trainer": ("shoe",),
    "trainers": ("shoes",),
    "trouser": ("pants",),
    "trousers": ("pants",),
    "grey": ("gray",),
}


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def normalize(value: object) -> str:
    raw = unicodedata.normalize("NFKC", text(value)).replace("&", " and ")
    return SPACE_RE.sub(" ", raw).strip()


def terms(value: object, *, expand: bool = False, limit: int = 64) -> list[str]:
    result: list[str] = []
    for token in TOKEN_RE.findall(normalize(value).lower()):
        if len(token) <= 1 or token in STOPWORDS:
            continue
        result.append(token)
        if expand:
            result.extend(SYNONYMS.get(token, ()))
    return list(dict.fromkeys(result))[:limit]


@dataclass(frozen=True)
class CatalogFields:
    parent_asin: str
    title: str
    leaf_category: str
    category_path: str
    brand: str
    attributes: str
    features: str
    details: str
    description: str
    price: float | None


def catalog_fields(product: dict) -> CatalogFields:
    categories: list[str] = []
    for value in product.get("categories") or []:
        cleaned = normalize(value)
        if cleaned:
            categories.append(cleaned)
    title = normalize(product.get("title"))
    # Large free-text fields do not need Unicode/whitespace canonicalization for
    # SQLite's tokenizer. Avoiding it substantially reduces index startup time.
    features = text(product.get("features")).strip()
    details = text(product.get("details")).strip()
    description = text(product.get("description")).strip()
    brand = normalize(product.get("store"))
    # One token pass is much cheaper than scanning the full product repeatedly
    # with separate regular expressions.
    attribute_corpus = " ".join(
        (title, features, details, description, " ".join(categories))
    ).lower()
    attribute_parts = [
        " ".join(dict.fromkeys(
            token for token in TOKEN_RE.findall(attribute_corpus) if token in ATTRIBUTE_TERMS
        ))
    ]
    details_value = product.get("details")
    if isinstance(details_value, dict):
        for key, value in details_value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("department", "material", "fabric", "color", "size")):
                attribute_parts.append(normalize(value))
    raw_price = product.get("price")
    try:
        price = float(raw_price) if raw_price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    return CatalogFields(
        parent_asin=str(product["parent_asin"]),
        title=title,
        leaf_category=" ".join(categories[-3:]),
        category_path=" ".join(categories),
        brand=brand,
        attributes=" ".join(part for part in attribute_parts if part),
        features=features,
        details=details,
        description=description,
        price=price,
    )


def query_parts(message: str) -> tuple[list[str], list[str], list[str]]:
    """Split simulator-style messages while remaining safe for ordinary text."""
    category_text = ""
    category_match = re.search(r"\blooking\s+for\s+(.+?)(?:,|\.|$)", message, re.IGNORECASE)
    if category_match:
        category_text = category_match.group(1)
    constraint_text = ""
    if ":" in message:
        constraint_text = message.split(":", 1)[1]
    elif "what matters is" in message.lower():
        constraint_text = message.lower().split("what matters is", 1)[1]

    all_terms = terms(message, expand=True, limit=48)
    category_terms = terms(category_text, expand=True, limit=16)
    constraint_terms = terms(constraint_text, expand=True, limit=32)
    return all_terms, category_terms, constraint_terms
