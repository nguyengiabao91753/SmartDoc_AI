from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, List, Sequence, TypeVar

T = TypeVar("T")

JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")
JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = float(low)
    return max(low, min(high, numeric_value))


def normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return " ".join(normalized.strip().lower().split())


def tokenize_text(text: str | None, *, min_len: int = 3) -> List[str]:
    tokens = TOKEN_PATTERN.findall(normalize_text(text))
    return [token for token in tokens if len(token) >= min_len]


def unique_preserve_order(items: Sequence[T]) -> List[T]:
    seen: set[str] = set()
    result: List[T] = []
    for item in items:
        key = normalize_text(str(item))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def parse_json_array(raw: str, fallback: Iterable[Any] | None = None) -> List[Any]:
    fallback_list = list(fallback or [])
    text = (raw or "").strip()
    if not text:
        return fallback_list

    for candidate in (text, _extract_by_pattern(JSON_ARRAY_PATTERN, text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return parsed
    return fallback_list


def parse_json_object(raw: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback_obj = dict(fallback or {})
    text = (raw or "").strip()
    if not text:
        return fallback_obj

    for candidate in (text, _extract_by_pattern(JSON_OBJECT_PATTERN, text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return fallback_obj


def _extract_by_pattern(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(0)
