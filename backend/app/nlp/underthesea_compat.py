"""Compatibility layer for underthesea.

The project originally depends on `underthesea` for:
- sent_tokenize (Vietnamese sentence splitting)
- word_tokenize (Vietnamese word tokenization)
- stopwords loading via `underthesea.__file__`

In some environments `underthesea` may be missing. This module ensures the
application can still import and run by providing safe fallbacks.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Set, List

from app.core.logger import LOG


try:
    import underthesea as _underthesea  # type: ignore

    # Re-export
    sent_tokenize = _underthesea.sent_tokenize  # type: ignore[attr-defined]
    word_tokenize = _underthesea.word_tokenize  # type: ignore[attr-defined]

    def load_stopwords() -> Set[str]:
        """Load stopwords from underthesea package datasets (if available)."""

        try:
            stopwords_path = (
                Path(_underthesea.__file__).resolve().parent
                / "datasets"
                / "stopwords"
                / "stopwords.txt"
            )
            if not stopwords_path.exists():
                return set()
            return {
                w.strip() for w in stopwords_path.read_text(encoding="utf-8").splitlines() if w.strip()
            }
        except Exception as exc:  # pragma: no cover
            LOG.warning("Unable to load Vietnamese stopwords from underthesea: %s", exc)
            return set()

except ModuleNotFoundError:  # pragma: no cover
    _underthesea = None

    _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+", flags=re.UNICODE)

    # Rough token pattern: keep word characters (including unicode)
    _TOKEN_RE = re.compile(r"\w+", re.UNICODE)

    def sent_tokenize(text: str) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        # Split by punctuation then further split by newlines if needed.
        parts = _SENT_SPLIT_RE.split(cleaned)
        # Also split hard newlines.
        out: List[str] = []
        for p in parts:
            for line in re.split(r"\n+", p):
                line = line.strip()
                if line:
                    out.append(line)
        return out

    def word_tokenize(text: str, format: str = "text") -> str:  # match underthesea signature loosely
        # underthesea.word_tokenize returns different formats; for this project we call with format="text"
        if text is None:
            return ""
        normalized = unicodedata.normalize("NFC", text or "").lower()
        tokens = _TOKEN_RE.findall(normalized)
        return " ".join(tokens)

    def load_stopwords() -> Set[str]:
        # No package => no built-in stopwords.
        return set()


__all__ = ["sent_tokenize", "word_tokenize", "load_stopwords"]

