from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any, Dict, List, Tuple

from app.core.logger import LOG

try:
    from underthesea import ner, sent_tokenize

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    LOG.warning("Thu vien underthesea khong co san. Hay cai dat: pip install underthesea")

logger = logging.getLogger(__name__)


class NERService:
    """
    Fast entity/relation extraction service.

    Notes:
    - Uses underthesea NER when available.
    - Adds lightweight regex fallback for technical terms.
    - Relations are sentence co-occurrence to avoid incorrect semantic labels.
    """

    TECH_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_./-]{2,}\b")

    def __init__(self):
        self.type_mapping = {
            "PER": "Nguoi",
            "ORG": "ToChuc",
            "LOC": "DiaDiem",
            "MISC": "KhaiNiem",
        }
        self.entity_stopwords = {
            "va",
            "hoac",
            "la",
            "duoc",
            "cho",
            "trong",
            "cua",
            "nhung",
            "cac",
            "mot",
            "tai",
            "voi",
            "neu",
            "khi",
            "buoc",
            "step",
            "page",
            "the",
            "this",
            "that",
        }
        LOG.info("Khoi tao NERService (underthesea + regex fallback).")

    def extract_graph_elements(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"entities": [], "relationships": []}

        if not UNDERTHESEA_AVAILABLE:
            LOG.error("underthesea chua duoc cai dat, khong the trich xuat graph elements.")
            return {"entities": [], "relationships": []}

        try:
            return self._extract_fast(text)
        except Exception as exc:
            LOG.error("Loi khi trich xuat do thi: %s", exc)
            return {"entities": [], "relationships": []}

    def _extract_fast(self, text: str) -> Dict[str, Any]:
        entities: List[Dict[str, str]] = []
        relationships: List[Dict[str, str]] = []
        entity_map: Dict[str, Dict[str, str]] = {}

        sentences = sent_tokenize(text)

        for sentence in sentences:
            sentence_clean = self._clean_sentence(sentence)
            if not sentence_clean:
                continue

            sentence_entities = self._extract_sentence_entities(sentence_clean)

            for name, raw_type in sentence_entities:
                normalized_name = self._normalize_entity_name(name)
                if not self._is_valid_entity_name(normalized_name):
                    continue

                entity_type = self.type_mapping.get(raw_type, "KhaiNiem")
                if normalized_name not in entity_map:
                    payload = {
                        "name": normalized_name,
                        "type": entity_type,
                        "description": f"Duoc nhac den trong ngu canh: {sentence_clean[:220]}",
                    }
                    entity_map[normalized_name] = payload
                    entities.append(payload)

            # Build pairwise co-occurrence relations inside sentence.
            unique_names = sorted({self._normalize_entity_name(name) for name, _ in sentence_entities})
            unique_names = [name for name in unique_names if self._is_valid_entity_name(name)]

            for source, target in combinations(unique_names, 2):
                if source == target:
                    continue

                relationships.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": "DONG_XUAT_HIEN",
                        "description": f"Cung xuat hien trong cau: {sentence_clean[:220]}",
                    }
                )

        deduped_relationships = self._dedupe_relationships(relationships)
        return {"entities": entities, "relationships": deduped_relationships}

    def _extract_sentence_entities(self, sentence: str) -> List[Tuple[str, str]]:
        extracted: List[Tuple[str, str]] = []

        # 1) underthesea NER entities
        tokens = ner(sentence)
        current_entity: List[str] = []
        current_type = "MISC"

        for word, _pos, _chunk, tag in tokens:
            if tag.startswith("B-"):
                if current_entity:
                    extracted.append((" ".join(current_entity), current_type))
                current_entity = [word]
                current_type = tag[2:]
            elif tag.startswith("I-") and current_entity:
                current_entity.append(word)
            else:
                if current_entity:
                    extracted.append((" ".join(current_entity), current_type))
                    current_entity = []
                    current_type = "MISC"

        if current_entity:
            extracted.append((" ".join(current_entity), current_type))

        # 2) regex fallback for technical terms
        for match in self.TECH_TERM_PATTERN.findall(sentence):
            extracted.append((match, "MISC"))

        return extracted

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        return re.sub(r"\s+", " ", (sentence or "")).strip()

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        normalized = re.sub(r"\s+", " ", str(name or "").strip())
        normalized = normalized.strip(".,:;!?()[]{}\"'")
        return normalized

    def _is_valid_entity_name(self, name: str) -> bool:
        if not name or len(name) < 2:
            return False

        lowered = name.lower()
        if lowered in self.entity_stopwords:
            return False

        # Reject pure numeric IDs and tiny symbols.
        if re.fullmatch(r"[0-9_.-]+", name):
            return False

        return True

    @staticmethod
    def _dedupe_relationships(relationships: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        deduped: List[Dict[str, str]] = []

        for rel in relationships:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            relation = str(rel.get("relation", "DONG_XUAT_HIEN")).strip() or "DONG_XUAT_HIEN"
            if not source or not target:
                continue

            # Undirected key for co-occurrence relation.
            a, b = sorted((source, target))
            key = f"{a}|{relation}|{b}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rel)

        return deduped
