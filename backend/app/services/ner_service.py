from __future__ import annotations

import json
import logging
import re
import unicodedata
from itertools import combinations
from typing import Any, Dict, List, Tuple

from app.core.logger import LOG

try:
    from underthesea import sent_tokenize

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False

logger = logging.getLogger(__name__)


class NERService:
    """
    LLM-first extractor for Vietnamese Knowledge Graph construction.

    Notes:
    - Primary extraction is done by local Ollama model via get_llm().
    - No manual relation ontology or mapping file is required.
    - Lightweight lexical fallback is used only when LLM parsing fails.
    """

    MAX_INPUT_CHARS = 2000
    SEGMENT_CHARS = 900
    MAX_EVIDENCE_CHARS = 220
    DEFAULT_ENTITY_TYPE = "KhaiNiem"
    DEFAULT_RELATION = "LIEN_QUAN"
    DEFAULT_RELATION_CONFIDENCE = 0.45
    MIN_RELATION_CONFIDENCE = 0.2
    TECH_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_./-]{2,}\b")

    def __init__(self, llm_model: str | None = None):
        self.llm_model = llm_model
        self.llm = None
        try:
            from app.ai.llm import get_llm

            self.llm = get_llm(temperature=0.0, model=llm_model)
            LOG.info("Khoi tao NERService (LLM-first extraction, model=%s).", llm_model or "default")
        except Exception as exc:
            LOG.warning("Khong khoi tao duoc LLM extractor, su dung lexical fallback: %s", exc)

    def extract_graph_elements(self, text: str) -> Dict[str, Any]:
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return {"entities": [], "relationships": []}

        if self.llm is None:
            return self._extract_fallback_lexical(cleaned_text)

        try:
            extracted = self._extract_with_llm(cleaned_text)
            if extracted["entities"] or extracted["relationships"]:
                return extracted
        except Exception as exc:
            LOG.warning("LLM extraction that bai, chuyen sang fallback lexical: %s", exc)

        return self._extract_fallback_lexical(cleaned_text)

    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        segments = self._split_for_llm(text)
        entities_map: Dict[str, Dict[str, str]] = {}
        relationships: List[Dict[str, str]] = []

        for segment in segments:
            prompt = self._build_extraction_prompt(segment)
            raw = str(self.llm.invoke(prompt)).strip()
            payload = self._parse_json_payload(raw)
            if not payload:
                continue

            segment_entities = self._normalize_entities(payload.get("entities", []), evidence=segment)
            segment_relationships = self._normalize_relationships(payload.get("relationships", []), evidence=segment)

            for entity in segment_entities:
                canonical = self._entity_key(entity["name"])
                if not canonical:
                    continue
                existing = entities_map.get(canonical)
                if existing is None:
                    entities_map[canonical] = entity
                else:
                    existing_desc = str(existing.get("description", "")).strip()
                    if len(entity.get("description", "")) > len(existing_desc):
                        existing["description"] = entity["description"]
                    if existing.get("type") == self.DEFAULT_ENTITY_TYPE and entity.get("type"):
                        existing["type"] = entity["type"]

            relationships.extend(segment_relationships)

        entities = list(entities_map.values())

        # Guarantee endpoint entities from relations also exist.
        for rel in relationships:
            for endpoint in (rel.get("source", ""), rel.get("target", "")):
                endpoint_name = self._normalize_entity_name(endpoint)
                if not endpoint_name:
                    continue
                key = self._entity_key(endpoint_name)
                if key not in entities_map:
                    entity = {
                        "name": endpoint_name,
                        "type": self.DEFAULT_ENTITY_TYPE,
                        "description": "Duoc nhac den trong quan he tri thuc.",
                    }
                    entities_map[key] = entity
                    entities.append(entity)

        deduped_relationships = self._dedupe_relationships(relationships)
        return {"entities": entities, "relationships": deduped_relationships}

    def _build_extraction_prompt(self, segment: str) -> str:
        return (
            "Ban la he thong trich xuat tri thuc cho tai lieu tieng Viet.\n"
            "Nhiem vu: trich xuat thuc the va quan he CHI dua tren doan van ban duoc cung cap.\n"
            "Khong duoc them kien thuc ben ngoai.\n"
            "Chi tra ve JSON hop le, khong markdown, khong giai thich them.\n\n"
            "Yeu cau:\n"
            "- entities: danh sach thuc the quan trong.\n"
            "- relationships: danh sach quan he co huong giua cac thuc the.\n"
            "- confidence trong [0,1].\n"
            "- relation nen la nhan ngan gon (vi du: THUOC_VE, GAY_RA, SU_DUNG, BAO_VE, TAC_DONG_DEN).\n"
            "- Neu khong co gi thi tra ve mang rong.\n\n"
            "Schema JSON:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name":"...", "type":"...", "description":"..."}\n'
            "  ],\n"
            '  "relationships": [\n'
            '    {"source":"...", "target":"...", "relation":"...", "description":"...", "confidence":0.0}\n'
            "  ]\n"
            "}\n\n"
            f"Doan van ban:\n{segment}\n\n"
            "JSON:"
        )

    def _parse_json_payload(self, raw: str) -> Dict[str, Any] | None:
        if not raw:
            return None

        # 1) strict JSON
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        # 2) fenced block
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            try:
                payload = json.loads(fenced_match.group(1))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

        # 3) first JSON object
        object_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if object_match:
            try:
                payload = json.loads(object_match.group(0))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return None

        return None

    def _normalize_entities(self, entities: List[Dict[str, Any]], *, evidence: str) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen = set()

        for item in entities or []:
            name = self._normalize_entity_name(item.get("name"))
            if not self._is_valid_entity_name(name):
                continue

            key = self._entity_key(name)
            if key in seen:
                continue
            seen.add(key)

            entity_type = self._normalize_entity_type(item.get("type"))
            description = self._clean_text(item.get("description"))
            if not description:
                description = f"Duoc nhac den trong ngu canh: {evidence[:self.MAX_EVIDENCE_CHARS]}"

            normalized.append(
                {
                    "name": name,
                    "type": entity_type,
                    "description": description,
                }
            )

        return normalized

    def _normalize_relationships(
        self,
        relationships: List[Dict[str, Any]],
        *,
        evidence: str,
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []

        for item in relationships or []:
            source = self._normalize_entity_name(item.get("source"))
            target = self._normalize_entity_name(item.get("target"))
            if not self._is_valid_entity_name(source) or not self._is_valid_entity_name(target):
                continue
            if self._entity_key(source) == self._entity_key(target):
                continue

            relation = self._normalize_relation_label(item.get("relation"))
            description = self._clean_text(item.get("description"))
            if not description:
                description = f"Suy ra tu doan: {evidence[:self.MAX_EVIDENCE_CHARS]}"

            confidence = self._to_confidence(item.get("confidence"))
            if confidence < self.MIN_RELATION_CONFIDENCE:
                continue

            normalized.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "description": description,
                    "confidence": f"{confidence:.4f}",
                    "direction": "forward",
                }
            )

        return normalized

    def _extract_fallback_lexical(self, text: str) -> Dict[str, Any]:
        """
        Last-resort fallback when LLM output cannot be parsed.
        """
        sentences = self._split_sentences(text)
        entity_map: Dict[str, Dict[str, str]] = {}
        relationships: List[Dict[str, str]] = []

        for sentence in sentences:
            sentence_clean = self._clean_text(sentence)
            if not sentence_clean:
                continue

            raw_terms = [term for term in self.TECH_TERM_PATTERN.findall(sentence_clean) if self._is_technical_term(term)]
            sentence_entities = []
            for term in raw_terms:
                name = self._normalize_entity_name(term)
                if not self._is_valid_entity_name(name):
                    continue
                sentence_entities.append(name)

                key = self._entity_key(name)
                if key not in entity_map:
                    entity_map[key] = {
                        "name": name,
                        "type": self.DEFAULT_ENTITY_TYPE,
                        "description": f"Duoc nhac den trong ngu canh: {sentence_clean[:self.MAX_EVIDENCE_CHARS]}",
                    }

            for source, target in combinations(sorted(set(sentence_entities)), 2):
                relationships.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": self.DEFAULT_RELATION,
                        "description": f"Cung xuat hien trong cau: {sentence_clean[:self.MAX_EVIDENCE_CHARS]}",
                        "confidence": f"{self.DEFAULT_RELATION_CONFIDENCE:.4f}",
                        "direction": "undirected",
                    }
                )

        return {"entities": list(entity_map.values()), "relationships": self._dedupe_relationships(relationships)}

    def _split_for_llm(self, text: str) -> List[str]:
        trimmed = text[: self.MAX_INPUT_CHARS]
        sentences = self._split_sentences(trimmed)
        if not sentences:
            return [trimmed]

        segments: List[str] = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= self.SEGMENT_CHARS:
                current = candidate
            else:
                segments.append(current)
                current = sentence

        if current:
            segments.append(current)
        return segments

    def _split_sentences(self, text: str) -> List[str]:
        if UNDERTHESEA_AVAILABLE:
            try:
                return [self._clean_text(s) for s in sent_tokenize(text) if self._clean_text(s)]
            except Exception as exc:
                LOG.warning("Tach cau bang underthesea that bai: %s", exc)

        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
        return [self._clean_text(s) for s in raw_sentences if self._clean_text(s)]

    @staticmethod
    def _clean_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _strip_accents(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def _normalize_entity_name(self, name: Any) -> str:
        cleaned = self._clean_text(name)
        cleaned = cleaned.strip(".,:;!?()[]{}\"'")
        return cleaned

    def _entity_key(self, name: str) -> str:
        lowered = self._strip_accents(self._clean_text(name).lower())
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _normalize_entity_type(self, entity_type: Any) -> str:
        cleaned = self._clean_text(entity_type)
        if not cleaned:
            return self.DEFAULT_ENTITY_TYPE
        if len(cleaned) > 40:
            return self.DEFAULT_ENTITY_TYPE
        return cleaned

    def _normalize_relation_label(self, relation: Any) -> str:
        raw = self._clean_text(relation)
        if not raw:
            return self.DEFAULT_RELATION
        stripped = self._strip_accents(raw)
        label = re.sub(r"[^A-Za-z0-9]+", "_", stripped).strip("_").upper()
        if not label:
            return self.DEFAULT_RELATION
        return label[:48]

    def _to_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return self.DEFAULT_RELATION_CONFIDENCE
        return max(0.0, min(confidence, 1.0))

    def _is_valid_entity_name(self, name: str) -> bool:
        if not name or len(name) < 2:
            return False

        normalized = self._entity_key(name)
        if len(normalized) < 2:
            return False

        if re.fullmatch(r"[0-9_.-]+", name):
            return False

        return True

    @staticmethod
    def _is_technical_term(name: str) -> bool:
        candidate = str(name or "").strip()
        if not candidate:
            return False
        if re.fullmatch(r"[A-Z]{2,10}", candidate):
            return True
        if re.search(r"[0-9_/.\-]", candidate):
            return True
        if re.search(r"[A-Z]", candidate) and re.search(r"[a-z]", candidate):
            return True
        return False

    def _dedupe_relationships(self, relationships: List[Dict[str, str]]) -> List[Dict[str, str]]:
        best_by_key: Dict[str, Dict[str, str]] = {}

        for rel in relationships:
            source = self._normalize_entity_name(rel.get("source"))
            target = self._normalize_entity_name(rel.get("target"))
            relation = self._normalize_relation_label(rel.get("relation"))
            direction = str(rel.get("direction", "forward")).strip() or "forward"

            if not source or not target:
                continue
            if self._entity_key(source) == self._entity_key(target):
                continue

            if direction == "undirected":
                a, b = sorted((self._entity_key(source), self._entity_key(target)))
                key = f"{a}|{relation}|{b}"
            else:
                key = f"{self._entity_key(source)}|{relation}|{self._entity_key(target)}"

            incoming_conf = self._to_confidence(rel.get("confidence"))
            current = best_by_key.get(key)
            if current is None:
                best_by_key[key] = {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "description": str(rel.get("description", "")).strip(),
                    "confidence": f"{incoming_conf:.4f}",
                }
                continue

            current_conf = self._to_confidence(current.get("confidence"))
            if incoming_conf > current_conf:
                current["source"] = source
                current["target"] = target
                current["relation"] = relation
                current["description"] = str(rel.get("description", "")).strip()
                current["confidence"] = f"{incoming_conf:.4f}"

        return list(best_by_key.values())
