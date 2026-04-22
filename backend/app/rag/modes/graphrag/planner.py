from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from app.ai.llm import get_llm
from app.core.logger import LOG
from app.rag.models import RAGQueryRequest


@dataclass
class GraphRAGPlan:
    """Execution plan for GraphRAG query-time pipeline."""

    question: str
    search_strategy: str  # "local" or "global"
    search_type: str      # "vector" or "hybrid"
    top_k: int
    document_id: Optional[int] = None
    document_ids: Optional[List[int]] = None
    session_id: Optional[int] = None
    llm_model: Optional[str] = None


class GraphRAGPlanner:
    """
    Planner for GraphRAG.
    Decides between:
    - local: entity/relation/detail questions
    - global: summary/theme/trend questions
    """

    def plan(self, request: RAGQueryRequest) -> GraphRAGPlan:
        LOG.info(f"[Planner] Question: '{request.question}'")
        strategy = self._classify_intent(request.question, request.llm_model)

        plan_obj = GraphRAGPlan(
            question=request.question,
            search_strategy=strategy,
            search_type=request.search_type,
            top_k=request.top_k,
            document_id=request.document_id,
            document_ids=request.document_ids,
            session_id=request.session_id,
            llm_model=request.llm_model,
        )
        LOG.info(f"[Planner] Selected strategy: {strategy}")
        return plan_obj

    def _classify_intent(self, question: str, llm_model: str | None = None) -> str:
        """
        Strategy classifier with robust fallback:
        1) quick heuristic
        2) LLM JSON classification
        3) heuristic fallback
        """
        heuristic_strategy = self._classify_by_heuristic(question)

        prompt = (
            "You are a GraphRAG query classifier.\n"
            "Return JSON only, format: {\"strategy\":\"local\"} or {\"strategy\":\"global\"}.\n"
            "- local: specific entities, relations, fine-grained facts.\n"
            "- global: summaries, themes, trends, high-level synthesis.\n\n"
            f"Question: {question}\n"
            "JSON:"
        )

        try:
            llm = get_llm(temperature=0.0, model=llm_model)
            raw_response = llm.invoke(prompt).strip()
            parsed_strategy = self._parse_strategy(raw_response)
            if parsed_strategy:
                return parsed_strategy
        except Exception as exc:
            LOG.warning(f"[Planner] LLM strategy classification failed: {exc}")

        return heuristic_strategy or "local"

    def _classify_by_heuristic(self, question: str) -> str | None:
        q = self._normalize_text(question)
        if not q:
            return "local"

        global_patterns = [
            r"\btom tat\b",
            r"\btong quan\b",
            r"\btoan canh\b",
            r"\bxu huong\b",
            r"\bchu de\b",
            r"\btong the\b",
            r"\bnoi ve gi\b",
            r"\btai lieu.*ve gi\b",
            r"\bmuc dich\b",
            r"\bkhai quat\b",
            r"\boverall\b",
            r"\bsummary\b",
            r"\boverview\b",
            r"\bwhat.*about\b",
            r"\bmain (topic|idea)\b",
            r"\bbig picture\b",
            r"\bhigh level\b",
        ]
        local_patterns = [
            r"\bai\b",
            r"\bla gi\b",
            r"\bkhi nao\b",
            r"\bo dau\b",
            r"\bquan he\b",
            r"\bchi tiet\b",
            r"\bso lieu\b",
            r"\bbao nhieu\b",
            r"\bnhu the nao\b",
            r"\bwhich\b",
            r"\bwho\b",
            r"\bwhen\b",
            r"\bwhere\b",
        ]

        global_hits = sum(1 for pattern in global_patterns if re.search(pattern, q))
        local_hits = sum(1 for pattern in local_patterns if re.search(pattern, q))

        if global_hits > local_hits:
            return "global"
        if local_hits > global_hits:
            return "local"

        return None

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        raw = (text or "").strip().lower()
        if not raw:
            return ""
        normalized = unicodedata.normalize("NFD", raw)
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return normalized

    def _parse_strategy(self, raw_response: str) -> str | None:
        raw = (raw_response or "").strip()
        if not raw:
            return None

        # 1) Parse strict JSON object.
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                strategy = str(payload.get("strategy", "")).strip().lower()
                if strategy in {"local", "global"}:
                    return strategy
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2) Extract JSON object from free text.
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
                if isinstance(payload, dict):
                    strategy = str(payload.get("strategy", "")).strip().lower()
                    if strategy in {"local", "global"}:
                        return strategy
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 3) Last fallback: keyword match.
        lowered = raw.lower()
        if re.search(r"\bglobal\b", lowered):
            return "global"
        if re.search(r"\blocal\b", lowered):
            return "local"
        return None
