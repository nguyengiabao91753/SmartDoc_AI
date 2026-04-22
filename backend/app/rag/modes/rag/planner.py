from dataclasses import dataclass
from typing import List

from app.rag.models import RAGQueryRequest


@dataclass
class RAGPlan:
    question: str
    search_type: str
    top_k: int
    document_id: int | None = None
    document_ids: List[int] | None = None
    session_id: int | None = None
    llm_model: str | None = None


class RAGPlanner:
    """Planner for vanilla RAG. Keeps current behavior as a direct passthrough."""

    def plan(self, request: RAGQueryRequest) -> RAGPlan:
        return RAGPlan(
            question=request.question,
            search_type=request.search_type,
            top_k=request.top_k,
            document_id=request.document_id,
            document_ids=request.document_ids,
            session_id=request.session_id,
            llm_model=request.llm_model,
        )
