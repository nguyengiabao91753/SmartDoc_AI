from dataclasses import dataclass

from app.rag.models import RAGQueryRequest


@dataclass
class RAGPlan:
    question: str
    search_type: str
    top_k: int
    document_id: int | None = None
    llm_model: str | None = None


class RAGPlanner:
    """Planner for vanilla RAG. Keeps current behavior as a direct passthrough."""

    def plan(self, request: RAGQueryRequest) -> RAGPlan:
        return RAGPlan(
            question=request.question,
            search_type=request.search_type,
            top_k=request.top_k,
            document_id=request.document_id,
            llm_model=request.llm_model,
        )
