from __future__ import annotations

from app.rag.base import BaseRAGModeEngine
from app.rag.models import RAGEngineResult, RAGQueryRequest

from .planner import CoRAGPlanner
from .responder import CoRAGResponder
from .retriever import CoRAGRetriever


class CoRAGEngine(BaseRAGModeEngine):
    """
    CoRAG Engine - Chain of Retrieval Augmented Generation.

    Pipeline:
    1. Planner  : LLM phân tách câu hỏi → N sub-queries độc lập
    2. Retriever: Retrieve FAISS+BM25 riêng cho từng sub-query → RRF merge + dedupe
    3. Responder: Build context có cấu trúc theo khía cạnh → LLM trả lời bám sát tài liệu
    """

    mode = "corag"
    display_name = "CoRAG"

    def __init__(self, store, embedding_service):
        super().__init__(store=store, embedding_service=embedding_service)
        self.planner   = CoRAGPlanner()
        self.retriever = CoRAGRetriever(store=store, embedding_service=embedding_service)
        self.responder = CoRAGResponder()

    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        # Bước 1: Decompose câu hỏi thành sub-queries
        plan = self.planner.plan(request)

        # Bước 2: Retrieve cho từng sub-query, RRF merge + dedupe
        merged_docs, per_query_docs = self.retriever.retrieve_all(plan)

        # Bước 3: Build context có cấu trúc → LLM tổng hợp bám sát tài liệu
        answer = self.responder.answer(
            question=plan.question,
            sub_queries=plan.sub_queries,
            source_documents=merged_docs,
            per_query_docs=per_query_docs,   # truyền thêm để build structured context
            llm_model=plan.llm_model,
        )

        metadata = {
            "mode": "corag",
            "sub_queries": plan.sub_queries,
            "per_query_doc_counts": {q: len(docs) for q, docs in per_query_docs.items()},
            "merged_doc_count": len(merged_docs),
        }

        return RAGEngineResult(
            answer=answer,
            source_documents=merged_docs,
            metadata=metadata,
        )