from app.rag.base import BaseRAGModeEngine
from app.rag.models import RAGEngineResult, RAGQueryRequest

from .planner import RAGPlanner
from .responder import RAGResponder
from .retriever import RAGRetriever


class VanillaRAGEngine(BaseRAGModeEngine):
    mode = "rag"
    display_name = "RAG"

    def __init__(self, store, embedding_service):
        super().__init__(store=store, embedding_service=embedding_service)
        self.planner = RAGPlanner()
        self.retriever = RAGRetriever(store=store, embedding_service=embedding_service)
        self.responder = RAGResponder()

    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        plan = self.planner.plan(request)
        source_documents = self.retriever.retrieve(plan)
        answer = self.responder.answer(
            question=plan.question,
            source_documents=source_documents,
            llm_model=plan.llm_model,
        )
        return RAGEngineResult(answer=answer, source_documents=source_documents)
