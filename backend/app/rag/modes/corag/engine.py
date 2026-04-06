from app.rag.base import BaseRAGModeEngine, ModeNotImplementedError
from app.rag.models import RAGEngineResult, RAGQueryRequest


class CoRAGEngine(BaseRAGModeEngine):
    mode = "corag"
    display_name = "CoRAG"

    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        raise ModeNotImplementedError("CoRAG mode chua duoc implement.")
