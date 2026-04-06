from app.rag.base import BaseRAGModeEngine, ModeNotImplementedError
from app.rag.models import RAGEngineResult, RAGQueryRequest


class GraphRAGEngine(BaseRAGModeEngine):
    mode = "graphrag"
    display_name = "GraphRAG"

    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        raise ModeNotImplementedError("GraphRAG mode chua duoc implement.")
