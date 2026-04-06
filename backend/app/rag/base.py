from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.rag.models import RAGEngineResult, RAGQueryRequest
from app.vectorstore.faiss_store import FaissStore


class ModeNotImplementedError(RuntimeError):
    """Raised when a mode is registered but intentionally not implemented yet."""


class BaseRAGModeEngine(ABC):
    mode = "rag"
    display_name = "RAG"

    def __init__(self, store: FaissStore, embedding_service: Any):
        self.store = store
        self.embedding_service = embedding_service

    @abstractmethod
    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        """Execute one query for this mode."""
