from __future__ import annotations

from typing import Dict, Type

from app.rag.base import BaseRAGModeEngine
from app.rag.modes.corag.engine import CoRAGEngine
from app.rag.modes.graphrag.engine import GraphRAGEngine
from app.rag.modes.rag.engine import VanillaRAGEngine

ENGINE_BY_MODE: Dict[str, Type[BaseRAGModeEngine]] = {
    "rag": VanillaRAGEngine,
    "corag": CoRAGEngine,
    "graphrag": GraphRAGEngine,
}

MODE_ALIASES = {
    "default": "rag",
    "vanilla": "rag",
    "co-rag": "corag",
    "graph": "graphrag",
    "graph-rag": "graphrag",
}

MODE_LABELS = {
    "rag": "RAG",
    "corag": "CoRAG",
    "graphrag": "GraphRAG",
}

AVAILABLE_RAG_MODES = tuple(ENGINE_BY_MODE.keys())


def normalize_rag_mode(mode: str | None) -> str:
    normalized = (mode or "rag").strip().lower()
    normalized = MODE_ALIASES.get(normalized, normalized)
    if normalized not in ENGINE_BY_MODE:
        return "rag"
    return normalized


def build_engine_registry(store, embedding_service) -> Dict[str, BaseRAGModeEngine]:
    return {
        mode: engine_cls(store=store, embedding_service=embedding_service)
        for mode, engine_cls in ENGINE_BY_MODE.items()
    }
