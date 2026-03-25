from typing import Any, Dict

import numpy as np

from app.core.config import settings
from app.vectorstore.faiss_store import FaissStore


def retrieve_topk(store: FaissStore, q_vector: np.ndarray, k: int | None = None):
    return store.search(q_vector, k or settings.TOP_K)


def get_retriever(
    store: FaissStore,
    search_type: str = "vector",
    top_k: int | None = None,
    filters: Dict[str, Any] | None = None,
):
    normalized_type = (search_type or "vector").lower()
    if normalized_type in {"hybrid", "keyword"}:
        return HybridRetriever(store, top_k=top_k, filters=filters)
    return VectorRetriever(store, top_k=top_k, filters=filters)


class VectorRetriever:
    def __init__(
        self,
        store: FaissStore,
        top_k: int | None = None,
        filters: Dict[str, Any] | None = None,
    ):
        self.store = store
        self.top_k = top_k
        self.filters = filters

    def retrieve(self, query_text: str, q_vector: np.ndarray, top_k: int | None = None):
        k = top_k or self.top_k or settings.TOP_K
        return self.store.search(q_vector, k, filters=self.filters)


class HybridRetriever:
    def __init__(
        self,
        store: FaissStore,
        top_k: int | None = None,
        filters: Dict[str, Any] | None = None,
    ):
        self.store = store
        self.top_k = top_k
        self.filters = filters

    def retrieve(self, query_text: str, q_vector: np.ndarray, top_k: int | None = None, alpha: float = 0.6):
        k = top_k or self.top_k or settings.TOP_K
        return self.store.hybrid_search(query_text, q_vector, k, alpha=alpha, filters=self.filters)
