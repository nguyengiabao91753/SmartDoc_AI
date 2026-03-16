from app.vectorstore.faiss_store import FaissStore
from app.core.config import settings
from app.core.logger import LOG
import numpy as np
from typing import List

def retrieve_topk(store: FaissStore, q_vector: np.ndarray, k: int = None):
    k = k or settings.TOP_K
    return store.search(q_vector, k)

def get_retriever(store: FaissStore, search_type: str = "vector"):
    if search_type == "hybrid":
        return HybridRetriever(store)
    return VectorRetriever(store)

class VectorRetriever:
    def __init__(self, store: FaissStore):
        self.store = store

    def retrieve(self, query_text: str, q_vector: np.ndarray, top_k: int = None):
        k = top_k or settings.TOP_K
        return self.store.search(q_vector, k)

class HybridRetriever:
    def __init__(self, store: FaissStore):
        self.store = store

    def retrieve(self, query_text: str, q_vector: np.ndarray, top_k: int = None, alpha: float = 0.5):
        k = top_k or settings.TOP_K
        return self.store.hybrid_search(query_text, q_vector, k, alpha)
