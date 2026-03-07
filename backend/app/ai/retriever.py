# from app.vectorstore.faiss_store import FaissStore
# from app.core.config import settings
# from app.core.logger import LOG
# import numpy as np
# from typing import List

# # Simple wrapper to perform retrieval using FaissStore instance
# def retrieve_topk(store: FaissStore, q_vector: np.ndarray, k: int = None):
#     k = k or settings.TOP_K
#     return store.search(q_vector, k)