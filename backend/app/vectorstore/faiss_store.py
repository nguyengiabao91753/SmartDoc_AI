import os
import faiss
import numpy as np
import pickle
from app.core.config import settings
from app.core.logger import LOG

INDEX_FILE = os.path.join(settings.VECTOR_DIR, "faiss.index")
META_FILE = os.path.join(settings.VECTOR_DIR, "meta.pkl")

class FaissStore:
    def __init__(self, dim: int):
        os.makedirs(settings.VECTOR_DIR, exist_ok=True)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # use IP on normalized vectors -> cosine
        self.meta = []

    def add(self, vectors: np.ndarray, metas: list):
        # vectors: (N, dim), float32 normalized
        if vectors.size == 0:
            return
        self.index.add(vectors)
        self.meta.extend(metas)
        LOG.info("Added %d vectors; total vectors: %d", len(vectors), self.index.ntotal)

    def search(self, q_vector: np.ndarray, top_k: int):
        q = np.array([q_vector]).astype('float32')
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append({"score": float(score), "meta": self.meta[idx], "id": int(idx)})
        return results

    def save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(self.meta, f)
        LOG.info("Saved FAISS index and meta")

    def load(self):
        try:
            if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
                return False
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                self.meta = pickle.load(f)
            self.dim = self.index.d
            LOG.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            return True
        except Exception as e:
            LOG.exception("Failed to load index: %s", e)
            return False