import os
import pickle
import string
from typing import Any, Dict, List

import faiss
import nltk
import numpy as np
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logger import LOG

INDEX_FILE = os.path.join(settings.VECTOR_DIR, "faiss.index")
META_FILE = os.path.join(settings.VECTOR_DIR, "meta.pkl")
BM25_FILE = os.path.join(settings.VECTOR_DIR, "bm25.pkl")


def _ensure_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception as exc:
            LOG.warning("Unable to download NLTK stopwords: %s", exc)


def preprocess_text(text: str) -> List[str]:
    _ensure_stopwords()
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    normalized_text = (text or "").lower()
    normalized_text = normalized_text.translate(str.maketrans("", "", string.punctuation))
    return [word for word in normalized_text.split() if word and word not in stop_words]


class BM25Retriever:
    def __init__(self, corpus: List[str] | None = None):
        self.corpus = corpus or []
        self.bm25: BM25Okapi | None = None
        if self.corpus:
            self.fit(self.corpus)

    def fit(self, corpus: List[str]):
        self.corpus = corpus
        tokenized_corpus = [preprocess_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, top_k: int | None = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []

        tokenized_query = preprocess_text(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        limit = len(doc_scores) if top_k is None else min(top_k, len(doc_scores))
        top_indices = np.argsort(doc_scores)[-limit:][::-1]
        return [{"score": float(doc_scores[i]), "id": int(i)} for i in top_indices]

    def save(self, filepath: str):
        with open(filepath, "wb") as file_handle:
            pickle.dump(self, file_handle)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as file_handle:
            return pickle.load(file_handle)


class FaissStore:
    def __init__(self, dim: int, index_type: str = "flat"):
        os.makedirs(settings.VECTOR_DIR, exist_ok=True)
        self.dim = dim
        self.index_type = index_type.lower()
        self.index = self._create_index(dim, self.index_type)
        self.meta: List[Dict[str, Any]] = []
        self.bm25_retriever = BM25Retriever()

    def _create_index(self, dim: int, index_type: str):
        if index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if index_type == "hnsw":
            m = 32
            return faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexFlatIP(dim)

    def _matches_filters(self, meta: Dict[str, Any], filters: Dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            actual = meta.get(key)

            # Support {"field": {"$in": [...]}} filter style.
            if isinstance(expected, dict) and "$in" in expected:
                allowed_values = expected.get("$in") or []
                if actual not in allowed_values:
                    return False
                continue

            # If meta field is a list/set, membership match is accepted.
            if isinstance(actual, (list, tuple, set)):
                if expected not in actual:
                    return False
                continue

            if actual != expected:
                return False

        return True

    def _refresh_bm25(self):
        texts = [meta.get("text", "") for meta in self.meta if meta.get("text")]
        if texts:
            self.bm25_retriever.fit(texts)
        else:
            self.bm25_retriever = BM25Retriever()

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> List[int]:
        if vectors.size == 0:
            return []

        if len(vectors) != len(metas):
            raise ValueError("Vectors and metadata length mismatch")

        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(vectors)

        start_id = self.index.ntotal
        self.index.add(vectors.astype("float32"))
        self.meta.extend(metas)
        self._refresh_bm25()

        added_ids = list(range(start_id, self.index.ntotal))
        LOG.info("Added %d vectors; total vectors: %d", len(added_ids), self.index.ntotal)
        return added_ids

    def search(self, q_vector: np.ndarray, top_k: int, filters: Dict[str, Any] | None = None):
        if self.index.ntotal == 0:
            return []

        candidate_k = self.index.ntotal if filters else min(top_k, self.index.ntotal)
        q = np.asarray([q_vector], dtype="float32")
        scores, indices = self.index.search(q, candidate_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            meta = self.meta[idx]
            if not self._matches_filters(meta, filters):
                continue
            results.append({"score": float(score), "meta": meta, "id": int(idx)})
            if len(results) >= top_k:
                break
        return results

    def keyword_search(self, query_text: str, top_k: int, filters: Dict[str, Any] | None = None):
        if not self.meta:
            return []

        candidate_k = len(self.meta) if filters else top_k
        bm25_results = self.bm25_retriever.search(query_text, candidate_k)

        results = []
        for result in bm25_results:
            idx = result["id"]
            if idx < 0 or idx >= len(self.meta):
                continue
            meta = self.meta[idx]
            if not self._matches_filters(meta, filters):
                continue
            results.append({"score": float(result["score"]), "meta": meta, "id": idx})
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(
        self,
        query_text: str,
        q_vector: np.ndarray,
        top_k: int,
        alpha: float = 0.6,
        filters: Dict[str, Any] | None = None,
    ):
        vector_results = self.search(q_vector, top_k, filters=filters)
        keyword_results = self.keyword_search(query_text, top_k, filters=filters)

        if not vector_results and not keyword_results:
            return []

        max_vector_score = max((result["score"] for result in vector_results), default=1.0) or 1.0
        max_keyword_score = max((result["score"] for result in keyword_results), default=1.0) or 1.0

        combined_results: Dict[int, Dict[str, Any]] = {}
        for result in vector_results:
            combined_results[result["id"]] = {
                "meta": result["meta"],
                "vector_score": result["score"] / max_vector_score,
                "keyword_score": 0.0,
            }

        for result in keyword_results:
            entry = combined_results.setdefault(
                result["id"],
                {"meta": result["meta"], "vector_score": 0.0, "keyword_score": 0.0},
            )
            entry["keyword_score"] = result["score"] / max_keyword_score

        ranked = []
        for item_id, scores in combined_results.items():
            final_score = alpha * scores["vector_score"] + (1 - alpha) * scores["keyword_score"]
            ranked.append({"score": float(final_score), "meta": scores["meta"], "id": item_id})

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "wb") as file_handle:
            pickle.dump(self.meta, file_handle)
        self.bm25_retriever.save(BM25_FILE)
        LOG.info("Saved FAISS index, metadata, and BM25 model to %s", settings.VECTOR_DIR)

    def load(self) -> bool:
        try:
            if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
                return False

            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as file_handle:
                self.meta = pickle.load(file_handle)

            if os.path.exists(BM25_FILE):
                self.bm25_retriever = BM25Retriever.load(BM25_FILE)
            else:
                self._refresh_bm25()

            self.dim = self.index.d
            LOG.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            return True
        except Exception as exc:
            LOG.exception("Failed to load index: %s", exc)
            return False
