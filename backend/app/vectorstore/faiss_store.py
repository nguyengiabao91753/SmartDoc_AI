import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import underthesea
from underthesea import word_tokenize

from app.core.config import settings
from app.core.logger import LOG

INDEX_FILE = os.path.join(settings.VECTOR_DIR, "faiss.index")
META_FILE = os.path.join(settings.VECTOR_DIR, "meta.pkl")
BM25_FILE = os.path.join(settings.VECTOR_DIR, "bm25.pkl")

PREPROCESS_VERSION = "vi_underthesea_stopwords_v1"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _load_vietnamese_stopwords() -> set[str]:
    stopwords_path = Path(underthesea.__file__).resolve().parent / "datasets" / "stopwords" / "stopwords.txt"
    try:
        return {word.strip() for word in stopwords_path.read_text(encoding="utf-8").splitlines() if word.strip()}
    except (OSError, UnicodeError) as exc:
        LOG.warning("Unable to load Vietnamese stopwords from underthesea: %s", exc)
        return set()


VIETNAMESE_STOPWORDS = _load_vietnamese_stopwords()


def _tokenize_vietnamese(text: str) -> List[str]:
    normalized_text = unicodedata.normalize("NFC", text or "").lower()
    if not normalized_text.strip():
        return []

    try:
        normalized_text = word_tokenize(normalized_text, format="text")
    except Exception as exc:
        LOG.debug("Vietnamese word tokenization failed; using regex fallback: %s", exc)

    return TOKEN_PATTERN.findall(normalized_text)


def preprocess_text(text: str) -> List[str]:
    return [token for token in _tokenize_vietnamese(text) if token not in VIETNAMESE_STOPWORDS]


class BM25Retriever:
    def __init__(self, corpus: List[str] | None = None):
        self.corpus = corpus or []
        self.bm25: BM25Okapi | None = None
        self.preprocess_version = PREPROCESS_VERSION
        if self.corpus:
            self.fit(self.corpus)

    def fit(self, corpus: List[str]):
        self.corpus = corpus
        tokenized_corpus = [preprocess_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None
        self.preprocess_version = PREPROCESS_VERSION

    def search(self, query: str, top_k: int | None = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []

        tokenized_query = preprocess_text(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        matched_indices = np.flatnonzero(doc_scores > 0)
        if matched_indices.size == 0:
            return []

        limit = matched_indices.size if top_k is None else min(max(top_k, 0), matched_indices.size)
        if limit == 0:
            return []

        top_indices = matched_indices[np.argsort(doc_scores[matched_indices])[-limit:]][::-1]
        return [{"score": float(doc_scores[i]), "id": int(i)} for i in top_indices]

    def save(self, filepath: str):
        with open(filepath, "wb") as file_handle:
            pickle.dump(self, file_handle)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as file_handle:
            return pickle.load(file_handle)


class FaissStore:
    def __init__(self, dim: int, index_type: str = "hnsw"):
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
        texts = [meta.get("text", "") for meta in self.meta]
        if texts:
            self.bm25_retriever.fit(texts)
        else:
            self.bm25_retriever = BM25Retriever()

    def _reconstruct_vectors(self) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        total = min(int(self.index.ntotal), len(self.meta))
        for idx in range(total):
            vector = np.zeros((self.dim,), dtype="float32")
            self.index.reconstruct(idx, vector)
            vectors.append(vector)
        return vectors

    def _replace_contents(self, vectors: List[np.ndarray], metas: List[Dict[str, Any]]):
        self.index = self._create_index(self.dim, self.index_type)
        self.meta = list(metas)

        if vectors:
            matrix = np.asarray(vectors, dtype="float32")
            if self.index_type == "ivf" and not self.index.is_trained:
                self.index.train(matrix)
            self.index.add(matrix)

        self._refresh_bm25()

    @staticmethod
    def _normalize_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def remove_by_predicate(self, predicate: Callable[[Dict[str, Any]], bool]) -> int:
        """Remove vectors whose metadata matches predicate by rebuilding the FAISS index."""
        if self.index.ntotal == 0 or not self.meta:
            return 0

        vectors = self._reconstruct_vectors()
        kept_vectors: List[np.ndarray] = []
        kept_metas: List[Dict[str, Any]] = []
        removed_count = 0

        for vector, meta in zip(vectors, self.meta):
            if predicate(meta):
                removed_count += 1
                continue
            kept_vectors.append(vector)
            kept_metas.append(meta)

        if removed_count == 0:
            return 0

        self._replace_contents(kept_vectors, kept_metas)
        LOG.info("Removed %d vectors; total vectors: %d", removed_count, self.index.ntotal)
        return removed_count

    def remove_by_document_ids(self, document_ids: List[int]) -> int:
        normalized_ids = {int(doc_id) for doc_id in document_ids if doc_id is not None}
        if not normalized_ids:
            return 0

        return self.remove_by_predicate(
            lambda meta: self._normalize_int(meta.get("document_id")) in normalized_ids
        )

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
        if self.index.ntotal == 0 or top_k <= 0:
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
        if not self.meta or top_k <= 0:
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
                bm25_retriever = BM25Retriever.load(BM25_FILE)
                if getattr(bm25_retriever, "preprocess_version", None) == PREPROCESS_VERSION:
                    self.bm25_retriever = bm25_retriever
                else:
                    LOG.info("Rebuilding BM25 cache with Vietnamese preprocessing")
                    self._refresh_bm25()
            else:
                self._refresh_bm25()

            self.dim = self.index.d
            LOG.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            return True
        except Exception as exc:
            LOG.exception("Failed to load index: %s", exc)
            return False
