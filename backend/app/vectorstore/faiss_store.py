import os
import faiss
import numpy as np
import pickle
from app.core.config import settings
from app.core.logger import LOG
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
# Constants
INDEX_FILE = os.path.join(settings.VECTOR_DIR, "faiss.index")
META_FILE = os.path.join(settings.VECTOR_DIR, "meta.pkl")
BM25_FILE = os.path.join(settings.VECTOR_DIR, "bm25.pkl")

# Helper for text processing
def preprocess_text(text: str) -> List[str]:
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in stop_words]

class BM25Retriever:
    def __init__(self, corpus: List[str] = None):
        self.corpus = corpus or []
        self.bm25 = None
        if self.corpus:
            self.fit(self.corpus)

    def fit(self, corpus: List[str]):
        self.corpus = corpus
        tokenized_corpus = [preprocess_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        tokenized_query = preprocess_text(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        results = []
        for i in top_k_indices:
            results.append({
                "score": doc_scores[i],
                "meta": {"text": self.corpus[i]},
                "id": i
            })
        return results

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)

class FaissStore:
    def __init__(self, dim: int, index_type: str = 'flat'):
        os.makedirs(settings.VECTOR_DIR, exist_ok=True)
        self.dim = dim
        self.index_type = index_type.lower()
        self.index = self._create_index(dim, self.index_type)
        self.meta = []
        self.bm25_retriever = BM25Retriever()
        
    def _create_index(self, dim: int, index_type: str):
        if index_type == 'ivf':
            # IVF requires a quantizer and number of clusters (nlist)
            nlist = 100  # Example value, should be tuned
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif index_type == 'hnsw':
            # HNSW is a graph-based index, good for speed
            m = 32  # Number of neighbors for each node, affects memory/speed trade-off
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        else: # flat
            index = faiss.IndexFlatIP(dim)
        return index

    def add(self, vectors: np.ndarray, metas: list):
        if vectors.size == 0:
            return
        
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(vectors)
        
        self.index.add(vectors)
        self.meta.extend(metas)

        # Update BM25 index with new text data
        new_texts = [m['text'] for m in metas if 'text' in m]
        if new_texts:
            all_texts = [m['text'] for m in self.meta if 'text' in m]
            self.bm25_retriever.fit(all_texts)

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

    def hybrid_search(self, query_text: str, q_vector: np.ndarray, top_k: int, alpha: float = 0.5):
        # Vector search
        vector_results = self.search(q_vector, top_k)
        
        # Keyword search
        bm25_results = self.bm25_retriever.search(query_text, top_k)

        # Combine results (simple re-ranking for now)
        # This is a naive combination. A better approach would be to use a reciprocal rank fusion
        combined_results = {}
        for res in vector_results:
            combined_results[res['id']] = {'vector_score': res['score'], 'meta': res['meta']}

        for res in bm25_results:
            if res['id'] in combined_results:
                combined_results[res['id']]['bm25_score'] = res['score']
            else:
                # This case is tricky as we don't have vector score.
                # For simplicity, we can ignore or assign a default.
                pass
        
        final_scores = []
        for id, scores in combined_results.items():
            vec_score = scores.get('vector_score', 0)
            bm25_score = scores.get('bm25_score', 0)
            # Weighted sum for ranking
            final_score = alpha * vec_score + (1 - alpha) * bm25_score
            final_scores.append((final_score, scores['meta'], id))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [{"score": score, "meta": meta, "id": id} for score, meta, id in final_scores[:top_k]]

    def save(self):
        index_path = os.path.join(settings.VECTOR_DIR, "faiss.index")
        meta_path = os.path.join(settings.VECTOR_DIR, "meta.pkl")
        bm25_path = os.path.join(settings.VECTOR_DIR, "bm25.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f)
        self.bm25_retriever.save(bm25_path)
        LOG.info("Saved FAISS index, meta, and BM25 model to %s", settings.VECTOR_DIR)

    def load(self):
        try:
            if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
                return False
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                self.meta = pickle.load(f)
            if os.path.exists(BM25_FILE):
                self.bm25_retriever = BM25Retriever.load(BM25_FILE)
            
            self.dim = self.index.d
            LOG.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            return True
        except Exception as e:
            LOG.exception("Failed to load index: %s", e)
            return False
