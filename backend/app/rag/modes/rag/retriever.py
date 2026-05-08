from typing import Any, Dict, List

import numpy as np
from langchain_core.documents import Document

from app.ai.retriever import get_retriever
from app.core.logger import LOG
from app.vectorstore.faiss_store import FaissStore

from .planner import RAGPlan


class RAGRetriever:
    """Retriever for vanilla RAG mode."""

    RERANK_CANDIDATE_MULTIPLIER = 4
    MAX_RERANK_CANDIDATES = 40

    def __init__(self, store: FaissStore, embedding_service):
        self.store = store
        self.embedding_service = embedding_service
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            LOG.info("[RAGRetriever] Lazy-loading reranker model BAAI/bge-reranker-base...")
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder("BAAI/bge-reranker-base")
        return self._reranker

    def _build_filters(
        self,
        document_id: int | None,
        document_ids: List[int] | None,
        session_id: int | None,
    ) -> Dict | None:
        normalized_ids = [int(doc_id) for doc_id in (document_ids or []) if doc_id is not None]
        if not normalized_ids and document_id is not None:
            normalized_ids = [int(document_id)]

        if normalized_ids:
            unique_ids = sorted(set(normalized_ids))
            if len(unique_ids) == 1:
                return {"document_id": unique_ids[0]}
            return {"document_id": {"$in": unique_ids}}

        if session_id is not None:
            return {"session_id": int(session_id)}

        return None

    def _embed_query(self, question: str) -> np.ndarray:
        q_vector = self.embedding_service.embed_query(question)
        q_vector = np.asarray(q_vector, dtype="float32")
        norm = np.linalg.norm(q_vector)
        if norm > 0:
            q_vector = q_vector / norm
        return q_vector

    def _results_to_documents(self, results: List[Dict]) -> List[Document]:
        documents: List[Document] = []
        for result in results:
            metadata = dict(result.get("meta", {}))
            page_content = metadata.pop("text", "")
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def _candidate_count(self, top_k: int) -> int:
        return min(
            self.MAX_RERANK_CANDIDATES,
            max(top_k, top_k * self.RERANK_CANDIDATE_MULTIPLIER),
        )

    def _rerank_results(self, question: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if len(results) <= 1:
            return results[:top_k]

        pairs = [[question, str(result.get("meta", {}).get("text", ""))] for result in results]
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as exc:
            LOG.warning("[RAGRetriever] Cross-encoder rerank failed: %s", exc)
            return results[:top_k]

        reranked: List[Dict[str, Any]] = []
        for result, rerank_score in zip(results, rerank_scores):
            metadata = dict(result.get("meta", {}))
            metadata["retrieval_score"] = float(result.get("score", 0.0))
            metadata["rerank_score"] = float(rerank_score)

            reranked_result = dict(result)
            reranked_result["score"] = float(rerank_score)
            reranked_result["meta"] = metadata
            reranked.append(reranked_result)

        reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return reranked[:top_k]

    def retrieve(self, plan: RAGPlan) -> List[Document]:
        q_vector = self._embed_query(plan.question)
        top_k = max(int(plan.top_k), 1)
        candidate_k = self._candidate_count(top_k)
        retriever = get_retriever(
            self.store,
            search_type=plan.search_type,
            top_k=candidate_k,
            filters=self._build_filters(plan.document_id, plan.document_ids, plan.session_id),
        )
        results = retriever.retrieve(plan.question, q_vector, top_k=candidate_k)
        reranked_results = self._rerank_results(plan.question, results, top_k)
        return self._results_to_documents(reranked_results)
