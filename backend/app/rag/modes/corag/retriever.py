from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document

from app.ai.retriever import get_retriever
from app.vectorstore.faiss_store import FaissStore

from .planner import CoRAGPlan


class CoRAGRetriever:
    """
    Retriever cho CoRAG.
    - Retrieve riêng cho từng sub-query.
    - Merge, dedupe, rerank kết quả theo RRF (Reciprocal Rank Fusion).
    """

    def __init__(self, store: FaissStore, embedding_service):
        self.store = store
        self.embedding_service = embedding_service

    def retrieve_all(self, plan: CoRAGPlan) -> Tuple[List[Document], Dict[str, List[Document]]]:
        """
        Trả về:
        - merged_docs: danh sách tài liệu đã merge/rerank từ tất cả sub-queries
        - per_query_docs: map từ sub-query -> docs riêng (dùng cho metadata/debug)
        """
        per_query_docs: Dict[str, List[Document]] = {}

        for sub_query in plan.sub_queries:
            docs = self._retrieve_one(sub_query, plan)
            per_query_docs[sub_query] = docs

        merged = self._rrf_merge(per_query_docs, top_n=plan.top_k * 2)
        return merged, per_query_docs

    def _retrieve_one(self, query: str, plan: CoRAGPlan) -> List[Document]:
        q_vector = self._embed_query(query)
        filters = self._build_filters(plan.document_id)
        retriever = get_retriever(
            self.store,
            search_type=plan.search_type,
            top_k=plan.top_k,
            filters=filters,
        )
        results = retriever.retrieve(query, q_vector)
        return self._results_to_documents(results)

    def _embed_query(self, question: str) -> np.ndarray:
        q_vector = self.embedding_service.embed_query(question)
        q_vector = np.asarray(q_vector, dtype="float32")
        norm = np.linalg.norm(q_vector)
        if norm > 0:
            q_vector = q_vector / norm
        return q_vector

    def _build_filters(self, document_id: int | None) -> Dict | None:
        if document_id is None:
            return None
        return {"document_id": document_id}

    def _results_to_documents(self, results: List[Dict]) -> List[Document]:
        documents: List[Document] = []
        for result in results:
            metadata = dict(result.get("meta", {}))
            page_content = metadata.pop("text", "")
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def _rrf_merge(
        self,
        per_query_docs: Dict[str, List[Document]],
        top_n: int,
        k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion: gộp nhiều ranked list thành 1 list chất lượng cao.
        Mỗi doc được score = sum(1 / (k + rank_i)) across all query lists.
        Dedupe theo page_content.
        """
        rrf_scores: Dict[str, float] = {}
        doc_by_key: Dict[str, Document] = {}

        for _query, docs in per_query_docs.items():
            for rank, doc in enumerate(docs, start=1):
                key = doc.page_content.strip()[:300]  # key duy nhất cho dedup
                if not key:
                    continue
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                if key not in doc_by_key:
                    doc_by_key[key] = doc

        # Sort theo RRF score giảm dần
        ranked_keys = sorted(rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True)
        return [doc_by_key[key] for key in ranked_keys[:top_n]]