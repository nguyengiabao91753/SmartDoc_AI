from typing import Dict, List

import numpy as np
from langchain_core.documents import Document

from app.ai.retriever import get_retriever
from app.vectorstore.faiss_store import FaissStore

from .planner import RAGPlan


class RAGRetriever:
    """Retriever for vanilla RAG mode."""

    def __init__(self, store: FaissStore, embedding_service):
        self.store = store
        self.embedding_service = embedding_service

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

    def retrieve(self, plan: RAGPlan) -> List[Document]:
        q_vector = self._embed_query(plan.question)
        retriever = get_retriever(
            self.store,
            search_type=plan.search_type,
            top_k=plan.top_k,
            filters=self._build_filters(plan.document_id, plan.document_ids, plan.session_id),
        )
        results = retriever.retrieve(plan.question, q_vector)
        return self._results_to_documents(results)
