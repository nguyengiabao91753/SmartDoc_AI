from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from langchain_core.documents import Document

from app.rag.modes.rag.planner import RAGPlan
from app.rag.modes.rag.retriever import RAGRetriever

from .planner import SelfRAGPlan
from .utils import unique_preserve_order


class SelfRAGRetriever:
    def __init__(self, store, embedding_service):
        self.base_retriever = RAGRetriever(store=store, embedding_service=embedding_service)

    def retrieve(
        self,
        *,
        plan: SelfRAGPlan,
        query: str,
        top_k: int | None = None,
    ) -> List[Document]:
        docs = self.base_retriever.retrieve(self._to_rag_plan(plan, query=query, top_k=top_k))
        return self._annotate_documents(docs, query=query)

    def retrieve_many(
        self,
        *,
        plan: SelfRAGPlan,
        queries: Sequence[str],
        top_k: int | None = None,
    ) -> Dict[str, List[Document]]:
        per_query_docs: Dict[str, List[Document]] = {}
        for query in unique_preserve_order(list(queries)):
            per_query_docs[query] = self.retrieve(plan=plan, query=query, top_k=top_k)
        return per_query_docs

    def merge_documents(
        self,
        *document_groups: Iterable[Document],
        limit: int,
    ) -> List[Document]:
        merged: Dict[str, Document] = {}
        score_by_key: Dict[str, float] = {}
        hit_count_by_key: Dict[str, int] = {}
        queries_by_key: Dict[str, List[str]] = {}

        for docs in document_groups:
            for doc in docs:
                key = self._document_key(doc)
                base_score = self._document_score(doc)
                hit_count_by_key[key] = hit_count_by_key.get(key, 0) + 1
                score_by_key[key] = max(score_by_key.get(key, base_score), base_score)

                current = merged.get(key)
                if current is None:
                    merged[key] = doc
                    queries_by_key[key] = list((doc.metadata or {}).get("selfrag_queries") or [])
                else:
                    queries_by_key[key] = unique_preserve_order(
                        [*queries_by_key.get(key, []), *((doc.metadata or {}).get("selfrag_queries") or [])]
                    )

        ranked: List[Document] = []
        for key, doc in merged.items():
            metadata = dict(doc.metadata or {})
            metadata["selfrag_queries"] = queries_by_key.get(key, [])
            metadata["selfrag_support_count"] = hit_count_by_key.get(key, 1)
            metadata["selfrag_score"] = score_by_key.get(key, 0.0) + 0.05 * (hit_count_by_key.get(key, 1) - 1)
            ranked.append(Document(page_content=doc.page_content, metadata=metadata))

        ranked.sort(key=lambda item: float((item.metadata or {}).get("selfrag_score", 0.0)), reverse=True)
        return ranked[: max(1, limit)]

    def select_top_documents(self, documents: Sequence[Document], *, limit: int) -> List[Document]:
        if not documents:
            return []
        ranked = sorted(
            list(documents),
            key=lambda item: float((item.metadata or {}).get("selfrag_score", self._document_score(item))),
            reverse=True,
        )
        return ranked[: max(1, limit)]

    def _to_rag_plan(self, plan: SelfRAGPlan, *, query: str, top_k: int | None) -> RAGPlan:
        return RAGPlan(
            question=query,
            search_type=plan.search_type,
            top_k=max(1, int(top_k or plan.retrieval_top_k)),
            document_id=plan.document_id,
            document_ids=plan.document_ids,
            session_id=plan.session_id,
            llm_model=plan.llm_model,
        )

    def _annotate_documents(self, documents: List[Document], *, query: str) -> List[Document]:
        annotated: List[Document] = []
        for rank, doc in enumerate(documents, start=1):
            metadata = dict(doc.metadata or {})
            queries = unique_preserve_order([*((metadata.get("selfrag_queries") or [])), query])
            metadata["selfrag_queries"] = queries
            metadata["selfrag_rank"] = rank
            metadata["selfrag_score"] = self._document_score(doc)
            annotated.append(Document(page_content=doc.page_content, metadata=metadata))
        return annotated

    def _document_key(self, doc: Document) -> str:
        metadata = doc.metadata or {}
        source = str(metadata.get("source", "unknown"))
        document_id = str(metadata.get("document_id", ""))
        chunk = str(metadata.get("chunk", ""))
        page_start = str(metadata.get("page_start", ""))
        content_head = doc.page_content.strip()[:240]
        return "||".join([document_id, source, chunk, page_start, content_head])

    def _document_score(self, doc: Document) -> float:
        metadata = doc.metadata or {}
        for key in ("selfrag_score", "rerank_score", "retrieval_score"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0
