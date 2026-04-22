from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

import numpy as np
from langchain_core.documents import Document
from neo4j import GraphDatabase

from app.ai.retriever import get_retriever
from app.core.config import settings
from app.core.logger import LOG
from app.rag.modes.graphrag.planner import GraphRAGPlan


class GraphRAGRetriever:
    """
    Retriever for GraphRAG.

    Improvements:
    - Better local retrieval quality (relation semantics + rerank + dedupe)
    - Semantic global retrieval over community reports
    - Uses search_type (vector/hybrid) to optionally blend FAISS text evidence
    """

    LOCAL_CANDIDATE_MULTIPLIER = 8
    GLOBAL_CANDIDATE_MULTIPLIER = 6
    MAX_LOCAL_CANDIDATES = 80
    MAX_GLOBAL_CANDIDATES = 40
    MAX_TEXT_SUPPORT = 12

    def __init__(self, store, embedding_service):
        self.store = store
        self.embedding_service = embedding_service

        self._uri = settings.NEO4J_URI
        self._user = settings.NEO4J_USERNAME
        self._password = settings.NEO4J_PASSWORD
        self._driver = None
        self._reranker = None

    @property
    def driver(self):
        if self._driver is None:
            LOG.info("[Retriever] Connecting to Neo4j...")
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    @property
    def reranker(self):
        if self._reranker is None:
            LOG.info("[Retriever] Lazy-loading reranker model BAAI/bge-reranker-base...")
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder("BAAI/bge-reranker-base")
        return self._reranker

    def __del__(self):
        if getattr(self, "_driver", None) is not None:
            self._driver.close()

    def _embed_query(self, question: str) -> np.ndarray:
        q_vector = self.embedding_service.embed_query(question)
        q_vector = np.asarray(q_vector, dtype="float32")
        norm = np.linalg.norm(q_vector)
        if norm > 0:
            q_vector = q_vector / norm
        return q_vector

    @staticmethod
    def _to_float_list(vector: np.ndarray) -> List[float]:
        return [float(x) for x in vector.astype("float32").tolist()]

    @staticmethod
    def _build_faiss_filters(plan: GraphRAGPlan) -> Dict[str, Any] | None:
        normalized_ids = [int(doc_id) for doc_id in (plan.document_ids or []) if doc_id is not None]
        if not normalized_ids and plan.document_id is not None:
            normalized_ids = [int(plan.document_id)]

        if normalized_ids:
            unique_ids = sorted(set(normalized_ids))
            if len(unique_ids) == 1:
                return {"document_id": unique_ids[0]}
            return {"document_id": {"$in": unique_ids}}

        if plan.session_id is not None:
            return {"session_id": int(plan.session_id)}

        return None

    @staticmethod
    def _normalize_doc_filters(plan: GraphRAGPlan) -> List[str]:
        normalized_ids = [str(int(doc_id)) for doc_id in (plan.document_ids or []) if doc_id is not None]
        if not normalized_ids and plan.document_id is not None:
            normalized_ids = [str(int(plan.document_id))]
        # Preserve deterministic order and remove duplicates.
        return sorted(set(normalized_ids))

    def retrieve(self, plan: GraphRAGPlan) -> List[Document]:
        LOG.info(
            "[Retriever] Strategy=%s, SearchType=%s, Question='%s'",
            plan.search_strategy,
            plan.search_type,
            plan.question,
        )

        query_vector = self._embed_query(plan.question)
        if plan.search_strategy == "global":
            docs = self._global_search(plan, query_vector)
        else:
            docs = self._local_search(plan, query_vector)

        LOG.info("[Retriever] Returned %d document(s)", len(docs))
        return docs

    def _local_search(self, plan: GraphRAGPlan, query_vector: np.ndarray) -> List[Document]:
        doc_filters = self._normalize_doc_filters(plan)
        top_k = plan.top_k or 5
        candidate_k = min(
            self.MAX_LOCAL_CANDIDATES,
            max(top_k * self.LOCAL_CANDIDATE_MULTIPLIER, 30),
        )

        rows = self._run_local_vector_query(
            doc_filters=doc_filters,
            query_vector=query_vector,
            candidate_k=candidate_k,
        )
        if not rows:
            LOG.info("[Retriever][Local] Vector graph search returned no rows, using lexical fallback.")
            rows = self._fallback_search(plan=plan, doc_filters=doc_filters, candidate_k=candidate_k)

        graph_docs = self._rows_to_graph_documents(question=plan.question, rows=rows)
        graph_docs = self._rerank_graph_documents(plan.question, graph_docs)

        text_docs: List[Document] = []
        if (plan.search_type or "vector").lower() in {"hybrid", "keyword"}:
            text_docs = self._retrieve_text_support(plan, query_vector)

        return self._merge_local_documents(graph_docs, text_docs, limit=top_k)

    def _run_local_vector_query(
        self,
        *,
        doc_filters: List[str],
        query_vector: np.ndarray,
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        cypher_query = """
        CALL db.index.vector.queryNodes('entity_vector_index', $candidate_k, $question_vector)
        YIELD node AS startNode, score AS vector_score
        WHERE (
            size($doc_ids) = 0
            OR startNode.document_id IN $doc_ids
            OR any(doc_id IN coalesce(startNode.document_ids, []) WHERE doc_id IN $doc_ids)
        )
        MATCH (startNode)-[r]-(neighbor:Entity)
        WHERE (
            size($doc_ids) = 0
            OR neighbor.document_id IN $doc_ids
            OR any(doc_id IN coalesce(neighbor.document_ids, []) WHERE doc_id IN $doc_ids)
        )
        RETURN startNode.name AS source,
               coalesce(startNode.original_name, startNode.name) AS source_display,
               coalesce(r.type, type(r)) AS relationship,
               neighbor.name AS target,
               coalesce(neighbor.original_name, neighbor.name) AS target_display,
               coalesce(r.description, '') AS rel_description,
               coalesce(startNode.description, '') AS source_description,
               coalesce(neighbor.description, '') AS target_description,
               vector_score
        ORDER BY vector_score DESC
        LIMIT $candidate_k
        """

        params = {
            "question_vector": self._to_float_list(query_vector),
            "doc_ids": doc_filters,
            "candidate_k": candidate_k,
        }

        try:
            with self.driver.session() as session:
                return session.run(cypher_query, **params).data()
        except Exception as exc:
            LOG.warning("[Retriever][Local] Vector index query failed: %s", exc)
            return []

    def _rows_to_graph_documents(self, *, question: str, rows: List[Dict[str, Any]]) -> List[Document]:
        if not rows:
            return []

        deduped: Dict[tuple[str, str, str], Document] = {}

        for row in rows:
            source = str(row.get("source") or "").strip()
            target = str(row.get("target") or "").strip()
            if not source or not target:
                continue

            relationship = str(row.get("relationship") or "RELATED").strip() or "RELATED"
            source_display = str(row.get("source_display") or source).strip()
            target_display = str(row.get("target_display") or target).strip()
            rel_description = str(row.get("rel_description") or "").strip()
            source_description = str(row.get("source_description") or "").strip()
            target_description = str(row.get("target_description") or "").strip()

            semantic_score = float(row.get("vector_score", row.get("score", 0.0)) or 0.0)
            lexical_score = self._keyword_overlap_score(
                question,
                " ".join(
                    [
                        source_display,
                        target_display,
                        relationship,
                        rel_description,
                        source_description,
                        target_description,
                    ]
                ),
            )
            retrieval_score = (0.7 * semantic_score) + (0.3 * lexical_score)

            context = self._build_graph_context(
                source=source_display,
                relationship=relationship,
                target=target_display,
                rel_description=rel_description,
                source_description=source_description,
                target_description=target_description,
            )

            metadata = {
                "source_type": "graph",
                "source": source_display,
                "target": target_display,
                "relationship": relationship,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "retrieval_score": retrieval_score,
            }
            doc = Document(page_content=context, metadata=metadata)

            key = tuple(sorted((source, target)) + [relationship])
            previous = deduped.get(key)
            if previous is None:
                deduped[key] = doc
                continue

            prev_score = float(previous.metadata.get("retrieval_score", 0.0))
            if retrieval_score > prev_score:
                deduped[key] = doc

        docs = list(deduped.values())
        docs.sort(key=lambda d: float(d.metadata.get("retrieval_score", 0.0)), reverse=True)
        return docs

    @staticmethod
    def _build_graph_context(
        *,
        source: str,
        relationship: str,
        target: str,
        rel_description: str,
        source_description: str,
        target_description: str,
    ) -> str:
        parts = [f"Entity '{source}' has relation [{relationship}] with '{target}'."]
        if rel_description:
            parts.append(f"Relation detail: {rel_description}.")
        if source_description:
            parts.append(f"About {source}: {source_description}.")
        if target_description:
            parts.append(f"About {target}: {target_description}.")
        return " ".join(parts)

    def _rerank_graph_documents(self, question: str, graph_docs: List[Document]) -> List[Document]:
        if len(graph_docs) <= 1:
            return graph_docs

        pairs = [[question, doc.page_content] for doc in graph_docs]
        try:
            rerank_scores = self.reranker.predict(pairs)
            for idx, doc in enumerate(graph_docs):
                doc.metadata["rerank_score"] = float(rerank_scores[idx])
            graph_docs.sort(key=lambda d: float(d.metadata.get("rerank_score", 0.0)), reverse=True)
            return graph_docs
        except Exception as exc:
            LOG.warning("[Retriever][Local] Cross-encoder rerank failed: %s", exc)
            graph_docs.sort(key=lambda d: float(d.metadata.get("retrieval_score", 0.0)), reverse=True)
            return graph_docs

    def _retrieve_text_support(self, plan: GraphRAGPlan, query_vector: np.ndarray) -> List[Document]:
        top_k = max(plan.top_k or 5, 5)
        try:
            retriever = get_retriever(
                self.store,
                search_type=plan.search_type,
                top_k=top_k,
                filters=self._build_faiss_filters(plan),
            )
            results = retriever.retrieve(plan.question, query_vector, top_k=top_k)
        except Exception as exc:
            LOG.warning("[Retriever][Local] FAISS support retrieval failed: %s", exc)
            return []

        documents = self._results_to_text_documents(results, source_type="text")
        documents.sort(key=lambda d: float(d.metadata.get("retrieval_score", 0.0)), reverse=True)
        return documents

    def _results_to_text_documents(self, results: List[Dict[str, Any]], *, source_type: str) -> List[Document]:
        documents: List[Document] = []
        seen = set()

        for result in results:
            meta = dict(result.get("meta", {}))
            text = str(meta.pop("text", "")).strip()
            if not text:
                continue

            key = text[:240]
            if key in seen:
                continue
            seen.add(key)

            doc = Document(
                page_content=f"[Text evidence] {text}",
                metadata={
                    **meta,
                    "source_type": source_type,
                    "retrieval_score": float(result.get("score", 0.0)),
                },
            )
            documents.append(doc)

        return documents

    def _merge_local_documents(
        self,
        graph_docs: List[Document],
        text_docs: List[Document],
        *,
        limit: int,
    ) -> List[Document]:
        if limit <= 0:
            return []
        if not text_docs:
            return graph_docs[:limit]
        if not graph_docs:
            return text_docs[:limit]

        merged: List[Document] = []
        seen = set()
        graph_idx = 0
        text_idx = 0

        while len(merged) < limit and (graph_idx < len(graph_docs) or text_idx < len(text_docs)):
            # Prefer graph evidence first: 2 graph docs then 1 text doc.
            for _ in range(2):
                if len(merged) >= limit or graph_idx >= len(graph_docs):
                    break
                doc = graph_docs[graph_idx]
                graph_idx += 1
                key = doc.page_content[:240]
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)

            if len(merged) >= limit or text_idx >= len(text_docs):
                continue

            doc = text_docs[text_idx]
            text_idx += 1
            key = doc.page_content[:240]
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)

        return merged[:limit]

    def _global_search(self, plan: GraphRAGPlan, query_vector: np.ndarray) -> List[Document]:
        doc_filters = self._normalize_doc_filters(plan)
        top_k = plan.top_k or 5
        candidate_k = min(
            self.MAX_GLOBAL_CANDIDATES,
            max(top_k * self.GLOBAL_CANDIDATE_MULTIPLIER, 20),
        )

        rows = self._run_global_vector_query(
            doc_filters=doc_filters,
            query_vector=query_vector,
            candidate_k=candidate_k,
        )
        if not rows:
            LOG.info("[Retriever][Global] Vector community search unavailable, using fallback query.")
            rows = self._run_global_fallback_query(doc_filters=doc_filters, candidate_k=candidate_k)

        community_docs = self._rows_to_global_documents(question=plan.question, rows=rows)
        text_docs = self._retrieve_global_text_support(plan, query_vector)
        return self._merge_global_documents(
            plan=plan,
            community_docs=community_docs,
            text_docs=text_docs,
            limit=top_k,
        )

    def _retrieve_global_text_support(self, plan: GraphRAGPlan, query_vector: np.ndarray) -> List[Document]:
        top_k = min(self.MAX_TEXT_SUPPORT, max((plan.top_k or 5) * 2, 6))
        search_type = "hybrid" if (plan.search_type or "vector").lower() in {"hybrid", "keyword"} else "vector"

        text_docs: List[Document] = []
        try:
            retriever = get_retriever(
                self.store,
                search_type=search_type,
                top_k=top_k,
                filters=self._build_faiss_filters(plan),
            )
            results = retriever.retrieve(plan.question, query_vector, top_k=top_k)
            text_docs = self._results_to_text_documents(results, source_type="text_global")
        except Exception as exc:
            LOG.warning("[Retriever][Global] FAISS support retrieval failed: %s", exc)

        # For broad summary questions, prepend head chunks to anchor topic.
        if self._is_overview_question(plan.question):
            doc_ids = [int(doc_id) for doc_id in (plan.document_ids or []) if doc_id is not None]
            if not doc_ids and plan.document_id is not None:
                doc_ids = [int(plan.document_id)]
            head_docs: List[Document] = []
            for doc_id in doc_ids[:4]:
                head_docs.extend(self._collect_document_head_chunks(doc_id, limit=2))
            if head_docs:
                text_docs = head_docs + text_docs

        deduped: List[Document] = []
        seen = set()
        for doc in text_docs:
            key = doc.page_content[:260]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)

        deduped.sort(key=lambda d: float(d.metadata.get("retrieval_score", 0.0)), reverse=True)
        return deduped[:top_k]

    def _collect_document_head_chunks(self, document_id: int, limit: int = 4) -> List[Document]:
        metas = getattr(self.store, "meta", []) or []
        candidates = []

        for meta in metas:
            if meta.get("document_id") != document_id:
                continue
            text = str(meta.get("text", "")).strip()
            if not text:
                continue
            chunk = int(meta.get("chunk", 0) or 0)
            page_start = int(meta.get("page_start", 0) or 0)
            candidates.append((chunk, page_start, meta, text))

        candidates.sort(key=lambda item: (item[0], item[1]))
        docs: List[Document] = []
        for idx, (_, _, meta, text) in enumerate(candidates[:limit]):
            # Force high score so global summarization sees document opening context.
            base_score = 1.0 - (idx * 0.03)
            docs.append(
                Document(
                    page_content=f"[Text evidence] {text}",
                    metadata={
                        **meta,
                        "source_type": "text_global",
                        "retrieval_score": max(0.7, base_score),
                    },
                )
            )
        return docs

    def _merge_global_documents(
        self,
        *,
        plan: GraphRAGPlan,
        community_docs: List[Document],
        text_docs: List[Document],
        limit: int,
    ) -> List[Document]:
        if limit <= 0:
            return []
        if not community_docs and not text_docs:
            return []

        prefer_text = self._is_overview_question(plan.question)
        merged: List[Document] = []
        seen = set()

        text_idx = 0
        community_idx = 0
        while len(merged) < limit and (text_idx < len(text_docs) or community_idx < len(community_docs)):
            take_text = prefer_text and text_idx < len(text_docs)
            take_community = community_idx < len(community_docs)

            if take_text:
                doc = text_docs[text_idx]
                text_idx += 1
                key = doc.page_content[:260]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
                    if len(merged) >= limit:
                        break

            if take_community:
                doc = community_docs[community_idx]
                community_idx += 1
                key = doc.page_content[:260]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
                    if len(merged) >= limit:
                        break

            if not prefer_text and text_idx < len(text_docs):
                doc = text_docs[text_idx]
                text_idx += 1
                key = doc.page_content[:260]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        return merged[:limit]

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        raw = (question or "").strip().lower()
        normalized = unicodedata.normalize("NFD", raw)
        q = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        patterns = [
            r"\btai lieu.*noi ve gi\b",
            r"\bnoi ve gi\b",
            r"\btom tat\b",
            r"\btong quan\b",
            r"\boverview\b",
            r"\bsummary\b",
            r"\bmain idea\b",
            r"\bwhat is.*about\b",
        ]
        return any(re.search(pattern, q) for pattern in patterns)

    def _run_global_vector_query(
        self,
        *,
        doc_filters: List[str],
        query_vector: np.ndarray,
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        cypher_query = """
        CALL db.index.vector.queryNodes('community_vector_index', $candidate_k, $question_vector)
        YIELD node AS c, score AS vector_score
        WHERE c:Community
          AND (
              size($doc_ids) = 0
              OR c.document_id IN $doc_ids
              OR any(doc_id IN coalesce(c.document_ids, []) WHERE doc_id IN $doc_ids)
          )
        RETURN c.local_id AS id,
               c.report AS report,
               c.document_id AS document_id,
               vector_score
        ORDER BY vector_score DESC
        LIMIT $candidate_k
        """

        params = {
            "question_vector": self._to_float_list(query_vector),
            "doc_ids": doc_filters,
            "candidate_k": candidate_k,
        }

        try:
            with self.driver.session() as session:
                rows = session.run(cypher_query, **params).data()
                LOG.info("[Retriever][Global] Vector result count: %d", len(rows))
                return rows
        except Exception as exc:
            LOG.warning("[Retriever][Global] Community vector query failed: %s", exc)
            return []

    def _run_global_fallback_query(self, *, doc_filters: List[str], candidate_k: int) -> List[Dict[str, Any]]:
        cypher_query = """
        MATCH (c:Community)
        WHERE (
            size($doc_ids) = 0
            OR c.document_id IN $doc_ids
            OR any(doc_id IN coalesce(c.document_ids, []) WHERE doc_id IN $doc_ids)
        )
        RETURN c.local_id AS id,
               c.report AS report,
               c.document_id AS document_id,
               0.0 AS vector_score
        LIMIT $candidate_k
        """
        params = {"doc_ids": doc_filters, "candidate_k": candidate_k}

        try:
            with self.driver.session() as session:
                rows = session.run(cypher_query, **params).data()
                LOG.info("[Retriever][Global] Fallback result count: %d", len(rows))
                return rows
        except Exception as exc:
            LOG.error("[Retriever][Global] Fallback query failed: %s", exc)
            return []

    def _rows_to_global_documents(self, *, question: str, rows: List[Dict[str, Any]]) -> List[Document]:
        documents: List[Document] = []
        seen = set()

        for row in rows:
            report = str(row.get("report") or "").strip()
            if not report:
                continue

            community_id = row.get("id")
            key = f"{community_id}:{report[:160]}"
            if key in seen:
                continue
            seen.add(key)

            semantic_score = float(row.get("vector_score", 0.0) or 0.0)
            lexical_score = self._keyword_overlap_score(question, report)
            retrieval_score = (0.7 * semantic_score) + (0.3 * lexical_score)

            doc = Document(
                page_content=f"[Community {community_id}] {report}",
                metadata={
                    "source_type": "community",
                    "community_id": community_id,
                    "document_id": row.get("document_id"),
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "retrieval_score": retrieval_score,
                },
            )
            documents.append(doc)

        documents.sort(key=lambda d: float(d.metadata.get("retrieval_score", 0.0)), reverse=True)
        return documents

    def _fallback_search(
        self,
        *,
        plan: GraphRAGPlan,
        doc_filters: List[str],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        cypher_query = """
        MATCH (e:Entity)-[r]-(neighbor:Entity)
        WHERE (
            size($doc_ids) = 0
            OR e.document_id IN $doc_ids
            OR any(doc_id IN coalesce(e.document_ids, []) WHERE doc_id IN $doc_ids)
        )
        RETURN e.name AS source,
               coalesce(e.original_name, e.name) AS source_display,
               coalesce(r.type, type(r)) AS relationship,
               neighbor.name AS target,
               coalesce(neighbor.original_name, neighbor.name) AS target_display,
               coalesce(r.description, '') AS rel_description,
               coalesce(e.description, '') AS source_description,
               coalesce(neighbor.description, '') AS target_description
        LIMIT $raw_limit
        """

        params = {
            "doc_ids": doc_filters,
            "raw_limit": candidate_k * 4,
        }

        try:
            with self.driver.session() as session:
                rows = session.run(cypher_query, **params).data()
        except Exception as exc:
            LOG.error("[Retriever][Fallback] Query failed: %s", exc)
            return []

        scored_rows: List[Dict[str, Any]] = []
        for row in rows:
            content = " ".join(
                [
                    str(row.get("source_display") or ""),
                    str(row.get("target_display") or ""),
                    str(row.get("relationship") or ""),
                    str(row.get("rel_description") or ""),
                    str(row.get("source_description") or ""),
                    str(row.get("target_description") or ""),
                ]
            )
            score = self._keyword_overlap_score(plan.question, content)
            if score <= 0:
                continue
            scored_rows.append({**row, "score": score})

        scored_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return scored_rows[:candidate_k]

    @staticmethod
    def _keyword_overlap_score(query: str, text: str) -> float:
        query_tokens = set(re.findall(r"\w+", (query or "").lower()))
        text_tokens = set(re.findall(r"\w+", (text or "").lower()))

        query_tokens = {token for token in query_tokens if len(token) >= 3}
        text_tokens = {token for token in text_tokens if len(token) >= 3}

        if not query_tokens or not text_tokens:
            return 0.0

        overlap = len(query_tokens.intersection(text_tokens))
        return overlap / max(1, len(query_tokens))
