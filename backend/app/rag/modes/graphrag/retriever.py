from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from neo4j import GraphDatabase

from app.core.config import settings
from app.rag.modes.graphrag.planner import GraphRAGPlan
from app.core.logger import LOG

class GraphRAGRetriever:
    """
    Retriever chuyên dụng cho GraphRAG có tích hợp Re-ranking.
    """

    def __init__(self, store, embedding_service):
        self.store = store
        self.embedding_service = embedding_service

        # Thiết lập kết nối Neo4j (Lazy load)
        self._uri = settings.NEO4J_URI
        self._user = settings.NEO4J_USERNAME
        self._password = settings.NEO4J_PASSWORD
        self._driver = None

        # KHỞI TẠO MÔ HÌNH RE-RANKER (Chỉ load 1 lần vào RAM - CHUYỂN SANG LAZY LOAD)
        self._reranker = None

    @property
    def driver(self):
        if self._driver is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Dang ket noi Neo4j...")
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    @property
    def reranker(self):
        if self._reranker is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Dang lazy load CrossEncoder (BAAI/bge-reranker-base)...")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder('BAAI/bge-reranker-base')
            logger.info("Load CrossEncoder thanh cong.")
        return self._reranker

    def __del__(self):
        if hasattr(self, '_driver') and self._driver is not None:
            self._driver.close()

    def _embed_query(self, question: str) -> List[float]:
        q_vector = self.embedding_service.embed_query(question)
        q_vector = np.asarray(q_vector, dtype="float32")
        norm = np.linalg.norm(q_vector)
        if norm > 0:
            q_vector = q_vector / norm
        return [float(x) for x in q_vector]

    def retrieve(self, plan: GraphRAGPlan) -> List[Document]:
        LOG.info(f"[Retriever] Strategy: {plan.search_strategy}, Question: '{plan.question}'")
        if plan.search_strategy == 'global':
            docs = self._global_search(plan)
        else:
            docs = self._local_search(plan)
        LOG.info(f"[Retriever] Returned {len(docs)} document(s)")
        return docs

    def _local_search(self, plan: GraphRAGPlan) -> List[Document]:
        """
        CHIẾN LƯỢC CỤC BỘ: Vector Search + Graph Traversal + Cross-Encoder Re-ranking
        """
        query_vector = self._embed_query(plan.question)
        doc_filter = str(plan.document_id) if plan.document_id is not None else None

        results = []
        try:
            cypher_query = """
            CALL db.index.vector.queryNodes('entity_vector_index', 50, $question_vector)
            YIELD node AS startNode, score AS vector_score
            MATCH (startNode)-[r]-(neighbor:Entity)
            WHERE startNode.document_id = $doc_id OR $doc_id IS NULL OR startNode.document_id = 'global_knowledge_graph'
            RETURN startNode.name AS source,
                   type(r) AS relationship,
                   neighbor.name AS target,
                   r.description AS description,
                   vector_score
            ORDER BY vector_score DESC
            LIMIT 50
            """

            params = {
                "question_vector": query_vector,
                "doc_id": doc_filter
            }

            with self.driver.session() as session:
                results = session.run(cypher_query, **params).data()

        except Exception as e:
            LOG.warning(f"[Retriever] Vector index query failed: {e}. Using fallback search...")

        if not results:
            LOG.info("[Retriever] Vector search returned no results. Using fallback (keyword-based search)...")
            fallback_results = self._fallback_search(plan, doc_filter)

            if not fallback_results:
                return []

            documents = []
            pairs = []

            for row in fallback_results:
                context_str = f"Thực thể '{row['source']}' có quan hệ [{row['relationship']}] với '{row['target']}'. "
                if row.get('description'):
                    context_str += f"Mô tả: {row['description']}"

                pairs.append([plan.question, context_str])

                doc = Document(
                    page_content=context_str,
                    metadata={
                        "vector_score": row.get("score", 0.0),
                        "source": row["source"],
                        "target": row["target"]
                    }
                )
                documents.append(doc)

            documents.sort(key=lambda x: x.metadata.get("vector_score", 0), reverse=True)
            final_top_k = plan.top_k or 5
            return documents[:final_top_k]

        # Đóng gói tạm thành text để đưa vào Re-ranker
        documents = []
        pairs = []

        for row in results:
            context_str = f"Thực thể '{row['source']}' có quan hệ [{row['relationship']}] với '{row['target']}'. "
            if row.get('description'):
                context_str += f"Mô tả: {row['description']}"

            # Cặp [câu_hỏi, đoạn_ngữ_cảnh] dùng cho Cross-Encoder
            pairs.append([plan.question, context_str])

            doc = Document(
                page_content=context_str,
                metadata={
                    "vector_score": row["vector_score"],
                    "source": row["source"],
                    "target": row["target"]
                }
            )
            documents.append(doc)

        # Bước 2.2: Re-ranking bằng Cross-Encoder
        # Model sẽ trả về một mảng điểm số cho từng cặp
        rerank_scores = self.reranker.predict(pairs)

        # Cập nhật điểm mới vào metadata của Document
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(rerank_scores[i])

        # Bước 2.3: Sắp xếp lại dựa trên điểm rerank_score (từ cao xuống thấp)
        documents.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        # Trả về đúng số lượng top_k mà UI yêu cầu (thường là 5 hoặc 10)
        final_top_k = plan.top_k or 5
        return documents[:final_top_k]

    def _global_search(self, plan: GraphRAGPlan) -> List[Document]:
        doc_filter = str(plan.document_id) if plan.document_id is not None else None
        results = []

        try:
            cypher_query = """
            MATCH (c:Community)
            WHERE c.document_id = $doc_id OR c.document_id = 'global_knowledge_graph' OR $doc_id IS NULL
            RETURN c.local_id AS id, c.report AS report
            ORDER BY size(c.report) DESC
            """
            params = {"doc_id": doc_filter}
            with self.driver.session() as session:
                results = session.run(cypher_query, **params).data()
                LOG.info(f"[Retriever][Global] Neo4j raw result count: {len(results)}")
        except Exception as e:
            LOG.error(f"[Retriever][Global] Query failed: {e}")
            return []

        documents = []
        for row in results:
            if not row.get('report'):
                continue
            doc = Document(
                page_content=f"[Báo cáo Cộng đồng {row['id']}]:\n{row['report']}",
                metadata={"community_id": row["id"]}
            )
            documents.append(doc)

        # Lấy top_k báo cáo lớn nhất
        return documents[:plan.top_k] if plan.top_k else documents

    def _fallback_search(self, plan: GraphRAGPlan, doc_filter: str = None) -> List[Dict]:
        """
        Fallback search: Tìm kiếm entity dựa trên tên và description không dùng vector index.
        Lấy tất cả entities và relationships rồi lọc bằng keyword matching.
        """
        import re
        query_lower = plan.question.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        cypher_query = """
        MATCH (e:Entity)-[r]-(neighbor:Entity)
        WHERE $doc_id IS NULL OR e.document_id = $doc_id OR e.document_id = 'global_knowledge_graph'
        RETURN e.name AS source,
               type(r) AS relationship,
               neighbor.name AS target,
               e.description AS source_desc,
               r.description AS rel_desc
        LIMIT 100
        """

        params = {"doc_id": doc_filter}

        try:
            with self.driver.session() as session:
                results = session.run(cypher_query, **params).data()
        except Exception as e:
            LOG.error(f"[Retriever][Fallback] Query failed: {e}")
            return []

        scored_results = []
        for row in results:
            source = row.get('source', '')
            target = row.get('target', '')
            source_desc = row.get('source_desc', '')
            rel_desc = row.get('rel_desc', '')

            text_to_search = f"{source} {target} {source_desc} {rel_desc}".lower()
            text_words = set(re.findall(r'\w+', text_to_search))

            overlap = query_words.intersection(text_words)
            if overlap:
                scored_results.append({
                    'source': source,
                    'relationship': row.get('relationship'),
                    'target': target,
                    'description': f"{source_desc} {rel_desc}".strip(),
                    'score': len(overlap)
                })

        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:50]
