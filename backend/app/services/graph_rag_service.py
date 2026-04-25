import os
import logging
import networkx as nx
from cdlib import algorithms
from typing import List, Dict, Any

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# Import các thành phần từ repo của bạn
from app.core.config import settings
from app.ai.llm import get_llm 
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService

LOG = logging.getLogger(__name__)

class GraphRAGService:
    def __init__(self):
            # 1. Khởi tạo Services phụ trợ TRƯỚC
            self.llm = get_llm(temperature=0)
            self.transformer = LLMGraphTransformer(llm=self.llm)
            self.doc_service = DocumentService()
            self.embedding_service = EmbeddingService()

            # 2. Kết nối Neo4j và Setup Indexes
            self.url = settings.NEO4J_URI
            self.user = settings.NEO4J_USERNAME
            self.password = settings.NEO4J_PASSWORD
            self.database = settings.NEO4J_DATABASE

            try:
                self.graph_db = Neo4jGraph(url=self.url, username=self.user, password=self.password, database=self.database)
                LOG.info("[GraphRAGService] ✅ Đã kết nối thành công tới Neo4j!")
                # Tự động thiết lập Vector Index khi khởi tạo
                # Lúc này self.embedding_service đã tồn tại nên sẽ không bị lỗi nữa
                self._setup_indexes()
            except Exception as e:
                LOG.error(f"[GraphRAGService] ❌ Lỗi kết nối Neo4j: {e}")
                self.graph_db = None

            # 2. Khởi tạo Services phụ trợ
            self.llm = get_llm(temperature=0)
            self.transformer = LLMGraphTransformer(llm=self.llm)
            self.doc_service = DocumentService()
            self.embedding_service = EmbeddingService()

    def _setup_indexes(self):
        """Tạo các Index cần thiết cho Graph RAG nếu chưa có."""
        dim = self.embedding_service.get_dimension()
        try:
            # Tạo index cho Entity (Local Search)
            self.graph_db.query(f"""
                CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                FOR (n:__Entity__) ON (n.embedding)
                OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine' }} }}
            """)
            # Tạo index cho Community (Global Search)
            self.graph_db.query(f"""
                CREATE VECTOR INDEX community_vector_index IF NOT EXISTS
                FOR (n:Community) ON (n.embedding)
                OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine' }} }}
            """)
            LOG.info("[GraphRAGService] ✅ Đã kiểm tra và thiết lập các Vector Indexes.")
        except Exception as e:
            LOG.warning(f"[GraphRAGService] ⚠️ Lỗi khi thiết lập Index: {e}")

    # --- PHẦN 1: XÂY DỰNG ĐỒ THỊ (INGESTION) ---
    def process_and_save_graph(self, file_path: str, session_id: int):
        """Pipeline: Đọc file -> Chunking (DocumentService) -> Extract -> Community -> Save."""
        lc_docs = self.doc_service.load_document(file_path) # Dùng chung chunking với RAG
        if not lc_docs: return

        # Trích xuất Đồ thị
        graph_docs = self.transformer.convert_to_graph_documents(lc_docs)
        
        # ---> GẮN SESSION_ID VÀO CÁC THỰC THỂ (ENTITIES) <---
        for doc in graph_docs:
            for node in doc.nodes:
                if not hasattr(node, "properties") or node.properties is None:
                    node.properties = {}
                node.properties["session_id"] = session_id
        
        # Phân cụm (Leiden)
        G = nx.Graph()
        for doc in graph_docs:
            for node in doc.nodes: G.add_node(node.id, type=node.type)
            for rel in doc.relationships: G.add_edge(rel.source.id, rel.target.id, label=rel.type)
        
        communities = []
        for component in nx.connected_components(G):
            sub = G.subgraph(component)
            if len(sub.nodes) > 1:
                try:
                    for c in algorithms.leiden(sub).communities: communities.append(list(c))
                except: communities.append(list(sub.nodes))
            else: communities.append(list(sub.nodes))

        # Tóm tắt cụm & Nhúng Vector cho báo cáo
        community_summaries = []
        for idx, community in enumerate(communities):
            sub = G.subgraph(community)
            desc = f"Entities: {', '.join(list(sub.nodes))}"
            summary = self.llm.invoke(f"Tóm tắt ngắn gọn các thực thể này:\n{desc}").content.strip()
            # Quan trọng: Tạo embedding cho tóm tắt để Global Search hoạt động
            vector = self.embedding_service.embed_query(summary)
            
            community_summaries.append({
                "id": f"{os.path.basename(file_path)}_{idx}",
                "summary": summary,
                "embedding": vector,
                "nodes": list(sub.nodes)
            })

        # Lưu vào Neo4j
        self.graph_db.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        
        # ---> LƯU SESSION_ID VÀO COMMUNITY <---
        cypher_comm = """
        UNWIND $data AS row
        MERGE (c:Community {id: row.id})
        SET c.report = row.summary, 
            c.embedding = row.embedding,
            c.session_id = $session_id
        WITH c, row.nodes AS entities
        UNWIND entities AS e_name
        MATCH (e:__Entity__ {id: e_name})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        self.graph_db.query(cypher_comm, params={"data": community_summaries, "session_id": session_id})

    # --- PHẦN 2: TRUY XUẤT (RETRIEVAL) ---
    def search_local(self, query: str, session_id: int, top_k=5) -> str:
        """Tìm kiếm thực thể lân cận (Local Search) có lọc theo đoạn chat."""
        query_vector = self.embedding_service.embed_query(query)
        cypher = """
        CALL db.index.vector.queryNodes('entity_vector_index', $top_k, $vector)
        YIELD node AS n, score
        WHERE n.session_id = $session_id
        MATCH (n)-[r]->(m)
        WHERE m.session_id = $session_id
        RETURN n.id + ' -- ' + type(r) + ' --> ' + m.id AS context
        LIMIT 10
        """
        # Tăng top_k ở vector search để phòng hờ kết quả bị rớt sau khi lọc bằng WHERE
        results = self.graph_db.query(cypher, params={"vector": query_vector, "top_k": top_k * 5, "session_id": session_id})
        return "\n".join([r['context'] for r in results])

    def search_global(self, query: str, session_id: int, top_k=3) -> str:
        """Tìm kiếm báo cáo tóm tắt cụm (Global Search) có lọc theo đoạn chat."""
        query_vector = self.embedding_service.embed_query(query)
        cypher = """
        CALL db.index.vector.queryNodes('community_vector_index', $top_k, $vector)
        YIELD node AS c, score
        WHERE c.session_id = $session_id
        RETURN c.report AS report
        """
        results = self.graph_db.query(cypher, params={"vector": query_vector, "top_k": top_k * 5, "session_id": session_id})
        return "\n\n".join([r['report'] for r in results])
