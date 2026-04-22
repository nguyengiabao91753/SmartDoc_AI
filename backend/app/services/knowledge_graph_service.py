import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
import numpy as np

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.core.logger import LOG

class KnowledgeGraphService:
    """Service quản lý Knowledge Graph trên Neo4j tích hợp Vector Index."""

    def __init__(self, embedding_service=None):
        # Thông số kết nối lấy từ settings (bạn hãy thêm vào config.py)
        url = settings.NEO4J_URI
        user = settings.NEO4J_USERNAME
        password = settings.NEO4J_PASSWORD
        
        self.driver = GraphDatabase.driver(
            url, 
            auth=(user, password),
            connection_timeout=3.0,
            max_connection_lifetime=60.0
        )
        # Tái sử dụng EmbeddingService đã có sẵn thay vì load lại model
        self.embedding_service = embedding_service or EmbeddingService()
        self.dim = self.embedding_service.get_dimension()
        
        self.is_online = True # Giả định ban đầu là có mạng
        
        # Khởi tạo Vector Index trên Neo4j (Chạy nền để không làm đơ UI)
        import threading
        threading.Thread(target=self._setup_vector_index, daemon=True).start()

    def close(self):
        self.driver.close()

    def _setup_vector_index(self):
        LOG.info("[KnowledgeGraphService] Bắt đầu tạo Vector Index (nếu chưa tồn tại)…")
        """Tạo Vector Index trên Neo4j để tìm kiếm thực thể theo ngữ nghĩa."""
        try:
            with self.driver.session() as session:
                result = session.run("SHOW INDEXES")
                indexes = [record["name"] for record in result]
                LOG.info(f"[KnowledgeGraphService] Các index hiện có: {indexes}")

                if 'entity_vector_index' not in indexes:
                    session.run(f"""
                        CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                        FOR (n:Entity) ON (n.embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: {self.dim},
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}
                    """)
                    LOG.info("[KnowledgeGraphService] Đã tạo vector index mới")
                else:
                    LOG.info("[KnowledgeGraphService] Vector index đã tồn tại")

                if 'entity_doc_id_index' not in indexes:
                    session.run("CREATE INDEX entity_doc_id_index IF NOT EXISTS FOR (n:Entity) ON (n.document_id)")
                    LOG.info("[KnowledgeGraphService] Đã tạo b-tree index cho document_id")
                
                result = session.run("SHOW INDEXES")
                indexes = [record["name"] for record in result]
                LOG.info(f"[KnowledgeGraphService] Các index sau khi tạo: {indexes}")
        except Exception as e:
            LOG.warning(f"Lỗi khi cấu hình Neo4j Vector Index: {e}")
            LOG.warning("Vector search sẽ sử dụng fallback (tìm kiếm không có vector index)")

    def check_if_graph_exists(self) -> bool:
        """Kiểm tra xem đã có dữ liệu thực thể nào trong Neo4j chưa."""
        if not self.is_online:
            return False
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN count(n) > 0 AS exists")
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            LOG.error(f"Lỗi khi kiểm tra sự tồn tại của đồ thị: {e}")
            return False

    def get_indexed_document_ids(self) -> List[str]:
        """Lấy danh sách các document_id đã được đánh chỉ mục trong Neo4j (Dùng cho get_new_files)."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN DISTINCT n.document_id AS doc_id")
                return [record["doc_id"] for record in result if record["doc_id"]]
        except Exception as e:
            LOG.error(f"Lỗi khi lấy danh sách document_id từ Neo4j: {e}")
            return []

    def is_document_indexed(self, document_id: str) -> bool:
        """Kiểm tra O(1) xem một tài liệu cụ thể đã có trong đồ thị chưa."""
        if not self.is_online:
            return False
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n:Entity {document_id: $doc_id}) RETURN count(n) > 0 AS exists",
                    doc_id=str(document_id)
                )
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            LOG.error(f"Lỗi khi kiểm tra document_id trong Neo4j: {e}")
            return False

    def upsert_graph(self, document_id: str, graph_data: Dict[str, Any]):
        """Đẩy dữ liệu thực thể và mối quan hệ vào Neo4j bằng kỹ thuật Batching và UNWIND."""
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])

        if not entities and not relationships:
            return

        # 1. Chuẩn bị dữ liệu thực thể (Nhúng vector theo lô nếu có thể)
        # Ở đây ta dùng ThreadPool để nhúng vector song song cho nhanh
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        entity_payloads = []
        
        def prepare_entity(ent):
            name = ent["name"]
            desc = ent.get("description", "")
            label = ent.get("type", "Unknown")
            emb = self.embedding_service.embed_query(f"{name}: {desc}")
            vector = [float(x) for x in emb]
            return {
                "name": name,
                "label": label,
                "description": desc,
                "embedding": vector,
                "document_id": document_id
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            entity_payloads = list(executor.map(prepare_entity, entities))

        # 2. Thực hiện Batch Upsert vào Neo4j
        with self.driver.session() as session:
            # Batch cho Entities
            session.run("""
                UNWIND $batch AS data
                MERGE (e:Entity {name: data.name})
                SET e.type = data.label,
                    e.description = data.description,
                    e.embedding = data.embedding,
                    e.document_id = data.document_id
            """, batch=entity_payloads)

            # Batch cho Relationships
            rel_payloads = []
            for rel in relationships:
                rel_payloads.append({
                    "source": rel["source"],
                    "target": rel["target"],
                    "relation": rel["relation"],
                    "description": rel.get("description", ""),
                    "document_id": document_id
                })

            if rel_payloads:
                session.run("""
                    UNWIND $batch AS data
                    MATCH (a:Entity {name: data.source})
                    MATCH (b:Entity {name: data.target})
                    MERGE (a)-[r:RELATED {type: data.relation}]->(b)
                    SET r.description = data.description,
                        r.document_id = data.document_id
                """, batch=rel_payloads)
        
        LOG.info(f"Đã cập nhật xong {len(entities)} thực thể và {len(relationships)} mối quan hệ vào Neo4j (Batch mode).")

    @staticmethod
    def _create_entity_node(tx, name, label, description, embedding, doc_id):
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $label,
            e.description = $description,
            e.embedding = $embedding,
            e.document_id = $doc_id
        """
        tx.run(query, name=name, label=label, description=description, embedding=embedding, doc_id=doc_id)

    @staticmethod
    def _create_relationship(tx, source, target, relation, description, doc_id):
        # Sử dụng APOC hoặc Cypher động để tạo quan hệ
        query = """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        MERGE (a)-[r:RELATED {type: $relation}]->(b)
        SET r.description = $description,
            r.document_id = $doc_id
        """
        tx.run(query, source=source, target=target, relation=relation, description=description, doc_id=doc_id)

    def search_local_context(self, query: str, top_k: int = 5) -> str:
        """
        Thực hiện 'Local Search': Tìm thực thể bằng Vector, sau đó lấy quan hệ xung quanh.
        """
        emb = self.embedding_service.embed_query(query)
        query_vector = [float(x) for x in emb]
        
        with self.driver.session() as session:
            # Tìm các thực thể giống nhất qua Vector Index và lấy các hàng xóm cấp 1
            result = session.run("""
                CALL db.index.vector.queryNodes('entity_vector_index', $top_k, $query_vector)
                YIELD node AS startNode, score
                MATCH (startNode)-[r]->(neighbor)
                RETURN startNode.name AS source, 
                       type(r) AS relation, 
                       neighbor.name AS target, 
                       r.description AS details,
                       score
                ORDER BY score DESC
            """, query_vector=query_vector, top_k=top_k)
            
            context_parts = []
            for record in result:
                context_parts.append(
                    f"- {record['source']} có quan hệ {record['relation']} với {record['target']}: {record['details']}"
                )
            
            return "\n".join(context_parts)

    def upsert_communities(self, document_id: str, community_reports: Dict[int, Dict[str, Any]]):
        """Lưu báo cáo cộng đồng vào Neo4j bằng Batching."""
        from concurrent.futures import ThreadPoolExecutor
        
        community_data = []
        
        def prepare_community(item):
            comm_id, data = item
            report = data["report"]
            vector = self.embedding_service.embed_query(report).tolist()
            return {
                "comm_id": comm_id,
                "unique_comm_id": f"{document_id}_comm_{comm_id}",
                "report": report,
                "embedding": vector,
                "document_id": document_id,
                "nodes": data["nodes"]
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            community_data = list(executor.map(prepare_community, community_reports.items()))

        with self.driver.session() as session:
            # Batch tạo Community nodes
            session.run("""
                UNWIND $batch AS data
                MERGE (c:Community {id: data.unique_comm_id})
                SET c.report = data.report,
                    c.embedding = data.embedding,
                    c.document_id = data.document_id,
                    c.local_id = data.comm_id
            """, batch=community_data)

            # Batch tạo liên kết Entity -> Community
            link_payloads = []
            for comm in community_data:
                for node_name in comm["nodes"]:
                    link_payloads.append({
                        "entity_name": node_name,
                        "unique_comm_id": comm["unique_comm_id"]
                    })
            
            if link_payloads:
                session.run("""
                    UNWIND $batch AS data
                    MATCH (e:Entity {name: data.entity_name})
                    MATCH (c:Community {id: data.unique_comm_id})
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                """, batch=link_payloads)

        LOG.info(f"Đã lưu {len(community_reports)} cộng đồng vào Neo4j cho tài liệu {document_id} (Batch mode)")

    @staticmethod
    def _create_community_node(tx, comm_id, report, embedding, doc_id):
        # Tạo id duy nhất cho community tránh trùng lặp giữa các tài liệu
        unique_comm_id = f"{doc_id}_comm_{comm_id}"
        query = """
        MERGE (c:Community {id: $unique_comm_id})
        SET c.report = $report,
            c.embedding = $embedding,
            c.document_id = $doc_id,
            c.local_id = $comm_id
        """
        tx.run(query, unique_comm_id=unique_comm_id, report=report, embedding=embedding, doc_id=doc_id, comm_id=comm_id)

    @staticmethod
    def _link_entity_to_community(tx, comm_id, entity_name, doc_id):
        unique_comm_id = f"{doc_id}_comm_{comm_id}"
        query = """
        MATCH (e:Entity {name: $entity_name})
        MATCH (c:Community {id: $unique_comm_id})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        tx.run(query, entity_name=entity_name, unique_comm_id=unique_comm_id)
