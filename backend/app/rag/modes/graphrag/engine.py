from typing import Any, Dict, List

from app.rag.base import BaseRAGModeEngine
from app.rag.models import RAGEngineResult, RAGQueryRequest
from app.core.logger import LOG

# Import các bộ phận dưới quyền chỉ huy (Query time)
from .planner import GraphRAGPlanner
from .retriever import GraphRAGRetriever
from .responder import GraphRAGResponder

# Import các service xây dựng đồ thị (Indexing time) — lazy import bên dưới
# để tránh load graspologic, neo4j, networkx, NERService khi khởi động app.


class GraphRAGEngine(BaseRAGModeEngine):
    """
    GraphRAGEngine - Người chỉ huy tổng thể cho luồng GraphRAG.

    Nhiệm vụ:
    1. Điều phối Planner để xác định chiến lược (Local vs Global).
    2. Ra lệnh cho Retriever truy xuất dữ liệu từ Neo4j/Vector.
    3. Yêu cầu Responder tổng hợp câu trả lời dựa trên đồ thị.

    Các service xây dựng đồ thị (Indexing time):
    - DocumentGraphService: Trích xuất thực thể và quan hệ từ text, dựng đồ thị In-Memory (NetworkX).
    - DetectCommunityService: Phân cụm Leiden và tóm tắt cộng đồng bằng LLM.
    - KnowledgeGraphService: Lưu trữ đồ thị vào Neo4j kèm Vector Index.
    """

    mode = "graphrag"
    display_name = "GraphRAG"

    def __init__(self, store, embedding_service):
        super().__init__(store=store, embedding_service=embedding_service)

        # Khởi tạo các "cấp dưới" để phục vụ việc truy vấn (Query time)
        self.planner = GraphRAGPlanner()
        self.retriever = GraphRAGRetriever(store=store, embedding_service=embedding_service)
        self.responder = GraphRAGResponder()

        # Thay thế GraphIndexingService bằng lazy load properties
        self._doc_graph_service = None
        self._kg_service = None
        self._community_service = None

    @property
    def doc_graph_service(self):
        if self._doc_graph_service is None:
            from app.services.document_graph_service import DocumentGraphService
            self._doc_graph_service = DocumentGraphService()
        return self._doc_graph_service

    @property
    def kg_service(self):
        if self._kg_service is None:
            from app.services.knowledge_graph_service import KnowledgeGraphService
            self._kg_service = KnowledgeGraphService()
        return self._kg_service

    @property
    def community_service(self):
        if self._community_service is None:
            from app.services.detect_community_service import DetectCommunityService
            self._community_service = DetectCommunityService()
        return self._community_service

    # ---------- GIAI ĐOẠN XÂY DỰNG (Indexing Time) ----------
    def build_graph_index(self, document_id: int, text_units: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Chỉ huy toàn bộ pipeline xây dựng đồ thị tri thức cho một tài liệu.

        Quy trình:
        1. DocumentGraphService: Gọi LLM (NER) trích xuất thực thể & quan hệ → đồ thị NetworkX trên RAM.
        2. DetectCommunityService: Chạy thuật toán Leiden phân cụm → LLM tóm tắt từng cộng đồng.
        3. KnowledgeGraphService: Lưu thực thể, quan hệ, cộng đồng vào Neo4j kèm vector embedding.

        Args:
            document_id: ID định danh của tài liệu.
            text_units: Danh sách các chunk dạng [{"page_content": "..."}, ...].

        Returns:
            Dict chứa thống kê (số node, edge, community).
        """
        doc_id_str = str(document_id)

        # Bước 1: Trích xuất thực thể và quan hệ → dựng đồ thị In-Memory (NetworkX)
        LOG.info(f"[GraphRAGEngine] Bước 1: Trích xuất đồ thị cho tài liệu {doc_id_str}...")
        nx_graph = self.doc_graph_service.build_in_memory_graph(doc_id_str, text_units)

        # Bước 2: Phân cụm Leiden + LLM tóm tắt cộng đồng
        LOG.info(f"[GraphRAGEngine] Bước 2: Phân cụm và tóm tắt cộng đồng...")
        community_reports = self.community_service.detect_and_summarize(nx_graph)
        LOG.info(f"[GraphRAGEngine] Phát hiện {len(community_reports)} cộng đồng.")

        # Bước 3: Chuyển đồ thị NetworkX → dict format tương thích với KnowledgeGraphService
        graph_data = self._nx_graph_to_dict(nx_graph)

        # Bước 4: Lưu thực thể + quan hệ vào Neo4j (kèm vector embedding)
        LOG.info(f"[GraphRAGEngine] Bước 3: Lưu đồ thị vào Neo4j...")
        self.kg_service.upsert_graph(doc_id_str, graph_data)

        # Bước 5: Lưu báo cáo cộng đồng vào Neo4j và liên kết với các thực thể
        self.kg_service.upsert_communities(doc_id_str, community_reports)

        LOG.info(f"[GraphRAGEngine] Hoàn tất xây dựng đồ thị cho tài liệu {doc_id_str}.")
        return {
            "nodes": nx_graph.number_of_nodes(),
            "edges": nx_graph.number_of_edges(),
            "communities": len(community_reports),
        }

    @staticmethod
    def _nx_graph_to_dict(nx_graph) -> Dict[str, Any]:
        """Chuyển đổi đồ thị NetworkX sang dict format {entities, relationships}."""
        graph_data: Dict[str, Any] = {"entities": [], "relationships": []}

        for node, data in nx_graph.nodes(data=True):
            graph_data["entities"].append({
                "name": node,
                "type": data.get("type", "Unknown"),
                "description": data.get("description", ""),
            })

        for u, v, data in nx_graph.edges(data=True):
            graph_data["relationships"].append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATED"),
                "description": data.get("description", ""),
            })

        return graph_data

    # ---------- GIAI ĐOẠN TRUY VẤN (Query Time) ----------
    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        """
        Quy trình chỉ huy 3 bước chuẩn mực.
        """

        # Bước 1: Lên kế hoạch (Giao cho Planner phân loại Local hay Global)
        plan = self.planner.plan(request)

        # Bước 2: Truy xuất (Giao cho Retriever lục tìm trong Neo4j)
        # Retriever sẽ dựa vào plan.search_strategy để tự chạy Cypher tương ứng
        source_documents = self.retriever.retrieve(plan)

        # Bước 3: Tổng hợp (Giao cho Responder đóng gói context và gọi LLM)
        if request.stream:
            answer_generator = self.responder.stream_answer(
                question=plan.question,
                source_documents=source_documents,
                llm_model=plan.llm_model,
                strategy=plan.search_strategy
            )
            answer = ""
        else:
            answer = self.responder.answer(
                question=plan.question,
                source_documents=source_documents,
                llm_model=plan.llm_model,
                strategy=plan.search_strategy  # Truyền thêm strategy để responder biết cách format prompt
            )
            answer_generator = None

        # Metadata để giao diện UI có thể hiển thị "Thinking process"
        metadata = {
            "mode": "graphrag",
            "strategy": plan.search_strategy,
            "doc_count": len(source_documents)
        }

        return RAGEngineResult(
            answer=answer,
            source_documents=source_documents,
            metadata=metadata,
            answer_generator=answer_generator,
        )
