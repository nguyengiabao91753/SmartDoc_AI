from typing import Any, Dict, List

from langchain_core.documents import Document # Import thêm Document để bọc dữ liệu

from app.rag.base import BaseRAGModeEngine
from app.rag.models import RAGEngineResult, RAGQueryRequest
from app.core.logger import LOG

# Import các bộ phận dưới quyền chỉ huy (Query time)
from .planner import GraphRAGPlanner
from .retriever import GraphRAGRetriever
from .responder import GraphRAGResponder

# Import thêm service mới để khởi tạo Retriever
from app.services.graph_rag_service import GraphRAGService

class GraphRAGEngine(BaseRAGModeEngine):
    """
    GraphRAGEngine - Người chỉ huy tổng thể cho luồng GraphRAG.
    """

    mode = "graphrag"
    display_name = "GraphRAG"

    def __init__(self, store, embedding_service):
        super().__init__(store=store, embedding_service=embedding_service)

        # Khởi tạo các "cấp dưới" để phục vụ việc truy vấn (Query time)
        self.planner = GraphRAGPlanner()
        
        # SỬA LỖI Ở ĐÂY: Khởi tạo GraphRAGService và truyền vào Retriever mới
        self.graph_rag_service = GraphRAGService()
        self.retriever = GraphRAGRetriever(graph_rag_service=self.graph_rag_service)
        
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
        doc_id_str = str(document_id)

        LOG.info(f"[GraphRAGEngine] Bước 1: Trích xuất đồ thị cho tài liệu {doc_id_str}...")
        nx_graph = self.doc_graph_service.build_in_memory_graph(doc_id_str, text_units)

        LOG.info(f"[GraphRAGEngine] Bước 2: Phân cụm và tóm tắt cộng đồng...")
        community_reports = self.community_service.detect_and_summarize(nx_graph)
        LOG.info(f"[GraphRAGEngine] Phát hiện {len(community_reports)} cộng đồng.")

        graph_data = self._nx_graph_to_dict(nx_graph)

        LOG.info(f"[GraphRAGEngine] Bước 3: Lưu đồ thị vào Neo4j...")
        self.kg_service.upsert_graph(doc_id_str, graph_data)
        self.kg_service.upsert_communities(doc_id_str, community_reports)

        LOG.info(f"[GraphRAGEngine] Hoàn tất xây dựng đồ thị cho tài liệu {doc_id_str}.")
        return {
            "nodes": nx_graph.number_of_nodes(),
            "edges": nx_graph.number_of_edges(),
            "communities": len(community_reports),
        }

    @staticmethod
    def _nx_graph_to_dict(nx_graph) -> Dict[str, Any]:
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
        # Bước 1: Kế hoạch
        plan = self.planner.plan(request)

        # SỬA LỖI Ở ĐÂY: Truy xuất bằng Retriever mới (nhận string thay vì object)
        retrieval_result = self.retriever.retrieve(query=plan.question)
        
        # SỬA LỖI ĐÓNG GÓI: Retriever mới trả về Dict, nhưng Responder cần List[Document]
        # Ta tạo một Document giả lập để tương thích với luồng cũ
        source_documents = [
            Document(
                page_content=retrieval_result["context"],
                metadata={
                    "strategy": retrieval_result["strategy"],
                    "context_type": retrieval_result["context_type"]
                }
            )
        ]
        
        actual_strategy = retrieval_result["strategy"].lower()

        # Bước 3: Tổng hợp
        if request.stream:
            answer_generator = self.responder.stream_answer(
                question=plan.question,
                source_documents=source_documents,
                llm_model=plan.llm_model,
                strategy=actual_strategy
            )
            answer = ""
        else:
            answer = self.responder.answer(
                question=plan.question,
                source_documents=source_documents,
                llm_model=plan.llm_model,
                strategy=actual_strategy
            )
            answer_generator = None

        metadata = {
            "mode": "graphrag",
            "strategy": actual_strategy,
            "doc_count": len(source_documents)
        }

        return RAGEngineResult(
            answer=answer,
            source_documents=source_documents,
            metadata=metadata,
            answer_generator=answer_generator,
        )
