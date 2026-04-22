import logging
from typing import List, Dict, Any
import networkx as nx

from app.services.ner_service import NERService
from app.core.logger import LOG

logger = logging.getLogger(__name__)

class DocumentGraphService:
    """Service điều phối việc xây dựng đồ thị tri thức trên RAM (In-Memory)."""

    def __init__(self, ner_service=None):
        # Tái sử dụng NERService nếu đã có sẵn
        self.ner_service = ner_service or NERService()
        # Khởi tạo một đồ thị rỗng trên RAM
        self.graph = nx.Graph() 

    def build_in_memory_graph(self, document_id: str, chunks: List[Dict[str, Any]]) -> nx.Graph:
        """
        Trích xuất và xây dựng đồ thị ngay trong bộ nhớ từ các chunks.
        Trả về đối tượng networkx Graph để truy vấn sau này.
        """
        LOG.info(f"Bắt đầu trích xuất đồ thị (In-Memory) cho Document ID: {document_id}")
        
        # Bước 1: Trích xuất dữ liệu từ từng chunk
        for i, chunk in enumerate(chunks):
            text = chunk.get("page_content", "")
            if not text:
                continue
                
            LOG.info(f"Đang xử lý chunk {i+1}/{len(chunks)} bằng LLM...")
            graph_data = self.ner_service.extract_graph_elements(text)
            
            # Bước 2: Đưa các Thực thể (Entities) vào đồ thị làm Nodes
            for entity in graph_data.get("entities", []):
                node_name = entity.get("name", "").strip().lower()
                if not node_name:
                    continue
                    
                # Nếu node chưa có, thêm mới. Nếu có rồi, cập nhật thêm metadata
                if not self.graph.has_node(node_name):
                    self.graph.add_node(
                        node_name, 
                        original_name=entity.get("name"),
                        type=entity.get("type", "Unknown"),
                        description=entity.get("description", ""),
                        document_ids={document_id}
                    )
                else:
                    # Gộp document_id để biết node này xuất hiện ở những file nào
                    self.graph.nodes[node_name]["document_ids"].add(document_id)

            # Bước 3: Đưa Mối quan hệ (Relationships) vào đồ thị làm Edges
            for rel in graph_data.get("relationships", []):
                source = rel.get("source", "").strip().lower()
                target = rel.get("target", "").strip().lower()
                
                if not source or not target:
                    continue
                
                # Nếu hai đầu mút chưa tồn tại (do LLM trích xuất lỗi), tạo node ẩn cho chúng
                if not self.graph.has_node(source):
                    self.graph.add_node(source, original_name=rel.get("source"), type="Unknown", description="")
                if not self.graph.has_node(target):
                    self.graph.add_node(target, original_name=rel.get("target"), type="Unknown", description="")

                # Thêm đường nối (Cạnh)
                self.graph.add_edge(
                    source, 
                    target, 
                    relation=rel.get("relation", ""),
                    description=rel.get("description", ""),
                    document_id=document_id
                )

        LOG.info(f"Hoàn thành đồ thị! Tổng số Nút (Thực thể): {self.graph.number_of_nodes()}, "
                 f"Tổng số Cạnh (Mối quan hệ): {self.graph.number_of_edges()}")
        
        return self.graph

    def get_neighbors_context(self, entity_name: str) -> str:
        """
        Hàm tiện ích: Lấy toàn bộ hàng xóm của một thực thể để nhồi vào LLM.
        """
        node = entity_name.strip().lower()
        if not self.graph.has_node(node):
            return ""
            
        neighbors = list(self.graph.neighbors(node))
        context = f"Thông tin về '{entity_name}':\n"
        
        for neighbor in neighbors:
            edge_data = self.graph.get_edge_data(node, neighbor)
            relation = edge_data.get("relation", "")
            desc = edge_data.get("description", "")
            context += f"- Có quan hệ [{relation}] với '{neighbor}': {desc}\n"
            
        return context
