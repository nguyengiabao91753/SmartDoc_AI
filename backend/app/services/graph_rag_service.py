import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from app.core.config import settings
from app.core.logger import LOG
from app.services.ner_service import NERService
from app.services.document_graph_service import DocumentGraphService
from app.services.detect_community_service import DetectCommunityService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.document_service import DocumentService

class GraphRAGService:
    """
    Service xây dựng Đồ thị Tri thức (GraphRAG) toàn diện.
    Quy trình: NER đa luồng -> Xây dựng Đồ thị -> Phát hiện Cộng đồng -> Lưu vào Neo4j.
    """

    def __init__(self, embedding_service=None):
        self.ner_service = NERService()
        self.doc_service = DocumentService()
        # DocumentGraphService quản lý đồ thị NetworkX trong RAM
        self.doc_graph_service = DocumentGraphService(ner_service=self.ner_service)
        self.community_service = DetectCommunityService(ner_service=self.ner_service)
        self.kg_service = KnowledgeGraphService(embedding_service=embedding_service)
        self.graph_lock = threading.Lock()

    def _get_original_filename(self, filename: str) -> str:
        """
        Trích xuất tên file gốc từ tên file đã bị sanitize (timestamp_uuid_name).
        Dựa trên logic sanitize_filename trong streamlit_app.py.
        """
        # Format: timestamp_uuid8_realname.ext
        # Ví dụ: 20240419_120000_abcd1234_contract_a.pdf
        parts = filename.split('_', 3) # Split thành tối đa 4 phần
        if len(parts) >= 4:
            # Bỏ timestamp (parts[0], parts[1]) và uuid (parts[2])
            # Tuy nhiên sanitize_filename dùng timestamp = %Y%m%d_%H%M%S (2 phần nếu split '_')
            # uuid8 là 1 phần.
            return parts[3]
        return filename

    def process_document(self, file_path: str):
        """Xử lý một tài liệu đơn lẻ: Trích xuất tri thức và đưa vào đồ thị."""
        from app.services.database_service import db_service
        
        doc_record = db_service.get_document_by_filepath(file_path)
        if doc_record:
            doc_id = str(doc_record["id"])
            display_name = doc_record["filename"]
        else:
            filename = os.path.basename(file_path)
            doc_id = filename 
            display_name = self._get_original_filename(filename)
        
        LOG.info(f"=== Bắt đầu trích xuất tri thức cho: {display_name} ===")
        
        try:
            # 1. Load và chia nhỏ tài liệu thành các chunks
            langchain_docs = self.doc_service.load_document(file_path)
            if not langchain_docs:
                LOG.warning(f"Tài liệu {display_name} không có nội dung.")
                return

            # 2. Trích xuất Thực thể & Mối quan hệ từ từng chunk (Đa luồng LLM)
            doc_entities = []
            doc_relationships = []
            
            # Hàm bổ trợ để gọi NER cho 1 chunk
            def process_chunk(chunk_idx, chunk_doc):
                LOG.info(f"   -> Đang xử lý chunk {chunk_idx+1}/{len(langchain_docs)}...")
                return self.ner_service.extract_graph_elements(chunk_doc.page_content)

            # Sử dụng ThreadPoolExecutor để gọi Ollama song song cho các chunk
            # Với Local LLM như qwen2.5:7b, KHÔNG ĐƯỢC để số luồng quá cao để tránh OOM hoặc nghẽn cổ chai
            max_ner_workers = int(os.getenv("MAX_NER_WORKERS", 1)) # Khuyến nghị: 1 cho GPU phổ thông
            with ThreadPoolExecutor(max_workers=max_ner_workers) as executor:
                futures = {executor.submit(process_chunk, i, d): i for i, d in enumerate(langchain_docs)}
                
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        graph_data = future.result()
                        
                        # Gom nhóm dữ liệu để đẩy vào Neo4j sau
                        entities = graph_data.get("entities", [])
                        rels = graph_data.get("relationships", [])
                        doc_entities.extend(entities)
                        doc_relationships.extend(rels)
                        
                        LOG.info(f"   ✓ Đã xong chunk {chunk_idx+1}: Trích xuất được {len(entities)} thực thể.")
                        
                        # Cập nhật vào đồ thị NetworkX toàn cục (Thread-safe)
                        with self.graph_lock:
                            self._merge_into_networkx(doc_id, graph_data)
                            
                    except Exception as e:
                        LOG.error(f"Lỗi khi xử lý chunk của {display_name}: {e}")

            # 3. Đẩy dữ liệu của tài liệu này vào Neo4j
            if doc_entities or doc_relationships:
                self.kg_service.upsert_graph(doc_id, {
                    "entities": doc_entities,
                    "relationships": doc_relationships
                })
                LOG.info(f"--- Đã lưu {len(doc_entities)} thực thể từ '{display_name}' vào Neo4j ---")
            else:
                LOG.warning(f"Không trích xuất được tri thức nào từ {display_name}")

        except Exception as e:
            LOG.error(f"Thất bại khi xử lý tài liệu {display_name}: {e}")

    def _merge_into_networkx(self, doc_id: str, graph_data: Dict[str, Any]):
        """Hợp nhất dữ liệu NER vào NetworkX Graph (tương tự DocumentGraphService)."""
        nx_graph = self.doc_graph_service.graph
        
        # Thêm Entities
        for entity in graph_data.get("entities", []):
            name = entity.get("name", "").strip().lower()
            if not name: continue
            
            if not nx_graph.has_node(name):
                nx_graph.add_node(
                    name,
                    original_name=entity.get("name"),
                    type=entity.get("type", "Unknown"),
                    description=entity.get("description", ""),
                    document_ids={doc_id}
                )
            else:
                nx_graph.nodes[name]["document_ids"].add(doc_id)

        # Thêm Relationships
        for rel in graph_data.get("relationships", []):
            src = rel.get("source", "").strip().lower()
            tgt = rel.get("target", "").strip().lower()
            if not src or not tgt: continue
            
            # Đảm bảo node tồn tại
            if not nx_graph.has_node(src):
                nx_graph.add_node(src, original_name=rel.get("source"), type="Unknown", description="")
            if not nx_graph.has_node(tgt):
                nx_graph.add_node(tgt, original_name=rel.get("target"), type="Unknown", description="")
            
            nx_graph.add_edge(
                src, tgt,
                relation=rel.get("relation", ""),
                description=rel.get("description", ""),
                document_id=doc_id
            )

    def run_full_pipeline(self):
        """
        Khởi chạy toàn bộ pipeline xây dựng GraphRAG cho tất cả tài liệu trong data/documents.
        """
        doc_dir = settings.DOCUMENT_DIR
        if not os.path.exists(doc_dir):
            LOG.error(f"Thư mục tài liệu không tồn tại: {doc_dir}")
            return

        # Lấy danh sách các file được hỗ trợ
        files = [
            os.path.join(doc_dir, f) for f in os.listdir(doc_dir)
            if f.lower().endswith(('.pdf', '.docx')) and os.path.isfile(os.path.join(doc_dir, f))
        ]

        if not files:
            LOG.warning("Không có tài liệu nào trong data/documents để xử lý.")
            return

        LOG.info(f">>> Bắt đầu quy trình GraphRAG Pipeline với {len(files)} tài liệu <<<")

        # BƯỚC 1: Xử lý đa luồng cho các tài liệu
        # Sử dụng ThreadPoolExecutor để xử lý song song nhiều tài liệu
        # Để chạy nhanh nhất với 1 GPU cục bộ, max_workers nên là 1
        with ThreadPoolExecutor(max_workers=1) as doc_executor:
            doc_executor.map(self.process_document, files)

        # BƯỚC 2: Phát hiện Cộng đồng (Community Detection)
        # Bước này chạy trên đồ thị NetworkX đã tích lũy từ tất cả tài liệu
        LOG.info(">>> Đang chạy thuật toán phát hiện cộng đồng (Leiden) trên toàn bộ đồ thị...")
        try:
            community_reports = self.community_service.detect_and_summarize(self.doc_graph_service.graph)
            
            # BƯỚC 3: Lưu báo cáo cộng đồng vào Neo4j
            if community_reports:
                LOG.info(f">>> Đang lưu {len(community_reports)} báo cáo cộng đồng vào Neo4j...")
                self.kg_service.upsert_communities("global_knowledge_graph", community_reports)
                LOG.info(">>> Đã hoàn tất lưu tri thức cộng đồng. <<<")
            else:
                LOG.warning("Không có cộng đồng nào được phát hiện.")
                
        except Exception as e:
            LOG.error(f"Lỗi trong quá trình phát hiện cộng đồng: {e}")

        LOG.info("✅ QUY TRÌNH GRAPHRAG ĐÃ HOÀN TẤT THÀNH CÔNG!")

    def update_graph_with_file(self, file_path: str):
        """
        Cập nhật đồ thị tri thức khi người dùng đưa file mới vào.
        Quy trình: Xử lý file mới -> Cập nhật NetworkX -> Upsert Neo4j -> Chạy lại Community Detection.
        """
        if not os.path.exists(file_path):
            LOG.error(f"File không tồn tại: {file_path}")
            return

        display_name = self._get_original_filename(os.path.basename(file_path))
        LOG.info(f">>> Đang cập nhật đồ thị với file mới: {display_name} <<<")

        # 1. Xử lý tài liệu mới (NER + Cập nhật NetworkX + Lưu Neo4j)
        self.process_document(file_path)

        # 2. Cập nhật lại Cộng đồng
        # Lưu ý: Việc chạy lại Community Detection giúp cấu trúc tri thức toàn cục luôn được làm mới
        # khi có các mối liên kết mới giữa các tài liệu.
        LOG.info(">>> Đang tái cấu trúc cộng đồng sau khi thêm dữ liệu mới...")
        try:
            community_reports = self.community_service.detect_and_summarize(self.doc_graph_service.graph)
            
            if community_reports:
                self.kg_service.upsert_communities("global_knowledge_graph", community_reports)
                LOG.info(f"✅ Đã cập nhật xong tri thức và cộng đồng cho: {display_name}")
            else:
                LOG.warning("Không phát hiện cộng đồng mới.")
        except Exception as e:
            LOG.error(f"Lỗi khi cập nhật cộng đồng cho file mới: {e}")

    def is_graph_ready(self) -> bool:
        """Kiểm tra xem đồ thị tri thức đã được xây dựng chưa (bất kỳ dữ liệu nào)."""
        return self.kg_service.check_if_graph_exists()

    def is_document_indexed(self, document_id: int) -> bool:
        """Kiểm tra O(1) xem một tài liệu cụ thể đã được đánh chỉ mục trong đồ thị chưa."""
        return self.kg_service.is_document_indexed(str(document_id))

    def get_new_files(self) -> List[str]:
        """Lấy danh sách các đường dẫn file chưa được đưa vào đồ thị tri thức."""
        doc_dir = settings.DOCUMENT_DIR
        if not os.path.exists(doc_dir):
            return []
            
        all_files = [f for f in os.listdir(doc_dir) 
                     if f.lower().endswith(('.pdf', '.docx')) and os.path.isfile(os.path.join(doc_dir, f))]
        
        indexed_ids = self.kg_service.get_indexed_document_ids()
        # So sánh tên file với document_id đã lưu trong Neo4j
        new_files = [f for f in all_files if f not in indexed_ids]
        
        return [os.path.join(doc_dir, f) for f in new_files]
