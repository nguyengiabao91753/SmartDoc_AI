import logging
from typing import Any, Dict

from app.core.config import settings
from app.core.logger import LOG

try:
    from underthesea import ner, sent_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    LOG.warning("Thư viện underthesea không có sẵn. Vui lòng chạy: pip install underthesea")

logger = logging.getLogger(__name__)

class NERService:
    """Service trích xuất Thực thể và Mối quan hệ tốc độ cao bằng thuật toán NLP (O(n))."""

    def __init__(self):
        self.type_mapping = {
            "PER": "Người",
            "ORG": "Tổ chức",
            "LOC": "Địa điểm",
            "MISC": "Khái niệm khác"
        }
        LOG.info("Khởi tạo NERService siêu tốc sử dụng mô hình NLP truyền thống (O(n)).")

    def extract_graph_elements(self, text: str) -> Dict[str, Any]:
        """
        Trích xuất Entities và Relationships siêu tốc sử dụng mô hình NLP thay vì LLM.
        Độ phức tạp O(n) với n là số từ trong văn bản.
        """
        if not UNDERTHESEA_AVAILABLE:
            LOG.error("Thư viện underthesea chưa được cài đặt. Hãy đảm bảo đã cài đặt underthesea.")
            return {"entities": [], "relationships": []}

        try:
            return self._extract_fast(text)
        except Exception as exc:
            LOG.error(f"Lỗi khi trích xuất đồ thị bằng thuật toán nhanh: {exc}")
            return {"entities": [], "relationships": []}

    def _extract_fast(self, text: str) -> Dict[str, Any]:
        entities = []
        relationships = []
        entity_map = {}
        
        # O(n) - Tách câu
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # O(n_i) - Trích xuất NER bằng CRF (Rất nhanh)
            tokens = ner(sentence)
            
            current_entity = None
            current_type = None
            sentence_entities = []
            
            # O(n_i) - Ghép nối các IOB tags để lấy thực thể
            for word, pos, chunk, tag in tokens:
                if tag.startswith('B-'):
                    if current_entity:
                        sentence_entities.append({"name": current_entity.strip(), "type": current_type})
                    current_entity = word
                    current_type = tag[2:]
                elif tag.startswith('I-') and current_entity:
                    current_entity += " " + word
                else:
                    if current_entity:
                        sentence_entities.append({"name": current_entity.strip(), "type": current_type})
                        current_entity = None
                        current_type = None
            
            if current_entity:
                 sentence_entities.append({"name": current_entity.strip(), "type": current_type})
                 
            # Lọc và format các thực thể
            valid_sentence_entities = []
            for ent in sentence_entities:
                name = ent["name"]
                t = self.type_mapping.get(ent["type"], "Khái niệm")
                
                # Bỏ qua các từ quá ngắn hoặc sai sót
                if len(name) <= 1:
                    continue
                    
                valid_sentence_entities.append(ent)
                
                if name not in entity_map:
                    entity_map[name] = {
                        "name": name, 
                        "type": t, 
                        "description": f"Thực thể {t} được nhắc đến trong văn bản."
                    }
                    entities.append(entity_map[name])
                    
            # Trích xuất mối quan hệ theo cụm cùng xuất hiện (Co-occurrence)
            # Độ phức tạp O(k^2) với k là số thực thể trong 1 câu (thường rất nhỏ k < 5)
            for i in range(len(valid_sentence_entities)):
                for j in range(i + 1, len(valid_sentence_entities)):
                    e1 = valid_sentence_entities[i]
                    e2 = valid_sentence_entities[j]
                    
                    if e1["name"] == e2["name"]:
                        continue
                        
                    rel_type = "LIÊN_QUAN_ĐẾN"
                    
                    # Logic heuristic cho tiếng Việt
                    if e1["type"] == "PER" and e2["type"] == "ORG":
                        rel_type = "LÀM_VIỆC_TẠI"
                    elif e1["type"] == "ORG" and e2["type"] == "LOC":
                        rel_type = "CÓ_TRỤ_SỞ_TẠI"
                    elif e1["type"] == "PER" and e2["type"] == "LOC":
                        rel_type = "SỐNG_TẠI"
                    # Đảo chiều
                    elif e2["type"] == "PER" and e1["type"] == "ORG":
                        e1, e2 = e2, e1
                        rel_type = "LÀM_VIỆC_TẠI"
                    elif e2["type"] == "ORG" and e1["type"] == "LOC":
                        e1, e2 = e2, e1
                        rel_type = "CÓ_TRỤ_SỞ_TẠI"
                    elif e2["type"] == "PER" and e1["type"] == "LOC":
                        e1, e2 = e2, e1
                        rel_type = "SỐNG_TẠI"
                        
                    # Cập nhật mảng quan hệ
                    relationships.append({
                        "source": e1["name"],
                        "target": e2["name"],
                        "relation": rel_type,
                        "description": f"{e1['name']} và {e2['name']} cùng xuất hiện trong ngữ cảnh."
                    })
                    
        # Loại bỏ các quan hệ bị trùng lặp chính xác
        unique_relationships = []
        seen = set()
        for rel in relationships:
            rel_str = f"{rel['source']}|{rel['relation']}|{rel['target']}"
            if rel_str not in seen:
                seen.add(rel_str)
                unique_relationships.append(rel)
                
        return {"entities": entities, "relationships": unique_relationships}

