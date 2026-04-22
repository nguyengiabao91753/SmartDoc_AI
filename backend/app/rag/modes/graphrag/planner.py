from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from app.ai.llm import get_llm
from app.rag.models import RAGQueryRequest


@dataclass
class GraphRAGPlan:
    """
    Kế hoạch thực thi cho GraphRAG.
    Xác định chiến lược tìm kiếm dựa trên ý định câu hỏi.
    """
    question: str
    search_strategy: str  # 'local' hoặc 'global'
    search_type: str      # 'vector', 'hybrid', v.v.
    top_k: int
    document_id: Optional[int] = None
    llm_model: Optional[str] = None


from app.core.logger import LOG

class GraphRAGPlanner:
    """
    Bộ lập kế hoạch cho GraphRAG.
    Dùng LLM để phân loại câu hỏi vào 2 nhóm:
    1. Local Search: Câu hỏi về thực thể, quan hệ hoặc chi tiết cụ thể.
    2. Global Search: Câu hỏi tóm tắt, tổng quát hoặc chủ đề trên toàn bộ tài liệu.
    """

    def plan(self, request: RAGQueryRequest) -> GraphRAGPlan:
        # Log câu hỏi và chiến lược được phân loại
        LOG.info(f"[Planner] Question: '{request.question}'")
        # Bước quan trọng: Xác định chiến lược truy xuất
        strategy = self._classify_intent(request.question, request.llm_model)
        
        plan_obj = GraphRAGPlan(
            question=request.question,
            search_strategy=strategy,
            search_type=request.search_type,
            top_k=request.top_k,
            document_id=request.document_id,
            llm_model=request.llm_model,
        )
        LOG.info(f"[Planner] Selected strategy: {strategy}")
        return plan_obj

    def _classify_intent(self, question: str, llm_model: str | None = None) -> str:
        """
        Sử dụng LLM để phân loại câu hỏi vào Local hoặc Global search.
        """
        prompt = (
            "Bạn là chuyên gia phân loại câu hỏi cho hệ thống GraphRAG.\n"
            "Nhiệm vụ: Dựa vào câu hỏi, hãy chọn chiến lược tìm kiếm phù hợp nhất.\n\n"
            "CHIẾN LƯỢC:\n"
            "- 'local': Nếu câu hỏi hỏi về các thực thể cụ thể (người, địa danh, khái niệm riêng biệt), "
            "mối quan hệ giữa chúng, hoặc thông tin chi tiết có trong tài liệu.\n"
            "- 'global': Nếu câu hỏi yêu cầu tóm tắt, phân tích chủ đề rộng, tìm xu hướng toàn cảnh "
            "hoặc đánh giá vĩ mô về toàn bộ nội dung tài liệu.\n\n"
            "YÊU CẦU: Chỉ trả về duy nhất một từ 'local' hoặc 'global' (dạng JSON).\n\n"
            f"Câu hỏi: \"{question}\"\n"
            "JSON: {\"strategy\": "
        )

        try:
            # Dùng LLM với temperature thấp để đảm bảo tính nhất quán
            llm = get_llm(temperature=0.0, model=llm_model)
            response = llm.invoke(prompt).strip()
            
            # Làm sạch response để lấy keyword
            if 'global' in response.lower():
                return 'global'
            return 'local' # Mặc định là local vì độ chính xác cao cho chi tiết
        except Exception:
            # Fallback an toàn về local search nếu LLM gặp lỗi
            return 'local'
