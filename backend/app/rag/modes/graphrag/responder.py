import logging
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from app.ai.llm import get_llm 

LOG = logging.getLogger(__name__)

class GraphRAGResponder:
    """
    Responder tiếp nhận Context đã được lọc từ Retriever và sinh ra câu trả lời.
    """
    def __init__(self):
        self.llm = get_llm(temperature=0.2)
        
        self.prompt_template = PromptTemplate(
            template="""Bạn là một chuyên gia phân tích dữ liệu AI.
Hệ thống đã tự động chọn lọc và cung cấp cho bạn một ngữ cảnh dạng {context_type}. 
Hãy sử dụng ngữ cảnh này để trả lời câu hỏi của người dùng một cách chính xác nhất.

--- NGỮ CẢNH CUNG CẤP ---
{context}

--- HƯỚNG DẪN ---
1. Chỉ dựa vào ngữ cảnh được cung cấp. Nếu ngữ cảnh không có thông tin, hãy nói "Tôi không tìm thấy dữ liệu về vấn đề này".
2. Trả lời rõ ràng, súc tích, đi thẳng vào trọng tâm câu hỏi.

Câu hỏi của người dùng: {query}
Câu trả lời:""",
            input_variables=["context_type", "context", "query"]
        )

    def generate_response(self, retrieved_data: Dict[str, Any]) -> str:
        query = retrieved_data.get("query", "")
        context = retrieved_data.get("context", "")
        context_type = retrieved_data.get("context_type", "Dữ liệu đồ thị")
        
        LOG.info(f"[GraphRAGResponder] Đang sinh câu trả lời dựa trên chiến lược {retrieved_data.get('strategy')}...")
        
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "context_type": context_type,
                "context": context,
                "query": query
            })
            
            return response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
        except Exception as e:
            LOG.error(f"[GraphRAGResponder] Lỗi sinh câu trả lời: {e}")
            return "Đã xảy ra lỗi trong quá trình tổng hợp dữ liệu để trả lời."
