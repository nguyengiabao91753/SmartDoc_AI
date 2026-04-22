from typing import List
from langchain_core.documents import Document
from app.ai.llm import get_llm

class GraphRAGResponder:
    """
    Bộ tổng hợp câu trả lời cho GraphRAG.
    Tự động điều chỉnh Prompt dựa trên chiến lược truy xuất (Local/Global).
    """

    def answer(
        self,
        *,
        question: str,
        source_documents: List[Document],
        llm_model: str | None = None,
        strategy: str = "local",
    ) -> str:
        # Kiểm tra nếu không có dữ liệu từ đồ thị
        if not source_documents:
            return "Không tìm thấy thông tin liên quan trong Đồ thị tri thức để trả lời câu hỏi này."

        # 1. Chuẩn bị ngữ cảnh và Prompt dựa trên chiến lược (giống cấu trúc rag/responder.py)
        if strategy == "global":
            # Ngữ cảnh cho Global Search là các báo cáo cộng đồng (tóm tắt vĩ mô)
            context = "\n\n".join(
                [f"--- BÁO CÁO {i + 1} ---\n{doc.page_content}" for i, doc in enumerate(source_documents)]
            )

            prompt = (
                "Bạn là một chuyên gia phân tích dữ liệu vĩ mô. "
                "Dưới đây là các BÁO CÁO CỘNG ĐỒNG được trích xuất từ Đồ thị tri thức. "
                "Nhiệm vụ của bạn là tổng hợp các báo cáo này để đưa ra câu trả lời toàn diện nhất cho câu hỏi của người dùng.\n\n"
                "YÊU CẦU:\n"
                "- Trả lời bằng cùng ngôn ngữ với câu hỏi.\n"
                "- Nếu thông tin trong các báo cáo mâu thuẫn, hãy nêu rõ các quan điểm khác nhau.\n"
                "- Chỉ dựa vào dữ liệu được cung cấp.\n"
                "- LƯU Ý QUAN TRỌNG: KHÔNG lặp lại câu hỏi, CHỈ đưa ra câu trả lời trực tiếp.\n\n"
                f"NGỮ CẢNH (CÁC BÁO CÁO):\n{context}\n\n"
                f"CÂU HỎI:\n{question}\n\n"
                "TRẢ LỜI TỔNG HỢP:"
            )
        else:
            # Ngữ cảnh cho Local Search là danh sách thực thể và quan hệ (chi tiết vi mô)
            context = "\n".join(
                [f"- {doc.page_content}" for doc in source_documents]
            )

            prompt = (
                "Bạn là trợ lý hỏi đáp dựa trên Đồ thị tri thức (Knowledge Graph). "
                "Dưới đây là danh sách các THỰC THỂ và MỐI QUAN HỆ liên quan trực tiếp đến câu hỏi. "
                "Hãy kết nối các thông tin này để đưa ra câu trả lời logic và chính xác.\n\n"
                "YÊU CẦU:\n"
                "- Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp.\n"
                "- Trả lời bằng cùng ngôn ngữ với câu hỏi (Tiếng Việt rõ ràng).\n"
                "- Nếu không đủ thông tin để kết luận, hãy nói rõ không tìm thấy trong đồ thị.\n"
                "- LƯU Ý QUAN TRỌNG: KHÔNG lặp lại câu hỏi, CHỈ đưa ra câu trả lời trực tiếp.\n\n"
                f"NGỮ CẢNH (THÔNG TIN ĐỒ THỊ):\n{context}\n\n"
                f"CÂU HỎI:\n{question}\n\n"
                "TRẢ LỜI CHI TIẾT:"
            )

        # 2. Gọi LLM để sinh câu trả lời (giống rag/responder.py)
        llm = get_llm(model=llm_model)
        return llm.invoke(prompt)

    def stream_answer(
        self,
        *,
        question: str,
        source_documents: List[Document],
        llm_model: str | None = None,
        strategy: str = "local",
    ):
        if not source_documents:
            yield "Không tìm thấy thông tin liên quan trong Đồ thị tri thức để trả lời câu hỏi này."
            return

        if strategy == "global":
            context = "\n\n".join(
                [f"--- BÁO CÁO {i + 1} ---\n{doc.page_content}" for i, doc in enumerate(source_documents)]
            )
            prompt = (
                "Bạn là một chuyên gia phân tích dữ liệu vĩ mô. "
                "Dưới đây là các BÁO CÁO CỘNG ĐỒNG được trích xuất từ Đồ thị tri thức. "
                "Nhiệm vụ của bạn là tổng hợp các báo cáo này để đưa ra câu trả lời toàn diện nhất cho câu hỏi của người dùng.\n\n"
                "YÊU CẦU:\n"
                "- Trả lời bằng cùng ngôn ngữ với câu hỏi.\n"
                "- Nếu thông tin trong các báo cáo mâu thuẫn, hãy nêu rõ các quan điểm khác nhau.\n"
                "- Chỉ dựa vào dữ liệu được cung cấp.\n"
                "- LƯU Ý QUAN TRỌNG: KHÔNG lặp lại câu hỏi, CHỈ đưa ra câu trả lời trực tiếp.\n\n"
                f"NGỮ CẢNH (CÁC BÁO CÁO):\n{context}\n\n"
                f"CÂU HỎI:\n{question}\n\n"
                "TRẢ LỜI TỔNG HỢP:"
            )
        else:
            context = "\n".join(
                [f"- {doc.page_content}" for doc in source_documents]
            )
            prompt = (
                "Bạn là trợ lý hỏi đáp dựa trên Đồ thị tri thức (Knowledge Graph). "
                "Dưới đây là danh sách các THỰC THỂ và MỐI QUAN HỆ liên quan trực tiếp đến câu hỏi. "
                "Hãy kết nối các thông tin này để đưa ra câu trả lời logic và chính xác.\n\n"
                "YÊU CẦU:\n"
                "- Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp.\n"
                "- Trả lời bằng cùng ngôn ngữ với câu hỏi (Tiếng Việt rõ ràng).\n"
                "- Nếu không đủ thông tin để kết luận, hãy nói rõ không tìm thấy trong đồ thị.\n"
                "- LƯU Ý QUAN TRỌNG: KHÔNG lặp lại câu hỏi, CHỈ đưa ra câu trả lời trực tiếp.\n\n"
                f"NGỮ CẢNH (THÔNG TIN ĐỒ THỊ):\n{context}\n\n"
                f"CÂU HỎI:\n{question}\n\n"
                "TRẢ LỜI CHI TIẾT:"
            )

        llm = get_llm(model=llm_model)
        for chunk in llm.stream(prompt):
            yield chunk
