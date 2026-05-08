from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm


class SelfRAGResponder:
    MAX_CONTEXT_DOCS = 6
    MAX_DOC_CHARS = 900

    def answer(
        self,
        *,
        question: str,
        source_documents: List[Document],
        llm_model: str | None = None,
        active_query: str | None = None,
        critique: str | None = None,
        attempt: int = 1,
    ) -> str:
        if not source_documents:
            return "Không tìm thấy nội dung phù hợp trong tài liệu hiện tại."

        prompt = (
            "Bạn là bộ tạo câu trả lời của hệ thống Self-RAG.\n"
            "\n"
            "## NGUYÊN TẮC BẮT BUỘC\n"
            "1. CHỈ được sử dụng thông tin có trong phần NGỮ CẢNH bên dưới.\n"
            "2. TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài tài liệu, dù bạn có biết.\n"
            "3. TUYỆT ĐỐI KHÔNG bịa đặt, suy diễn hoặc thêm thông tin không có trong NGỮ CẢNH.\n"
            "4. Chỉ trả lời đúng phạm vi câu hỏi — không mở rộng sang nội dung khác dù có trong NGỮ CẢNH.\n"
            "5. Nếu NGỮ CẢNH không đủ để trả lời, hãy nói rõ: "
            "\"Tài liệu không cung cấp đủ thông tin về [chủ đề cụ thể].\"\n"
            "6. Không đề cập đến điểm số nội bộ, quy trình đánh giá hoặc chuỗi suy luận ẩn.\n"
            "7. Trả lời bằng ngôn ngữ của câu hỏi người dùng.\n"
            "\n"
            f"## CÂU HỎI\n{question}\n\n"
            f"## TRUY VẤN TRUY XUẤT\n{active_query or question}\n\n"
            f"Lần thử: {attempt}\n\n"
            f"## VẤN ĐỀ CẦN KHẮC PHỤC TỪ LẦN TRƯỚC\n{critique or '(không có)'}\n\n"
            f"## NGỮ CẢNH\n{self._build_context(source_documents)}\n\n"
            "## CÂU TRẢ LỜI\n"
            "Dựa hoàn toàn vào NGỮ CẢNH, trả lời câu hỏi một cách súc tích và chính xác.\n"
            "Nếu một thông tin không có trong NGỮ CẢNH, không được đề cập đến nó:\n"
        )
        llm = get_llm(temperature=0.0, model=llm_model)
        return llm.invoke(prompt).strip()

    def _build_context(self, source_documents: List[Document]) -> str:
        blocks: List[str] = []
        for index, doc in enumerate(source_documents[: self.MAX_CONTEXT_DOCS], start=1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source", "unknown")
            chunk = metadata.get("chunk", "?")
            query_trace = ", ".join(metadata.get("selfrag_queries") or [])
            evidence = " ".join(doc.page_content.split())[: self.MAX_DOC_CHARS]
            blocks.append(
                f"[Bằng chứng {index}]\n"
                f"Nguồn: {source_name}\n"
                f"Chunk: {chunk}\n"
                f"Truy xuất bởi: {query_trace or '(trực tiếp)'}\n"
                f"Nội dung: {evidence}"
            )
        return "\n\n".join(blocks)