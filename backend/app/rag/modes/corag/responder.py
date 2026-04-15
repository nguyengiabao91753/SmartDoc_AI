from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.ai.llm import get_llm


class CoRAGResponder:
    """
    Responder cho CoRAG.
    Build context có cấu trúc: nhóm docs theo sub-query để LLM
    biết rõ đoạn nào trả lời khía cạnh nào → tránh hallucinate.
    """

    def answer(
        self,
        *,
        question: str,
        sub_queries: List[str],
        source_documents: List[Document],
        per_query_docs: Dict[str, List[Document]] | None = None,
        llm_model: str | None = None,
    ) -> str:
        if not source_documents:
            return "Không tìm thấy nội dung phù hợp trong tài liệu. Hãy thử lại với từ khóa khác."

        # Build context có cấu trúc: nhóm theo sub-query nếu có per_query_docs
        if per_query_docs:
            context = self._build_structured_context(sub_queries, per_query_docs)
        else:
            context = "\n\n".join(
                [f"[Đoạn {i + 1}]\n{doc.page_content}" for i, doc in enumerate(source_documents)]
            )

        sub_queries_text = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(sub_queries)])

        prompt = f"""Bạn là trợ lý hỏi đáp tài liệu. Trả lời câu hỏi dựa HOÀN TOÀN vào tài liệu được cung cấp.

QUY TẮC:
- CHỈ dùng thông tin trong phần TÀI LIỆU. KHÔNG dùng kiến thức bên ngoài.
- KHÔNG bịa, KHÔNG suy diễn thêm ngoài những gì tài liệu nêu.
- Nếu tài liệu không có thông tin → ghi rõ "Tài liệu không đề cập đến điều này".
- Trả lời bằng tiếng Việt nếu câu hỏi bằng tiếng Việt.

HỆ THỐNG ĐÃ PHÂN TÍCH CÂU HỎI THÀNH {len(sub_queries)} KHÍA CẠNH:
{sub_queries_text}

TÀI LIỆU THEO TỪNG KHÍA CẠNH:
{context}

CÂU HỎI GỐC: {question}

TRẢ LỜI (dựa hoàn toàn vào tài liệu, trả lời đủ {len(sub_queries)} khía cạnh trên):"""

        llm = get_llm(temperature=0.0, model=llm_model)
        return llm.invoke(prompt)

    def _build_structured_context(
        self,
        sub_queries: List[str],
        per_query_docs: Dict[str, List[Document]],
    ) -> str:
        """
        Tạo context có cấu trúc: gom docs theo sub-query.
        Giúp LLM biết đoạn nào liên quan đến khía cạnh nào.
        """
        parts = []
        seen_contents = set()

        for i, sub_query in enumerate(sub_queries, 1):
            docs = per_query_docs.get(sub_query, [])
            if not docs:
                parts.append(f"[Khía cạnh {i}: {sub_query}]\n(Không tìm thấy tài liệu liên quan)")
                continue

            section = f"[Khía cạnh {i}: {sub_query}]"
            doc_texts = []
            for doc in docs:
                content = doc.page_content.strip()
                # Dedupe cross-section
                content_key = content[:200]
                if content_key in seen_contents:
                    continue
                seen_contents.add(content_key)
                doc_texts.append(content)

            if doc_texts:
                section += "\n" + "\n---\n".join(doc_texts)
            else:
                section += "\n(Tất cả kết quả trùng với khía cạnh khác)"

            parts.append(section)

        return "\n\n".join(parts)