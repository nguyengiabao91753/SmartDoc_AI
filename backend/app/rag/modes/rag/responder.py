from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm


class RAGResponder:
    """Response generator for vanilla RAG mode."""

    def answer(
        self,
        *,
        question: str,
        source_documents: List[Document],
        llm_model: str | None = None,
    ) -> str:
        if not source_documents:
            return "Khong tim thay noi dung phu hop trong tai lieu hien tai."

        context = "\n\n".join(
            [f"[Nguon {i + 1}] {doc.page_content}" for i, doc in enumerate(source_documents)]
        )
        prompt = (
            "Ban la tro ly hoi dap dua tren tai lieu. "
            "Chi tra loi dua tren phan NGU CANH. "
            "Tra loi bang cung ngon ngu voi cau hoi; neu cau hoi bang tieng Viet thi tra loi bang tieng Viet ro rang. "
            "Neu khong du thong tin, noi ro khong tim thay trong tai lieu.\n\n"
            f"NGU CANH:\n{context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "TRA LOI:"
        )
        llm = get_llm(model=llm_model)
        return llm.invoke(prompt)
