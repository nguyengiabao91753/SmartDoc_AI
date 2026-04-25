from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm


class GraphRAGResponder:
    """Answer synthesizer for GraphRAG with strategy-aware prompting."""

    MAX_CONTEXT_CHARS = 800

    def answer(
        self,
        *,
        question: str,
        source_documents: List[Document],
        llm_model: str | None = None,
        strategy: str = "local",
    ) -> str:
        if not source_documents:
            return "Khong tim thay thong tin lien quan trong do thi tri thuc de tra loi cau hoi nay."

        prompt = self._build_prompt(question=question, source_documents=source_documents, strategy=strategy)
        llm = get_llm(model=llm_model, temperature=0.0)
        raw_answer = str(llm.invoke(prompt)).strip()
        return self._postprocess_answer(raw_answer)

    def stream_answer(
        self,
        *,
        question: str,
        source_documents: List[Document],
        llm_model: str | None = None,
        strategy: str = "local",
    ):
        if not source_documents:
            yield "Khong tim thay thong tin lien quan trong do thi tri thuc de tra loi cau hoi nay."
            return

        prompt = self._build_prompt(question=question, source_documents=source_documents, strategy=strategy)
        llm = get_llm(model=llm_model, temperature=0.0)
        for chunk in llm.stream(prompt):
            yield chunk

    def _build_prompt(self, *, question: str, source_documents: List[Document], strategy: str) -> str:
        if strategy == "global":
            return self._build_global_prompt(question=question, source_documents=source_documents)
        return self._build_local_prompt(question=question, source_documents=source_documents)

    def _build_global_prompt(self, *, question: str, source_documents: List[Document]) -> str:
        graph_items: List[str] = []
        text_items: List[str] = []
        community_items: List[str] = []

        for doc in source_documents:
            source_type = str(doc.metadata.get("source_type", "community")).lower()
            cleaned = self._normalize_doc_content(doc.page_content)
            if not cleaned:
                continue

            if source_type.startswith("text"):
                text_items.append(cleaned)
            elif "community" in source_type:
                community_id = doc.metadata.get("community_id")
                if community_id is None:
                    community_items.append(cleaned)
                else:
                    community_items.append(f"(cong dong {community_id}) {cleaned}")
            else:
                graph_items.append(cleaned)

        graph_context = self._format_context_section(graph_items, prefix="G")
        text_context = self._format_context_section(text_items, prefix="T")
        community_context = self._format_context_section(community_items, prefix="C")

        return (
            "Ban la tro ly tong hop cho GraphRAG.\n"
            "Nhiem vu: Tra loi cau hoi tong quan bang tieng Viet, tu nhien va co logic.\n\n"
            "Quy tac bat buoc:\n"
            "- Chi su dung thong tin trong NGU CANH.\n"
            "- Uu tien BANG CHUNG VAN BAN [Tx] de xac dinh chu de tong quat.\n"
            "- THONG TIN CONG DONG [Cx] va THONG TIN DO THI [Gx] dung de bo sung lap luan.\n"
            "- Moi luan diem quan trong phai gan it nhat mot ma nguon [Tx]/[Cx]/[Gx].\n"
            "- Neu du lieu thieu hoac mau thuan, phai noi ro muc do chac chan.\n"
            "- Khong lap lai cau hoi.\n"
            "- Khong duoc mo dau bang cac cum: 'Dua tren phan tich', 'Cau hoi nay', 'Tra loi la', 'Cau tra loi cong dong'.\n"
            "- Khong duoc xung ho theo kieu phan tich quy trinh, chi tra loi nhu tro ly binh thuong.\n"
            "- Tra loi thanh mot doan tu nhien, ro nghia; neu can thi them 1 cau ket luan ve muc do chac chan.\n\n"
            f"THONG TIN DO THI:\n{graph_context}\n\n"
            f"BANG CHUNG VAN BAN:\n{text_context}\n\n"
            f"BAO CAO CONG DONG:\n{community_context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "TRA LOI:"
        )

    def _build_local_prompt(self, *, question: str, source_documents: List[Document]) -> str:
        graph_items: List[str] = []
        text_items: List[str] = []

        for doc in source_documents:
            source_type = str(doc.metadata.get("source_type", "graph")).lower()
            cleaned = self._normalize_doc_content(doc.page_content)
            if not cleaned:
                continue

            if source_type.startswith("text"):
                text_items.append(cleaned)
            else:
                graph_items.append(cleaned)

        graph_context = self._format_context_section(graph_items, prefix="G")
        text_context = self._format_context_section(text_items, prefix="T")

        return (
            "Ban la tro ly hoi dap dua tren Knowledge Graph.\n"
            "Nhiem vu: Ket noi cac quan he trong do thi de tra loi cau hoi bang tieng Viet, ro rang va tu nhien.\n\n"
            "Quy tac bat buoc:\n"
            "- CHI su dung thong tin trong NGU CANH.\n"
            "- Moi nhan dinh quan trong phai dan nguon [Gx] hoac [Tx].\n"
            "- Neu ngu canh chua du de ket luan, noi ro pham vi chua du.\n"
            "- Neu co mau thuan giua cac nguon, neu ro thay vi doan mo rong.\n"
            "- Khong lap lai cau hoi.\n"
            "- Khong duoc mo dau bang cac cum: 'Dua tren phan tich', 'Cau hoi nay', 'Tra loi la', 'Cau tra loi cong dong'.\n"
            "- Tra loi thanh mot doan tu nhien, khong chia muc cau truc kieu bao cao.\n\n"
            f"THONG TIN DO THI:\n{graph_context}\n\n"
            f"BANG CHUNG VAN BAN BO SUNG:\n{text_context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "TRA LOI:"
        )

    def _format_context_section(self, items: List[str], *, prefix: str) -> str:
        if not items:
            return "(Khong co)"
        return "\n".join([f"[{prefix}{idx}] {item}" for idx, item in enumerate(items, start=1)])

    def _normalize_doc_content(self, content: str) -> str:
        text = re.sub(r"\s+", " ", str(content or "")).strip()
        if not text:
            return ""

        text = re.sub(r"^\[(text evidence|bang chung van ban)\]\s*", "", text, flags=re.IGNORECASE)
        if len(text) > self.MAX_CONTEXT_CHARS:
            return text[: self.MAX_CONTEXT_CHARS].rstrip() + " ..."
        return text

    def _postprocess_answer(self, answer: str) -> str:
        cleaned = str(answer or "").strip()
        if not cleaned:
            return cleaned

        leading_patterns = [
            r"^\s*Dựa trên phân tích[^:\n]*:\s*",
            r"^\s*Dua tren phan tich[^:\n]*:\s*",
            r"^\s*Câu hỏi này[^.\n]*[.\n]+\s*",
            r"^\s*Cau hoi nay[^.\n]*[.\n]+\s*",
            r"^\s*Trả lời là\s*:?\s*",
            r"^\s*Tra loi la\s*:?\s*",
        ]
        for pattern in leading_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(
            r"\bCâu trả lời\s+(Cộng|Công)\s*đồng\s*\d+\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\bCau tra loi\s+Cong\s*dong\s*\d+\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
