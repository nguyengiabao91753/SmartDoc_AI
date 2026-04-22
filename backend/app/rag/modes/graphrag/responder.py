from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm


class GraphRAGResponder:
    """Answer synthesizer for GraphRAG with strategy-aware prompting."""

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
        text_sections = []
        community_sections = []

        for idx, doc in enumerate(source_documents, start=1):
            source_type = str(doc.metadata.get("source_type", "community"))
            if source_type.startswith("text"):
                text_sections.append(f"[Text {idx}]\n{doc.page_content}")
            else:
                community_id = doc.metadata.get("community_id", idx)
                community_sections.append(f"[Bao cao {community_id}]\n{doc.page_content}")

        text_context = "\n\n".join(text_sections) if text_sections else "(Khong co)"
        community_context = "\n\n".join(community_sections) if community_sections else "(Khong co)"

        return (
            "Ban la tro ly tong hop cho GraphRAG.\n"
            "Nhiem vu: Tra loi cau hoi tong quan ve tai lieu.\n\n"
            "Quy tac bat buoc:\n"
            "- Chi su dung thong tin trong NGU CANH.\n"
            "- Uu tien muc TEXT EVIDENCE de xac dinh chu de tai lieu.\n"
            "- Muc COMMUNITY REPORT chi dung de bo sung boi canh.\n"
            "- Neu khong du thong tin thi noi ro: 'Khong du du lieu trong do thi'.\n"
            "- Khi dua ra y chinh, ghi kem ma nguon [Text X] hoac [Bao cao X].\n"
            "- Khong lap lai cau hoi.\n"
            "- Tra loi cung ngon ngu voi cau hoi.\n\n"
            f"TEXT EVIDENCE:\n{text_context}\n\n"
            f"COMMUNITY REPORT:\n{community_context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "TRA LOI TONG HOP:"
        )

    def _build_local_prompt(self, *, question: str, source_documents: List[Document]) -> str:
        graph_facts = []
        text_facts = []

        for idx, doc in enumerate(source_documents, start=1):
            source_type = str(doc.metadata.get("source_type", "graph"))
            if source_type == "text":
                text_facts.append(f"[Text {idx}] {doc.page_content}")
            else:
                graph_facts.append(f"[Graph {idx}] {doc.page_content}")

        context_parts = []
        if graph_facts:
            context_parts.append("THONG TIN DO THI:\n" + "\n".join(graph_facts))
        if text_facts:
            context_parts.append("BANG CHUNG VAN BAN BO SUNG:\n" + "\n".join(text_facts))

        context = "\n\n".join(context_parts)

        return (
            "Ban la tro ly hoi dap dua tren Knowledge Graph.\n"
            "Nhiem vu: Ket noi cac quan he trong do thi de tra loi cau hoi.\n\n"
            "Quy tac bat buoc:\n"
            "- CHI su dung thong tin co trong NGU CANH.\n"
            "- Neu khong du thong tin thi noi ro: 'Khong tim thay thong tin du de ket luan'.\n"
            "- Neu co bang chung van ban bo sung, uu tien dung de lam ro ngu canh.\n"
            "- Khong lap lai cau hoi.\n"
            "- Tra loi ngan gon, logic, cung ngon ngu voi cau hoi.\n\n"
            f"NGU CANH:\n{context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "TRA LOI CHI TIET:"
        )
