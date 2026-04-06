from typing import List

import numpy as np
from langchain_core.documents import Document

from app.ai.llm import get_llm
from app.ai.retriever import get_retriever
from app.vectorstore.faiss_store import FaissStore


class SimpleRAGChain:
    """Compatibility wrapper that exposes invoke({"query": ...}) like RetrievalQA."""

    def __init__(self, custom_retriever, embeddings, llm):
        self.custom_retriever = custom_retriever
        self.embeddings = embeddings
        self.llm = llm

    def _embed_query(self, query: str) -> np.ndarray:
        if hasattr(self.embeddings, "embed_query"):
            q_vector = self.embeddings.embed_query(query)
        else:
            q_vector = self.embeddings.encode(query, convert_to_numpy=True)

        q_vector = np.asarray(q_vector, dtype="float32")
        norm = np.linalg.norm(q_vector)
        if norm > 0:
            q_vector = q_vector / norm
        return q_vector

    def _retrieve_documents(self, query: str) -> List[Document]:
        q_vector = self._embed_query(query)
        results = self.custom_retriever.retrieve(query, q_vector)

        documents: List[Document] = []
        for result in results:
            metadata = dict(result.get("meta", {}))
            page_content = metadata.pop("text", "")
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def invoke(self, inputs: dict) -> dict:
        query = inputs.get("query", "")
        source_documents = self._retrieve_documents(query)
        if not source_documents:
            return {
                "result": "Khong tim thay noi dung phu hop trong tai lieu hien tai.",
                "source_documents": [],
            }

        context = "\n\n".join(
            [f"[Nguon {index + 1}] {document.page_content}" for index, document in enumerate(source_documents)]
        )
        prompt = (
            "Ban la tro ly hoi dap dua tren tai lieu. "
            "Chi tra loi dua tren phan NGU CANH. "
            "Tra loi bang cung ngon ngu voi cau hoi; neu cau hoi bang tieng Viet thi tra loi bang tieng Viet ro rang. "
            "Neu khong du thong tin, noi ro khong tim thay trong tai lieu.\n\n"
            f"NGU CANH:\n{context}\n\n"
            f"CAU HOI:\n{query}\n\n"
            "TRA LOI:"
        )
        answer = self.llm.invoke(prompt)
        return {
            "result": answer,
            "source_documents": source_documents,
        }


def build_chain(
    store: FaissStore,
    embeddings,
    search_type: str = "vector",
    model: str | None = None,
    top_k: int | None = None,
    filters: dict | None = None,
):
    custom_retriever = get_retriever(store, search_type, top_k=top_k, filters=filters)
    llm = get_llm(model=model)
    return SimpleRAGChain(custom_retriever=custom_retriever, embeddings=embeddings, llm=llm)
