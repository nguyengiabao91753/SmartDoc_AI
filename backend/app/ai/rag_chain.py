from langchain.chains import RetrievalQA
from app.ai.llm import get_llm
from app.ai.retriever import get_retriever
from app.vectorstore.faiss_store import FaissStore
from langchain.schema import BaseRetriever, Document
from typing import List, Any, Dict
from pydantic import Field
import re


def _normalize_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [tok for tok in cleaned.split() if len(tok) >= 3]


def _evaluate_grounding(answer: str, source_documents: List[Document]) -> Dict[str, Any]:
    source_count = len(source_documents or [])
    if not answer:
        return {
            "is_hallucination": False,
            "supported_ratio": 1.0,
            "overlap_tokens": 0,
            "answer_tokens": 0,
            "source_count": source_count,
            "reason": "empty_answer",
        }

    answer_tokens = _normalize_tokens(answer)
    source_text = " ".join((doc.page_content or "") for doc in (source_documents or []))
    source_tokens = set(_normalize_tokens(source_text))

    if not answer_tokens:
        return {
            "is_hallucination": False,
            "supported_ratio": 1.0,
            "overlap_tokens": 0,
            "answer_tokens": 0,
            "source_count": source_count,
            "reason": "no_meaningful_tokens",
        }

    overlap = sum(1 for tok in answer_tokens if tok in source_tokens)
    supported_ratio = overlap / max(1, len(answer_tokens))

    # Conservative threshold to avoid false positives and keep existing flow unchanged.
    has_enough_evidence = source_count > 0 and (supported_ratio >= 0.12 or overlap >= 8)

    return {
        "is_hallucination": not has_enough_evidence,
        "supported_ratio": round(supported_ratio, 4),
        "overlap_tokens": overlap,
        "answer_tokens": len(answer_tokens),
        "source_count": source_count,
        "reason": "grounded" if has_enough_evidence else "low_overlap_with_sources",
    }


class GuardedRetrievalQA:
    """Wrapper that keeps RetrievalQA behavior but appends hallucination metadata."""

    def __init__(self, qa_chain: RetrievalQA):
        self._qa_chain = qa_chain

    def _augment_result(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload

        answer = payload.get("result") or payload.get("answer") or ""
        source_documents = payload.get("source_documents") or []
        payload["hallucination_check"] = _evaluate_grounding(answer, source_documents)
        return payload

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        result = self._qa_chain.invoke(input, config=config, **kwargs)
        return self._augment_result(result)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        result = await self._qa_chain.ainvoke(input, config=config, **kwargs)
        return self._augment_result(result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._qa_chain(*args, **kwargs)
        return self._augment_result(result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._qa_chain, name)

class LangChainRetriever(BaseRetriever):
    # Sử dụng Pydantic Field thay vì __init__ để tương thích với LangChain/Pydantic
    custom_retriever: Any = Field(description="Custom retriever implementation")
    embeddings: Any = Field(description="Embeddings model")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        q_vector = self.embeddings.embed_query(query)
        results = self.custom_retriever.retrieve(query, q_vector)
        
        documents = []
        for res in results:
            # Sao chép meta để tránh thay đổi dữ liệu gốc trong bộ nhớ cache (nếu có)
            meta = res.get('meta', {}).copy()
            page_content = meta.pop('text', '') # Tách nội dung văn bản ra khỏi metadata
            documents.append(Document(page_content=page_content, metadata=res['meta']))
            
        return documents

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Asynchronous version can be implemented if needed
        return self._get_relevant_documents(query, run_manager=run_manager)

def build_chain(store: FaissStore, embeddings, search_type: str = "vector"):
    custom_retriever = get_retriever(store, search_type)
    # Khởi tạo qua keyword arguments cho Pydantic model
    langchain_retriever = LangChainRetriever(custom_retriever=custom_retriever, embeddings=embeddings)
    
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=langchain_retriever, return_source_documents=True)
    return GuardedRetrievalQA(qa)