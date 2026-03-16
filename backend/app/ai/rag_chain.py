from langchain.chains import RetrievalQA
from app.ai.llm import get_llm
from app.ai.retriever import get_retriever
from app.vectorstore.faiss_store import FaissStore
from langchain.schema import BaseRetriever, Document
from typing import List, Any
from pydantic import Field

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
    return qa
