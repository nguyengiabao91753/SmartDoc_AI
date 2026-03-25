from typing import Any, Dict, List
import json
import os
from urllib import error, request

from langchain_core.documents import Document

from app.ai.rag_chain import build_chain
from app.core.config import settings
from app.core.logger import LOG
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.vectorstore.faiss_store import FaissStore


class RAGService:
    """Coordinates document loading, embedding, vector storage, retrieval, and LLM answering."""

    def __init__(self):
        LOG.info("Khởi tạo RAGService...")
        self.doc_service = DocumentService()
        self.embedding_service = EmbeddingService()
        self.embedding_dim = self.embedding_service.get_dimension()
        self.vectorstore = FaissStore(self.embedding_dim)
        self.vectorstore.load()
        LOG.info("RAGService khởi tạo xong")

    def _resolve_search_type(self, search_type: str | None) -> str:
        normalized = (search_type or "vector").lower()
        if normalized in {"keyword", "hybrid"}:
            return "hybrid"
        return "vector"

    def _resolve_top_k(self, detail_level: str = "fast", top_k: int | None = None) -> int:
        if top_k is not None:
            return top_k
        if detail_level in {"detailed", "ky", "kỹ"}:
            return settings.TOP_K_DETAILED
        return settings.TOP_K

    def _build_filters(self, document_id: int | None = None) -> Dict[str, Any] | None:
        if document_id is None:
            return None
        return {"document_id": document_id}

    def _build_query_chain(
        self,
        search_type: str,
        top_k: int,
        document_id: int | None = None,
        model: str | None = None,
    ):
        return build_chain(
            self.vectorstore,
            self.embedding_service,
            search_type=search_type,
            model=model,
            top_k=top_k,
            filters=self._build_filters(document_id),
        )

    def _get_ollama_models(self) -> List[str]:
        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            with request.urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return []

    def _try_low_memory_fallback(
        self,
        question: str,
        search_type: str,
        top_k: int,
        document_id: int | None,
    ) -> Dict[str, Any] | None:
        available = self._get_ollama_models()
        if not available:
            return None

        candidate_priority = [
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "phi3:mini",
            "gemma2:2b",
            "llama3.2:1b",
        ]
        available_lower = [model.lower() for model in available]
        fallback_model = None
        for candidate in candidate_priority:
            candidate_lower = candidate.lower()
            for model_name, model_lower in zip(available, available_lower):
                if model_lower == candidate_lower or model_lower.startswith(candidate_lower) or candidate_lower in model_lower:
                    fallback_model = model_name
                    break
            if fallback_model is not None:
                break

        if fallback_model is None:
            return None

        try:
            chain = self._build_query_chain(
                search_type=search_type,
                top_k=top_k,
                document_id=document_id,
                model=fallback_model,
            )
            result = chain.invoke({"query": question})
            return {
                "status": "success",
                "answer": f"[Dang dung model nhe {fallback_model}]\n\n{result.get('result', '')}",
                "sources": self._format_sources(result.get("source_documents", [])),
            }
        except Exception as exc:
            LOG.error("Fallback model thất bại: %s", exc)
            return None

    def _format_sources(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        formatted_sources = []
        for doc in source_docs:
            formatted_sources.append(
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "page_start": doc.metadata.get("page_start"),
                    "page_end": doc.metadata.get("page_end"),
                    "document_id": doc.metadata.get("document_id"),
                }
            )
        return formatted_sources

    def add_documents(self, file_path: str, document_id: int | None = None) -> Dict[str, Any]:
        try:
            LOG.info("Đang xử lý tài liệu: %s", file_path)
            documents = self.doc_service.load_document(
                file_path,
                extra_metadata={"document_id": document_id} if document_id is not None else None,
            )
            if not documents:
                raise ValueError("Không có nội dung được tải từ file")

            vectors = self.embedding_service.embed_documents(documents)
            metas = []
            for document in documents:
                metas.append(
                    {
                        "text": document.page_content,
                        "source": document.metadata.get("source", os.path.basename(file_path)),
                        "chunk": document.metadata.get("chunk", 0),
                        "page_start": document.metadata.get("page_start"),
                        "page_end": document.metadata.get("page_end"),
                        "document_id": document.metadata.get("document_id"),
                    }
                )

            vector_ids = self.vectorstore.add(vectors, metas)
            self.vectorstore.save()

            chunk_rows = []
            for vector_id, meta in zip(vector_ids, metas):
                chunk_rows.append(
                    {
                        "chunk_index": meta.get("chunk", 0),
                        "page": meta.get("page_start"),
                        "text_excerpt": meta.get("text", "")[:240],
                        "vector_id": vector_id,
                    }
                )

            result = {
                "status": "success",
                "file": os.path.basename(file_path),
                "chunks_added": len(documents),
                "message": f"Đã thêm {len(documents)} chunks từ {os.path.basename(file_path)}",
                "chunks": chunk_rows,
            }
            LOG.info(result["message"])
            return result
        except Exception as exc:
            error_msg = f"Lỗi xử lý file {file_path}: {exc}"
            LOG.error(error_msg)
            return {"status": "error", "message": error_msg}

    def query(
        self,
        question: str,
        search_type: str = "vector",
        top_k: int | None = None,
        document_id: int | None = None,
        detail_level: str = "fast",
    ) -> Dict[str, Any]:
        try:
            if self.vectorstore.index.ntotal == 0:
                return {
                    "status": "error",
                    "answer": "Chưa có documents. Vui lòng tải tài liệu trước.",
                    "sources": [],
                }

            resolved_search_type = self._resolve_search_type(search_type)
            resolved_top_k = self._resolve_top_k(detail_level=detail_level, top_k=top_k)
            chain = self._build_query_chain(
                search_type=resolved_search_type,
                top_k=resolved_top_k,
                document_id=document_id,
            )
            result = chain.invoke({"query": question})
            answer = result.get("result", "")
            sources = self._format_sources(result.get("source_documents", []))
            return {"status": "success", "answer": answer, "sources": sources}
        except Exception as exc:
            error_msg = f"Lỗi query: {exc}"
            LOG.error(error_msg)

            if "requires more system memory" in str(exc).lower():
                fallback_result = self._try_low_memory_fallback(
                    question=question,
                    search_type=self._resolve_search_type(search_type),
                    top_k=self._resolve_top_k(detail_level=detail_level, top_k=top_k),
                    document_id=document_id,
                )
                if fallback_result is not None:
                    return fallback_result

                installed_models = self._get_ollama_models()
                installed_text = ", ".join(installed_models[:5]) if installed_models else "(khong doc duoc danh sach model)"
                return {
                    "status": "error",
                    "answer": (
                        "Model Ollama hiện tại vượt quá RAM khả dụng của máy. "
                        "Hãy pull model nhẹ hơn, ví dụ: `ollama pull qwen2.5:0.5b` hoặc `ollama pull qwen2.5:1.5b`, "
                        "sau đó đặt LLM_MODEL tương ứng rồi thử lại. "
                        f"Model hiện có: {installed_text}"
                    ),
                    "sources": [],
                }

            return {"status": "error", "answer": f"Có lỗi xảy ra: {exc}", "sources": []}

    def clear_vectorstore(self):
        try:
            if os.path.exists(settings.VECTOR_DIR):
                import shutil

                shutil.rmtree(settings.VECTOR_DIR)
            self.vectorstore = FaissStore(self.embedding_dim)
            LOG.info("Xóa vectorstore thành công")
            return {"status": "success", "message": "Vectorstore cleared"}
        except Exception as exc:
            LOG.error("Lỗi xóa vectorstore: %s", exc)
            return {"status": "error", "message": str(exc)}

    def get_status(self) -> Dict[str, Any]:
        return {
            "vectorstore_ready": self.vectorstore.index.ntotal > 0,
            "rag_chain_ready": self.vectorstore.index.ntotal > 0,
            "total_documents": int(self.vectorstore.index.ntotal),
            "embedding_model": self.embedding_service.model_name,
            "llm_model": settings.LLM_MODEL,
            "llm_url": settings.OLLAMA_BASE_URL,
        }
