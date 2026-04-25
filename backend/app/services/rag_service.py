from typing import Any, Dict, List
import json
import os
from urllib import error, request

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import LOG
from app.rag.base import ModeNotImplementedError
from app.rag.models import RAGQueryRequest
from app.rag.registry import AVAILABLE_RAG_MODES, build_engine_registry, normalize_rag_mode
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.vectorstore.faiss_store import FaissStore


class RAGService:
    """Coordinates ingestion and delegates querying to mode-specific RAG engines."""

    def __init__(self):
        LOG.info("Khoi tao RAGService...")
        self.doc_service = DocumentService()
        self.embedding_service = EmbeddingService()
        self.embedding_dim = self.embedding_service.get_dimension()
        self.vectorstore = FaissStore(self.embedding_dim)
        self.vectorstore.load()
        self._rebuild_engines()
        LOG.info("RAGService khoi tao xong")

    def _rebuild_engines(self):
        try:
            self.engines = build_engine_registry(self.vectorstore, self.embedding_service)
        except Exception as exc:
            import traceback
            LOG.error("RAG init error chi tiet:\n%s", traceback.format_exc()) # Sẽ hiện lỗi đỏ lòm rất chi tiết
            raise exc

    def _resolve_rag_mode(self, rag_mode: str | None = None) -> str:
        resolved = normalize_rag_mode(rag_mode)
        if resolved not in self.engines:
            return "rag"
        return resolved

    def _resolve_search_type(self, search_type: str | None) -> str:
        normalized = (search_type or "vector").lower()
        if normalized in {"keyword", "hybrid"}:
            return "hybrid"
        return "vector"

    def _resolve_top_k(self, detail_level: str = "fast", top_k: int | None = None) -> int:
        if top_k is not None:
            return top_k
        if detail_level in {"detailed", "ky", "ki"}:
            return settings.TOP_K_DETAILED
        return settings.TOP_K

    def _execute_query(
        self,
        *,
        question: str,
        rag_mode: str,
        search_type: str,
        top_k: int,
        document_id: int | None,
        document_ids: List[int] | None = None,
        session_id: int | None = None,
        llm_model: str | None = None,
    ) -> Dict[str, Any]:
        engine = self.engines[rag_mode]
        result = engine.query(
            RAGQueryRequest(
                question=question,
                search_type=search_type,
                top_k=top_k,
                document_id=document_id,
                document_ids=document_ids,
                session_id=session_id,
                llm_model=llm_model,
            )
        )
        return {
            "status": "success",
            "answer": result.answer,
            "sources": self._format_sources(result.source_documents),
            "rag_mode": rag_mode,
            "metadata": result.metadata,
        }

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
        *,
        question: str,
        rag_mode: str,
        search_type: str,
        top_k: int,
        document_id: int | None,
        document_ids: List[int] | None = None,
        session_id: int | None = None,
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
            fallback_result = self._execute_query(
                question=question,
                rag_mode=rag_mode,
                search_type=search_type,
                top_k=top_k,
                document_id=document_id,
                document_ids=document_ids,
                session_id=session_id,
                llm_model=fallback_model,
            )
            fallback_result["answer"] = f"[Dang dung model nhe {fallback_model}]\n\n{fallback_result['answer']}"
            return fallback_result
        except Exception as exc:
            LOG.error("Fallback model that bai: %s", exc)
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

    def add_documents(
        self,
        file_path: str,
        document_id: int | None = None,
        session_id: int | None = None,
    ) -> Dict[str, Any]:
        try:
            LOG.info("Dang xu ly tai lieu: %s", file_path)
            documents = self.doc_service.load_document(
                file_path,
                extra_metadata={"document_id": document_id} if document_id is not None else None,
            )
            if not documents:
                raise ValueError("Khong co noi dung duoc tai tu file")

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
                        "session_id": session_id,
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
                "message": f"Da them {len(documents)} chunks tu {os.path.basename(file_path)}",
                "chunks": chunk_rows,
            }
            LOG.info(result["message"])
            return result
        except Exception as exc:
            error_msg = f"Loi xu ly file {file_path}: {exc}"
            LOG.error(error_msg)
            return {"status": "error", "message": error_msg}

    def query(
        self,
        question: str,
        search_type: str = "vector",
        top_k: int | None = None,
        document_id: int | None = None,
        document_ids: List[int] | None = None,
        session_id: int | None = None,
        detail_level: str = "fast",
        rag_mode: str | None = None,
    ) -> Dict[str, Any]:
        if self.vectorstore.index.ntotal == 0:
            return {
                "status": "error",
                "answer": "Chua co documents. Vui long tai tai lieu truoc.",
                "sources": [],
                "rag_mode": self._resolve_rag_mode(rag_mode),
            }

        resolved_search_type = self._resolve_search_type(search_type)
        resolved_top_k = self._resolve_top_k(detail_level=detail_level, top_k=top_k)
        resolved_mode = self._resolve_rag_mode(rag_mode)

        try:
            return self._execute_query(
                question=question,
                rag_mode=resolved_mode,
                search_type=resolved_search_type,
                top_k=resolved_top_k,
                document_id=document_id,
                document_ids=document_ids,
                session_id=session_id,
            )
        except ModeNotImplementedError as exc:
            return {
                "status": "error",
                "answer": str(exc),
                "sources": [],
                "rag_mode": resolved_mode,
            }
        except Exception as exc:
            error_msg = f"Loi query ({resolved_mode}): {exc}"
            LOG.error(error_msg)

            if "requires more system memory" in str(exc).lower():
                fallback_result = self._try_low_memory_fallback(
                    question=question,
                    rag_mode=resolved_mode,
                    search_type=resolved_search_type,
                    top_k=resolved_top_k,
                    document_id=document_id,
                    document_ids=document_ids,
                    session_id=session_id,
                )
                if fallback_result is not None:
                    return fallback_result

                installed_models = self._get_ollama_models()
                installed_text = ", ".join(installed_models[:5]) if installed_models else "(khong doc duoc danh sach model)"
                return {
                    "status": "error",
                    "answer": (
                        "Model Ollama hien tai vuot qua RAM kha dung cua may. "
                        "Hay pull model nhe hon, vi du: `ollama pull qwen2.5:0.5b` hoac `ollama pull qwen2.5:1.5b`, "
                        "sau do dat LLM_MODEL tuong ung roi thu lai. "
                        f"Model hien co: {installed_text}"
                    ),
                    "sources": [],
                    "rag_mode": resolved_mode,
                }

            return {
                "status": "error",
                "answer": f"Co loi xay ra: {exc}",
                "sources": [],
                "rag_mode": resolved_mode,
            }

    def clear_vectorstore(self):
        try:
            if os.path.exists(settings.VECTOR_DIR):
                import shutil

                shutil.rmtree(settings.VECTOR_DIR)
            self.vectorstore = FaissStore(self.embedding_dim)
            self._rebuild_engines()
            LOG.info("Xoa vectorstore thanh cong")
            return {"status": "success", "message": "Vectorstore cleared"}
        except Exception as exc:
            LOG.error("Loi xoa vectorstore: %s", exc)
            return {"status": "error", "message": str(exc)}

    def get_status(self) -> Dict[str, Any]:
        return {
            "vectorstore_ready": self.vectorstore.index.ntotal > 0,
            "rag_chain_ready": self.vectorstore.index.ntotal > 0,
            "total_documents": int(self.vectorstore.index.ntotal),
            "embedding_model": self.embedding_service.model_name,
            "llm_model": settings.LLM_MODEL,
            "llm_url": settings.OLLAMA_BASE_URL,
            "available_modes": list(AVAILABLE_RAG_MODES),
        }

