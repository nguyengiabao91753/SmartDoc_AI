from typing import List, Dict, Any
import os
import json
from urllib import request, error

# Limit BLAS thread usage on low-memory Windows environments.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from app.core.config import settings
from app.core.logger import LOG
from app.services.document_service import DocumentService
from langchain_core.documents import Document


class RAGService:
    """
    Service chính để xử lý RAG (Retrieval-Augmented Generation) với Ollama
    
    Flow:
    1. Document Loading → DocumentService
    2. Embedding → Sentence Transformers
    3. Vector Storage → FAISS
    4. Retrieval + LLM → Ollama (thông qua rag_chain)
    """
    
    def __init__(self):
        LOG.info("Khởi tạo RAGService...")

        # Force HF cache to project data dir (often D:), tránh đầy ổ C: của user profile.
        hf_cache_dir = os.path.join(settings.DATA_DIR, "hf_cache")
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
        
        # Document loader
        self.doc_service = DocumentService()
        
        # Embeddings
        LOG.info(f"Tải embedding model: {settings.EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer

        self.embeddings = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embedding_dim = self.embeddings.get_sentence_embedding_dimension()
        
        # Vector store
        self.vectorstore = None
        self.rag_chain = None
        
        # Load vectorstore nếu đã có
        self._load_vectorstore()
        
        LOG.info("RAGService khởi tạo xong ✓")
    
    def _load_vectorstore(self):
        """Load existing vectorstore từ disk (nếu có)"""
        try:
            if os.path.exists(settings.VECTOR_DIR):
                import faiss
                from app.vectorstore.faiss_store import FaissStore

                index_file = os.path.join(settings.VECTOR_DIR, "faiss.index")
                meta_file = os.path.join(settings.VECTOR_DIR, "meta.pkl")
                
                if os.path.exists(index_file):
                    self.vectorstore = FaissStore(self.embedding_dim)
                    index = faiss.read_index(index_file)
                    self.vectorstore.index = index
                    
                    # Load metadata
                    if os.path.exists(meta_file):
                        import pickle
                        with open(meta_file, "rb") as f:
                            self.vectorstore.meta = pickle.load(f)
                    
                    LOG.info(f"Tải vectorstore từ disk ✓ ({self.vectorstore.index.ntotal} documents)")
                    self._rebuild_rag_chain()
        except Exception as e:
            LOG.warning(f"Không thể load vectorstore: {str(e)}")
            self.vectorstore = None
    
    def _rebuild_rag_chain(self):
        """Tái tạo RAG chain (gọi khi vectorstore thay đổi)"""
        if self.vectorstore and self.vectorstore.index.ntotal > 0:
            try:
                from app.ai.rag_chain import build_chain

                self.rag_chain = build_chain(self.vectorstore, self.embeddings)
                LOG.info("RAG chain tái tạo ✓")
            except Exception as e:
                LOG.error(f"Lỗi tái tạo RAG chain: {str(e)}")

    def _get_ollama_models(self) -> List[str]:
        """Đọc danh sách models đang có từ Ollama local."""
        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            with request.urlopen(url, timeout=2) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return []

    def _try_low_memory_fallback(self, question: str) -> Dict[str, Any] | None:
        """Thử model nhẹ hơn nếu model hiện tại bị lỗi thiếu RAM."""
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
        available_lower = [m.lower() for m in available]
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
            from app.ai.rag_chain import build_chain

            fallback_chain = build_chain(self.vectorstore, self.embeddings, model=fallback_model)
            result = fallback_chain.invoke({"query": question})
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])

            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0)
                })

            LOG.warning(f"Fallback sang model nhẹ: {fallback_model}")
            return {
                "status": "success",
                "answer": f"[Dang dung model nhe {fallback_model}]\n\n{answer}",
                "sources": sources,
            }
        except Exception as fallback_error:
            LOG.error(f"Fallback model thất bại: {str(fallback_error)}")
            return None
    
    def add_documents(self, file_path: str) -> Dict[str, Any]:
        """
        Thêm documents từ file vào vectorstore
        
        Args:
            file_path: Đường dẫn file (PDF/DOCX)
            
        Returns:
            Dict với thông tin về document được thêm
        """
        try:
            LOG.info(f"Đang xử lý: {file_path}")
            
            # Load document
            documents: List[Document] = self.doc_service.load_document(file_path)
            
            if not documents:
                raise ValueError("Không có nội dung được tải từ file")
            
            # Embedding
            LOG.info(f"Tạo embedding cho {len(documents)} chunks...")
            texts = [doc.page_content for doc in documents]
            vectors = self.embeddings.encode(texts, convert_to_numpy=True)
            
            # Chuẩn hóa vectors (FAISS dùng inner product nên cần chuẩn hóa)
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
            
            # Khởi tạo vectorstore nếu chưa có
            if self.vectorstore is None:
                from app.vectorstore.faiss_store import FaissStore

                self.vectorstore = FaissStore(self.embedding_dim)
            
            # Thêm vào vectorstore
            metas = []
            for doc in documents:
                metas.append({
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0)
                })
            
            self.vectorstore.add(vectors, metas)
            
            # Lưu vectorstore
            self._save_vectorstore()
            
            # Tái tạo RAG chain
            self._rebuild_rag_chain()
            
            result = {
                "status": "success",
                "file": os.path.basename(file_path),
                "chunks_added": len(documents),
                "message": f"Đã thêm {len(documents)} chunks từ {os.path.basename(file_path)}"
            }
            
            LOG.info(result["message"])
            return result
            
        except Exception as e:
            error_msg = f"Lỗi xử lý file {file_path}: {str(e)}"
            LOG.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _save_vectorstore(self):
        """Lưu vectorstore xuống disk"""
        try:
            if self.vectorstore is None:
                return
            
            os.makedirs(settings.VECTOR_DIR, exist_ok=True)
            
            import faiss
            index_file = os.path.join(settings.VECTOR_DIR, "faiss.index")
            faiss.write_index(self.vectorstore.index, index_file)
            
            # Lưu metadata
            import pickle
            meta_file = os.path.join(settings.VECTOR_DIR, "meta.pkl")
            with open(meta_file, "wb") as f:
                pickle.dump(self.vectorstore.meta, f)
            
            LOG.info(f"Lưu vectorstore: {index_file}")
        except Exception as e:
            LOG.error(f"Lỗi lưu vectorstore: {str(e)}")
    
    def query(self, question: str, search_type: str = "vector", top_k: int = None) -> Dict[str, Any]:
        """
        Query RAG chain với Ollama
        
        Args:
            question: Câu hỏi của user
            search_type: "vector" hoặc "hybrid" (vector + keyword)
            top_k: Số documents để retrieve (mặc định từ config)
            
        Returns:
            Dict với answer và source documents
        """
        try:
            if self.rag_chain is None:
                return {
                    "status": "error",
                    "answer": "Chưa có documents. Vui lòng tải tài liệu trước.",
                    "sources": []
                }
            
            LOG.info(f"Query: {question}")
            
            # RAG chain sẽ:
            # 1. Retrieve documents từ vectorstore
            # 2. Gọi Ollama LLM với retrieved documents
            # 3. Trả về answer + source documents
            result = self.rag_chain.invoke({
                "query": question
            })
            
            # Xử lý kết quả
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0)
                })
            
            return {
                "status": "success",
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            error_msg = f"Lỗi query: {str(e)}"
            LOG.error(error_msg)

            err_lower = str(e).lower()
            if "requires more system memory" in err_lower:
                fallback_result = self._try_low_memory_fallback(question)
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
                    "sources": []
                }

            return {
                "status": "error",
                "answer": f"Có lỗi xảy ra: {str(e)}",
                "sources": []
            }
    
    def clear_vectorstore(self):
        """Xóa vectorstore hiện tại"""
        try:
            import shutil
            if os.path.exists(settings.VECTOR_DIR):
                shutil.rmtree(settings.VECTOR_DIR)
            
            self.vectorstore = None
            self.rag_chain = None
            
            LOG.info("Xóa vectorstore thành công")
            return {"status": "success", "message": "Vectorstore cleared"}
        except Exception as e:
            LOG.error(f"Lỗi xóa vectorstore: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Lấy thông tin hiện tại của RAG service"""
        status = {
            "vectorstore_ready": self.vectorstore is not None,
            "rag_chain_ready": self.rag_chain is not None,
            "total_documents": 0,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "llm_url": settings.OLLAMA_BASE_URL
        }
        
        if self.vectorstore:
            status["total_documents"] = self.vectorstore.index.ntotal
        
        return status
