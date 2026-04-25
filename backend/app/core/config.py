from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[3]  # smartdoc-ai/backend/app/..
class Settings(BaseSettings):
    # Ollama / LLM
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5:2b")
    LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", 1024))
    LLM_NUM_PREDICT: int = int(os.getenv("LLM_NUM_PREDICT", 256))
    LLM_NUM_BATCH: int = int(os.getenv("LLM_NUM_BATCH", 512))
    LLM_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", "5m")

    # Embedding model (sentence-transformers)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Paths
    DATA_DIR: str = str(ROOT / "data")
    DOCUMENT_DIR: str = str(ROOT / "data" / "documents")
    VECTOR_DIR: str = str(ROOT / "data" / "vectorstore")
    SQLITE_PATH: str = str(ROOT / "data" / "sqlite" / "chat.db")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    OVERLAP_SENTENCES: int = int(os.getenv("OVERLAP_SENTENCES", 2))

    # FAISS params
    TOP_K: int = int(os.getenv("TOP_K", 2))
    TOP_K_DETAILED: int = int(os.getenv("TOP_K_DETAILED", 6))
    RAG_MODE: str = os.getenv("RAG_MODE", "rag")
    
    #NEO4J
    link_graphrag: str = str(ROOT / "backend" / "app" / "rag" / "modes" / "graphrag" / ".env")
    load_dotenv(link_graphrag) 
    NEO4J_URI: str = str(os.getenv("NEO4J_URI"))
    NEO4J_USERNAME: str = str(os.getenv("NEO4J_USERNAME"))
    NEO4J_PASSWORD: str = str(os.getenv("NEO4J_PASSWORD"))
    NEO4J_DATABASE: str = str(os.getenv("NEO4J_DATABASE"))

    model_config = SettingsConfigDict(
        env_file=(str(ROOT / ".env"), str(ROOT / "backend" / ".env"))
    )

settings = Settings()

###

# from pydantic import BaseSettings
# from pathlib import Path
# import os

# # Đường dẫn gốc của project
# ROOT = Path(__file__).resolve().parents[3]  # smartdoc-ai/backend/app/..

# class Settings(BaseSettings):
#     # --- LLM Cloud Config (OpenRouter / OpenAI compatible) ---
#     # Mặc định dùng OpenRouter nếu không có trong .env
#     LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
#     LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3-coder:free")
#     LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

#     # Tên định danh cho ứng dụng (OpenRouter yêu cầu để thống kê, tùy chọn)
#     HTTP_REFERER: str = os.getenv("HTTP_REFERER", "http://localhost:3000")
#     X_TITLE: str = os.getenv("X_TITLE", "SmartDoc AI")

#     # --- Embedding model (Vẫn giữ local để bảo mật dữ liệu vector) ---
#     EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

#     # --- Paths ---
#     DATA_DIR: str = str(ROOT / "data")
#     DOCUMENT_DIR: str = str(ROOT / "data" / "documents")
#     VECTOR_DIR: str = str(ROOT / "data" / "vectorstore")
#     SQLITE_PATH: str = str(ROOT / "data" / "sqlite" / "chat.db")

#     # --- RAG Params ---
#     CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
#     CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
#     TOP_K: int = int(os.getenv("TOP_K", 3))

#     class Config:
#         env_file = str(ROOT / ".env")
#         # Cho phép lấy dữ liệu từ môi trường hệ thống nếu không có trong .env
#         env_file_encoding = 'utf-8'

# settings = Settings()
