from pydantic_settings import BaseSettings
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]  # smartdoc-ai/backend/app/..

class Settings(BaseSettings):
    # Ollama / LLM
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")

    # Embedding model (sentence-transformers)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Paths
    DATA_DIR: str = str(ROOT / "data")
    DOCUMENT_DIR: str = str(ROOT / "data" / "documents")
    VECTOR_DIR: str = str(ROOT / "data" / "vectorstore")
    SQLITE_PATH: str = str(ROOT / "data" / "sqlite" / "chat.db")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # FAISS params
    TOP_K: int = int(os.getenv("TOP_K", 3))

    class Config:
        env_file = str(ROOT / ".env")

settings = Settings()