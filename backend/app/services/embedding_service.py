import logging
import os
from typing import Iterable, List

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    DEFAULT_BATCH_SIZE = 64

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ):
        hf_cache_dir = os.path.join(settings.DATA_DIR, "hf_cache")
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()

    def get_dimension(self) -> int:
        return self.dim

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [text for text in texts if text and text.strip()]
        if not text_list:
            return np.empty((0, self.dim), dtype="float32")

        embeddings = self.model.encode(
            text_list,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        return self._normalize(np.asarray(embeddings, dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        embeddings = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        vector = np.asarray(embeddings, dtype="float32")
        return self._normalize(vector)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors.astype("float32")

        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return (vectors / norm if norm else vectors).astype("float32")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vectors / norms).astype("float32")
