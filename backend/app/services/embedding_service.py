import hashlib
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from app.vectorstore.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class EmbeddingService:
    DEFAULT_BATCH_SIZE = 64
    MAX_TEXT_LENGTH = 512  #Giới hạn độ dài mỗi chunk khi truyền vào, vì model chỉ xử lý được 512 tokens/lần

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ):
        self.model = SentenceTransformer(model_name)
        self.vector_store = FaissStore()
        self.batch_size = batch_size
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_and_store(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Nhận danh sách texts đã được xử lý từ PDF/DOCX processor,
        embed rồi lưu xuống vector store.

        Args:
            texts:     Danh sách văn bản đã chunk.
            metadatas: Metadata tương ứng mỗi text (source, page, ...).
                       Nếu None thì tự build metadata tối thiểu.

        Returns:
            {"stored": int, "skipped": int, "failed": int}
        """
        if not texts:
            logger.warning("embed_and_store called with empty texts list.")
            return {"stored": 0, "skipped": 0, "failed": 0}

        valid_texts, valid_metas, skipped = self._validate_and_filter(
            texts, metadatas
        )

        if not valid_texts:
            logger.warning("No valid texts after filtering.")
            return {"stored": 0, "skipped": skipped, "failed": 0}

        stored, failed = self._embed_in_batches(valid_texts, valid_metas)

        logger.info(
            "embed_and_store complete | stored=%d skipped=%d failed=%d",
            stored, skipped, failed,
        )
        return {"stored": stored, "skipped": skipped, "failed": failed}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_filter(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]],
    ):
        """Loại bỏ text rỗng / None, build metadata nếu thiếu."""
        valid_texts, valid_metas = [], []
        skipped = 0

        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.debug("Skipping empty text at index %d", i)
                skipped += 1
                continue

            meta = (metadatas[i] if metadatas and i < len(metadatas) else {})
            valid_texts.append(text.strip())
            valid_metas.append(self._build_metadata(text, meta))

        return valid_texts, valid_metas, skipped

    def _build_metadata(self, text: str, extra: Dict) -> Dict:
        """Đảm bảo metadata luôn có các trường chuẩn."""
        return {
            "text": text,
            "text_hash": hashlib.md5(text.encode()).hexdigest(),
            "char_length": len(text),
            "embedded_at": datetime.utcnow().isoformat(),
            **extra,  # source, page_number, chunk_index, ... từ processor
        }

    def _embed_in_batches(
        self,
        texts: List[str],
        metas: List[Dict],
    ):
        """Encode theo batch để tránh OOM, lưu từng batch."""
        stored = failed = 0
        total = len(texts)

        for start in range(0, total, self.batch_size):
            batch_texts = texts[start: start + self.batch_size]
            batch_metas = metas[start: start + self.batch_size]

            try:
                vectors = self._encode(batch_texts)
                self.vector_store.add(vectors, batch_metas)
                self.vector_store.save()
                stored += len(batch_texts)
                logger.debug(
                    "Batch %d-%d stored (%d/%d)",
                    start, start + len(batch_texts) - 1, stored, total,
                )
            except Exception:
                failed += len(batch_texts)
                logger.exception(
                    "Failed to embed/store batch %d-%d",
                    start, start + len(batch_texts) - 1,
                )

        return stored, failed

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode + optional L2 normalize."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = np.array(embeddings, dtype="float32")

        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # tránh chia 0
            vectors = vectors / norms

        return vectors