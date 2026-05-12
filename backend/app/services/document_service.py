from typing import Any, Dict, List
import os

import re
import docx
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import LOG
from app.core.chunk_params import validate_chunk_params
from app.nlp.underthesea_compat import sent_tokenize

# --- Tokenization (tiktoken is optional at import-time) ---
try:
    import tiktoken  # type: ignore
except ImportError:
    tiktoken = None  # type: ignore


# --- LOGIC HIỆN TẠI (GIỮ THỜI TƯƠNG THÍCH, SẼ CHỈNH PARAM THEO UI Ở BƯỚC SAU) ---


try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

if tiktoken is None:
    # Fallback token estimation when tiktoken is unavailable
    def token_len(text: str) -> int:
        return max(1, len(text) // 4)

    def trim_to_token_limit(text: str, max_tokens: int) -> str:
        # Roughly align with the token_len fallback
        approx_chars = max_tokens * 4
        return text[:approx_chars].strip()
else:
    enc = tiktoken.get_encoding("cl100k_base")



if tiktoken is not None:
    def token_len(text: str) -> int:
        return len(enc.encode(text))

    def trim_to_token_limit(text: str, max_tokens: int) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])




def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def extract_text_from_page(page) -> str:
    text = page.extract_text() or ""
    if not text.strip() and OCR_AVAILABLE:
        try:
            img = page.to_image(resolution=300).original
            text = pytesseract.image_to_string(img, lang="eng+vie")
            LOG.debug("Used OCR for page")
        except Exception as e:
            LOG.warning(f"OCR failed: {e}")
    return clean_text(text)


def process_pdf(
    file_path: str,
    *,
    chunk_size: int,
    chunk_overlap_sentences: int,
) -> List[Dict]:
    LOG.info("Processing PDF: %s", file_path)
    chunks: List[Dict] = []
    chunk_id = 0
    try:
        try:
            import pdfplumber  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pdfplumber is required to process PDF files. Install it with `pip install pdfplumber`."
            ) from e

        with pdfplumber.open(file_path) as pdf:

            num_pages = len(pdf.pages)
            LOG.info("Total pages: %s", num_pages)
            current_sentences: List[str] = []
            current_token_count = 0
            current_page_start = 1

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = extract_text_from_page(page)
                if not page_text:
                    continue

                sentences = sent_tokenize(page_text)
                for sentence in sentences:
                    sentence_tokens = token_len(sentence)

                    if sentence_tokens > chunk_size:
                        LOG.warning("Long sentence detected -> trimming by token")
                        sentence = trim_to_token_limit(sentence, chunk_size)
                        sentence_tokens = token_len(sentence)

                    if current_token_count + sentence_tokens > chunk_size:
                        if current_sentences:
                            chunk_text = " ".join(current_sentences)
                            chunks.append(
                                {
                                    "id": chunk_id,
                                    "text": chunk_text,
                                    "tokens": current_token_count,
                                    "page_start": current_page_start,
                                    "page_end": page_num,
                                }
                            )
                            chunk_id += 1

                        # overlap theo số sentence
                        overlap_sentences = (
                            current_sentences[-chunk_overlap_sentences:]
                            if chunk_overlap_sentences > 0
                            else []
                        )
                        current_sentences = overlap_sentences.copy()
                        current_token_count = sum(token_len(s) for s in current_sentences)
                        current_page_start = page_num

                    current_sentences.append(sentence)
                    current_token_count += sentence_tokens

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "tokens": current_token_count,
                        "page_start": current_page_start,
                        "page_end": num_pages,
                    }
                )
    except Exception as e:
        LOG.error(f"PDF processing error: {e}")
        return []

    LOG.info("Generated %s chunks", len(chunks))
    return chunks


def process_docx(
    file_path: str,
    *,
    chunk_size: int,
    chunk_overlap_sentences: int,
) -> List[Dict]:
    LOG.info("Processing DOCX: %s", file_path)
    doc = docx.Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    sentences = sent_tokenize(full_text)
    chunks: List[Dict] = []
    chunk_id = 0
    current_sentences: List[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = token_len(sentence)

        if sentence_tokens > chunk_size:
            sentence = trim_to_token_limit(sentence, chunk_size)
            sentence_tokens = token_len(sentence)

        if current_token_count + sentence_tokens > chunk_size:
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({"id": chunk_id, "text": chunk_text, "tokens": current_token_count})
                chunk_id += 1

            overlap_sentences = (
                current_sentences[-chunk_overlap_sentences:]
                if chunk_overlap_sentences > 0
                else []
            )
            current_sentences = overlap_sentences.copy()
            current_token_count = sum(token_len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({"id": chunk_id, "text": chunk_text, "tokens": current_token_count})

    LOG.info("Generated %s chunks from DOCX", len(chunks))
    return chunks


class DocumentService:
    """Dịch vụ load tài liệu và chunk hóa.

    Mục tiêu: giữ pipeline hiện có, đồng thời cho phép inject chunk params từ UI ở tầng trên.
    """

    def load_document(
        self,
        file_path: str,
        extra_metadata: Dict[str, Any] | None = None,
        *,
        chunk_size: int | None = None,
        chunk_overlap_sentences: int | None = None,
    ) -> List[Document]:
        source = os.path.basename(file_path)
        _, extension = os.path.splitext(file_path.lower())

        resolved_chunk_size = int(chunk_size) if chunk_size is not None else int(settings.CHUNK_SIZE)
        resolved_overlap = (
            int(chunk_overlap_sentences)
            if chunk_overlap_sentences is not None
            else int(settings.OVERLAP_SENTENCES)
        )

        # Validate chunk params for every ingestion flow (prevents overlap out-of-range).
        config = validate_chunk_params(
            chunk_size=resolved_chunk_size,
            chunk_overlap_sentences=resolved_overlap,
        )

        processed_chunks: List[Dict] = []
        if extension == ".pdf":
            processed_chunks = process_pdf(
                file_path,
                chunk_size=config.chunk_size,
                chunk_overlap_sentences=config.chunk_overlap_sentences,
            )
        elif extension == ".docx":
            processed_chunks = process_docx(
                file_path,
                chunk_size=config.chunk_size,
                chunk_overlap_sentences=config.chunk_overlap_sentences,
            )

        else:
            raise ValueError(f"Unsupported file format: {extension}")

        langchain_docs: List[Document] = []
        for i, chunk in enumerate(processed_chunks):
            metadata = {
                "source": source,
                "chunk": chunk.get("id", i),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}

            doc = Document(page_content=chunk["text"], metadata=metadata)
            langchain_docs.append(doc)

        return langchain_docs

