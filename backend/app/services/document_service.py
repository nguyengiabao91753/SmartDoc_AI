from __future__ import annotations

from typing import Any, Dict, List
import os
import re

import docx
from langchain_core.documents import Document

from app.core.chunk_params import validate_chunk_params
from app.core.config import settings
from app.core.logger import LOG
from app.nlp.underthesea_compat import sent_tokenize


try:
    import tiktoken  # type: ignore
except ImportError:
    tiktoken = None  # type: ignore


try:
    import pytesseract

    if getattr(settings, "TESSERACT_CMD", None):
        pytesseract.pytesseract_cmd = settings.TESSERACT_CMD
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None  # type: ignore
    OCR_AVAILABLE = False


if tiktoken is not None:
    enc = tiktoken.get_encoding("cl100k_base")

    def token_len(text: str) -> int:
        return len(enc.encode(text))

    def trim_to_token_limit(text: str, max_tokens: int) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])

else:

    def token_len(text: str) -> int:
        return max(1, len(text) // 4)

    def trim_to_token_limit(text: str, max_tokens: int) -> str:
        return text[: max_tokens * 4].strip()


IMAGES_DIR = os.path.join(settings.DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def _resolve_chunk_params(
    chunk_size: int | None,
    chunk_overlap_sentences: int | None,
):
    return validate_chunk_params(
        chunk_size=int(chunk_size) if chunk_size is not None else int(settings.CHUNK_SIZE),
        chunk_overlap_sentences=(
            int(chunk_overlap_sentences)
            if chunk_overlap_sentences is not None
            else int(settings.OVERLAP_SENTENCES)
        ),
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _overlap_tail(sentences: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or not sentences:
        return []

    overlap_count = min(overlap, max(len(sentences) - 1, 0))
    if overlap_count <= 0:
        return []
    return sentences[-overlap_count:]


def save_image_from_page(page, page_num: int, document_id: int, chunk_id: int) -> str | None:
    try:
        img = page.to_image(resolution=200).original
        doc_img_dir = os.path.join(IMAGES_DIR, f"doc_{document_id}")
        os.makedirs(doc_img_dir, exist_ok=True)

        img_name = f"chunk_{chunk_id}_page_{page_num}.png"
        img_path = os.path.join(doc_img_dir, img_name)
        img.save(img_path, "PNG", quality=85)

        rel_path = os.path.join("images", f"doc_{document_id}", img_name)
        LOG.debug("Saved image: %s", rel_path)
        return rel_path
    except Exception as exc:
        LOG.warning("Failed to save image from page %s: %s", page_num, exc)
        return None


def _ocr_embedded_images(page, page_num: int) -> str:
    if not OCR_AVAILABLE:
        return ""

    embedded = page.images
    if not embedded:
        return ""

    LOG.debug("Page %s: found %s embedded image(s)", page_num, len(embedded))
    ocr_parts: List[str] = []

    for idx, img_info in enumerate(embedded):
        try:
            x0 = img_info.get("x0", 0)
            top = img_info.get("top", 0)
            x1 = img_info.get("x1", 0)
            bottom = img_info.get("bottom", 0)

            width_pt = x1 - x0
            height_pt = bottom - top
            if width_pt < 80 or height_pt < 80:
                LOG.debug(
                    "Image %s on page %s too small (%.0fx%.0fpt), skipping",
                    idx,
                    page_num,
                    width_pt,
                    height_pt,
                )
                continue

            cropped = page.crop((x0, top, x1, bottom))
            pil_img = cropped.to_image(resolution=300).original
            img_text = pytesseract.image_to_string(pil_img, lang="vie+eng", config="--psm 6")
            img_text = clean_text(img_text)

            if img_text:
                LOG.debug("Image %s on page %s OCR extracted %s chars", idx, page_num, len(img_text))
                ocr_parts.append(img_text)
        except Exception as exc:
            LOG.warning("OCR failed on embedded image %s, page %s: %s", idx, page_num, exc)

    return "\n".join(ocr_parts)


def extract_text_from_page(
    page,
    page_num: int | None = None,
    document_id: int | None = None,
    chunk_id: int | None = None,
) -> Dict[str, Any]:
    native_text = clean_text(page.extract_text() or "")
    has_ocr = False
    image_path = None

    if document_id is not None and page_num is not None and chunk_id is not None:
        image_path = save_image_from_page(page, page_num, document_id, chunk_id)

    embedded_ocr_text = _ocr_embedded_images(page, page_num or 0)
    if embedded_ocr_text:
        has_ocr = True

    full_page_ocr_text = ""
    if not native_text and not embedded_ocr_text and OCR_AVAILABLE:
        try:
            LOG.debug("Page %s: no text found, running full-page OCR", page_num)
            pil_img = page.to_image(resolution=300).original
            full_page_ocr_text = clean_text(
                pytesseract.image_to_string(pil_img, lang="vie+eng", config="--psm 6")
            )
            if full_page_ocr_text:
                has_ocr = True
                LOG.debug("Page %s: full-page OCR extracted %s chars", page_num, len(full_page_ocr_text))
        except Exception as exc:
            LOG.warning("Full-page OCR failed for page %s: %s", page_num, exc)

    final_parts = [text for text in [native_text, embedded_ocr_text, full_page_ocr_text] if text]
    return {
        "text": "\n\n".join(final_parts),
        "has_ocr": has_ocr,
        "image_path": image_path,
    }


def process_pdf(
    file_path: str,
    document_id: int | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap_sentences: int | None = None,
) -> List[Dict[str, Any]]:
    config = _resolve_chunk_params(chunk_size, chunk_overlap_sentences)
    LOG.info("Processing PDF: %s", file_path)

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    pages_with_no_text = 0

    try:
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required to process PDF files. Install it with `pip install pdfplumber`."
            ) from exc

        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            LOG.info("Total pages: %s", num_pages)

            current_sentences: List[str] = []
            current_token_count = 0
            current_page_start = 1
            chunk_images: List[str] = []
            chunk_has_ocr = False
            all_ocr_texts: List[str] = []

            for page_num, page in enumerate(pdf.pages, 1):
                page_extract = extract_text_from_page(page, page_num, document_id, chunk_id)
                page_text = page_extract["text"]
                page_has_ocr = bool(page_extract["has_ocr"])
                page_image_path = page_extract["image_path"]

                if not page_text:
                    pages_with_no_text += 1
                    if page_image_path:
                        chunk_has_ocr = True
                        chunk_images.append(page_image_path)
                    continue

                if page_has_ocr:
                    chunk_has_ocr = True
                    all_ocr_texts.append(page_text)
                    if page_image_path:
                        chunk_images.append(page_image_path)

                for sentence in sent_tokenize(page_text):
                    sentence_tokens = token_len(sentence)
                    if sentence_tokens > config.chunk_size:
                        LOG.warning("Long sentence detected; trimming by token")
                        sentence = trim_to_token_limit(sentence, config.chunk_size)
                        sentence_tokens = token_len(sentence)

                    if current_token_count + sentence_tokens > config.chunk_size and current_sentences:
                        chunk_text = " ".join(current_sentences)
                        chunks.append(
                            {
                                "id": chunk_id,
                                "text": chunk_text,
                                "tokens": current_token_count,
                                "page_start": current_page_start,
                                "page_end": page_num,
                                "has_ocr": chunk_has_ocr,
                                "image_paths": chunk_images.copy(),
                            }
                        )
                        chunk_id += 1

                        current_sentences = _overlap_tail(
                            current_sentences,
                            config.chunk_overlap_sentences,
                        )
                        current_token_count = sum(token_len(item) for item in current_sentences)
                        current_page_start = page_num
                        chunk_images = []
                        chunk_has_ocr = False

                    current_sentences.append(sentence)
                    current_token_count += sentence_tokens

            if current_sentences:
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": " ".join(current_sentences),
                        "tokens": current_token_count,
                        "page_start": current_page_start,
                        "page_end": num_pages,
                        "has_ocr": chunk_has_ocr,
                        "image_paths": chunk_images.copy(),
                    }
                )

            if not chunks and all_ocr_texts:
                LOG.warning("Creating chunk from collected OCR text (%s pages)", len(all_ocr_texts))
                combined_ocr_text = " ".join(all_ocr_texts)
                chunks.append(
                    {
                        "id": 0,
                        "text": combined_ocr_text,
                        "tokens": sum(token_len(text) for text in all_ocr_texts),
                        "page_start": 1,
                        "page_end": num_pages,
                        "has_ocr": True,
                        "image_paths": chunk_images.copy(),
                    }
                )
            elif not chunks and chunk_images:
                LOG.warning(
                    "PDF has %s pages but no extractable text. Creating image-only chunk.",
                    pages_with_no_text,
                )
                chunks.append(
                    {
                        "id": 0,
                        "text": f"[Scanned document - {num_pages} page(s) with images for visual reference]",
                        "tokens": 10,
                        "page_start": 1,
                        "page_end": num_pages,
                        "has_ocr": True,
                        "image_paths": chunk_images.copy(),
                    }
                )
    except Exception as exc:
        LOG.error("PDF processing error: %s", exc)
        return []

    LOG.info("Generated %s chunks, %s pages without text", len(chunks), pages_with_no_text)
    return chunks


def process_docx(
    file_path: str,
    document_id: int | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap_sentences: int | None = None,
) -> List[Dict[str, Any]]:
    del document_id
    config = _resolve_chunk_params(chunk_size, chunk_overlap_sentences)
    LOG.info("Processing DOCX: %s", file_path)

    doc = docx.Document(file_path)
    full_text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
    sentences = sent_tokenize(full_text)

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    current_sentences: List[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = token_len(sentence)
        if sentence_tokens > config.chunk_size:
            sentence = trim_to_token_limit(sentence, config.chunk_size)
            sentence_tokens = token_len(sentence)

        if current_token_count + sentence_tokens > config.chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "tokens": current_token_count,
                    "has_ocr": False,
                    "image_paths": [],
                }
            )
            chunk_id += 1

            current_sentences = _overlap_tail(
                current_sentences,
                config.chunk_overlap_sentences,
            )
            current_token_count = sum(token_len(item) for item in current_sentences)

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    if current_sentences:
        chunks.append(
            {
                "id": chunk_id,
                "text": " ".join(current_sentences),
                "tokens": current_token_count,
                "has_ocr": False,
                "image_paths": [],
            }
        )

    LOG.info("Generated %s chunks from DOCX", len(chunks))
    return chunks


class DocumentService:
    """Load documents and convert extracted chunks to LangChain documents."""

    def load_document(
        self,
        file_path: str,
        document_id: int | None = None,
        extra_metadata: Dict[str, Any] | None = None,
        *,
        chunk_size: int | None = None,
        chunk_overlap_sentences: int | None = None,
    ) -> List[Document]:
        source = os.path.basename(file_path)
        _, extension = os.path.splitext(file_path.lower())
        config = _resolve_chunk_params(chunk_size, chunk_overlap_sentences)

        if extension == ".pdf":
            processed_chunks = process_pdf(
                file_path,
                document_id=document_id,
                chunk_size=config.chunk_size,
                chunk_overlap_sentences=config.chunk_overlap_sentences,
            )
        elif extension == ".docx":
            processed_chunks = process_docx(
                file_path,
                document_id=document_id,
                chunk_size=config.chunk_size,
                chunk_overlap_sentences=config.chunk_overlap_sentences,
            )
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        langchain_docs: List[Document] = []
        for index, chunk in enumerate(processed_chunks):
            metadata: Dict[str, Any] = {
                "source": source,
                "chunk": chunk.get("id", index),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "has_ocr": chunk.get("has_ocr", False),
                "image_paths": chunk.get("image_paths", []),
            }
            if document_id is not None:
                metadata["document_id"] = document_id
            if extra_metadata:
                metadata.update(extra_metadata)

            metadata = {key: value for key, value in metadata.items() if value is not None}
            langchain_docs.append(Document(page_content=chunk["text"], metadata=metadata))

        return langchain_docs
