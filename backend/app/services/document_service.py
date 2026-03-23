from typing import List, Dict
import pdfplumber
import re

from app.core.config import settings
from app.core.logger import LOG

# Sentence tokenizer (Vietnamese)
from underthesea import sent_tokenize

# Tokenizer (LLM-based)
import tiktoken

# Optional OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

# Init tokenizer
enc = tiktoken.get_encoding("cl100k_base")


# =========================
# Utils
# =========================
def token_len(text: str) -> int:
    return len(enc.encode(text))


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])


def clean_text(text: str) -> str:
    """Normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


# =========================
# Extract text
# =========================
def extract_text_from_page(page) -> str:
    """Extract text with optional OCR fallback"""
    text = page.extract_text() or ""

    if not text.strip() and OCR_AVAILABLE:
        try:
            img = page.to_image(resolution=300).original
            text = pytesseract.image_to_string(img, lang="eng+vie")
            LOG.debug("Used OCR for page")
        except Exception as e:
            LOG.warning(f"OCR failed: {e}")

    return clean_text(text)


# =========================
# Main processing
# =========================
def process_pdf(file_path: str) -> List[Dict]:
    """
    Advanced PDF processing pipeline:
    - Page-by-page extraction
    - Vietnamese sentence splitting
    - Token-based chunking
    - Sentence overlap
    - Rich metadata
    """

    LOG.info(f"Processing PDF: {file_path}")

    chunks: List[Dict] = []
    chunk_id = 0

    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            LOG.info(f"Total pages: {num_pages}")

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

                    # Nếu sentence quá dài → trim theo token (FIXED)
                    if sentence_tokens > settings.CHUNK_SIZE:
                        LOG.warning("Long sentence detected → trimming by token")
                        sentence = trim_to_token_limit(sentence, settings.CHUNK_SIZE)
                        sentence_tokens = token_len(sentence)

                    # Nếu vượt quá chunk size → flush chunk
                    if current_token_count + sentence_tokens > settings.CHUNK_SIZE:
                        if current_sentences:
                            chunk_text = " ".join(current_sentences)

                            chunks.append({
                                "id": chunk_id,
                                "text": chunk_text,
                                "tokens": current_token_count,
                                "page_start": current_page_start,
                                "page_end": page_num
                            })

                            chunk_id += 1

                        # Overlap (sentence-based)
                        overlap_sentences = current_sentences[-settings.OVERLAP_SENTENCES:]
                        current_sentences = overlap_sentences.copy()
                        current_token_count = sum(token_len(s) for s in current_sentences)

                        current_page_start = page_num

                    current_sentences.append(sentence)
                    current_token_count += sentence_tokens

            # Flush chunk cuối
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "tokens": current_token_count,
                    "page_start": current_page_start,
                    "page_end": num_pages
                })

    except Exception as e:
        LOG.error(f"PDF processing error: {e}")
        return []

    LOG.info(f"Generated {len(chunks)} chunks")
    return chunks


# =========================
# TEST
# =========================
if __name__ == "__main__":
    test_pdf = "../test.pdf"

    result = process_pdf(test_pdf)

    print(f"Total chunks: {len(result)}")

    for chunk in result[:2]:
        print("\n---")
        print(f"ID: {chunk['id']}")
        print(f"Pages: {chunk['page_start']} -> {chunk['page_end']}")
        print(f"Tokens: {chunk['tokens']}")
        print(chunk["text"][:200], "...")