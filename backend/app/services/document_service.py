from typing import Any, Dict, List
import os
import pdfplumber
import re
import docx
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import LOG
from underthesea import sent_tokenize
import tiktoken

# --- LOGIC HIỆN TẠI (GIỮ NGUYÊN) ---

try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

enc = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(enc.encode(text))

def trim_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
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

def process_pdf(file_path: str) -> List[Dict]:
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
                    if sentence_tokens > settings.CHUNK_SIZE:
                        LOG.warning("Long sentence detected → trimming by token")
                        sentence = trim_to_token_limit(sentence, settings.CHUNK_SIZE)
                        sentence_tokens = token_len(sentence)
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
                        overlap_sentences = current_sentences[-settings.OVERLAP_SENTENCES:]
                        current_sentences = overlap_sentences.copy()
                        current_token_count = sum(token_len(s) for s in current_sentences)
                        current_page_start = page_num
                    current_sentences.append(sentence)
                    current_token_count += sentence_tokens
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

# --- LOGIC BỔ SUNG CHO DOCX (HẠN CHẾ VIẾT LẠI) ---

def process_docx(file_path: str) -> List[Dict]:
    LOG.info(f"Processing DOCX: {file_path}")
    doc = docx.Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    
    # Tận dụng lại logic chia chunk đã có
    sentences = sent_tokenize(full_text)
    chunks: List[Dict] = []
    chunk_id = 0
    current_sentences: List[str] = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = token_len(sentence)
        if sentence_tokens > settings.CHUNK_SIZE:
            sentence = trim_to_token_limit(sentence, settings.CHUNK_SIZE)
            sentence_tokens = token_len(sentence)
        if current_token_count + sentence_tokens > settings.CHUNK_SIZE:
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({"id": chunk_id, "text": chunk_text, "tokens": current_token_count})
                chunk_id += 1
            overlap_sentences = current_sentences[-settings.OVERLAP_SENTENCES:]
            current_sentences = overlap_sentences.copy()
            current_token_count = sum(token_len(s) for s in current_sentences)
        current_sentences.append(sentence)
        current_token_count += sentence_tokens
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({"id": chunk_id, "text": chunk_text, "tokens": current_token_count})
        
    LOG.info(f"Generated {len(chunks)} chunks from DOCX")
    return chunks

# --- LỚP WRAPPER ĐỂ TƯƠNG THÍCH VỚI RAGService ---

class DocumentService:
    """
    Lớp tương thích để ráp nối logic xử lý file với RAGService.
    Không chứa logic xử lý, chỉ điều phối và chuyển đổi dữ liệu.
    """
    def load_document(self, file_path: str, extra_metadata: Dict[str, Any] | None = None) -> List[Document]:
        """
        Tải và xử lý file, trả về định dạng List[Document] mà RAGService cần.
        """
        source = os.path.basename(file_path)
        _, extension = os.path.splitext(file_path.lower())
        
        processed_chunks: List[Dict] = []
        if extension == ".pdf":
            processed_chunks = process_pdf(file_path)
        elif extension == ".docx":
            processed_chunks = process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
            
        # Chuyển đổi từ List[Dict] sang List[Document]
        langchain_docs: List[Document] = []
        for i, chunk in enumerate(processed_chunks):
            metadata = {
                "source": source,
                "chunk": chunk.get("id", i),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end")
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            doc = Document(page_content=chunk["text"], metadata=metadata)
            langchain_docs.append(doc)
            
        return langchain_docs
