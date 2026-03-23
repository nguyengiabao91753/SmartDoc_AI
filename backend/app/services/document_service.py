from pathlib import Path
from typing import List, Dict, Any
from app.core.config import settings
from app.core.logger import LOG
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.loaders.pdf_loader import load_pdf_from_path
from app.loaders.docx_loader import load_docx_from_path
import os

class DocumentService:
    """Service để xử lý tài liệu: tải, chunking, chuẩn bị cho embedding"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Tải document từ file path (PDF hoặc DOCX)
        
        Args:
            file_path: Đường dẫn đầy đủ tới file
            
        Returns:
            List[Document]: Danh sách các Document objects
        """
        file_path = str(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        raw_content = []
        
        try:
            if file_ext == '.pdf':
                raw_content = load_pdf_from_path(file_path)
            elif file_ext in ['.docx', '.doc']:
                raw_content = load_docx_from_path(file_path)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {file_ext}")
            
            if not raw_content:
                LOG.warning(f"File {file_path} rỗng hoặc không có nội dung")
                return []
            
            # Ghép nội dung lại
            full_text = "\n\n".join([item['text'] for item in raw_content])
            
            # Chunking
            chunks = self.splitter.split_text(full_text)
            
            # Tạo Document objects với metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(file_path),
                        "chunk": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            LOG.info(f"Tải {len(documents)} chunks từ {os.path.basename(file_path)}")
            return documents
            
        except Exception as e:
            LOG.error(f"Lỗi khi tải document {file_path}: {str(e)}")
            raise
    
    def process_batch(self, file_paths: List[str]) -> List[Document]:
        """
        Xử lý nhiều files
        
        Args:
            file_paths: Danh sách đường dẫn files
            
        Returns:
            List[Document]: Tất cả documents từ các files
        """
        all_documents = []
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                LOG.error(f"Lỗi xử lý file {file_path}: {str(e)}")
                continue
        
        return all_documents
