# SmartDoc AI - Ollama RAG Integration Guide

## 📋 Tổng Quan

Hệ thống đã được tích hợp hoàn chỉnh **Ollama LLM + RAG (Retrieval-Augmented Generation)** để:
- Tải và xử lý documents (PDF, DOCX)
- Lưu text embeddings vào FAISS vectorstore
- Truy vấn thông qua Ollama LLM với context từ documents

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (ui/streamlit_app.py)       │
│              File Upload & Chat Interface                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Document    │ │  RAG         │ │  Ollama      │
│  Service     │ │  Service     │ │  LLM Chain   │
│              │ │              │ │              │
│ - Load PDF   │ │ - Orchestrate│ │ - Generate   │
│ - Load DOCX  │ │ - Chunk text │ │ - Answer     │
│ - Split text │ │ - Embed      │ │ - Support    │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  FAISS Vector Store           │
        │  + Embeddings                 │
        │  + Document Metadata          │
        └───────────────────────────────┘
```

## 📦 Files Created/Modified

### 1. **Document Processing**
- `backend/app/services/document_service.py` ✨ NEW
  - Load PDF/DOCX files
  - Chunk documents
  - Prepare for embedding

- `backend/app/loaders/pdf_loader.py` 📝 Modified
  - Uncommented PDF loading implementation

- `backend/app/loaders/docx_loader.py` 📝 Modified
  - Uncommented DOCX loading implementation

### 2. **RAG Service** 
- `backend/app/services/rag_service.py` ✨ NEW
  - Main RAG orchestration service
  - Integrates: DocumentService + Embeddings + FAISS + Ollama
  - Handles document management & querying

### 3. **LLM Integration**
- `backend/app/ai/llm.py` ✅ Already configured
  - Uses Ollama with settings from config

- `backend/app/ai/rag_chain.py` 📝 Modified
  - Fixed input key specification for RetrievalQA

### 4. **UI Integration**
- `ui/streamlit_app.py` 📝 Modified
  - Integrated RAGService
  - Handle file uploads → vectorstore
  - Handle queries → Ollama responses

### 5. **Configuration**
- `backend/app/core/config.py` ✅ Already configured
  - OLLAMA_BASE_URL = "http://localhost:11434"
  - LLM_MODEL = "qwen2.5:7b"
  - EMBEDDING_MODEL = sentence-transformers model

## 🚀 Cách Sử Dụng

### 1️⃣ Cài Đặt & Khởi Động Ollama

**Windows:**
- Tải Ollama từ: https://ollama.ai
- Cài đặt từ installer (OllamaSetup.exe)
- Khởi động service: Ollama sẽ chạy tự động ở background

**Hoặc chạy từ terminal:**
```bash
ollama serve
```

### 2️⃣ Kiểm Tra Kết Nối Ollama

```bash
# Kiểm tra Ollama chạy
curl http://localhost:11434/api/tags

# Hoặc chạy test script
python test_rag_integration.py
```

### 3️⃣ Chạy Streamlit UI

Từ thư mục project root:
```bash
# Windows
streamlit run ui/streamlit_app.py

# Linux/Mac
python -m streamlit run ui/streamlit_app.py
```

### 4️⃣ Sử Dụng Ứng Dụng

1. **Tải Documents**
   - Kéo thả file PDF/DOCX vào upload box
   - Hoặc click "Chọn file từ máy"
   - Documents sẽ được xử lý tự động

2. **Chat với AI**
   - Nhập câu hỏi liên quan tới documents
   - AI sẽ truy vấn FAISS + Ollama để trả lời
   - Hiển thị source documents

## 🔧 Configuration

**File: `backend/app/core/config.py`**

```python
# Ollama LLM
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"  # Thay đổi model nếu cần

# Embedding (sentence-transformers)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Document Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 3  # số documents lấy ra khi query
```

**Thay đổi Model Ollama:**
```python
# config.py
LLM_MODEL = "llama2"      # hoặc
LLM_MODEL = "neural-chat" # hoặc model khác
```

## 📝 API Reference

### RAGService

```python
from app.services.rag_service import RAGService

rag = RAGService()

# 1. Add documents to vectorstore
result = rag.add_documents("/path/to/document.pdf")
# Returns: {"status": "success/error", "message": "...", "chunks_added": N}

# 2. Query with RAG
result = rag.query("Your question here", search_type="vector")
# Returns: {"status": "success", "answer": "...", "sources": [...]}

# 3. Get status
status = rag.get_status()
# Returns: {"vectorstore_ready": bool, "total_documents": N, ...}

# 4. Clear vectorstore
rag.clear_vectorstore()
```

### DocumentService

```python
from app.services.document_service import DocumentService

doc_service = DocumentService()

# Load single document
documents = doc_service.load_document("/path/to/file.pdf")
# Returns: List[Document] with chunked content

# Load batch
docs = doc_service.process_batch(["/file1.pdf", "/file2.docx"])
```

## 🐛 Troubleshooting

### Problem: "Ollama is not recognized"
**Solution:**
1. Cài Ollama: https://ollama.ai
2. Restart terminal/command prompt
3. Verify: `ollama --version`

### Problem: "Model not found"
**Solution:**
```bash
# Pull the model
ollama pull qwen2.5:7b

# Or list available models
ollama list
```

### Problem: Connection refused (111)
**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, test
curl http://localhost:11434/api/tags
```

### Problem: Out of memory when loading embeddings
**Solution:**
- Giảm CHUNK_SIZE trong config.py
- Hoặc dùng embedding model nhẹ hơn:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Nhẹ hơn
```

### Problem: Slow inference
**Solution:**
1. Dùng modelo Ollama nhẹ hơn:
   ```bash
   ollama pull neural-chat
   # Thay LLM_MODEL = "neural-chat" trong config
   ```

2. Giảm TOP_K trong config.py (lấy ít Document hơn)

3. Tăng compute resources

## 📊 Data Flow Chi Tiết

1. **Upload Document**
   ```
   File (PDF/DOCX)
   ↓ DocumentService
   Raw text by page/paragraph
   ↓ Chunking
   Text chunks (size=1000, overlap=200)
   ↓ SentenceTransformer
   Vector embeddings (384 dimensions)
   ↓ FAISS
   Stored + Indexed
   ```

2. **Query with RAG**
   ```
   User Question
   ↓ SentenceTransformer
   Query vector (same dim as docs)
   ↓ FAISS
   Top-K similar documents retrieved
   ↓ Ollama LLM (with docs as context)
   Generated answer
   ↓ Streamlit UI
   Display answer + sources
   ```

## 🔄 Flow Diagram

```
┌─────────────────────┐
│  User Upload File   │
└──────────┬──────────┘
           │
           ↓ streamlit_app.py
    ┌──────────────────┐
    │ RAGService       │
    │ .add_documents() │
    └─────┬────────────┘
          │
    ┌─────┴──────────────────┐
    │ 1. DocumentService     │
    │    .load_document()    │ → Tải PDF/DOCX
    └─────┬──────────────────┘
          │
    ┌─────┴──────────────────┐
    │ 2. Text Chunking       │
    │    (1000 chars)        │
    └─────┬──────────────────┘
          │
    ┌─────┴──────────────────┐
    │ 3. SentenceTransformer  │
    │    .encode()           │ → Embeddings
    └─────┬──────────────────┘
          │
    ┌─────┴──────────────────┐
    │ 4. FAISS Vector Store  │
    │    .add()              │ → Save indexed
    └─────┬──────────────────┘
          │
          ↓ Vectorstore Saved
         
┌──────────────────────┐
│  User Ask Question   │
└──────────┬───────────┘
           │
    ┌──────┴────────────────┐
    │ RAGService.query()    │
    └──────┬─────────────────┘
           │
    ┌──────┴────────────────────────┐
    │ 1. Embed Question (same model) │
    └──────┬────────────────────────┘
           │
    ┌──────┴────────────────────┐
    │ 2. FAISS Search           │
    │    Top-K Documents        │
    └──────┬────────────────────┘
           │
    ┌──────┴────────────────────────┐
    │ 3. LangChain RetrievalQA      │
    │    (Ollama + Retrieved Docs)  │
    └──────┬────────────────────────┘
           │
    ┌──────┴────────────────────┐
    │ 4. Ollama LLM             │
    │    Generate Answer        │
    └──────┬────────────────────┘
           │
           ↓ Return to UI
    ┌──────────────────────────────┐
    │ Display Answer + Sources     │
    │ (from metadata)              │
    └──────────────────────────────┘
```

## ✅ Tính Năng

- ✅ Support PDF, DOCX file formats
- ✅ Automatic text chunking & embedding
- ✅ FAISS vector indexing for fast retrieval
- ✅ Ollama LLM integration for generation
- ✅ Source document tracking
- ✅ Hybrid search (vector + keyword)
- ✅ Persistent vectorstore on disk
- ✅ Easy-to-use Streamlit UI

## 📝 Notes

- Embedding model downloads ~400MB on first run
- FAISS index grows with number of documents
- Ollama models stored in `~/.ollama/models`
- Vectorstore cached in `backend/data/vectorstore/`

## 🤝 Support

For issues or questions:
1. Check logs in `backend/data/logs/`
2. Verify Ollama is running: `ollama serve`
3. Run test script: `python test_rag_integration.py`
