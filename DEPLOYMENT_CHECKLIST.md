# SmartDoc AI - Setup & Deployment Checklist

## ✅ Implementation Completed

### 1. Core Services Created

- [x] **DocumentService** (`backend/app/services/document_service.py`)
  - Loads PDF and DOCX files
  - Chunks documents with configurable size/overlap
  - Prepares LangChain Document objects with metadata

- [x] **RAGService** (`backend/app/services/rag_service.py`)
  - Orchestrates entire RAG pipeline
  - Manages embeddings (SentenceTransformer)
  - Stores/retrieves from FAISS vectorstore
  - Queries with Ollama LLM
  - Persists vectorstore to disk

### 2. Loaders Fixed

- [x] PDF Loader (`backend/app/loaders/pdf_loader.py`)
- [x] DOCX Loader (`backend/app/loaders/docx_loader.py`)

### 3. LLM Integration

- [x] Ollama LLM (`backend/app/ai/llm.py`)
- [x] RAG Chain (`backend/app/ai/rag_chain.py`)
  - Uses RetrievalQA from LangChain
  - Properly configured with Ollama

### 4. UI Integration

- [x] Streamlit App (`ui/streamlit_app.py`)
  - File upload handler
  - Chat query handler
  - RAGService integration
  - Display results with sources

### 5. Configuration

- [x] Core Config (`backend/app/core/config.py`)
  - Ollama settings
  - Embedding model
  - Chunk sizing
  - Vector store paths

### 6. Testing & Documentation

- [x] Test Script (`test_rag_integration.py`)
- [x] Integration Guide (`RAG_OLLAMA_INTEGRATION.md`)
- [x] Quick Start Scripts
  - `start.bat` (Windows)
  - `start.sh` (Linux/Mac)

---

## 🚀 Pre-Deployment Setup

### Step 1: Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai
# Run OllamaSetup.exe
# Restart Terminal/CMD
ollama --version  # Verify installation
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama --version
```

**macOS:**
```bash
# Download from https://ollama.ai
# Open .dmg file and drag to Applications
ollama --version
```

### Step 2: Pull Model

```bash
# Pull Ollama model (one time)
ollama pull qwen2.5:7b

# Or use different model (optional)
ollama pull llama2
ollama pull neural-chat
ollama pull mistral
```

### Step 3: Install Python Dependencies

```bash
# From project root
cd d:\do_an\python\SmartDoc_AI

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test RAG integration
python test_rag_integration.py

# Expected output:
# ✓ Checking Configuration...
# ✓ Initializing RAGService...
# ✓ RAGService Status...
# ✓ Testing Ollama Connection...
# ✅ ALL TESTS PASSED!
```

---

## ▶️ Running the System

### Terminal 1: Start Ollama

```bash
ollama serve
# Keep running - this is the backend service
```

### Terminal 2: Run Streamlit

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Navigate to project
cd d:\do_an\python\SmartDoc_AI

# Start Streamlit
streamlit run ui/streamlit_app.py

# Should see:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

### Terminal 3 (Optional): Monitor Logs

```bash
# Watch application logs for debugging
tail -f backend/data/logs/app.log  # Linux/Mac
Get-Content backend/data/logs/app.log -Wait  # Windows PowerShell
```

---

## 📋 Post-Deployment Checklist

- [ ] Ollama running: `curl http://localhost:11434/api/tags` returns JSON
- [ ] Streamlit accessible: http://localhost:8501 loads without errors
- [ ] Upload test: Try uploading a PDF/DOCX
- [ ] Query test: Ask a question about uploaded document
- [ ] Response test: AI returns answer with source documents

---

## 🔧 Configuration Options

### Change LLM Model

Edit `backend/app/core/config.py`:
```python
# Current (fast, good quality)
LLM_MODEL = "qwen2.5:7b"

# Alternatives:
LLM_MODEL = "llama2"           # Good balance
LLM_MODEL = "neural-chat"      # Conversational
LLM_MODEL = "mistral"          # Fast
LLM_MODEL = "openchat"         # Good quality
```

### Adjust Chunk Size (for longer/shorter documents)

```python
# Larger chunks = fewer documents, faster but less precise
CHUNK_SIZE = 2000           # For long documents
CHUNK_SIZE = 500            # For short documents (default: 1000)
CHUNK_OVERLAP = 100         # Less overlap for speed
```

### Change Embedding Model (for other languages)

```python
# Current (multilingual, balanced)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Faster, lighter:
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# English specific:
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Chinese optimized:
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

---

## 🐛 Troubleshooting

### Ollama Won't Start
```bash
# Check if Ollama is already running
ps aux | grep ollama  # Linux/Mac
Get-Process ollama    # Windows PowerShell

# Kill if stuck, then restart
kill <PID>
ollama serve
```

### Model Download Fails
```bash
# Check internet connection
ping github.com

# Try specific model
ollama pull llama2:7b  # Smaller model to test

# Check disk space
df -h  # Linux/Mac
dir    # Windows
```

### High Memory Usage
```python
# Reduce chunk size and top_k in config.py
CHUNK_SIZE = 500
TOP_K = 2  # Instead of 3
```

### Slow Responses
1. Check if Ollama GPU is working
2. Use lighter model: `ollama pull neural-chat`
3. Reduce TOP_K documents
4. Reduce CHUNK_SIZE for fewer embeddings

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│         Streamlit UI (Port 8501)            │
│  - File Upload                              │
│  - Chat Interface                           │
│  - Result Display                           │
└────────────────┬────────────────────────────┘
                 │
         ┌───────┴────────┐
         ↓                ↓
    ┌─────────┐      ┌──────────┐
    │Document │      │RAG       │
    │Service  │      │Service   │
    └─────────┘      └──────────┘
         │                │
         └───────┬────────┘
                 ↓
    ┌─────────────────────┐
    │ Sentence-           │
    │ Transformers        │
    │ (Embeddings)        │
    └──────────┬──────────┘
               ↓
    ┌──────────────────────┐
    │ FAISS Vector Store   │
    │ (Persisted to Disk)  │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ Ollama LLM           │
    │ (Port 11434)         │
    └──────────────────────┘
```

---

## 📝 Key Implementation Details

### Document Processing Flow
1. User uploads file → `RAGService.add_documents()`
2. `DocumentService` loads and chunks document
3. `SentenceTransformer` creates embeddings
4. FAISS indexes vectors for fast retrieval
5. Metadata stored for source tracking

### Query Flow
1. User asks question → `RAGService.query()`
2. Question embedded with same model
3. FAISS retrieves top-K similar documents
4. LangChain RetrievalQA formats prompt with context
5. Ollama generates response
6. Sources returned to UI

### Persistence
- FAISS index: `backend/data/vectorstore/faiss.index`
- Metadata: `backend/data/vectorstore/meta.pkl`
- Logs: `backend/data/logs/app.log`

---

## ✨ Features

- ✅ Multi-format support (PDF, DOCX)
- ✅ Automatic chunking and embedding
- ✅ Fast vector similarity search (FAISS)
- ✅ LLM-powered generation (Ollama)
- ✅ Source document tracking
- ✅ Persistent vectorstore
- ✅ Configurable models/parameters
- ✅ Error handling and logging
- ✅ Clean UI with Streamlit

---

## 📞 Support

For issues:
1. Run test script: `python test_rag_integration.py`
2. Check logs: `backend/data/logs/app.log`
3. Verify Ollama: `curl http://localhost:11434/api/tags`
4. Review documentation: `RAG_OLLAMA_INTEGRATION.md`
