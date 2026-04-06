# CoRAG + GraphRAG Dev Guide

## 1) Muc tieu
Tai lieu nay huong dan dev tiep tuc code `CoRAG` va `GraphRAG` tren kien truc moi da refactor, trong khi van giu on dinh luong `RAG` hien tai.

## 2) Kien truc hien tai (query side)
File chinh:

- `backend/app/services/rag_service.py`: orchestration chung (resolve mode, top_k, fallback low-memory, format sources).
- `backend/app/rag/registry.py`: map `rag_mode -> engine`.
- `backend/app/rag/base.py`: interface engine chung (`BaseRAGModeEngine`).
- `backend/app/rag/models.py`: request/response contract:
  - `RAGQueryRequest`
  - `RAGEngineResult`

Mode implementations:

- `backend/app/rag/modes/rag/engine.py` (da implement)
- `backend/app/rag/modes/rag/planner.py` (da implement)
- `backend/app/rag/modes/rag/retriever.py` (da implement)
- `backend/app/rag/modes/rag/responder.py` (da implement)
- `backend/app/rag/modes/corag/engine.py` (stub)
- `backend/app/rag/modes/graphrag/engine.py` (stub)

## 3) Luong thuc thi chung
1. UI gui query vao `RAGService.query(...)` kem `rag_mode`.
2. `rag_service.py` chuan hoa mode (`normalize_rag_mode`) va tao `RAGQueryRequest`.
3. Engine duoc lay tu registry va goi `engine.query(request)`.
4. Engine tra ve `RAGEngineResult(answer, source_documents, metadata)`.
5. `rag_service.py` format `sources` va tra output ve UI.

## 4) Luong RAG tham chieu (dang chay)
`VanillaRAGEngine.query()` dang theo pipeline:

1. `planner.plan(request)` -> tao `RAGPlan`.
2. `retriever.retrieve(plan)` -> dung:
   - `embedding_service.embed_query`
   - `app.ai.retriever.get_retriever(...)`
   - `FaissStore.search` hoac `FaissStore.hybrid_search`
3. `responder.answer(...)` -> dung `app.ai.llm.get_llm(...)` + prompt hien tai.
4. Tra `RAGEngineResult`.

Ghi chu quan trong: day la baseline de CoRAG/GraphRAG noi theo, khong duoc pha behavior mode `rag`.

## 5) Huong dan code CoRAG
### 5.1 File nen tao them trong mode CoRAG
- `backend/app/rag/modes/corag/planner.py`
- `backend/app/rag/modes/corag/retriever.py`
- `backend/app/rag/modes/corag/responder.py`
- cap nhat `backend/app/rag/modes/corag/engine.py`

### 5.2 Pipeline de xay dung
1. Planner:
   - Tach cau hoi thanh sub-queries (decomposition).
2. Retriever:
   - Retrieve cho tung sub-query (co the dung chung `get_retriever` + `FaissStore`).
   - Merge/rerank/dedupe ket qua.
3. Responder:
   - Tong hop ngu canh da merge.
   - Prompt co instruction cho "multi-perspective synthesis".
4. Engine:
   - Goi planner -> retriever -> responder.
   - Tra `RAGEngineResult`.

### 5.3 Nguyen tac
- Khong sua logic RAG mode da co.
- Khong if/else mode trong `rag_service.py`.
- Toan bo logic CoRAG dat trong `backend/app/rag/modes/corag/*`.

## 6) Huong dan code GraphRAG
### 6.1 File nen tao them trong mode GraphRAG
- `backend/app/rag/modes/graphrag/graph.py` (option)
- `backend/app/rag/modes/graphrag/retriever.py`
- `backend/app/rag/modes/graphrag/responder.py`
- cap nhat `backend/app/rag/modes/graphrag/engine.py`

### 6.2 Pipeline de xay dung
1. Seed retrieval:
   - Lay seed chunks bang vector/hybrid (dung chung ha tang retrieval hien co).
2. Graph expansion:
   - Mo rong node lien quan (neighbor chunks, entity links, relation links...).
3. Re-ranking:
   - Sap xep lai context sau expansion.
4. Responder:
   - Prompt theo kieu "connected evidence".
5. Engine:
   - seed -> expand -> rerank -> answer.

### 6.3 Nguyen tac
- Vector/hybrid trong `faiss_store.py` la retrieval primitive.
- Graph logic la lop bo sung rieng trong mode GraphRAG, khong nhiet logic do vao `faiss_store.py`.

## 7) Contract bat buoc cho moi mode engine
Ham `query(self, request: RAGQueryRequest) -> RAGEngineResult` phai dam bao:

- Khong raise loi runtime thong thuong neu co the tra ket qua hop le.
- `answer` luon la `str`.
- `source_documents` la list `langchain_core.documents.Document`.
- `metadata` la dict de debug/trace.

## 8) Registry va config
- Dang ky engine trong `backend/app/rag/registry.py`:
  - `ENGINE_BY_MODE`
  - `MODE_ALIASES`
  - `MODE_LABELS`
- Default mode doc tu env:
  - `backend/app/core/config.py` -> `RAG_MODE`
  - `backend/.env` co the set `RAG_MODE=rag|corag|graphrag`

## 9) Checklist truoc khi merge CoRAG/GraphRAG
1. `python -m compileall backend/app ui/streamlit_app.py` pass.
2. Mode `rag` van tra loi binh thuong nhu truoc.
3. Chon mode moi tren UI khong lam crash app.
4. Luu lich su chat co `rag_mode` dung.
5. Khong sua hanh vi ingest tai lieu (`add_documents`).

## 10) Noi can doc nhanh truoc khi code
- `backend/app/services/rag_service.py`
- `backend/app/rag/registry.py`
- `backend/app/rag/modes/rag/*`
- `backend/app/ai/retriever.py`
- `backend/app/vectorstore/faiss_store.py`
