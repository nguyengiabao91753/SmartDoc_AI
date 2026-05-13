import os
import re
import shutil
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure project root and backend are importable.
# - `ui.*` imports rely on project root being on sys.path
# - backend imports use the backend/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


from app.core.config import settings
from app.core.logger import LOG
from app.database.sqlite_db import init_db
from app.rag.registry import AVAILABLE_RAG_MODES, MODE_LABELS, normalize_rag_mode
from app.services.database_service import db_service
from app.services.rag_service import RAGService

from ui.chunk_config_ui import chunk_params_ui


st.set_page_config(layout="wide", page_title="SmartDoc AI")

RAG_MODE_OPTIONS = [MODE_LABELS[mode] for mode in AVAILABLE_RAG_MODES]
RAG_MODE_BY_LABEL = {MODE_LABELS[mode]: mode for mode in AVAILABLE_RAG_MODES}
DEFAULT_RAG_MODE = normalize_rag_mode(getattr(settings, "RAG_MODE", "rag"))
DEFAULT_RAG_MODE_LABEL = MODE_LABELS.get(DEFAULT_RAG_MODE, "RAG")
GRAPH_RAG_MODE = "graphrag"
GRAPH_RAG_LABEL = MODE_LABELS.get(GRAPH_RAG_MODE, "GraphRAG")
GRAPH_RAG_SEARCH_TYPE = "hybrid"
SEMANTIC_SEARCH_LABEL = "Ngữ nghĩa"
HYBRID_SEARCH_LABEL = "Hybrid"
LEGACY_KEYWORD_SEARCH_LABEL = "Từ khóa"
SEARCH_MODE_OPTIONS = [SEMANTIC_SEARCH_LABEL, HYBRID_SEARCH_LABEL]


def load_css_file() -> str:
    css_path = Path(__file__).parent / "css" / "app.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


@st.cache_resource
def get_rag_service():
    return RAGService()


def ensure_rag_service():
    try:
        return get_rag_service()
    except Exception as exc:
        LOG.error("RAG init error: %s", exc)
        st.error(f"Không thể khởi tạo RAG service: {exc}")
        return None


# Tắt spinner mặc định để tự hiển thị bên dưới đáy màn hình
@st.cache_resource(show_spinner=False)
def get_graph_rag_service():
    from app.services.graph_rag_service import GraphRAGService

    # Tái sử dụng EmbeddingService từ RAGService để tiết kiệm RAM và thời gian khởi tạo
    rag_svc = get_rag_service()
    embedding_svc = rag_svc.embedding_service if rag_svc else None
    return GraphRAGService(embedding_service=embedding_svc)


def init_state():
    defaults = {
        "active_session_id": None,
        "draft_session": False,
        "latest_sources_by_session": {},
        "uploader_nonce": 0,
        "flash_message": None,
        "editing_session_id": None,
        "queued_upload": None,
        "processing_upload": None,
        "queued_query": None,
        "processing_query": None,
        "last_upload_success": None,
        "last_graph_success": None,
        "current_rag_mode_label": DEFAULT_RAG_MODE_LABEL,
        "current_search_label": SEMANTIC_SEARCH_LABEL,
        "current_detail_label": "Nhanh",
        "composer_prompt": "",
        "open_session_menu_id": None,
        "scroll_to_bottom_nonce": 0,
        "scroll_applied_nonce": 0,
        "corag_metadata_by_session": {},
        "selfrag_metadata_by_session": {},
        "processing_graph": None,
        "initializing_graph": False,
        "_do_graph_load": False, 
        "_graph_just_initialized": False,
        "_graph_init_failed": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state.get("current_search_label") == LEGACY_KEYWORD_SEARCH_LABEL:
        st.session_state.current_search_label = HYBRID_SEARCH_LABEL
    if st.session_state.get("search_mode_select") == LEGACY_KEYWORD_SEARCH_LABEL:
        st.session_state.search_mode_select = HYBRID_SEARCH_LABEL


def schedule_bottom_scroll():
    st.session_state.scroll_to_bottom_nonce = int(st.session_state.get("scroll_to_bottom_nonce", 0)) + 1


def apply_scheduled_bottom_scroll():
    target_nonce = int(st.session_state.get("scroll_to_bottom_nonce", 0))
    applied_nonce = int(st.session_state.get("scroll_applied_nonce", 0))
    if target_nonce <= applied_nonce:
        return

    components.html(
        """
        <script>
        setTimeout(() => {
            const parentDoc = window.parent.document;
            const anchor = parentDoc.getElementById("content-bottom-anchor");
            if (anchor) {
                anchor.scrollIntoView({ behavior: "smooth", block: "end" });
            }
        }, 80);
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state.scroll_applied_nonce = target_nonce


def is_busy() -> bool:
    is_loading_graph = st.session_state.get("current_rag_mode_label") == GRAPH_RAG_LABEL and st.session_state.get("_graph_svc_instance") is None
    
    return any(
        [
            st.session_state.get("queued_upload"),
            st.session_state.get("processing_upload"),
            st.session_state.get("queued_query"),
            st.session_state.get("processing_query"),
            st.session_state.get("processing_graph"),
            st.session_state.get("initializing_graph"),
            st.session_state.get("_do_graph_load"), 
            is_loading_graph, 
        ]
    )


def show_flash_message(container=None):
    flash = st.session_state.get("flash_message")
    if not flash:
        return

    target = container if container else st
    kind = flash.get("type", "error")
    text = flash.get("text", "")
    if kind == "success":
        target.success(text)
    elif kind == "warning":
        target.warning(text)
    else:
        target.error(text)
    st.session_state.flash_message = None


def transition_pending_states():
    if st.session_state.get("queued_upload") and not st.session_state.get("processing_upload"):
        st.session_state.processing_upload = st.session_state.queued_upload
        st.session_state.queued_upload = None
        st.rerun()

    if st.session_state.get("queued_query") and not st.session_state.get("processing_query"):
        st.session_state.processing_query = st.session_state.queued_query
        st.session_state.queued_query = None
        st.rerun()


def ensure_active_session():
    sessions = db_service.get_chat_sessions()
    if not sessions:
        st.session_state.active_session_id = None
        st.session_state.draft_session = True
        return sessions

    # NẾU NGƯỜI DÙNG ĐANG YÊU CẦU DRAFT (VÍ DỤ VỪA XÓA CHAT HOẶC BẤM NEW CHAT)
    if st.session_state.get("draft_session"):
        st.session_state.active_session_id = None
        return sessions

    # NẾU ACTIVE_SESSION_ID BỊ NONE (DO VỪA XÓA), CHUYỂN VỀ DRAFT LUÔN
    if st.session_state.active_session_id is None:
        st.session_state.draft_session = True
        return sessions

    # NẾU ĐANG CÓ ID MÀ ID ĐÓ KHÔNG TỒN TẠI NỮA
    if not any(session["id"] == st.session_state.active_session_id for session in sessions):
        st.session_state.draft_session = True
        st.session_state.active_session_id = None

    return sessions


def get_active_session(sessions):
    for session in sessions:
        if session["id"] == st.session_state.active_session_id:
            return session
    return None


def create_new_chat():
    if is_busy():
        return
    st.session_state.active_session_id = None
    st.session_state.draft_session = True
    st.session_state.last_upload_success = None
    st.session_state.last_graph_success = None
    st.session_state.composer_prompt = ""
    st.session_state.editing_session_id = None
    st.session_state.open_session_menu_id = None


def set_active_session(session_id: int):
    if is_busy():
        return
    st.session_state.draft_session = False
    st.session_state.active_session_id = session_id
    st.session_state.editing_session_id = None
    st.session_state.last_upload_success = None
    st.session_state.last_graph_success = None
    st.session_state.composer_prompt = ""
    st.session_state.open_session_menu_id = None


def begin_session_rename(session_id: int, current_title: str):
    if is_busy():
        return
    st.session_state.editing_session_id = session_id
    st.session_state[f"rename_input_{session_id}"] = current_title
    st.session_state.open_session_menu_id = None


def cancel_session_rename():
    st.session_state.editing_session_id = None
    st.session_state.open_session_menu_id = None


def toggle_session_menu(session_id: int):
    current = st.session_state.get("open_session_menu_id")
    st.session_state.open_session_menu_id = None if current == session_id else session_id


def commit_session_rename(session_id: int):
    new_title = st.session_state.get(f"rename_input_{session_id}", "").strip()
    if new_title:
        db_service.rename_chat_session(session_id, new_title)
    st.session_state.editing_session_id = None
    st.session_state.open_session_menu_id = None


def remove_session(session_id: int, rag: RAGService | None = None):
    if is_busy():
        return

    cleanup_errors: list[str] = []
    delete_result = db_service.delete_chat_session(session_id)
    deleted_documents = delete_result.get("deleted_documents", [])
    deleted_document_ids = [int(document["id"]) for document in deleted_documents if document.get("id") is not None]

    if rag is not None:
        vector_result = rag.delete_documents(deleted_document_ids, session_id=session_id)
        if vector_result.get("status") != "success":
            cleanup_errors.append("vectorstore")

    try:
        if not cleanup_deleted_graph_documents(deleted_document_ids, session_id, rag):
            cleanup_errors.append("graph")
    except Exception as exc:
        LOG.warning("Graph cleanup failed while deleting session %s: %s", session_id, exc)
        cleanup_errors.append("graph")

    try:
        cleanup_deleted_document_files(deleted_documents)
    except Exception as exc:
        LOG.warning("File cleanup failed while deleting session %s: %s", session_id, exc)
        cleanup_errors.append("file")

    st.session_state.latest_sources_by_session.pop(session_id, None)
    st.session_state.corag_metadata_by_session.pop(session_id, None)
    st.session_state.selfrag_metadata_by_session.pop(session_id, None)
    st.session_state.pop(get_document_selection_key(session_id), None)
    st.session_state.pop(get_document_selection_catalog_key(session_id), None)

    # Clean UI state
    st.session_state.editing_session_id = None
    st.session_state.open_session_menu_id = None
    st.session_state.queued_query = None
    st.session_state.processing_query = None

    # Return to draft/new chat view
    st.session_state.active_session_id = None
    st.session_state.last_upload_success = None
    st.session_state.last_graph_success = None
    st.session_state.composer_prompt = ""
    st.session_state.draft_session = True

    if cleanup_errors:
        st.session_state.flash_message = {
            "type": "warning",
            "text": "Đã xóa đoạn chat khỏi lịch sử, nhưng một phần dữ liệu phụ chưa dọn xong. Xem log để kiểm tra.",
        }
    elif delete_result.get("deleted"):
        st.session_state.flash_message = {"type": "success", "text": "Đã xóa đoạn chat và toàn bộ dữ liệu liên quan!"}
    else:
        st.session_state.flash_message = {"type": "warning", "text": "Đoạn chat này không còn tồn tại."}

def sanitize_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    safe_stem = safe_stem or "document"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}_{safe_stem}{suffix}"


def save_uploaded_bytes(filename: str, content: bytes) -> str:
    os.makedirs(settings.DOCUMENT_DIR, exist_ok=True)
    saved_name = sanitize_filename(filename)
    saved_path = os.path.join(settings.DOCUMENT_DIR, saved_name)
    with open(saved_path, "wb") as file_handle:
        file_handle.write(content)
    return saved_path


def is_path_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def remove_file_if_safe(path_value: str | None, root: Path):
    if not path_value:
        return
    raw_path = Path(path_value)
    target = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    root = root.resolve()
    if target.exists() and target.is_file() and is_path_inside(target, root):
        target.unlink()


def remove_dir_if_safe(path_value: Path, root: Path):
    target = path_value.resolve()
    root = root.resolve()
    if target.exists() and target.is_dir() and is_path_inside(target, root):
        shutil.rmtree(target)


def cleanup_deleted_document_files(deleted_documents: list[dict]):
    data_root = Path(settings.DATA_DIR).resolve()
    images_root = (data_root / "images").resolve()
    document_root = Path(settings.DOCUMENT_DIR).resolve()

    for document in deleted_documents:
        document_id = document.get("id")
        for image_path in document.get("image_paths") or []:
            remove_file_if_safe(image_path, data_root)

        if document_id is not None:
            remove_dir_if_safe(images_root / f"doc_{int(document_id)}", images_root)

        remove_file_if_safe(document.get("filepath"), document_root)


def cleanup_deleted_graph_documents(document_ids: list[int], session_id: int, rag: RAGService | None = None) -> bool:
    if not document_ids:
        return True

    cleanup_ok = True
    graph_service = st.session_state.get("_graph_svc_instance")
    if graph_service is None:
        neo4j_uri = str(getattr(settings, "NEO4J_URI", "") or "").strip()
        if neo4j_uri and neo4j_uri.lower() != "none" and rag is not None:
            kg_service = None
            try:
                from app.services.knowledge_graph_service import KnowledgeGraphService

                kg_service = KnowledgeGraphService(
                    embedding_service=rag.embedding_service,
                    setup_indexes=False,
                )
                kg_service.delete_documents(document_ids)
            except Exception as exc:
                LOG.warning("Graph cleanup skipped because KnowledgeGraphService is unavailable: %s", exc)
                cleanup_ok = False
            finally:
                if kg_service is not None:
                    kg_service.close()
            graph_service = None

    if graph_service is not None:
        try:
            graph_service.delete_documents(document_ids)
        except Exception as exc:
            LOG.warning("Graph cleanup failed for document_ids=%s: %s", document_ids, exc)
            cleanup_ok = False

    for key in list(st.session_state.keys()):
        if key == f"graph_session_ready_{session_id}" or key.startswith(f"graph_indexed_{session_id}_"):
            st.session_state.pop(key, None)

    return cleanup_ok


def get_session_document_ids(active_session: dict | None) -> list[int]:
    if not active_session:
        return []
    raw_ids = active_session.get("document_ids") or []
    return [int(doc_id) for doc_id in raw_ids if doc_id is not None]


def get_document_selection_key(session_id: int) -> str:
    return f"selected_document_ids_{session_id}"


def get_document_selection_catalog_key(session_id: int) -> str:
    return f"selected_document_catalog_{session_id}"


def sync_selected_document_ids(active_session: dict | None) -> list[int]:
    if not active_session:
        return []

    session_id = active_session["id"]
    available_ids = get_session_document_ids(active_session)
    selection_key = get_document_selection_key(session_id)
    catalog_key = get_document_selection_catalog_key(session_id)
    previous_available_ids = st.session_state.get(catalog_key)
    raw_selection = st.session_state.get(selection_key)

    normalized_selection: list[int] = []
    seen_ids: set[int] = set()
    for raw_value in raw_selection or []:
        try:
            resolved_id = int(raw_value)
        except (TypeError, ValueError):
            continue
        if resolved_id not in available_ids or resolved_id in seen_ids:
            continue
        seen_ids.add(resolved_id)
        normalized_selection.append(resolved_id)

    had_full_previous_selection = (
        raw_selection is not None
        and previous_available_ids is not None
        and list(previous_available_ids) == list(raw_selection)
    )
    if raw_selection is None:
        normalized_selection = list(available_ids)
    elif had_full_previous_selection and normalized_selection == list(previous_available_ids):
        normalized_selection = list(available_ids)
    elif raw_selection and not normalized_selection:
        normalized_selection = list(available_ids)

    st.session_state[selection_key] = normalized_selection
    st.session_state[catalog_key] = list(available_ids)
    return normalized_selection


def get_selected_document_ids(active_session: dict | None) -> list[int]:
    return sync_selected_document_ids(active_session)


def get_selected_document_id(active_session: dict | None) -> int | None:
    if not active_session:
        return None

    selected_ids = get_selected_document_ids(active_session)
    if len(selected_ids) == 1:
        return selected_ids[0]

    primary_document_id = active_session.get("document_id")
    try:
        primary_document_id = int(primary_document_id) if primary_document_id is not None else None
    except (TypeError, ValueError):
        primary_document_id = None

    if primary_document_id in selected_ids:
        return primary_document_id

    return selected_ids[0] if selected_ids else None


def session_has_documents(active_session: dict | None) -> bool:
    return len(get_session_document_ids(active_session)) > 0


def normalize_uploaded_files(uploaded_files) -> list:
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, list):
        return [file for file in uploaded_files if file is not None]
    return [uploaded_files]


def queue_document_upload(rag: RAGService | None, uploaded_files, active_session: dict | None):
    # Chunk params lấy từ UI (nếu đã render)
    pending_chunk_params = {
        "chunk_size": st.session_state.get("upload_chunk_size"),
        # UI state key đang dùng là upload_chunk_overlap (không phải upload_chunk_overlap_sentences)
        "chunk_overlap_sentences": st.session_state.get("upload_chunk_overlap"),
    }

    # Chỉ đưa vào payload khi đủ giá trị (tránh None)
    pending_chunk_params = {
        k: v for k, v in pending_chunk_params.items() if v is not None
    }

    if rag is None:
        st.session_state.flash_message = {"type": "error", "text": "RAG service chưa sẵn sàng."}
        st.rerun()

    files = normalize_uploaded_files(uploaded_files)
    if not files:
        st.session_state.flash_message = {"type": "warning", "text": "Hãy chọn file PDF hoặc DOCX trước."}
        st.rerun()

    st.session_state.queued_upload = {
        "session_id": active_session["id"] if active_session else None,
        "files": [
            {
                "file_name": uploaded_file.name,
                "file_bytes": uploaded_file.getvalue(),
            }
            for uploaded_file in files
        ],
        "display_name": files[0].name if len(files) == 1 else f"{len(files)} tài liệu",
        "chunk_params": pending_chunk_params,
    }
    st.session_state.last_upload_success = None
    st.session_state.last_graph_success = None
    st.rerun()


def process_pending_upload(rag: RAGService | None):
    payload = st.session_state.get("processing_upload")
    chunk_params = payload.get("chunk_params") or {} if isinstance(payload, dict) else {}

    # Inject vào settings để rag_service.add_documents() dùng được
    if chunk_params:
        settings.UI_CHUNK_SIZE = chunk_params.get("chunk_size")
        settings.UI_CHUNK_OVERLAP_SENTENCES = chunk_params.get("chunk_overlap_sentences")
    if not payload:
        return

    if rag is None:
        st.session_state.processing_upload = None
        st.session_state.flash_message = {"type": "error", "text": "RAG service chưa sẵn sàng."}
        st.rerun()

    target_session = db_service.get_chat_session(payload["session_id"]) if payload.get("session_id") else None
    files = payload.get("files") or []

    append_to_existing_session = target_session is not None
    create_single_session_for_batch = (not append_to_existing_session) and len(files) == 1

    batch_session_id = None
    if append_to_existing_session:
        batch_session_id = target_session["id"]
    elif create_single_session_for_batch and files:
        batch_session_id = db_service.create_chat_session(files[0]["file_name"])

    processed_count = 0
    last_session_id = None
    last_file_name = None
    last_message = None

    try:
        for file_payload in files:
            # Decide target session for this file.
            if batch_session_id is not None:
                session_id = batch_session_id
            else:
                session_id = db_service.create_chat_session(file_payload["file_name"])

            saved_path = save_uploaded_bytes(file_payload["file_name"], file_payload["file_bytes"])
            document_id = None
            try:
                document_id = db_service.create_document(file_payload["file_name"], saved_path)
                result = rag.add_documents(saved_path, document_id=document_id, session_id=session_id)
                if result["status"] != "success":
                    raise RuntimeError(result["message"])

                db_service.replace_document_chunks(document_id, result.get("chunks", []))

                # If this is the first document of the session, use filename as session title.
                session_before_attach = db_service.get_chat_session(session_id)
                had_docs_before = session_has_documents(session_before_attach)
                db_service.attach_document_to_session(
                    session_id,
                    document_id,
                    title=file_payload["file_name"] if not had_docs_before else None,
                )

                # Graph status cache is per session/document, invalidate after new upload.
                st.session_state.pop(f"graph_session_ready_{session_id}", None)
                st.session_state.pop(f"graph_indexed_{session_id}_{document_id}", None)

                st.session_state.latest_sources_by_session[session_id] = []
                last_session_id = session_id
                last_file_name = file_payload["file_name"]
                last_message = result["message"]
                processed_count += 1
            except Exception:
                if document_id is not None:
                    db_service.delete_document(document_id)
                if os.path.exists(saved_path):
                    os.remove(saved_path)
                raise

        st.session_state.draft_session = False
        st.session_state.active_session_id = last_session_id
        st.session_state.open_session_menu_id = None

        if processed_count == 1 and last_session_id is not None and last_file_name:
            st.session_state.last_upload_success = {
                "session_id": last_session_id,
                "file_name": last_file_name,
                "message": last_message,
            }
            st.session_state.last_graph_success = None
        elif processed_count > 1:
            st.session_state.last_upload_success = None
            st.session_state.last_graph_success = None
            if append_to_existing_session:
                st.session_state.flash_message = {
                    "type": "success",
                    "text": f"Đã bổ sung {processed_count} tài liệu vào đoạn chat hiện tại.",
                }
            else:
                st.session_state.flash_message = {
                    "type": "success",
                    "text": f"Đã xử lý {processed_count} tài liệu và tạo đoạn chat riêng cho từng tài liệu.",
                }

        st.session_state.uploader_nonce += 1
        st.session_state.processing_upload = None
        st.rerun()
    except Exception as exc:
        LOG.error("Upload flow failed: %s", exc)
        if create_single_session_for_batch and batch_session_id is not None and processed_count == 0:
            db_service.delete_chat_session(batch_session_id)
        st.session_state.processing_upload = None
        st.session_state.flash_message = {"type": "error", "text": f"Lỗi xử lý file: {exc}"}
        st.rerun()


def queue_query(prompt: str, active_session: dict | None, rerun: bool = True):
    document_ids = get_selected_document_ids(active_session)
    if active_session is None or not document_ids:
        st.session_state.flash_message = {
            "type": "warning",
            "text": "Hãy chọn ít nhất một tài liệu trong session này trước khi đặt câu hỏi.",
        }
        if rerun:
            st.rerun()
        return

    rag_mode = RAG_MODE_BY_LABEL.get(st.session_state.current_rag_mode_label, "rag")
    if rag_mode == GRAPH_RAG_MODE:
        search_type = GRAPH_RAG_SEARCH_TYPE
        detail_level = "fast"
    else:
        search_type = "vector" if st.session_state.current_search_label == SEMANTIC_SEARCH_LABEL else "hybrid"
        detail_level = "fast" if st.session_state.current_detail_label == "Nhanh" else "detailed"

    st.session_state.queued_query = {
        "session_id": active_session["id"],
        "document_id": get_selected_document_id(active_session),
        "document_ids": document_ids,
        "prompt": prompt,
        "rag_mode": rag_mode,
        "search_type": search_type,
        "detail_level": detail_level,
    }
    st.session_state.latest_sources_by_session[active_session["id"]] = []
    st.session_state.last_upload_success = None
    
    # Bấm gửi câu hỏi là ẩn luôn thông báo đồ thị thành công
    st.session_state.last_graph_success = None
    
    schedule_bottom_scroll()
    if rerun:
        st.rerun()


def submit_query_from_state(active_session: dict | None):
    if is_busy():
        return
    prompt = st.session_state.get("composer_prompt", "").strip()
    if not prompt:
        return
    st.session_state.composer_prompt = ""
    queue_query(prompt, active_session, rerun=False)


def on_rag_mode_change():
    new_label = st.session_state.get("rag_mode_select")
    st.session_state.current_rag_mode_label = new_label
    if new_label == GRAPH_RAG_LABEL:
        st.session_state.initializing_graph = True
        st.session_state._graph_init_failed = False


def process_pending_query(rag: RAGService | None):
    payload = st.session_state.get("processing_query")
    if not payload:
        return

    if rag is None:
        st.session_state.processing_query = None
        st.session_state.flash_message = {"type": "error", "text": "RAG service chưa sẵn sàng."}
        st.rerun()

    result = rag.query(
        payload["prompt"],
        rag_mode=payload.get("rag_mode"),
        search_type=payload["search_type"],
        document_id=payload["document_id"],
        document_ids=payload.get("document_ids"),
        session_id=payload.get("session_id"),
        detail_level=payload["detail_level"],
    )
    db_service.add_chat_history(
        session_id=payload["session_id"],
        question=payload["prompt"],
        answer=result["answer"],
        document_id=payload["document_id"],
        search_type=payload["search_type"],
        rag_mode=payload.get("rag_mode", "rag"),
    )
    st.session_state.latest_sources_by_session[payload["session_id"]] = result.get("sources", [])

    session_id = payload["session_id"]
    if "corag_metadata_by_session" not in st.session_state:
        st.session_state.corag_metadata_by_session = {}
    if "selfrag_metadata_by_session" not in st.session_state:
        st.session_state.selfrag_metadata_by_session = {}
    rag_mode = payload.get("rag_mode", "rag")
    metadata = result.get("metadata", {})
    session_snapshot = db_service.get_chat_session(session_id)
    session_documents = session_snapshot.get("documents", []) if session_snapshot else []
    document_name_by_id = {
        int(document["id"]): str(document.get("filename") or f"Tài liệu {document['id']}")
        for document in session_documents
        if document.get("id") is not None
    }

    if rag_mode == "corag" and metadata.get("sub_queries"):
        st.session_state.corag_metadata_by_session[session_id] = {
            "sub_queries": metadata.get("sub_queries", []),
            "per_query_doc_counts": metadata.get("per_query_doc_counts", {}),
            "merged_doc_count": metadata.get("merged_doc_count", 0),
            "question": payload["prompt"],
        }
    else:
        st.session_state.corag_metadata_by_session.pop(session_id, None)

    if rag_mode == "selfrag" and metadata.get("trace"):
        selected_document_ids = [
            int(document_id)
            for document_id in (payload.get("document_ids") or [])
            if document_id is not None
        ]
        st.session_state.selfrag_metadata_by_session[session_id] = {
            "question": payload["prompt"],
            "final_query": metadata.get("final_query"),
            "query_history": metadata.get("query_history", []),
            "accepted": metadata.get("accepted", False),
            "attempts": metadata.get("attempts", 0),
            "confidence_score": metadata.get("confidence_score", 0.0),
            "evaluation_score": metadata.get("evaluation_score", 0.0),
            "confidence_threshold": metadata.get("confidence_threshold"),
            "min_confidence_threshold": metadata.get("min_confidence_threshold"),
            "error_type": metadata.get("error_type"),
            "evaluation_reason": metadata.get("evaluation_reason"),
            "policy_reason": metadata.get("policy_reason"),
            "trace": metadata.get("trace", []),
            "selected_document_ids": selected_document_ids,
            "selected_document_names": [
                document_name_by_id[document_id]
                for document_id in selected_document_ids
                if document_id in document_name_by_id
            ],
        }
    else:
        st.session_state.selfrag_metadata_by_session.pop(session_id, None)

    st.session_state.processing_query = None
    schedule_bottom_scroll()
    st.rerun()


def process_pending_graph_build(placeholder=None):
    payload = st.session_state.get("processing_graph")
    if not payload:
        return

    # LẬP TỨC CLEAR CỜ ĐỂ TRÁNH TREO STATE NẾU BỊ RELOAD GIỮA CHỪNG
    st.session_state.processing_graph = None

    graph_service = get_graph_rag_service()
    if not graph_service:
        st.session_state.flash_message = {"type": "error", "text": "GraphRAG service chưa sẵn sàng."}
        st.rerun()
        return

    session_id = payload.get("session_id")
    documents = payload.get("documents") or []
    doc_ids = payload.get("doc_ids") or []
    label = payload.get("label", "Xử lý đồ thị")
    loading_text = payload.get("loading_text", f"Đang {label.lower()}...")

    try:
        if documents:
            if placeholder:
                with placeholder.container():
                    with st.spinner(loading_text):
                        graph_service.update_graph_with_documents(documents)
            else:
                with st.spinner(loading_text):
                    graph_service.update_graph_with_documents(documents)

            if session_id is not None:
                for doc_id in doc_ids:
                    st.session_state[f"graph_indexed_{session_id}_{doc_id}"] = True
             
            clean_label = label
            st.session_state.last_graph_success = f"Đã {clean_label.lower()} thành công!"
             
        else:
            st.session_state.flash_message = {"type": "error", "text": "Không tìm thấy dữ liệu tài liệu để xây đồ thị."}
    except Exception as exc:
        LOG.error("Graph build failed: %s", exc)
        st.session_state.flash_message = {"type": "error", "text": f"Lỗi xử lý đồ thị: {exc}"}
    
    st.rerun()


def render_sidebar(sessions, rag: RAGService | None = None):
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Lịch sử hội thoại</div>', unsafe_allow_html=True)
        st.button(
            "+ Đoạn chat mới",
            key="new_chat_btn",
            use_container_width=True,
            type="primary",
            on_click=create_new_chat,
            disabled=is_busy(),
        )

        if st.session_state.get("draft_session"):
            st.markdown(
                """
                <div class="draft-session-item">
                    <span class="draft-session-dot"></span>
                    <span>Đoạn chat mới</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for session in sessions:
            session_id = session["id"]
            is_active = not st.session_state.get("draft_session") and session_id == st.session_state.active_session_id
            is_editing = session_id == st.session_state.editing_session_id
            is_menu_open = session_id == st.session_state.get("open_session_menu_id")

            if is_editing:
                row_cols = st.columns([6.1, 0.9], gap="small")
                with row_cols[0]:
                    st.text_input(
                        "Rename chat",
                        key=f"rename_input_{session_id}",
                        label_visibility="collapsed",
                        on_change=commit_session_rename,
                        args=(session_id,),
                        placeholder="Đổi tên đoạn chat",
                        disabled=is_busy(),
                    )
                with row_cols[1]:
                    st.button(
                        "x",
                        key=f"cancel_edit_{session_id}",
                        use_container_width=True,
                        on_click=cancel_session_rename,
                        disabled=is_busy(),
                    )
                continue

            row_cols = st.columns([6.1, 0.9], gap="small")
            with row_cols[0]:
                st.button(
                    session["title"],
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    disabled=is_busy(),
                    on_click=set_active_session,
                    args=(session_id,)
                )
            with row_cols[1]:
                st.button(
                    "⋯",
                    key=f"menu_toggle_{session_id}",
                    use_container_width=False,
                    disabled=is_busy(),
                    on_click=toggle_session_menu,
                    args=(session_id,),
                )

            if is_menu_open and not is_busy():
                menu_cols = st.columns([1, 1], gap="small")
                with menu_cols[0]:
                    st.button(
                        "Đổi tên",
                        key=f"menu_rename_{session_id}",
                        use_container_width=True,
                        on_click=begin_session_rename,
                        args=(session_id, session["title"])
                    )
                with menu_cols[1]:
                    st.button(
                        "Xóa",
                        key=f"menu_delete_{session_id}",
                        use_container_width=True,
                        on_click=remove_session,
                        args=(session_id, rag)
                    )

def render_processing_card(file_name: str):
    st.markdown(
        f"""
        <div class="upload-card upload-processing">
            <div class="processing-spinner"></div>
            <div class="upload-card-title">Đang xử lý</div>
            <div class="upload-card-subtitle">{file_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success_card(active_session: dict | None):
    success = st.session_state.get("last_upload_success")
    if not success or active_session is None or success.get("session_id") != active_session["id"]:
        return

    st.markdown(
        f"""
        <div class="ready-card">
            <div class="ready-check">✓</div>
            <div>
                <div class="ready-title">Tài liệu sẵn sàng</div>
                <div class="ready-subtitle">{success["file_name"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing_hero():
    st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-hero-kicker">Workspace đọc hiểu tài liệu</div>
            <div class="landing-hero-title">SmartDoc <span>AI</span></div>
            <div class="landing-hero-subtitle">Tải tài liệu PDF hoặc DOCX, để hệ thống xử lý, rồi đặt câu hỏi theo cách tự nhiên như đang trò chuyện với một trợ lý nghiên cứu.</div>
            <div class="landing-hero-badges">
                <span class="hero-badge">📄 PDF &amp; DOCX</span>
                <span class="hero-badge">🔍 RAG · CoRAG · GraphRAG</span>
                <span class="hero-badge">⚡ Semantic &amp; Hybrid Search</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_files(uploaded_files):
    files = normalize_uploaded_files(uploaded_files)
    if not files:
        return

    file_items = "".join(
        f"""
        <div class="selected-file-item">
            <span class="selected-file-icon">&#128196;</span>
            <span class="selected-file-name">{escape(file.name)}</span>
        </div>
        """
        for file in files
    )

    st.markdown(
        f"""
        <div class="selected-files-panel">
            <div class="selected-files-header">Đã chọn {len(files)} tài liệu</div>
            <div class="selected-files-list">{file_items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_stage(rag: RAGService | None, active_session: dict | None):
    _, center_col, _ = st.columns([0.75, 2.2, 0.75])
    with center_col:
        processing_upload = st.session_state.get("processing_upload")
        if processing_upload:
            render_processing_card(processing_upload.get("display_name", "Tài liệu"))
            return

        st.markdown(
            """
           
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Chunking parameters")
        chunk_ui = chunk_params_ui(key_prefix="upload_chunk")
        st.session_state.upload_chunk_size = chunk_ui["chunk_size"]
        st.session_state.upload_chunk_overlap = chunk_ui["chunk_overlap"]

        uploaded_files = st.file_uploader(
            "PDF hoặc DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"document_uploader_{st.session_state.uploader_nonce}",
            label_visibility="collapsed",
            disabled=is_busy(),
        )
        render_selected_files(uploaded_files)
        selected_files = normalize_uploaded_files(uploaded_files)
        if st.button(
            "Xử lý tài liệu" if len(selected_files) <= 1 else f"Xử lý {len(selected_files)} tài liệu",
            use_container_width=True,
            type="primary",
            disabled=len(selected_files) == 0 or rag is None or is_busy(),
        ):
            queue_document_upload(rag, selected_files, active_session)

        st.markdown(
            """
            <div class="landing-steps landing-steps-compact">
                <div class="landing-step">
                    <span class="landing-step-index">1</span>
                    <span class="landing-step-text">Chọn nhiều tài liệu cùng lúc nếu cần</span>
                </div>
                <div class="landing-step">
                    <span class="landing-step-index">2</span>
                    <span class="landing-step-text">Mỗi tài liệu sẽ được tạo thành một đoạn chat riêng</span>
                </div>
                <div class="landing-step">
                    <span class="landing-step-index">3</span>
                    <span class="landing-step-text">Đặt câu hỏi ngay trong thanh chat bên dưới</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_display_history(active_session: dict | None):
    if active_session is None:
        return []

    history = db_service.get_chat_history(active_session["id"])
    pending = st.session_state.get("processing_query")
    if pending and pending["session_id"] == active_session["id"]:
        history = history + [
            {"role": "user", "content": pending["prompt"], "pending": True},
            {"role": "assistant", "content": "", "thinking": True},
        ]
    return history


def render_corag_steps(active_session_id: int | None, is_last_assistant: bool):
    """Hiển thị CoRAG thinking steps cho tin nhắn assistant cuối cùng."""
    if not is_last_assistant or active_session_id is None:
        return
    meta = st.session_state.get("corag_metadata_by_session", {}).get(active_session_id)
    if not meta or not meta.get("sub_queries"):
        return

    sub_queries = meta["sub_queries"]
    per_query_doc_counts = meta.get("per_query_doc_counts", {})
    merged_doc_count = meta.get("merged_doc_count", 0)

    steps_html = '''
    <div class="corag-steps">
      <div class="corag-steps-header">
        <span class="corag-steps-icon">🔍</span>
        <span class="corag-steps-title">CoRAG · Quá trình phân tích</span>
      </div>
      <div class="corag-steps-body">
    '''

    # Step 1: Question Decomposition
    steps_html += f'''
        <div class="corag-step">
          <div class="corag-step-badge">1</div>
          <div class="corag-step-content">
            <div class="corag-step-label">Phân tách câu hỏi (Question Decomposition)</div>
            <div class="corag-step-detail">Tách thành <strong>{len(sub_queries)}</strong> sub-questions:</div>
            <ul class="corag-subqueries">
    '''
    for i, q in enumerate(sub_queries, 1):
        steps_html += f'<li><span class="sq-num">Q{i}</span> {q}</li>'
    steps_html += "</ul></div></div>"

    # Step 2: Multi-retrieval
    steps_html += f'''
        <div class="corag-step">
          <div class="corag-step-badge">2</div>
          <div class="corag-step-content">
            <div class="corag-step-label">Truy xuất đa luồng (Multi-Retrieval)</div>
            <div class="corag-step-detail">Retrieve FAISS+BM25 cho từng sub-question:</div>
            <ul class="corag-subqueries">
    '''
    for q, count in per_query_doc_counts.items():
        short_q = (q[:55] + "…") if len(q) > 55 else q
        steps_html += f'<li><span class="sq-icon">↳</span> "{short_q}" → <strong>{count}</strong> docs</li>'
    steps_html += "</ul></div></div>"

    # Step 3: Fusion & Rerank
    steps_html += f'''
        <div class="corag-step">
          <div class="corag-step-badge">3</div>
          <div class="corag-step-content">
            <div class="corag-step-label">Hợp nhất & Xếp hạng (Fusion + Rerank)</div>
            <div class="corag-step-detail">
              RRF merge + deduplicate → chọn <strong>{merged_doc_count}</strong> docs tốt nhất làm context.
            </div>
          </div>
        </div>
    '''

    # Step 4: LLM Synthesis
    steps_html += '''
        <div class="corag-step">
          <div class="corag-step-badge">4</div>
          <div class="corag-step-content">
            <div class="corag-step-label">Tổng hợp câu trả lời (LLM Synthesis)</div>
            <div class="corag-step-detail">LLM đọc context tổng hợp và trả lời toàn diện bên dưới.</div>
          </div>
        </div>
    '''

    steps_html += "</div></div>"
    st.html(steps_html)


def render_selfrag_trace(active_session_id: int | None, is_last_assistant: bool):
    if not is_last_assistant or active_session_id is None:
        return

    meta = st.session_state.get("selfrag_metadata_by_session", {}).get(active_session_id)
    if not meta or not meta.get("trace"):
        return

    def _to_percent(value) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return f"{int(round(max(0.0, min(1.0, numeric)) * 100))}%"

    def _format_decimal(value) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "-"

    def _to_html(text: object) -> str:
        return escape(str(text or "")).replace("\n", "<br>")

    action_labels = {
        "accept": "Đạt ngưỡng",
        "rewrite": "Viết lại truy vấn",
        "rewrite_before_generate": "Viết lại trước khi trả lời",
        "multi_hop": "Truy xuất bổ sung",
        "regenerate": "Sinh lại câu trả lời",
        "return_last_answer": "Giữ câu trả lời hiện tại",
        "early_stop_stagnation": "Dừng sớm do điểm không cải thiện",
        "early_stop_no_evidence": "Dừng do không đủ bằng chứng",
        "early_stop_no_new_evidence": "Dừng do không có bằng chứng mới",
        "fallback": "Dùng luồng dự phòng",
    }
    retrieval_labels = {
        "proceed": "Tiếp tục trả lời",
        "rewrite": "Cần viết lại truy vấn",
    }

    selected_document_names = meta.get("selected_document_names") or []
    selected_scope_text = (
        " | ".join(selected_document_names[:3]) + (f" | ... (+{len(selected_document_names) - 3})" if len(selected_document_names) > 3 else "")
        if selected_document_names
        else "Toàn bộ tài liệu trong session"
    )
    accepted = bool(meta.get("accepted"))
    status_label = "Đạt ngưỡng" if accepted else "Cần làm lại"

    attempt_blocks: list[str] = []
    for attempt in meta.get("trace", []):
        detail_rows: list[str] = []
        detail_rows.append(
            f"""
            <div class="selfrag-row">
                <span class="selfrag-row-label">Truy vấn</span>
                <span class="selfrag-row-value">{_to_html(attempt.get("query") or meta.get("question"))}</span>
            </div>
            """
        )

        doc_count = attempt.get("doc_count")
        if doc_count is not None:
            detail_rows.append(
                f"""
                <div class="selfrag-chip-row">
                    <span class="selfrag-chip">Nguồn: {escape(str(doc_count))}</span>
                    <span class="selfrag-chip">Retrieval: {_format_decimal(attempt.get("retrieval_quality"))}</span>
                    <span class="selfrag-chip">Relevance: {_format_decimal(attempt.get("relevance_score"))}</span>
                    <span class="selfrag-chip">Retrieval decision: {escape(retrieval_labels.get(str(attempt.get("retrieval_decision") or ""), str(attempt.get("retrieval_decision") or "-")))}</span>
                </div>
                """
            )

        if attempt.get("draft_answer"):
            detail_rows.append(
                f"""
                <div class="selfrag-answer-box">
                    <div class="selfrag-answer-label">Câu trả lời ở bước này</div>
                    <div class="selfrag-answer-content">{_to_html(attempt.get("draft_answer"))}</div>
                </div>
                """
            )

        if attempt.get("score") is not None or attempt.get("confidence") is not None:
            detail_rows.append(
                f"""
                <div class="selfrag-chip-row">
                    <span class="selfrag-chip">Score: {_format_decimal(attempt.get("score"))}</span>
                    <span class="selfrag-chip">Confidence: {_to_percent(attempt.get("confidence"))}</span>
                    <span class="selfrag-chip">Threshold: {_format_decimal(attempt.get("effective_threshold"))}</span>
                    <span class="selfrag-chip">Decision: {escape(action_labels.get(str(attempt.get("decision") or ""), str(attempt.get("decision") or "-")))}</span>
                </div>
                """
            )

        reason_text = attempt.get("reason") or attempt.get("retrieval_reason")
        if reason_text:
            detail_rows.append(
                f"""
                <div class="selfrag-row">
                    <span class="selfrag-row-label">Đánh giá</span>
                    <span class="selfrag-row-value">{_to_html(reason_text)}</span>
                </div>
                """
            )

        if attempt.get("rewritten_query"):
            detail_rows.append(
                f"""
                <div class="selfrag-row">
                    <span class="selfrag-row-label">Query mới</span>
                    <span class="selfrag-row-value">{_to_html(attempt.get("rewritten_query"))}</span>
                </div>
                """
            )

        if attempt.get("follow_up_query"):
            detail_rows.append(
                f"""
                <div class="selfrag-row">
                    <span class="selfrag-row-label">Multi-hop query</span>
                    <span class="selfrag-row-value">{_to_html(attempt.get("follow_up_query"))}</span>
                </div>
                """
            )

        attempt_blocks.append(
            f"""
            <div class="selfrag-step-card">
                <div class="selfrag-step-header">
                    <span class="selfrag-step-badge">Lần {escape(str(attempt.get("attempt") or "?"))}</span>
                    <span class="selfrag-step-action">{escape(action_labels.get(str(attempt.get("next_action") or ""), str(attempt.get("next_action") or "-")))}</span>
                </div>
                <div class="selfrag-step-body">
                    {''.join(detail_rows)}
                </div>
            </div>
            """
        )

    html = f"""
    <div class="selfrag-trace">
        <div class="selfrag-trace-header">
            <div>
                <div class="selfrag-trace-kicker">Self-RAG · Chi tiết từng bước</div>
                <div class="selfrag-trace-title">{status_label}</div>
            </div>
            <div class="selfrag-summary-chip {'selfrag-summary-chip-ok' if accepted else 'selfrag-summary-chip-warn'}">{status_label}</div>
        </div>
        <div class="selfrag-summary-grid">
            <div class="selfrag-summary-item">
                <span class="selfrag-summary-label">Điểm đánh giá</span>
                <strong>{_format_decimal(meta.get("evaluation_score"))}</strong>
            </div>
            <div class="selfrag-summary-item">
                <span class="selfrag-summary-label">Độ tin cậy</span>
                <strong>{_to_percent(meta.get("confidence_score"))}</strong>
            </div>
            <div class="selfrag-summary-item">
                <span class="selfrag-summary-label">Số vòng</span>
                <strong>{escape(str(meta.get("attempts") or len(meta.get("trace", []))))}</strong>
            </div>
            <div class="selfrag-summary-item">
                <span class="selfrag-summary-label">Phạm vi tài liệu</span>
                <strong>{escape(selected_scope_text)}</strong>
            </div>
        </div>
        <div class="selfrag-summary-note">{_to_html(meta.get("evaluation_reason") or "")}</div>
        <div class="selfrag-trace-steps">
            {''.join(attempt_blocks)}
        </div>
    </div>
    """
    st.html(html)


def render_thinking_card(rag_mode: str = "rag"):
    """Hiển thị thinking card với các bước xử lý theo từng mode."""

    STEPS = {
        "rag": [
            ("🔎", "Embed câu hỏi", "Chuyển câu hỏi thành vector embedding..."),
            ("📚", "Truy xuất tài liệu", "Tìm kiếm FAISS + BM25 trên vectorstore..."),
            ("🤖", "Sinh câu trả lời", "LLM đọc context và tổng hợp câu trả lời..."),
        ],
        "selfrag": [
            ("SR", "Lap ke hoach", "Chuan hoa cau hoi va chon chien luoc tu danh gia..."),
            ("RR", "Retrieve + rerank", "Lay bang chung ban dau va xep hang lai nguon tai lieu..."),
            ("EV", "Tu danh gia", "LLM cham diem do phu hop, day du va grounded..."),
            ("MH", "Rewrite / multi-hop", "Tu cai thien truy van hoac mo rong them sub-query neu can..."),
            ("CF", "Confidence gating", "Chap nhan cau tra loi khi vuot nguong tin cay..."),
        ],
        "corag": [
            ("🧩", "Phân tách câu hỏi", "LLM tách thành các sub-questions độc lập..."),
            ("🔎", "Multi-retrieval", "Retrieve FAISS+BM25 cho từng sub-question..."),
            ("🔀", "Fusion & Rerank", "RRF merge, loại trùng lặp, xếp hạng context..."),
            ("🤖", "Tổng hợp đa nguồn", "LLM đọc context tổng hợp và trả lời toàn diện..."),
        ],
        "graphrag": [
            ("🌱", "Seed retrieval", "Lấy chunks ban đầu qua vector/hybrid search..."),
            ("🕸️", "Graph expansion", "Mở rộng sang các node liên quan trong đồ thị..."),
            ("📊", "Re-ranking", "Xếp hạng lại context sau graph expansion..."),
            ("🤖", "Sinh câu trả lời", "LLM đọc connected evidence và tổng hợp..."),
        ],
    }

    steps = STEPS.get(rag_mode, STEPS["rag"])
    mode_label = MODE_LABELS.get(rag_mode, rag_mode.upper())

    steps_items = ""
    for i, (icon, label, detail) in enumerate(steps):
        is_last = i == len(steps) - 1
        steps_items += f"""
        <div class="tk-step {'tk-step-active' if is_last else 'tk-step-done'}">
          <div class="tk-step-icon">{icon}</div>
          <div class="tk-step-body">
            <div class="tk-step-label">{label}</div>
            <div class="tk-step-detail">{detail}</div>
          </div>
          <div class="tk-step-status">{'<span class="tk-spinner"></span>' if is_last else '<span class="tk-check">✓</span>'}</div>
        </div>"""

    html = f"""
    <div class="thinking-card">
      <div class="thinking-card-header">
        <span class="thinking-card-mode">{mode_label}</span>
        <span class="thinking-card-label">Đang xử lý câu hỏi</span>
        <span class="thinking-dots"><span></span><span></span><span></span></span>
      </div>
      <div class="thinking-card-steps">
        {steps_items}
      </div>
    </div>
    """
    st.html(html)


def render_thinking_card(rag_mode: str = "rag"):
    """Hiá»ƒn thá»‹ thinking card vá»›i cÃ¡c bÆ°á»›c xá»­ lÃ½ theo tá»«ng mode."""

    steps_by_mode = {
        "rag": [
            ("RE", "Embed câu hỏi", "Chuyển câu hỏi thành vector embedding..."),
            ("RT", "Truy xuất tài liệu", "Tìm kiếm FAISS + BM25 trên vectorstore..."),
            ("LL", "Sinh câu trả lời", "LLM đọc context và tổng hợp câu trả lời..."),
        ],
        "selfrag": [
            ("SR", "Lập kế hoạch", "Chuẩn hóa câu hỏi và chọn chiến lược tự đánh giá..."),
            ("RR", "Retrieve + rerank", "Lấy bằng chứng ban đầu và xếp hạng lại nguồn tài liệu..."),
            ("EV", "Tự đánh giá", "LLM chấm điểm độ phù hợp, độ đầy đủ và grounded..."),
            ("MH", "Rewrite / multi-hop", "Tự cải thiện truy vấn hoặc mở rộng thêm sub-query nếu cần..."),
            ("CF", "Confidence gating", "Chấp nhận câu trả lời khi vượt ngưỡng tin cậy..."),
        ],
        "corag": [
            ("QD", "Phân tách câu hỏi", "LLM tách thành các sub-questions độc lập..."),
            ("MR", "Multi-retrieval", "Retrieve FAISS + BM25 cho từng sub-question..."),
            ("FR", "Fusion & rerank", "RRF merge, loại trùng lặp, xếp hạng context..."),
            ("LL", "Tổng hợp đa nguồn", "LLM đọc context tổng hợp và trả lời toàn diện..."),
        ],
        "graphrag": [
            ("SD", "Seed retrieval", "Lấy chunks ban đầu qua vector / hybrid search..."),
            ("GX", "Graph expansion", "Mở rộng sang các node liên quan trong đồ thị..."),
            ("RK", "Re-ranking", "Xếp hạng lại context sau graph expansion..."),
            ("LL", "Sinh câu trả lời", "LLM đọc connected evidence và tổng hợp..."),
        ],
    }
    step_seconds_by_mode = {
        "rag": 1.10,
        "selfrag": 1.55,
        "corag": 1.30,
        "graphrag": 1.40,
    }

    steps = steps_by_mode.get(rag_mode, steps_by_mode["rag"])
    mode_label = MODE_LABELS.get(rag_mode, rag_mode.upper())
    step_seconds = step_seconds_by_mode.get(rag_mode, 1.20)
    card_id = f"thinking-card-{uuid4().hex[:8]}"

    step_blocks = []
    for index, (icon, label, detail) in enumerate(steps):
        is_last = index == len(steps) - 1
        delay_seconds = index * step_seconds
        row_class = "tk-step-hold" if is_last else "tk-step-seq"
        row_style = f"--tk-delay:{delay_seconds:.2f}s; --tk-slot:{step_seconds:.2f}s;"
        spinner_html = (
            '<span class="tk-status-spinner tk-status-spinner-hold"><span class="tk-spinner"></span></span>'
            if is_last
            else '<span class="tk-status-spinner tk-status-spinner-seq"><span class="tk-spinner"></span></span>'
        )
        check_html = "" if is_last else '<span class="tk-status-check tk-check">✓</span>'
        step_blocks.append(
            f"""
            <div class="tk-step tk-step-pending {row_class}" style="{row_style}">
              <div class="tk-step-icon">{escape(icon)}</div>
              <div class="tk-step-body">
                <div class="tk-step-label">{escape(label)}</div>
                <div class="tk-step-detail">{escape(detail)}</div>
              </div>
              <div class="tk-step-status">{spinner_html}{check_html}</div>
            </div>
            """
        )

    html = f"""
    <style>
      #{card_id} .thinking-card-subhint {{
        color: #64748b;
        font-size: 0.72rem;
        padding: 0.12rem 1rem 0;
      }}
      #{card_id} .tk-step-pending {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid transparent;
        opacity: 0.34;
      }}
      #{card_id} .tk-step-status {{
        gap: 0;
        position: relative;
      }}
      #{card_id} .tk-status-spinner,
      #{card_id} .tk-status-check {{
        opacity: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }}
      #{card_id} .tk-step-seq {{
        animation: tk-step-seq-activate var(--tk-slot) ease forwards;
        animation-delay: var(--tk-delay);
      }}
      #{card_id} .tk-step-hold {{
        animation: tk-step-hold-activate 0.01s linear forwards;
        animation-delay: var(--tk-delay);
      }}
      #{card_id} .tk-status-spinner-seq {{
        animation: tk-status-spinner-phase var(--tk-slot) linear forwards;
        animation-delay: var(--tk-delay);
      }}
      #{card_id} .tk-status-spinner-hold {{
        animation: tk-status-spinner-hold 0.01s linear forwards;
        animation-delay: var(--tk-delay);
      }}
      #{card_id} .tk-status-check {{
        animation: tk-status-check-show 0.01s linear forwards;
        animation-delay: calc(var(--tk-delay) + (var(--tk-slot) * 0.92));
      }}
      @keyframes tk-step-seq-activate {{
        0% {{
          opacity: 1;
          background: rgba(99, 102, 241, 0.10);
          border-color: rgba(99, 102, 241, 0.20);
        }}
        78% {{
          opacity: 1;
          background: rgba(99, 102, 241, 0.10);
          border-color: rgba(99, 102, 241, 0.20);
        }}
        100% {{
          opacity: 0.68;
          background: rgba(255, 255, 255, 0.02);
          border-color: transparent;
        }}
      }}
      @keyframes tk-step-hold-activate {{
        0% {{
          opacity: 1;
          background: rgba(99, 102, 241, 0.10);
          border-color: rgba(99, 102, 241, 0.20);
        }}
        100% {{
          opacity: 1;
          background: rgba(99, 102, 241, 0.10);
          border-color: rgba(99, 102, 241, 0.20);
        }}
      }}
      @keyframes tk-status-spinner-phase {{
        0% {{ opacity: 1; }}
        82% {{ opacity: 1; }}
        100% {{ opacity: 0; }}
      }}
      @keyframes tk-status-spinner-hold {{
        0% {{ opacity: 1; }}
        100% {{ opacity: 1; }}
      }}
      @keyframes tk-status-check-show {{
        0% {{ opacity: 1; }}
        100% {{ opacity: 1; }}
      }}
    </style>
    <div class="thinking-card" id="{card_id}">
      <div class="thinking-card-header">
        <span class="thinking-card-mode">{escape(mode_label)}</span>
        <span class="thinking-card-label">Đang xử lý câu hỏi</span>
        <span class="thinking-dots"><span></span><span></span><span></span></span>
      </div>
      <div class="thinking-card-subhint">Bước đang sáng là bước hệ thống đang xử lý.</div>
      <div class="thinking-card-steps">
        {''.join(step_blocks)}
      </div>
    </div>
    """
    st.html(html)


def render_chat_history(active_session: dict | None):
    history = get_display_history(active_session)
    if not history:
        st.markdown(
            """
            <div class="empty-chat-card">
                <h3>💬 Tài liệu đã sẵn sàng</h3>
                <p>Hãy đặt câu hỏi về nội dung tài liệu. Bạn có thể hỏi về khái niệm, chi tiết, so sánh, hoặc tóm tắt bất kỳ phần nào.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Tìm index của assistant message cuối cùng (đã có nội dung)
    last_assistant_idx = -1
    for i, msg in enumerate(history):
        if msg["role"] == "assistant" and not msg.get("thinking") and msg.get("content"):
            last_assistant_idx = i

    # Lấy rag_mode của query đang xử lý (để render đúng thinking steps)
    pending_rag_mode = "rag"
    pending = st.session_state.get("processing_query")
    if pending:
        pending_rag_mode = pending.get("rag_mode", "rag")

    for idx, message in enumerate(history):
        with st.chat_message(message["role"]):
            if message.get("thinking"):
                render_thinking_card(pending_rag_mode)
            else:
                is_last_assistant = (idx == last_assistant_idx)
                render_corag_steps(active_session["id"] if active_session else None, is_last_assistant)
                render_selfrag_trace(active_session["id"] if active_session else None, is_last_assistant)
                st.markdown(message["content"], unsafe_allow_html=True)


def render_sources(active_session_id: int | None):
    if active_session_id is None:
        return

    sources = st.session_state.latest_sources_by_session.get(active_session_id) or []
    if not sources:
        return

    st.markdown('<div class="sources-title">Nguồn liên quan</div>', unsafe_allow_html=True)

    def _is_graphrag_sources(items: list[dict]) -> bool:
        for item in items:
            source_type = str(item.get("source_type", "")).lower()
            if source_type in {"graph", "community", "text_global", "text"}:
                if source_type != "text":
                    return True
                # text source in GraphRAG usually has no chunk metadata
                if item.get("community_id") is not None or item.get("relationship") is not None:
                    return True
        return False

    is_graphrag = _is_graphrag_sources(sources)

    def _build_source_title(index: int, source: dict) -> str:
        source_type = str(source.get("source_type", "text")).lower()
        source_name = str(source.get("source", "unknown"))
        chunk = source.get("chunk", 0)

        if is_graphrag:
            if source_type == "community" and source.get("community_id") is not None:
                return f"Nguồn {index} · Cộng đồng số {source.get('community_id')}"
            if source_type == "graph":
                graph_source = source.get("graph_source")
                graph_target = source.get("graph_target")
                relation = source.get("relationship")
                if graph_source and graph_target and relation:
                    return f"Nguồn {index} · Đồ thị: {graph_source} -[{relation}]- {graph_target}"
                return f"Nguồn {index} · Bằng chứng đồ thị"
            if source_type.startswith("text"):
                if source_name and source_name != "unknown":
                    return f"Nguồn {index} · Văn bản hỗ trợ · {source_name}"
                return f"Nguồn {index} · Văn bản hỗ trợ"
            return f"Nguồn {index} · Bằng chứng GraphRAG"

        title = f"Nguồn {index} · {source_name} · chunk {chunk}"
        if source.get("page_start") is not None:
            title += f" · page {source['page_start']}"
        return title

    for index, source in enumerate(sources, start=1):
        title = _build_source_title(index, source)
        with st.expander(title):
            # Show main content
            st.write(source["content"])

            # Show images if available
            image_paths = source.get("image_paths", [])
            has_ocr = source.get("has_ocr", False)

            if image_paths or has_ocr:
                st.divider()
                col1, col2 = st.columns([1, 4])
                with col1:
                    if has_ocr:
                        st.caption("🖼️ OCR")
                with col2:
                    st.caption("Văn bản được trích xuất từ ảnh tài liệu")

            # Display images if they exist
            if image_paths:
                for img_path in image_paths:
                    if isinstance(img_path, str):
                        full_path = os.path.join(settings.DATA_DIR, img_path)
                        if os.path.exists(full_path):
                            try:
                                st.image(full_path, width="stretch")
                            except Exception as e:
                                st.warning(f"Không thể tải ảnh: {e}")
                        else:
                            st.warning(f"Ảnh không tìm thấy: {img_path}")


def render_document_pill(active_session: dict | None):
    if not active_session:
        return
    documents = active_session.get("documents") or []
    if not documents:
        return

    doc_names = [str(doc.get("filename", "")).strip() for doc in documents if doc.get("filename")]
    if not doc_names:
        return

    if len(doc_names) <= 3:
        display_name = " | ".join(doc_names)
    else:
        display_name = " | ".join(doc_names[:3]) + f" | ... (+{len(doc_names) - 3})"

    st.markdown('<div id="document-header-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="document-pill">
            <span class="document-pill-label">Tài liệu</span>
            <span class="document-pill-name">{display_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_scope_filter(active_session: dict | None):
    if not active_session:
        return

    documents = active_session.get("documents") or []
    if len(documents) < 2:
        return

    selection_key = get_document_selection_key(active_session["id"])
    sync_selected_document_ids(active_session)
    label_by_id = {
        int(document["id"]): str(document.get("filename") or f"Tài liệu {document['id']}")
        for document in documents
        if document.get("id") is not None
    }
    available_ids = list(label_by_id.keys())

    st.markdown(
        """
        <div class="doc-scope-note">
            <div class="doc-scope-title">Phạm vi tài liệu xử lý</div>
            <div class="doc-scope-subtitle">
                Mặc định hệ thống dùng toàn bộ tài liệu trong đoạn chat. Bạn có thể bỏ chọn để chỉ xử lý những tài liệu cần thiết.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.multiselect(
        "Chọn tài liệu để dùng cho câu trả lời",
        options=available_ids,
        format_func=lambda doc_id: label_by_id.get(doc_id, f"Tài liệu {doc_id}"),
        key=selection_key,
        disabled=is_busy(),
        placeholder="Chọn ít nhất 1 tài liệu",
    )

    current_selected = st.session_state.get(selection_key) or []
    if not current_selected:
        st.caption("Chọn ít nhất 1 tài liệu trước khi gửi câu hỏi.")
        return

    if len(current_selected) == len(available_ids):
        st.caption(f"Đang xử lý toàn bộ {len(available_ids)} tài liệu trong session này.")
    else:
        st.caption(
            f"Đang xử lý {len(current_selected)}/{len(available_ids)} tài liệu đã chọn."
        )


def render_inline_upload_for_session(rag: RAGService | None, active_session: dict | None):
    if active_session is None:
        return

    with st.expander("Bổ sung tài liệu vào đoạn chat này", expanded=False):
        uploaded_files = st.file_uploader(
            "Bổ sung PDF hoặc DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"inline_document_uploader_{active_session['id']}_{st.session_state.uploader_nonce}",
            label_visibility="collapsed",
            disabled=is_busy(),
        )
        selected_files = normalize_uploaded_files(uploaded_files)
        if st.button(
            "Bổ sung tài liệu",
            key=f"inline_upload_btn_{active_session['id']}",
            use_container_width=True,
            type="secondary",
            disabled=len(selected_files) == 0 or rag is None or is_busy(),
        ):
            queue_document_upload(rag, selected_files, active_session)
    
    st.markdown('<div id="document-header-end"></div>', unsafe_allow_html=True)


def render_composer(active_session: dict | None):
    processing = st.session_state.get("processing_upload") or st.session_state.get("processing_query")
    has_docs = session_has_documents(active_session)
    disabled = is_busy() or active_session is None or not has_docs
    prompt_placeholder = (
        "Nhập câu hỏi về tài liệu..."
        if active_session is not None and has_docs
        else "Tải tài liệu lên để bắt đầu trò chuyện"
    )
    send_label = "■" if processing else "→"

    with st.container():
        st.markdown('<div id="composer-anchor"></div>', unsafe_allow_html=True)
        if processing:
            st.markdown('<div id="composer-processing-flag"></div>', unsafe_allow_html=True)

        selected_rag_label = st.session_state.get(
            "rag_mode_select",
            st.session_state.get("current_rag_mode_label", DEFAULT_RAG_MODE_LABEL),
        )
        is_graphrag_mode = selected_rag_label == GRAPH_RAG_LABEL
        if is_graphrag_mode:
            st.markdown('<div id="composer-graphrag-mode"></div>', unsafe_allow_html=True)

        composer_cols = st.columns([1.55, 1.55, 1.3, 5.5, 0.85], gap="small")
        with composer_cols[0]:
            st.selectbox(
                "RAGMode",
                options=RAG_MODE_OPTIONS,
                label_visibility="collapsed",
                disabled=disabled,
                key="rag_mode_select",
                on_change=on_rag_mode_change,
            )
            st.session_state.current_rag_mode_label = st.session_state.get("rag_mode_select", DEFAULT_RAG_MODE_LABEL)
            is_graphrag_mode = st.session_state.current_rag_mode_label == GRAPH_RAG_LABEL

        if is_graphrag_mode:
            st.session_state.current_detail_label = "Nhanh"
        else:
            with composer_cols[1]:
                st.session_state.current_search_label = st.selectbox(
                    "Mode",
                    options=SEARCH_MODE_OPTIONS,
                    label_visibility="collapsed",
                    disabled=disabled,
                    key="search_mode_select",
                )
            with composer_cols[2]:
                if st.session_state.current_search_label == SEMANTIC_SEARCH_LABEL:
                    st.session_state.current_detail_label = st.selectbox(
                        "Depth",
                        options=["Nhanh", "Kỹ"],
                        label_visibility="collapsed",
                        disabled=disabled,
                        key="detail_mode_select",
                    )
                else:
                    st.selectbox(
                        "DepthLocked",
                        options=["Tự động"],
                        label_visibility="collapsed",
                        disabled=True,
                        key="detail_mode_select_locked",
                    )
                    st.session_state.current_detail_label = "Nhanh"
        
        with composer_cols[3]:
            show_build_button = False
            missing_doc_ids: list[int] = []

            if is_graphrag_mode and active_session and has_docs:
                graph_service = st.session_state.get("_graph_svc_instance")
                if graph_service:
                    try:
                        session_id = active_session["id"]
                        for doc_id in get_session_document_ids(active_session):
                            cache_key = f"graph_indexed_{session_id}_{doc_id}"
                            if cache_key not in st.session_state:
                                st.session_state[cache_key] = graph_service.is_document_indexed(doc_id)
                            if not st.session_state[cache_key]:
                                missing_doc_ids.append(doc_id)
                        show_build_button = len(missing_doc_ids) > 0
                    except Exception as e:
                        LOG.error(f"Lỗi kiểm tra trạng thái Graph: {e}")

            is_processing = st.session_state.get("processing_graph")
            
            if show_build_button:
                if not is_processing:
                    all_doc_ids = get_session_document_ids(active_session)
                    is_first_build = len(missing_doc_ids) == len(all_doc_ids)
                    label = "Xây dựng đồ thị tri thức" if is_first_build else "Cập nhật đồ thị tri thức"
                    loading_text = "Đang xây dựng đồ thị tri thức..." if is_first_build else "Đang cập nhật đồ thị tri thức..."
                    
                    if st.button(label, type="primary", use_container_width=True, disabled=disabled):
                        docs_by_id = {int(doc["id"]): doc for doc in (active_session.get("documents") or []) if doc.get("id") is not None}
                        target_docs = [docs_by_id[doc_id] for doc_id in missing_doc_ids if doc_id in docs_by_id]
                        st.session_state.processing_graph = {
                            "session_id": active_session["id"],
                            "doc_ids": missing_doc_ids,
                            "documents": target_docs,
                            "label": label,
                            "loading_text": loading_text
                        }
                        st.rerun()
                else:
                    st.button("Đang xử lý...", type="primary", use_container_width=True, disabled=True)
            else:
                st.text_input(
                    "Nhập câu hỏi",
                    key="composer_prompt",
                    label_visibility="collapsed",
                    placeholder=prompt_placeholder,
                    disabled=disabled,
                    on_change=submit_query_from_state,
                    args=(active_session,),
                )

        with composer_cols[4]:
            if not show_build_button:
                st.button(
                    send_label,
                    key="composer_send_btn",
                    use_container_width=True,
                    disabled=disabled,
                    on_click=submit_query_from_state,
                    args=(active_session,),
                )


def render_main_area(rag: RAGService | None, active_session: dict | None):
    graph_action_ph = st.empty()

    if active_session is None or not session_has_documents(active_session):
        with st.container():
            st.markdown('<div id="landing-stage-flag"></div>', unsafe_allow_html=True)
            graph_action_ph = st.empty() 
            render_landing_hero()
            render_upload_stage(rag, active_session)
            render_composer(active_session)
    else:
        with st.container():
            st.markdown('<div id="chat-stage-flag"></div>', unsafe_allow_html=True)
            
            render_document_pill(active_session)
            render_document_scope_filter(active_session)
            render_inline_upload_for_session(rag, active_session)
            render_success_card(active_session)
            
            render_chat_history(active_session)
            render_sources(active_session["id"])

            graph_action_ph = st.empty() 

            # Hiện thẻ thông báo chuẩn màu xanh lá (dấu tick tròn) khi xây dựng/cập nhật xong
            is_graphrag_mode = st.session_state.get("current_rag_mode_label") == GRAPH_RAG_LABEL
            graph_success_msg = st.session_state.get("last_graph_success")

            if is_graphrag_mode and graph_success_msg and not st.session_state.get("processing_graph"):
                with graph_action_ph.container():
                    st.markdown(
                        f"""
                        <div class="ready-card">
                            <div class="ready-check">✓</div>
                            <div>
                                <div class="ready-title">{graph_success_msg}</div>
                                <div class="ready-subtitle">Sẵn sàng để trả lời câu hỏi</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            st.markdown('<div id="content-bottom-anchor"></div><div class="chat-safe-spacer"></div>', unsafe_allow_html=True)
        
        render_composer(active_session)

    return graph_action_ph


def main():
    init_db()  
    init_state()
    css = load_css_file()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    rag = ensure_rag_service()
    transition_pending_states()

    # --- CLEAR FLAGS IF SERVICE EXISTS ---
    if st.session_state.get("_graph_svc_instance") is not None:
        st.session_state.initializing_graph = False
        st.session_state._do_graph_load = False

    if st.session_state.current_rag_mode_label == GRAPH_RAG_LABEL and st.session_state.get("_graph_svc_instance") is None:
        if not st.session_state.get("initializing_graph") and not st.session_state.get("_do_graph_load") and not st.session_state.get("_graph_init_failed"):
            st.session_state.initializing_graph = True
            st.rerun()

    sessions = ensure_active_session()
    active_session = get_active_session(sessions)

    top_placeholder = st.empty()
    with top_placeholder.container():
        show_flash_message()

    render_sidebar(sessions, rag)
    
    graph_action_ph = render_main_area(rag, active_session)  
    apply_scheduled_bottom_scroll()

    # TẢI SERVICE VÀ HIỂN THỊ SPINNER GIẢ LẬP ĐỊNH DẠNG INLINE CODE
    is_loading_graph = st.session_state.current_rag_mode_label == GRAPH_RAG_LABEL and st.session_state.get("_graph_svc_instance") is None
    if is_loading_graph:
        with graph_action_ph.container():
            with st.spinner("Running `get_graph_rag_service()` ."):
                try:
                    svc = get_graph_rag_service() 
                    st.session_state._graph_svc_instance = svc
                    
                    # --- RESET FLAGS NGAY KHI INIT XONG ---
                    st.session_state.initializing_graph = False
                    st.session_state._do_graph_load = False
                except Exception as exc:
                    LOG.error("GraphRAG init failed: %s", exc)
                    st.session_state._graph_svc_instance = None
                    st.session_state.initializing_graph = False
                    st.session_state._do_graph_load = False
                    st.session_state.flash_message = {"type": "error", "text": f"Lỗi khởi tạo: {exc}"}
        st.rerun()

    if st.session_state.get("processing_graph"):
        process_pending_graph_build(graph_action_ph)

    if st.session_state.get("processing_upload"):
        process_pending_upload(rag)
    if st.session_state.get("processing_query"):
        process_pending_query(rag)


if __name__ == "__main__":
    main()


