import os
import re
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

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import settings
from app.core.logger import LOG
from app.rag.registry import AVAILABLE_RAG_MODES, MODE_LABELS, normalize_rag_mode
from app.services.database_service import db_service
from app.services.rag_service import RAGService

st.set_page_config(layout="wide", page_title="SmartDoc AI")

RAG_MODE_OPTIONS = [MODE_LABELS[mode] for mode in AVAILABLE_RAG_MODES]
RAG_MODE_BY_LABEL = {MODE_LABELS[mode]: mode for mode in AVAILABLE_RAG_MODES}
DEFAULT_RAG_MODE = normalize_rag_mode(getattr(settings, "RAG_MODE", "rag"))
DEFAULT_RAG_MODE_LABEL = MODE_LABELS.get(DEFAULT_RAG_MODE, "RAG")


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
        "current_rag_mode_label": DEFAULT_RAG_MODE_LABEL,
        "current_search_label": "Ngữ nghĩa",
        "current_detail_label": "Nhanh",
        "composer_prompt": "",
        "open_session_menu_id": None,
        "scroll_to_bottom_nonce": 0,
        "scroll_applied_nonce": 0,
        "corag_metadata_by_session": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


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
    return any(
        [
            st.session_state.get("queued_upload"),
            st.session_state.get("processing_upload"),
            st.session_state.get("queued_query"),
            st.session_state.get("processing_query"),
        ]
    )


def show_flash_message():
    flash = st.session_state.get("flash_message")
    if not flash:
        return

    kind = flash.get("type", "error")
    text = flash.get("text", "")
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.error(text)
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

    if st.session_state.get("draft_session"):
        st.session_state.active_session_id = None
        return sessions

    if st.session_state.active_session_id is None or not any(
        session["id"] == st.session_state.active_session_id for session in sessions
    ):
        st.session_state.active_session_id = sessions[0]["id"]
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
    st.session_state.composer_prompt = ""
    st.session_state.open_session_menu_id = None
    st.rerun()


def begin_session_rename(session_id: int, current_title: str):
    if is_busy():
        return
    st.session_state.editing_session_id = session_id
    st.session_state[f"rename_input_{session_id}"] = current_title
    st.session_state.open_session_menu_id = None
    st.rerun()


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


def remove_session(session_id: int):
    if is_busy():
        return

    db_service.delete_chat_session(session_id)
    st.session_state.latest_sources_by_session.pop(session_id, None)

    if st.session_state.get("editing_session_id") == session_id:
        st.session_state.editing_session_id = None
    if st.session_state.get("open_session_menu_id") == session_id:
        st.session_state.open_session_menu_id = None

    queued_query = st.session_state.get("queued_query")
    if queued_query and queued_query.get("session_id") == session_id:
        st.session_state.queued_query = None

    processing_query = st.session_state.get("processing_query")
    if processing_query and processing_query.get("session_id") == session_id:
        st.session_state.processing_query = None

    if st.session_state.get("active_session_id") == session_id:
        st.session_state.active_session_id = None
        st.session_state.last_upload_success = None
        st.session_state.composer_prompt = ""
        st.session_state.draft_session = False

    st.session_state.flash_message = {"type": "success", "text": "Đã xóa đoạn chat."}
    st.rerun()


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


def session_can_receive_document(active_session: dict | None) -> bool:
    if not active_session:
        return False
    if active_session.get("document_id") is not None:
        return False
    return int(active_session.get("exchange_count") or 0) == 0


def normalize_uploaded_files(uploaded_files) -> list:
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, list):
        return [file for file in uploaded_files if file is not None]
    return [uploaded_files]


def queue_document_upload(rag: RAGService | None, uploaded_files, active_session: dict | None):
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
    }
    st.session_state.last_upload_success = None
    st.rerun()


def process_pending_upload(rag: RAGService | None):
    payload = st.session_state.get("processing_upload")
    if not payload:
        return

    if rag is None:
        st.session_state.processing_upload = None
        st.session_state.flash_message = {"type": "error", "text": "RAG service chưa sẵn sàng."}
        st.rerun()

    target_session = db_service.get_chat_session(payload["session_id"]) if payload.get("session_id") else None
    files = payload.get("files") or []
    attach_to_existing_session = len(files) == 1 and session_can_receive_document(target_session)
    processed_count = 0
    last_session_id = None
    last_file_name = None
    last_message = None

    try:
        for file_payload in files:
            saved_path = save_uploaded_bytes(file_payload["file_name"], file_payload["file_bytes"])
            document_id = None
            try:
                document_id = db_service.create_document(file_payload["file_name"], saved_path)
                result = rag.add_documents(saved_path, document_id=document_id)
                if result["status"] != "success":
                    raise RuntimeError(result["message"])

                db_service.replace_document_chunks(document_id, result.get("chunks", []))

                if attach_to_existing_session and target_session is not None:
                    session_id = target_session["id"]
                    db_service.attach_document_to_session(
                        session_id,
                        document_id,
                        title=file_payload["file_name"],
                    )
                else:
                    session_id = db_service.create_chat_session(
                        file_payload["file_name"],
                        document_id=document_id,
                    )

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
        elif processed_count > 1:
            st.session_state.last_upload_success = None
            st.session_state.flash_message = {
                "type": "success",
                "text": f"Đã xử lý {processed_count} tài liệu và tạo đoạn chat riêng cho từng tài liệu.",
            }
        st.session_state.uploader_nonce += 1
        st.session_state.processing_upload = None
        st.rerun()
    except Exception as exc:
        LOG.error("Upload flow failed: %s", exc)
        st.session_state.processing_upload = None
        st.session_state.flash_message = {"type": "error", "text": f"Lỗi xử lý file: {exc}"}
        st.rerun()


def queue_query(prompt: str, active_session: dict | None, rerun: bool = True):
    if active_session is None or active_session.get("document_id") is None:
        st.session_state.flash_message = {
            "type": "warning",
            "text": "Hãy upload tài liệu vào đoạn chat này trước khi đặt câu hỏi.",
        }
        if rerun:
            st.rerun()
        return

    st.session_state.queued_query = {
        "session_id": active_session["id"],
        "document_id": active_session["document_id"],
        "prompt": prompt,
        "rag_mode": RAG_MODE_BY_LABEL.get(st.session_state.current_rag_mode_label, "rag"),
        "search_type": "vector" if st.session_state.current_search_label == "Ngữ nghĩa" else "hybrid",
        "detail_level": "fast" if st.session_state.current_detail_label == "Nhanh" else "detailed",
    }
    st.session_state.latest_sources_by_session[active_session["id"]] = []
    st.session_state.last_upload_success = None
    schedule_bottom_scroll()
    if rerun:
        st.rerun()


def submit_query_from_state(active_session: dict | None):
    prompt = st.session_state.get("composer_prompt", "").strip()
    if not prompt:
        return
    st.session_state.composer_prompt = ""
    queue_query(prompt, active_session, rerun=False)


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

    # Lưu CoRAG metadata để hiển thị step-by-step
    session_id = payload["session_id"]
    if "corag_metadata_by_session" not in st.session_state:
        st.session_state.corag_metadata_by_session = {}
    rag_mode = payload.get("rag_mode", "rag")
    metadata = result.get("metadata", {})
    if rag_mode == "corag" and metadata.get("sub_queries"):
        st.session_state.corag_metadata_by_session[session_id] = {
            "sub_queries": metadata.get("sub_queries", []),
            "per_query_doc_counts": metadata.get("per_query_doc_counts", {}),
            "merged_doc_count": metadata.get("merged_doc_count", 0),
            "question": payload["prompt"],
        }
    else:
        st.session_state.corag_metadata_by_session.pop(session_id, None)

    st.session_state.processing_query = None
    schedule_bottom_scroll()
    st.rerun()


def render_sidebar(sessions):
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
                if st.button(
                    session["title"],
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    disabled=is_busy(),
                ):
                    set_active_session(session_id)
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
                    if st.button(
                        "Đổi tên",
                        key=f"menu_rename_{session_id}",
                        use_container_width=True,
                    ):
                        begin_session_rename(session_id, session["title"])
                with menu_cols[1]:
                    if st.button(
                        "Xóa",
                        key=f"menu_delete_{session_id}",
                        use_container_width=True,
                    ):
                        remove_session(session_id)

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
            <div class="upload-card upload-card-compact">
                <div class="upload-card-icon">&uarr;</div>
                <div class="upload-card-title">Kéo thả tài liệu vào đây</div>
                <div class="upload-card-subtitle">Chọn một hoặc nhiều file PDF, DOCX để tạo chat riêng cho từng tài liệu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "PDF ho?c DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"document_uploader_{st.session_state.uploader_nonce}",
            label_visibility="collapsed",
            disabled=is_busy(),
        )
        render_selected_files(uploaded_files)
        selected_files = normalize_uploaded_files(uploaded_files)
        if st.button(
            "Xử lý tài liệu" if len(selected_files) <= 1 else f"X? l? {len(selected_files)} t?i li?u",
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


def render_thinking_card(rag_mode: str = "rag"):
    """Hiển thị thinking card với các bước xử lý theo từng mode."""

    STEPS = {
        "rag": [
            ("🔎", "Embed câu hỏi", "Chuyển câu hỏi thành vector embedding..."),
            ("📚", "Truy xuất tài liệu", "Tìm kiếm FAISS + BM25 trên vectorstore..."),
            ("🤖", "Sinh câu trả lời", "LLM đọc context và tổng hợp câu trả lời..."),
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
    mode_label = {"rag": "RAG", "corag": "CoRAG", "graphrag": "GraphRAG"}.get(rag_mode, rag_mode.upper())

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
                st.markdown(message["content"], unsafe_allow_html=True)


def render_sources(active_session_id: int | None):
    if active_session_id is None:
        return

    sources = st.session_state.latest_sources_by_session.get(active_session_id) or []
    if not sources:
        return

    st.markdown('<div class="sources-title">Nguồn liên quan</div>', unsafe_allow_html=True)
    for index, source in enumerate(sources, start=1):
        title = f"Nguồn {index} · {source['source']} · chunk {source['chunk']}"
        if source.get("page_start") is not None:
            title += f" · page {source['page_start']}"
        with st.expander(title):
            st.write(source["content"])


def render_document_pill(active_session: dict | None):
    if not active_session or not active_session.get("document_name"):
        return

    st.markdown(
        f"""
        <div class="document-pill">
            <span class="document-pill-label">Tài liệu</span>
            <span class="document-pill-name">{active_session["document_name"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_composer(active_session: dict | None):
    processing = st.session_state.get("processing_upload") or st.session_state.get("processing_query")
    disabled = is_busy() or active_session is None or active_session.get("document_id") is None
    prompt_placeholder = (
        "Nhập câu hỏi về tài liệu..."
        if active_session is not None and active_session.get("document_id") is not None
        else "Tải tài liệu lên để bắt đầu trò chuyện"
    )
    send_label = "■" if processing else "→"

    with st.container():
        st.markdown('<div id="composer-anchor"></div>', unsafe_allow_html=True)
        if processing:
            st.markdown('<div id="composer-processing-flag"></div>', unsafe_allow_html=True)

        composer_cols = st.columns([1.55, 1.55, 1.3, 5.5, 0.85], gap="small")
        with composer_cols[0]:
            st.session_state.current_rag_mode_label = st.selectbox(
                "RAGMode",
                options=RAG_MODE_OPTIONS,
                label_visibility="collapsed",
                disabled=disabled,
                key="rag_mode_select",
            )
        with composer_cols[1]:
            st.session_state.current_search_label = st.selectbox(
                "Mode",
                options=["Ngữ nghĩa", "Từ khóa"],
                label_visibility="collapsed",
                disabled=disabled,
                key="search_mode_select",
            )
        with composer_cols[2]:
            if st.session_state.current_search_label == "Ngữ nghĩa":
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
            st.button(
                send_label,
                key="composer_send_btn",
                use_container_width=True,
                disabled=disabled,
                on_click=submit_query_from_state,
                args=(active_session,),
            )


def render_main_area(rag: RAGService | None, active_session: dict | None):
    show_flash_message()

    if active_session is None or active_session.get("document_id") is None:
        with st.container():
            st.markdown('<div id="landing-stage-flag"></div>', unsafe_allow_html=True)
            render_landing_hero()
            render_upload_stage(rag, active_session)
            render_composer(active_session)
    else:
        with st.container():
            st.markdown('<div id="chat-stage-flag"></div>', unsafe_allow_html=True)
            render_document_pill(active_session)
            render_success_card(active_session)
            render_chat_history(active_session)
            render_sources(active_session["id"])
            st.markdown('<div id="content-bottom-anchor"></div><div class="chat-safe-spacer"></div>', unsafe_allow_html=True)
        render_composer(active_session)


def main():
    init_state()
    css = load_css_file()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    rag = ensure_rag_service()
    transition_pending_states()

    sessions = ensure_active_session()
    active_session = get_active_session(sessions)

    render_sidebar(sessions)
    render_main_area(rag, active_session)
    apply_scheduled_bottom_scroll()

    if st.session_state.get("processing_upload"):
        process_pending_upload(rag)
    if st.session_state.get("processing_query"):
        process_pending_query(rag)


if __name__ == "__main__":
    main()