import os
import re
import sys
from datetime import datetime
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
from app.services.database_service import db_service
from app.services.rag_service import RAGService

st.set_page_config(layout="wide", page_title="SmartDoc AI")


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
        "current_search_label": "Ngữ nghĩa",
        "current_detail_label": "Nhanh",
        "composer_prompt": "",
        "open_session_menu_id": None,
        "scroll_to_bottom_nonce": 0,
        "scroll_applied_nonce": 0,
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
    st.rerun()


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


def queue_document_upload(rag: RAGService | None, uploaded_file, active_session: dict | None):
    if rag is None:
        st.session_state.flash_message = {"type": "error", "text": "RAG service chưa sẵn sàng."}
        st.rerun()

    if uploaded_file is None:
        st.session_state.flash_message = {"type": "warning", "text": "Hãy chọn file PDF hoặc DOCX trước."}
        st.rerun()

    st.session_state.queued_upload = {
        "session_id": active_session["id"] if active_session else None,
        "file_name": uploaded_file.name,
        "file_bytes": uploaded_file.getvalue(),
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
    saved_path = save_uploaded_bytes(payload["file_name"], payload["file_bytes"])
    document_id = None

    try:
        document_id = db_service.create_document(payload["file_name"], saved_path)
        result = rag.add_documents(saved_path, document_id=document_id)
        if result["status"] != "success":
            raise RuntimeError(result["message"])

        db_service.replace_document_chunks(document_id, result.get("chunks", []))

        if session_can_receive_document(target_session):
            session_id = target_session["id"]
            db_service.attach_document_to_session(session_id, document_id, title=payload["file_name"])
        else:
            session_id = db_service.create_chat_session(payload["file_name"], document_id=document_id)

        st.session_state.draft_session = False
        st.session_state.active_session_id = session_id
        st.session_state.latest_sources_by_session[session_id] = []
        st.session_state.open_session_menu_id = None
        st.session_state.last_upload_success = {
            "session_id": session_id,
            "file_name": payload["file_name"],
            "message": result["message"],
        }
        st.session_state.uploader_nonce += 1
        st.session_state.processing_upload = None
        st.rerun()
    except Exception as exc:
        LOG.error("Upload flow failed: %s", exc)
        if document_id is not None:
            db_service.delete_document(document_id)
        if os.path.exists(saved_path):
            os.remove(saved_path)
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
    )
    st.session_state.latest_sources_by_session[payload["session_id"]] = result.get("sources", [])
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
            <div class="upload-card-title">Processing</div>
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
            <div class="landing-hero-title">Hệ thống Phân tích Tài liệu</div>
            <div class="landing-hero-subtitle">Tải tài liệu và bắt đầu trò chuyện cùng AI.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_stage(rag: RAGService | None, active_session: dict | None):
    _, center_col, _ = st.columns([1.0, 1.85, 1.0])
    with center_col:
        processing_upload = st.session_state.get("processing_upload")
        if processing_upload:
            render_processing_card(processing_upload["file_name"])
            return

        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-icon">↑</div>
                <div class="upload-card-title">Kéo thả file vào đây</div>
                <div class="upload-card-subtitle">Hỗ trợ định dạng PDF, DOCX</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "PDF hoặc DOCX",
            type=["pdf", "docx"],
            key=f"document_uploader_{st.session_state.uploader_nonce}",
            label_visibility="collapsed",
            disabled=is_busy(),
        )
        if st.button(
            "Xử lý tài liệu",
            use_container_width=True,
            type="primary",
            disabled=uploaded_file is None or rag is None or is_busy(),
        ):
            queue_document_upload(rag, uploaded_file, active_session)


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


def render_chat_history(active_session: dict | None):
    history = get_display_history(active_session)
    if not history:
        st.markdown(
            """
            <div class="empty-chat-card">
                <h3>Bắt đầu với câu hỏi đầu tiên</h3>
                <p>Tài liệu đã sẵn sàng. Hãy hỏi bất kỳ điều gì liên quan tới nội dung của nó.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in history:
        with st.chat_message(message["role"]):
            if message.get("thinking"):
                st.markdown(
                    """
                    <div class="thinking-row">
                        <span></span><span></span><span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(message["content"])


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
            <span class="document-pill-label">Document</span>
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

        composer_cols = st.columns([2.2, 1.8, 6.8, 1.2], gap="small")
        with composer_cols[0]:
            st.session_state.current_search_label = st.selectbox(
                "Mode",
                options=["Ngữ nghĩa", "Từ khóa"],
                label_visibility="collapsed",
                disabled=disabled,
                key="search_mode_select",
            )
        with composer_cols[1]:
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
        with composer_cols[2]:
            st.text_input(
                "Nhập câu hỏi",
                key="composer_prompt",
                label_visibility="collapsed",
                placeholder=prompt_placeholder,
                disabled=disabled,
                on_change=submit_query_from_state,
                args=(active_session,),
            )
        with composer_cols[3]:
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
        render_landing_hero()
        render_upload_stage(rag, active_session)
    else:
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
