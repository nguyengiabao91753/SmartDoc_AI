from __future__ import annotations

import streamlit as st


def chunk_params_ui(*, key_prefix: str = "chunk_params"):
    """UI cấu hình chunking ổn định, không lỗi session_state."""

    # ===== KEYS =====
    auto_key = "upload_chunk_auto_mode"
    size_key = f"{key_prefix}_chunk_size_input"
    overlap_key = f"{key_prefix}_chunk_overlap_input"

    # ===== DEFAULT STATE (APP READS THESE) =====
    if auto_key not in st.session_state:
        st.session_state[auto_key] = True

    if "upload_chunk_size" not in st.session_state:
        st.session_state["upload_chunk_size"] = 600

    if "upload_chunk_overlap" not in st.session_state:
        st.session_state["upload_chunk_overlap"] = 3

    # ===== WIDGET STATE =====
    if size_key not in st.session_state:
        st.session_state[size_key] = st.session_state["upload_chunk_size"]

    if overlap_key not in st.session_state:
        st.session_state[overlap_key] = st.session_state["upload_chunk_overlap"]

    # ===== ACTIONS (ONLY BUTTONS CHANGE STATE) =====
    def to_auto():
        st.session_state[auto_key] = True
        st.session_state["upload_chunk_size"] = 600
        st.session_state["upload_chunk_overlap"] = 3
        st.session_state[size_key] = 600
        st.session_state[overlap_key] = 3

    def to_manual():
        st.session_state[auto_key] = False

    # ===== CHECKBOX (READ ONLY) =====
    st.checkbox(
        "Tự động tối ưu chia đoạn (khuyến nghị)",
        key=auto_key,
        help="Khi bật: hệ thống dùng cấu hình tối ưu cho RAG.",
    )

    auto_mode = bool(st.session_state[auto_key])
    disabled = auto_mode

    # ===== UI INPUTS =====
    col1, col2, col3 = st.columns(3)

    with col1:
        st.number_input(
            "Chunk_Size (ký tự)",
            min_value=300,
            max_value=1200,
            step=50,
            key=size_key,
            disabled=disabled,
            format="%d",
        )

    with col2:
        st.number_input(
            "Overlap (câu)",
            min_value=1,
            max_value=10,
            step=1,
            key=overlap_key,
            disabled=disabled,
            format="%d",
        )

    with col3:
        if auto_mode:
            st.button(
                "Chỉnh thông số",
                use_container_width=True,
                on_click=to_manual,
            )
        else:
            st.button(
                "Reset",
                use_container_width=True,
                on_click=to_auto,
            )

    # ===== SYNC WIDGET → APP STATE =====
    chunk_size = int(st.session_state[size_key])
    chunk_overlap = int(st.session_state[overlap_key])

    st.session_state["upload_chunk_size"] = chunk_size
    st.session_state["upload_chunk_overlap"] = chunk_overlap

    # ===== VALIDATION (MANUAL MODE) =====
    if not auto_mode:
        if not (300 <= chunk_size <= 1200):
            st.error("Chunk size phải trong khoảng 300 → 1200.")
            st.stop()

        if not (1 <= chunk_overlap <= 10):
            st.error("Overlap phải trong khoảng 1 → 10.")
            st.stop()

        st.caption("Đang dùng cấu hình bạn chỉnh.")

    # Backend đang đọc key này
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_overlap_sentences": chunk_overlap,
    }