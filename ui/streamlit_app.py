import streamlit as st
import os
import base64
import sys
import json
from urllib import request, error
from pathlib import Path

# Reduce BLAS memory pressure on Windows when loading embedding stack.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.rag_service import RAGService
from app.core.logger import LOG

st.set_page_config(layout="wide", page_title="SmartDoc AI")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def get_ollama_status():
    """Kiểm tra nhanh Ollama local API và model hiện có."""
    url = "http://127.0.0.1:11434/api/tags"
    try:
        with request.urlopen(url, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in payload.get("models", [])]
            return True, models
    except (error.URLError, TimeoutError, ValueError):
        return False, []

# ========== INITIALIZE RAG SERVICE ==========
@st.cache_resource
def get_rag_service():
    """Cache RAG service để tránh khởi tạo lại"""
    return RAGService()


def ensure_rag_service():
    """Khởi tạo RAG service theo nhu cầu để UI không bị treo khi mở trang."""
    try:
        return get_rag_service()
    except Exception as e:
        st.error(f"Không thể khởi tạo RAG service: {str(e)}")
        LOG.error(f"RAG init error: {str(e)}")
        return None


ollama_ok, ollama_models = get_ollama_status()
if ollama_ok:
    model_text = ", ".join(ollama_models[:3]) if ollama_models else "(không có model)"
    st.success(f"Ollama đang chạy. Models: {model_text}")
else:
    st.warning("Ollama chưa chạy hoặc không truy cập được tại 127.0.0.1:11434")

# ==========================================================
# 1. ĐỌC FILE CSS TỪ THƯ MỤC 'css'
# ==========================================================

def load_css_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "css/app.css")
    
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# ==========================================================
# 2. GLOBAL CSS: Ẩn UI mặc định của Streamlit
# ==========================================================
global_css = """
<style>
header[data-testid="stHeader"], [data-testid="stSidebar"], footer { display: none !important; }
[data-testid="stFileUploader"] { 
    position: fixed !important; 
    top: 50% !important; 
    left: 55% !important; 
    transform: translate(-50%, -100%) !important; 
    width: 60% !important; 
    max-width: 600px !important; 
    z-index: 999999 !important; 
}
[data-testid="stFileUploader"] section {
    border: 2px dashed #6366f1 !important;
    border-radius: 16px !important;
    background: white !important;
    padding: 30px !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
iframe { border: none !important; }
</style>
"""
st.html(global_css)

# ==========================================================
# 3. HTML COMPONENT
# ==========================================================
html_code = """
<input type="checkbox" id="sidebar-toggle" style="display: none;">

<div id="custom-app">
    <div id="sidebar" class="sidebar">
        <div class="sidebar-section">
            <h3>
                <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> 
                Lịch sử hội thoại
            </h3>
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z"/></svg>
                Danh sách trống
            </div>
        </div>
        <div style="flex-grow: 1;"></div>
        <div class="sidebar-section">
            <h3>
                <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path stroke-linecap="round" stroke-linejoin="round" d="M12 16v-4m0-4h.01"/></svg>
                Hướng dẫn
            </h3>
            <ul class="instructions-list">
                <li>
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                    Tải tài liệu lên.
                </li>
                <li>
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
                    Chọn tham số.
                </li>
                <li>
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
                    Chat với AI.
                </li>
            </ul>
        </div>
    </div>

    <div class="main-content">
        <header class="top-nav">
            <label for="sidebar-toggle" class="toggle-btn">
                <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16"/></svg>
            </label>
            <div class="auth-group">
                <a href="/log_in" target="_self" class="btn btn-login">
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"/></svg>
                    Đăng nhập
                </a>
                <a href="/register" target="_self" class="btn btn-register">
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"/></svg>
                    Đăng ký
                </a>
            </div>
        </header>

        <div class="content-body">
            <div class="hero">
                <h1>Hệ thống Phân tích Tài liệu</h1>
                <p style="color: #64748b;">Tải tài liệu và bắt đầu trò chuyện cùng AI</p>
            </div>
            
            <input type="file" id="hidden-file-input" accept=".pdf,.doc,.docx" style="display: none;">
            <div class="upload-box" id="upload-box-trigger">
                <svg class="icon-xl" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"/></svg>
                <div style="font-weight: 600; font-size: 1.1rem;">Kéo thả file vào đây</div>
                <div style="color: #64748b; font-size: 14px; margin-top: 5px;">Hỗ trợ định dạng PDF, DOCX</div>
                <button class="upload-btn" id="upload-btn-trigger">
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                    Chọn file từ máy
                </button>
            </div>

            <div id="upload-status" style="display: none; margin-top: 20px; background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: left; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <span id="st-text" style="font-weight: 600; font-size: 14px; color: #0f172a;">Đang chuẩn bị...</span>
                    <span id="st-pct" style="font-size: 14px; color: #6366f1; font-weight: 600;">0%</span>
                </div>
                <div style="width: 100%; background: #f1f5f9; border-radius: 8px; height: 8px; overflow: hidden; margin-bottom: 20px;">
                    <div id="st-bar" style="width: 0%; height: 100%; background: #6366f1; transition: width 0.5s ease-out, background-color 0.3s;"></div>
                </div>
                <div style="font-size: 13.5px; color: #64748b; display: flex; flex-direction: column; gap: 10px;">
                    <div id="step-1" style="display: flex; gap: 8px; align-items: center;"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg> Đang scan tài liệu...</div>
                    <div id="step-2" style="display: flex; gap: 8px; align-items: center;"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg> Đang xử lý...</div>
                    <div id="step-3" style="display: flex; gap: 8px; align-items: center;"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg> Đang lưu...</div>
                    <div id="step-4" style="display: flex; gap: 8px; align-items: center;"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg> Tài liệu đã sẵn sàng</div>
                </div>
            </div>

            <div class="search-container" style="display: flex; flex-direction: column; align-items: center; margin-top: 30px;">
                <input type="radio" id="type-semantic" name="r1" class="search-type-input" checked>
                <input type="radio" id="type-keyword" name="r1" class="search-type-input">
                <div class="search-ui">
                    <div class="config-bar" style="margin-top: 0;">
                        <label for="type-semantic">
                            <span class="radio-custom"></span> 
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
                            Ngữ nghĩa
                        </label>
                        <label for="type-keyword">
                            <span class="radio-custom"></span> 
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
                            Từ khóa
                        </label>
                    </div>
                    <div class="semantic-sub">
                        <label>
                            <input type="radio" name="r2" checked> 
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                            Nhanh
                        </label>
                        <label>
                            <input type="radio" name="r2"> 
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7M5 19l4-4m10-8l-6 6"/></svg>
                            Kỹ
                        </label>
                    </div>
                </div>
            </div>
        </div>

        <div class="chat-container">
            <div class="chat-wrapper">
                <input type="text" class="chat-input" placeholder="Nhập câu hỏi...">
                <button class="send-btn">
                    <svg class="icon" style="color: white; width: 1.2rem; height: 1.2rem;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
                </button>
            </div>
        </div>
    </div>
</div>
"""

# ==========================================================
# 4. JS SCRIPT (Đã fix lỗi không nhận nút bấm)
# ==========================================================
js_code = """
export default function(component) {
    const { setTriggerValue, parentElement } = component;
    
    // --- XỬ LÝ CHAT ---
    const sendBtn = parentElement.querySelector(".send-btn");
    const chatInput = parentElement.querySelector(".chat-input");
    
    if (sendBtn && chatInput) {
        const sendMessage = () => {
            const message = chatInput.value.trim();
            if (message) {
                setTriggerValue("chat_event", {
                    "action": "send_message",
                    "text": message,
                    "nonce": Date.now()
                });
                chatInput.value = ""; 
            }
        };

        sendBtn.addEventListener('click', sendMessage);

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.code === 'NumpadEnter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // --- XỬ LÝ UPLOAD FILE ---
    // Sửa document thành parentElement để Streamlit có thể tìm thấy nút bấm
    const fileInput = parentElement.querySelector("#hidden-file-input");
    const uploadBox = parentElement.querySelector("#upload-box-trigger");
    const uploadBtn = parentElement.querySelector("#upload-btn-trigger");

    if (fileInput && uploadBox && uploadBtn) {
        
        // Mở hộp thoại chọn file khi click vào khu vực Upload
        const openFileDialog = (e) => {
            e.stopPropagation();
            fileInput.click();
        };
        uploadBox.addEventListener("click", openFileDialog);
        uploadBtn.addEventListener("click", openFileDialog);

        // Xử lý khi user chọn file xong
        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // 1. Chạy hiệu ứng Loading (Cũng phải dùng parentElement)
            const uploadStatus = parentElement.querySelector('#upload-status');
            if (uploadStatus) {
                uploadStatus.style.display = 'block';
                const steps = [
                    {text:'Đang đọc tài liệu...', pct:25, id:'#step-1'},
                    {text:'Đang xử lý...', pct:60, id:'#step-2'},
                    {text:'Đang lưu...', pct:90, id:'#step-3'},
                    {text:'Tài liệu đã sẵn sàng', pct:100, id:'#step-4'}
                ];
                
                const stBar = parentElement.querySelector('#st-bar');
                const stPct = parentElement.querySelector('#st-pct');
                const stText = parentElement.querySelector('#st-text');
                
                if(stBar) { stBar.style.width = '0%'; stBar.style.background = '#6366f1'; }
                if(stPct) { stPct.innerText = '0%'; stPct.style.color = '#6366f1'; }
                
                const svgCircle = `<svg class='icon' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><circle cx='12' cy='12' r='9'/></svg>`;
                const svgSpin = `<svg class='icon spin' style='color:#6366f1' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><path stroke-linecap='round' stroke-linejoin='round' d='M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z'/></svg>`;
                const svgCheck = `<svg class='icon' style='color:#10b981' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><path stroke-linecap='round' stroke-linejoin='round' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'/></svg>`;

                steps.forEach(s => {
                    const el = parentElement.querySelector(s.id);
                    if(el) {
                        el.style.color = '#64748b';
                        el.innerHTML = svgCircle + ' ' + s.text;
                    }
                });

                let currentStep = 0;
                function processStep() {
                    if(currentStep >= steps.length) return;
                    const step = steps[currentStep];
                    const el = parentElement.querySelector(step.id);
                    
                    if(el) {
                        el.innerHTML = svgSpin + ' ' + step.text;
                        el.style.color = '#0f172a';
                    }
                    if(stText) stText.innerText = step.text;
                    
                    setTimeout(() => {
                        if(el) {
                            el.innerHTML = svgCheck + ' ' + step.text;
                            el.style.color = '#10b981';
                        }
                        if(stBar) stBar.style.width = step.pct + '%';
                        if(stPct) stPct.innerText = step.pct + '%';
                        currentStep++;
                        
                        if(currentStep < steps.length) {
                            setTimeout(processStep, 800);
                        } else {
                            if(stBar) stBar.style.background = '#10b981';
                            if(stText) stText.innerText = 'Hoàn tất!';
                            if(stPct) stPct.style.color = '#10b981';
                        }
                    }, 1000);
                }
                setTimeout(processStep, 200);
            }

            // 2. Đọc file thành Base64 và gửi cho Python
            const reader = new FileReader();
            reader.onload = (event) => {
                setTriggerValue("chat_event", {
                    "action": "upload_file",
                    "file_name": file.name,
                    "file_type": file.type,
                    "file_data": event.target.result
                });
            };
            reader.readAsDataURL(file);
            
            fileInput.value = ""; 
        });
    }
}
"""

css_code = load_css_file()


@st.cache_resource
def get_main_component():
    """Khai báo component một lần để tránh cảnh báo overwrite khi rerun."""
    return st.components.v2.component(
        name="main_app_component",
        html=html_code,
        css=css_code,
        js=js_code
    )

# ==========================================================
# 5. KHỞI TẠO STREAMLIT COMPONENT & XỬ LÝ KẾT QUẢ TỪ UI
# ==========================================================
app_component_func = get_main_component()

app_result = app_component_func(height=1000)

# Chuẩn hóa payload từ component trigger để hỗ trợ cả dạng trực tiếp
# và dạng lồng trong key trigger (vd: {"chat_event": {...}}).
event_payload = None
if isinstance(app_result, dict):
    if "action" in app_result:
        event_payload = app_result
    elif "chat_event" in app_result and isinstance(app_result["chat_event"], dict):
        event_payload = app_result["chat_event"]
elif isinstance(app_result, str):
    try:
        parsed = json.loads(app_result)
        if isinstance(parsed, dict):
            if "action" in parsed:
                event_payload = parsed
            elif "chat_event" in parsed and isinstance(parsed["chat_event"], dict):
                event_payload = parsed["chat_event"]
    except ValueError:
        event_payload = None

# (Đã mở rộng phần này để xử lý cả tin nhắn và file)
if event_payload:
    action = event_payload.get("action")
    
    # Kịch bản 1: Người dùng chat
    if action == "send_message":
        rag = ensure_rag_service()
        if rag is None:
            st.stop()

        user_message = event_payload.get("text")
        if not user_message:
            st.stop()

        st.session_state.chat_messages.append({"role": "user", "text": user_message})
        
        with st.spinner("Đang suy luận..."):
            try:
                # Query RAG chain với Ollama
                result = rag.query(user_message, search_type="vector")
                
                if result["status"] == "success":
                    st.session_state.chat_messages.append({"role": "assistant", "text": result["answer"]})
                    
                    if result["sources"]:
                        st.write("### Tài liệu liên quan:")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Nguồn {i} - {source['source']} (Chunk {source['chunk']})"):
                                st.write(source["content"])
                else:
                    st.session_state.chat_messages.append({"role": "assistant", "text": result["answer"]})
                    st.error(result["answer"])
            except Exception as e:
                st.session_state.chat_messages.append({"role": "assistant", "text": f"Lỗi query: {str(e)}"})
                st.error(f"Lỗi query: {str(e)}")
                LOG.error(f"Chat error: {str(e)}")

            st.rerun()
        
    # Kịch bản 2: Người dùng upload file
    elif action == "upload_file":
        rag = ensure_rag_service()
        if rag is None:
            st.stop()

        file_name = event_payload.get("file_name")
        file_data_b64 = event_payload.get("file_data")
        
        with st.spinner(f"Đang xử lý {file_name}..."):
            try:
                # Cắt bỏ phần header của chuỗi Base64
                if "," in file_data_b64:
                    file_data_b64 = file_data_b64.split(",")[1]
                    
                file_bytes = base64.b64decode(file_data_b64)
                
                # Lưu tạm thời
                current_dir = os.path.dirname(os.path.abspath(__file__))
                temp_dir = os.path.join(current_dir, "temp_docs")
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_file_path = os.path.join(temp_dir, file_name)
                
                with open(temp_file_path, "wb") as f:
                    f.write(file_bytes)
                
                # Thêm documents vào RAG vectorstore
                result = rag.add_documents(temp_file_path)
                
                if result["status"] == "success":
                    st.success(f"✅ {result['message']}")
                    
                    # Show RAG status
                    status = rag.get_status()
                    st.info(f"📊 Vectorstore: {status['total_documents']} documents | LLM: {status['llm_model']}")
                else:
                    st.error(f"❌ {result['message']}")
                
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
            except Exception as e:
                st.error(f"Lỗi xử lý file: {str(e)}")
                LOG.error(f"Upload error: {str(e)}")

            st.rerun()

if st.session_state.chat_messages:
    st.markdown("### Hội thoại")
    for msg in st.session_state.chat_messages[-10:]:
        prefix = "Bạn" if msg["role"] == "user" else "AI"
        st.markdown(f"**{prefix}:** {msg['text']}")