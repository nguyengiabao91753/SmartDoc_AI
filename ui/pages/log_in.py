import streamlit as st
import base64
import os

st.set_page_config(page_title="Login", page_icon="🔐", layout="wide")

def get_base64_of_bin_file(bin_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, bin_file)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def load_css_file():
    # Tính toán đường dẫn trỏ ra ngoài thư mục css/style.css
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "../css/style.css")
    
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def login_page():
    # Sử dụng file .jpg theo như file đính kèm của bạn
    bg_base64 = get_base64_of_bin_file('login-bg.png')
    bg_image_css = f"background-image: url('data:image/jpeg;base64,{bg_base64}');" if bg_base64 else "background-color: #000;"

    # 1. GLOBAL CSS: Phải để ở đây để st.html ẩn UI của Streamlit
    global_css = """
    <style>
    [data-testid="stSidebar"] {display: none !important;}
    header[data-testid="stHeader"] {display: none !important; }
    footer {display: none !important; }
    [data-testid="stHeader"] { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { border: none !important; }
    </style>
    """
    st.html(global_css)

    # 2. Đọc file CSS bên ngoài và nối thêm ảnh nền động vào .login-container
    base_css = load_css_file()
    css_code = base_css + f"\n.login-container {{ {bg_image_css} }}"

    html_code = """
    <div class="login-container">
        <form class="login__form" onsubmit="event.preventDefault();">
            <h1 class="login__title">Login</h1>
            
            <div class="login__content">
                <div class="login__box">
                    <div class="login__box-input">
                        <input type="text" class="login__input" placeholder=" " required>
                        <label class="login__label">Username</label>
                    </div>
                </div>
                
                <div class="login__box">
                    <div class="login__box-input">
                        <input type="password" class="login__input" placeholder=" " required>
                        <label class="login__label">Password</label>
                    </div>
                </div>
            </div>

            <div class="login__check">
                <div class="login__check-group">
                    <input type="checkbox" class="login__check-input" id="remember">
                    <label for="remember" class="login__check-label">Remember me</label>
                </div>
            </div>

            <button type="submit" class="login__button">Login</button>

            <p class="login__register">
                Don't have an account? <a href="/register" target="_self">Register</a>
            </p>
        </form>
    </div>
    """

    js_code = """
    export default function(component) {
        const { setTriggerValue, parentElement } = component;
        
        const form = parentElement.querySelector(".login__form");
        form.addEventListener('submit', (e) => {
            const username = parentElement.querySelector('input[type="text"]').value;
            const password = parentElement.querySelector('input[type="password"]').value;
            const remember = parentElement.querySelector('#remember').checked;
            
            setTriggerValue("login_data", {
                "username": username,
                "password": password,
                "remember": remember
            });
        });
    }
    """

    login_component_func = st.components.v2.component(
        name="login_component",
        html=html_code,
        css=css_code,
        js=js_code
    )

    login_result = login_component_func(height=900)
    
    if login_result:
        st.write("Dữ liệu nhận từ JS:", login_result)

if __name__ == "__main__":
    login_page()