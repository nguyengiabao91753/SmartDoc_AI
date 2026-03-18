import streamlit as st
import base64
import os

st.set_page_config(page_title="Register", page_icon="🔐", layout="wide")

def get_base64_of_bin_file(bin_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, bin_file)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def load_css_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "../css/style.css")
    
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def register_page():
    bg_base64 = get_base64_of_bin_file('login-bg.png')
    bg_image_css = f"background-image: url('data:image/jpeg;base64,{bg_base64}');" if bg_base64 else "background-color: #000;"

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

    # Đọc CSS và chèn động background
    base_css = load_css_file()
    css_code = base_css + f"\n.login-container {{ {bg_image_css} }}"

    html_code = """
    <div class="login-container">
        <form class="login__form" onsubmit="event.preventDefault();">
            <h1 class="login__title">Register</h1>
            <div class="login__content">
                <div class="login__box">
                    <div class="login__box-input">
                        <input type="text" class="login__input" placeholder=" " required>
                        <label class="login__label">Username</label>
                    </div>
                </div>
                <div class="login__box">
                    <div class="login__box-input">
                        <input type="email" class="login__input" placeholder=" " required>
                        <label class="login__label">Email</label>
                    </div>
                </div>
                <div class="login__box">
                    <div class="login__box-input">
                        <input type="password" class="login__input" placeholder=" " required>
                        <label class="login__label">Password</label>
                    </div>
                </div>
                <div class="login__box">
                    <div class="login__box-input">
                        <input type="password" class="login__input" placeholder=" " required>
                        <label class="login__label">Confirm Password</label>
                    </div>
                </div>
            </div>
            <button type="submit" class="login__button">Register</button>
            <p class="login__register">
                Already have an account? <a href="/log_in" target="_self">Login</a>
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
            const email = parentElement.querySelector('input[type="email"]').value;
            const password_inputs = parentElement.querySelectorAll('input[type="password"]');
            const password = password_inputs[0].value;
            const confirm_password = password_inputs[1].value;
            
            setTriggerValue("register_data", {
                "username": username,
                "email": email,
                "password": password,
                "confirm_password": confirm_password
            });
        });
    }
    """

    register_component_func = st.components.v2.component(
        name="register_component",
        html=html_code,
        css=css_code,
        js=js_code
    )

    register_result = register_component_func(height=950)
    
    if register_result:
        st.write("Dữ liệu nhận từ JS:", register_result)

if __name__ == "__main__":
    register_page()