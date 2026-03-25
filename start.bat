@echo off
REM SmartDoc AI - Quick Start for Windows

echo.
echo ========================================
echo SmartDoc AI - RAG + Ollama Quick Start
echo ========================================
echo.

REM Check if Ollama is in PATH
where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ollama not found!
    echo.
    echo To fix:
    echo 1. Download from: https://ollama.ai
    echo 2. Install OllamaSetup.exe
    echo 3. Restart your terminal
    echo.
    pause
    exit /b 1
)

echo [OK] Ollama found
ollama --version
echo.

REM Check Python virtual environment
if not exist ".venv" (
    echo [WARNING] Virtual environment not found
    echo Creating .venv...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install/update requirements
echo.
echo [INFO] Installing requirements...
pip install -q -r requirements.txt

echo.
echo [INFO] Checking for qwen2.5:7b model...
ollama list | find "qwen2.5" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Model not found
    echo.
    echo Pulling qwen2.5:7b (this may take 5-10 minutes)...
    ollama pull qwen2.5:7b
)

echo.
echo [OK] All checks passed!
echo.
echo Next steps:
echo.
echo 1. Open a NEW terminal and run (from project root):
echo    ollama serve
echo.
echo 2. In ANOTHER terminal, run:
echo    streamlit run ui/streamlit_app.py
echo.
echo 3. Open browser at: http://localhost:8501
echo.
echo.
echo Optional: Run test first
echo    python test_rag_integration.py
echo.
pause
