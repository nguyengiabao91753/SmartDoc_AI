#!/bin/bash
# Quick Start Guide for SmartDoc AI + Ollama

echo "========================================"
echo "SmartDoc AI - RAG + Ollama Quick Start"
echo "========================================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found!"
    echo "👉 Download from: https://ollama.ai"
    echo "   After installation, restart your terminal."
    exit 1
fi

echo "✅ Ollama found: $(ollama --version)"
echo ""

# Check if model exists
echo "📦 Checking for qwen2.5:7b model..."
if ollama list | grep -q "qwen2.5"; then
    echo "✅ Model found!"
else
    echo "⏳ Model not found. Pulling qwen2.5:7b..."
    echo "   This may take 5-10 minutes..."
    ollama pull qwen2.5:7b
fi

echo ""
echo "🚀 Starting Ollama service..."
echo "   (You can keep this running in a separate terminal)"
ollama serve &
OLLAMA_PID=$!

sleep 3

# Test connection
echo ""
echo "🧪 Testing connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running and ready!"
else
    echo "❌ Ollama connection failed"
    kill $OLLAMA_PID
    exit 1
fi

echo ""
echo "✅ Everything is ready!"
echo ""
echo "In another terminal, run:"
echo "  cd $(pwd)"
echo "  streamlit run ui/streamlit_app.py"
echo ""
echo "Then open: http://localhost:8501"
