#!/usr/bin/env python
"""
Test script để kiểm tra RAG + Ollama integration
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core.config import settings
from app.core.logger import LOG
from app.services.rag_service import RAGService

def test_rag_service():
    """Test RAGService initialization và cấu hình"""
    print("\n" + "="*60)
    print("TEST RAG - OLLAMA INTEGRATION")
    print("="*60)
    
    try:
        # 1. Check config
        print("\n✓ Checking Configuration...")
        print(f"  LLM Model: {settings.LLM_MODEL}")
        print(f"  Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"  Chunk Size: {settings.CHUNK_SIZE}")
        print(f"  Top K: {settings.TOP_K}")
        
        # 2. Initialize RAGService
        print("\n✓ Initializing RAGService...")
        rag = RAGService()
        
        # 3. Check status
        print("\n✓ RAGService Status:")
        status = rag.get_status()
        print(f"  Vectorstore Ready: {status['vectorstore_ready']}")
        print(f"  RAG Chain Ready: {status['rag_chain_ready']}")
        print(f"  Total Documents: {status['total_documents']}")
        
        # 4. Test that Ollama is reachable
        print("\n✓ Testing Ollama Connection...")
        from langchain_community.llms import Ollama
        llm = Ollama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL)
        response = llm.invoke("Say 'Ollama is ready' briefly")
        print(f"  Ollama Response: {response[:100]}...")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nTo use in Streamlit:")
        print("  cd", Path(__file__).parent)
        print("  streamlit run ui/streamlit_app.py")
        print("\nOllama must be running:")
        print("  ollama serve")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        LOG.error(str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    test_rag_service()
