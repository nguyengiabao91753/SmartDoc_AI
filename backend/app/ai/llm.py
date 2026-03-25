from langchain_ollama import OllamaLLM
from app.core.config import settings

def get_llm(temperature: float = 0.2, model: str | None = None):
    selected_model = model or settings.LLM_MODEL
    llm = OllamaLLM(
        model=selected_model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=settings.LLM_NUM_CTX,
        num_predict=settings.LLM_NUM_PREDICT,
        keep_alive=settings.LLM_KEEP_ALIVE,
    )
    return llm
