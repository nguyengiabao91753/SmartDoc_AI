from langchain_community.llms import Ollama
from app.core.config import settings

def get_llm(temperature: float = 0.2, model: str | None = None):
    # langchain-community Ollama wrapper
    selected_model = model or settings.LLM_MODEL
    llm = Ollama(
        model=selected_model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=settings.LLM_NUM_CTX,
        num_predict=settings.LLM_NUM_PREDICT,
        num_batch=settings.LLM_NUM_BATCH,
        keep_alive=settings.LLM_KEEP_ALIVE,
    )
    return llm