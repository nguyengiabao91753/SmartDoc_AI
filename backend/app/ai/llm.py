from app.core.config import settings

try:
    # Preferred provider package for newer LangChain Ollama integration.
    from langchain_ollama import OllamaLLM  # type: ignore
except ImportError:
    try:
        # Backward-compatible fallback when langchain-ollama is not installed.
        from langchain_community.llms import Ollama as OllamaLLM  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing Ollama integration package. Install one of: "
            "`langchain-ollama` (recommended) or `langchain-community`."
        ) from exc

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
