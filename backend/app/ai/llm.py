from app.core.config import settings

try:
    from langchain_ollama import OllamaLLM  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OllamaLLM = None  # type: ignore



def get_llm(
    temperature: float = 0.2,
    model: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
):
    selected_model = model or settings.LLM_MODEL
    if OllamaLLM is None:
        raise ModuleNotFoundError(
            "langchain_ollama is not installed. Install dependencies or provide an LLM backend alternative. "
            "Run: pip install langchain-ollama"
        )

    llm = OllamaLLM(
        model=selected_model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=num_ctx if num_ctx is not None else settings.LLM_NUM_CTX,
        num_predict=num_predict if num_predict is not None else settings.LLM_NUM_PREDICT,
        keep_alive=settings.LLM_KEEP_ALIVE,
    )
    return llm

