from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.documents import Document


@dataclass
class RAGQueryRequest:
    question: str
    search_type: str
    top_k: int
    document_id: int | None = None
    llm_model: str | None = None


@dataclass
class RAGEngineResult:
    answer: str
    source_documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
