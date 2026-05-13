from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm
from app.core.logger import LOG
from app.rag.models import RAGQueryRequest

from .utils import parse_json_array, unique_preserve_order


@dataclass
class SelfRAGPlan:
    question: str
    search_type: str
    top_k: int
    document_id: int | None = None
    document_ids: List[int] | None = None
    session_id: int | None = None
    llm_model: str | None = None
    complexity: str = "standard"
    max_attempts: int = 3
    max_rewrites: int = 1
    max_multi_hops: int = 1
    confidence_threshold: float = 0.72
    min_confidence_threshold: float = 0.50
    retrieval_top_k: int = 4
    hop_top_k: int = 2
    merge_top_k: int = 6
    stagnation_tolerance: float = 0.02


class SelfRAGPlanner:
    MAX_FOLLOW_UP_QUERIES = 1

    def plan(self, request: RAGQueryRequest) -> SelfRAGPlan:
        question = (request.question or "").strip()
        base_top_k = max(int(request.top_k), 2)
        complexity = self._estimate_complexity(question)

        if complexity == "complex":
            max_attempts = 4
            max_multi_hops = 2
            confidence_threshold = 0.70
            min_confidence_threshold = 0.50
        else:
            max_attempts = 3
            max_multi_hops = 1
            confidence_threshold = 0.72
            min_confidence_threshold = 0.52

        retrieval_top_k = max(base_top_k, min(base_top_k + 2, 6))
        hop_top_k = max(2, min(base_top_k, 3))
        merge_top_k = max(retrieval_top_k, base_top_k + hop_top_k)

        return SelfRAGPlan(
            question=question,
            search_type=request.search_type,
            top_k=base_top_k,
            document_id=request.document_id,
            document_ids=request.document_ids,
            session_id=request.session_id,
            llm_model=request.llm_model,
            complexity=complexity,
            max_attempts=max_attempts,
            max_rewrites=1,
            max_multi_hops=max_multi_hops,
            confidence_threshold=confidence_threshold,
            min_confidence_threshold=min_confidence_threshold,
            retrieval_top_k=retrieval_top_k,
            hop_top_k=hop_top_k,
            merge_top_k=merge_top_k,
            stagnation_tolerance=0.02,
        )

    def plan_follow_up_queries(
        self,
        *,
        question: str,
        draft_answer: str,
        evaluation_reason: str,
        missing_topics: List[str] | None = None,
        suggested_queries: List[str] | None = None,
        source_documents: List[Document] | None = None,
        llm_model: str | None = None,
        limit: int | None = None,
    ) -> List[str]:
        query_limit = max(1, min(limit or self.MAX_FOLLOW_UP_QUERIES, self.MAX_FOLLOW_UP_QUERIES))
        fallback_queries = self._build_follow_up_fallbacks(
            question=question,
            draft_answer=draft_answer,
            missing_topics=missing_topics or [],
            suggested_queries=suggested_queries or [],
            limit=query_limit,
        )

        context_excerpt = self._build_context_excerpt(source_documents or [])
        prompt = (
    "Bạn đang lên kế hoạch cho bước truy xuất tiếp theo (next hop) trong pipeline Self-RAG.\n"
    "Câu trả lời hiện tại mới chỉ là câu trả lời một phần.\n"
    "Trả về đúng một mảng JSON chứa 1 truy vấn tìm kiếm ngắn cho bước tiếp theo, không thêm nội dung nào khác.\n"
    "Truy vấn tiếp theo phải dựa trên những gì câu trả lời đã trình bày và những gì còn thiếu.\n"
    "Không phân rã thành nhiều truy vấn song song — chỉ tạo một truy vấn tuần tự duy nhất.\n"
    "Không lặp lại nguyên văn câu hỏi gốc trừ khi không có lựa chọn nào tốt hơn.\n\n"
    f"Câu hỏi gốc:\n{question}\n\n"
    f"Câu trả lời một phần hiện tại:\n{draft_answer}\n\n"
    f"Phản hồi từ bộ đánh giá:\n{evaluation_reason}\n\n"
    f"Các chủ đề còn thiếu đã biết:\n{', '.join(missing_topics or []) or '(không có)'}\n\n"
    f"Hướng gợi ý:\n{'; '.join(suggested_queries or []) or '(không có)'}\n\n"
    f"Bằng chứng hiện tại:\n{context_excerpt}\n\n"
    'Chỉ trả về mảng JSON, ví dụ: ["truy vấn tiếp theo"]'
)

        try:
            llm = get_llm(temperature=0.0, model=llm_model)
            raw = llm.invoke(prompt)
            planned = [
                str(item).strip()
                for item in parse_json_array(raw, fallback=fallback_queries)
                if str(item).strip()
            ]
        except Exception as exc:
            LOG.warning("[SelfRAGPlanner] Follow-up planning failed: %s", exc)
            planned = fallback_queries

        merged = unique_preserve_order([*planned, *fallback_queries])
        return merged[:query_limit]

    def _estimate_complexity(self, question: str) -> str:
        lowered = (question or "").lower()
        complex_markers = (
    "tại sao",
    "vì sao",
    "như thế nào",
    "so sánh",
    "quan hệ",
    "liên hệ",
    "khác nhau",
    "giống nhau",
    "ảnh hưởng",
    "nguyên nhân",
    "hệ quả",
    "trước",
    "sau",
    "between",
    "compare",
    "relationship",
    "reason",
    "cause",
    "impact",
    "multi",
        )
        if any(marker in lowered for marker in complex_markers):
            return "complex"
        if lowered.count("?") > 1 or lowered.count(" va ") >= 2:
            return "complex"
        return "standard"

    def _build_context_excerpt(self, source_documents: List[Document]) -> str:
        if not source_documents:
            return "(chưa có bằng chứng nào được truy xuất)"

        items: List[str] = []
        for index, doc in enumerate(source_documents[:4], start=1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source", "unknown")
            snippet = " ".join(doc.page_content.split())[:240]
            items.append(f"[{index}] {source_name}: {snippet}")
        return "\n".join(items)

    def _build_follow_up_fallbacks(
        self,
        *,
        question: str,
        draft_answer: str,
        missing_topics: List[str],
        suggested_queries: List[str],
        limit: int,
    ) -> List[str]:
        fallbacks: List[str] = []

        for query in suggested_queries:
            cleaned = str(query).strip()
            if cleaned:
                fallbacks.append(cleaned)

        base_question = question.strip().rstrip("?.!")
        answer_tail = self._extract_answer_tail(draft_answer)
        for topic in missing_topics:
            cleaned = str(topic).strip(" -")
            if cleaned:
                if answer_tail:
                    fallbacks.append(f"{answer_tail} {cleaned}".strip())
                fallbacks.append(f"{base_question} {cleaned}".strip())

        if not fallbacks and base_question:
            if answer_tail:
                fallbacks.append(f"{answer_tail} chi tiết bổ sung")

            fallbacks.append(f"{base_question} chi tiết bổ sung")

        return unique_preserve_order(fallbacks)[:limit]

    def _extract_answer_tail(self, draft_answer: str) -> str:
        cleaned = " ".join((draft_answer or "").split()).strip()
        if not cleaned:
            return ""
        if len(cleaned) <= 160:
            return cleaned
        return cleaned[-160:].strip(" ,;")
