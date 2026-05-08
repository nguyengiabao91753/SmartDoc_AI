from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm
from app.core.logger import LOG

from .evaluator import SelfRAGEvaluation
from .utils import normalize_text, unique_preserve_order


class SelfRAGRewriter:
    ALLOWED_ERROR_TYPES = {"retrieval_issue", "query_issue"}

    def rewrite(
        self,
        *,
        question: str,
        evaluation: SelfRAGEvaluation,
        source_documents: List[Document],
        llm_model: str | None = None,
        previous_queries: List[str] | None = None,
    ) -> str:
        previous_queries = previous_queries or []
        if evaluation.error_type not in self.ALLOWED_ERROR_TYPES:
            return question

        rewrite_goal = self._rewrite_goal(evaluation.error_type)
        prompt = (
    "Bạn đang viết lại truy vấn tìm kiếm cho bộ truy xuất Self-RAG.\n"
    "Chỉ trả về đúng một truy vấn tìm kiếm đã viết lại dưới dạng văn bản thuần tuý.\n"
    "Truy vấn phải dễ truy xuất hơn, cụ thể hơn và bám sát nhu cầu thực sự của người dùng.\n"
    f"Chiến lược viết lại: {rewrite_goal}\n"
    "Không giải thích bất cứ điều gì.\n\n"
    f"Câu hỏi gốc:\n{question}\n\n"
    f"Loại lỗi:\n{evaluation.error_type}\n\n"
    f"Phản hồi từ bộ đánh giá:\n{evaluation.rationale}\n\n"
    f"Các chủ đề còn thiếu:\n{', '.join(evaluation.missing_topics) or '(không có)'}\n\n"
    f"Các truy vấn trước đó cần tránh lặp lại:\n{'; '.join(previous_queries) or '(không có)'}\n\n"
    f"Gợi ý từ bằng chứng hiện tại:\n{self._build_context_hint(source_documents)}\n\n"
    "Truy vấn đã viết lại:"
)

        try:
            llm = get_llm(temperature=0.0, model=llm_model)
            candidate = self._clean_query(llm.invoke(prompt))
        except Exception as exc:
            LOG.warning("[SelfRAGRewriter] Query rewrite failed: %s", exc)
            candidate = ""

        if not candidate:
            candidate = self._fallback_rewrite(question=question, evaluation=evaluation)

        if normalize_text(candidate) in {normalize_text(question), *[normalize_text(item) for item in previous_queries]}:
            fallback = self._fallback_rewrite(question=question, evaluation=evaluation)
            if normalize_text(fallback) not in {normalize_text(question), *[normalize_text(item) for item in previous_queries]}:
                return fallback
        return candidate or question

    def _clean_query(self, raw: str) -> str:
        cleaned = " ".join((raw or "").replace("Query:", "").replace("Rewritten query:", "").split())
        cleaned = cleaned.strip("`\"' ")
        if len(cleaned) > 280:
            cleaned = cleaned[:280].rstrip(" ,;")
        return cleaned

    def _fallback_rewrite(self, *, question: str, evaluation: SelfRAGEvaluation) -> str:
        base_question = question.strip().rstrip("?.!")
        additions = unique_preserve_order([*evaluation.missing_topics, *evaluation.follow_up_queries])
        if evaluation.error_type == "retrieval_issue":
            f"{base_question} nội dung chính từ khoá chính".strip()
        if evaluation.error_type == "query_issue" and additions:
            return f"{base_question} {' '.join(additions[:2])}".strip()
        if additions:
            return f"{base_question} {' '.join(additions[:2])}".strip()
        return f"{base_question} chi tiết cụ thể".strip()

    def _build_context_hint(self, source_documents: List[Document]) -> str:
        if not source_documents:
            return "(none)"
        hints: List[str] = []
        for doc in source_documents[:2]:
            metadata = doc.metadata or {}
            source_name = metadata.get("source", "unknown")
            snippet = " ".join(doc.page_content.split())[:180]
            hints.append(f"{source_name}: {snippet}")
        return "\n".join(hints)

    def _rewrite_goal(self, error_type: str) -> str:
        if error_type == "retrieval_issue":
            return "mở rộng truy vấn, làm rõ hơn và tập trung vào các khái niệm cốt lõi"

        if error_type == "query_issue":
            return "làm truy vấn cụ thể hơn, bớt mơ hồ và tập trung vào khía cạnh còn thiếu"
        return "cải thiện chất lượng truy xuất"
