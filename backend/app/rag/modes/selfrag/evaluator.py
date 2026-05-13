from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.ai.llm import get_llm
from app.core.logger import LOG

from .utils import clamp, parse_json_object, tokenize_text, unique_preserve_order


@dataclass
class SelfRAGEvaluation:
    score: float
    confidence: float
    decision: str
    error_type: str
    rationale: str
    missing_topics: List[str] = field(default_factory=list)
    follow_up_queries: List[str] = field(default_factory=list)
    retrieval_quality: float = 0.0
    relevance_score: float = 0.0
    grounding_score: float = 0.0
    evidence_coverage: float = 0.0
    used_llm_review: bool = False
    policy_reason: str = ""


class SelfRAGEvaluator:
    MISSING_PATTERNS = (
        "khong tim thay",
        "khong du thong tin",
        "khong de cap",
        "not enough information",
        "not mentioned",
        "insufficient information",
    )
    EARLY_ACCEPT_MARGIN = 0.08
    EARLY_REJECT_MARGIN = 0.12

    def assess_retrieval(
        self,
        *,
        question: str,
        source_documents: List[Document],
    ) -> SelfRAGEvaluation:
        metrics = self._compute_metrics(question=question, answer="", source_documents=source_documents)
        if not source_documents:
            return SelfRAGEvaluation(
                score=0.05,
                confidence=0.05,
                decision="rewrite",
                error_type="retrieval_issue",
                rationale="No supporting evidence was retrieved.",
                retrieval_quality=0.0,
                relevance_score=0.0,
                grounding_score=0.0,
                evidence_coverage=0.0,
                policy_reason="retrieval_empty",
            )

        if metrics["retrieval_quality"] < 0.12:
            return SelfRAGEvaluation(
                score=metrics["retrieval_quality"],
                confidence=metrics["retrieval_quality"],
                decision="rewrite",
                error_type="retrieval_issue",
                rationale=(
                    "Retrieved documents are too weak to support answer generation "
                    f"(retrieval_quality={metrics['retrieval_quality']:.2f})."
                ),
                retrieval_quality=metrics["retrieval_quality"],
                relevance_score=metrics["relevance_score"],
                grounding_score=0.0,
                evidence_coverage=metrics["evidence_coverage"],
                policy_reason="retrieval_quality_too_low",
            )

        if metrics["relevance_score"] < 0.18:
            return SelfRAGEvaluation(
                score=metrics["retrieval_quality"],
                confidence=metrics["retrieval_quality"],
                decision="rewrite",
                error_type="query_issue",
                rationale=(
                    "Retrieved documents exist but do not align well with the user question "
                    f"(relevance={metrics['relevance_score']:.2f})."
                ),
                retrieval_quality=metrics["retrieval_quality"],
                relevance_score=metrics["relevance_score"],
                grounding_score=0.0,
                evidence_coverage=metrics["evidence_coverage"],
                policy_reason="question_context_mismatch",
            )

        return SelfRAGEvaluation(
            score=metrics["retrieval_quality"],
            confidence=metrics["retrieval_quality"],
            decision="proceed",
            error_type="none",
            rationale=(
                "Retrieved evidence is strong enough to proceed with answer generation "
                f"(retrieval_quality={metrics['retrieval_quality']:.2f})."
            ),
            retrieval_quality=metrics["retrieval_quality"],
            relevance_score=metrics["relevance_score"],
            grounding_score=0.0,
            evidence_coverage=metrics["evidence_coverage"],
            policy_reason="retrieval_ready",
        )

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        source_documents: List[Document],
        llm_model: str | None = None,
        threshold: float = 0.72,
    ) -> SelfRAGEvaluation:
        if not answer.strip():
            return SelfRAGEvaluation(
                score=0.05,
                confidence=0.05,
                decision="rewrite",
                error_type="retrieval_issue",
                rationale="The answer is empty.",
                policy_reason="empty_answer",
            )

        if not source_documents:
            return SelfRAGEvaluation(
                score=0.10,
                confidence=0.10,
                decision="rewrite",
                error_type="retrieval_issue",
                rationale="No supporting evidence was retrieved.",
                policy_reason="no_source_documents",
            )

        heuristic = self._heuristic_evaluation(
            question=question,
            answer=answer,
            source_documents=source_documents,
            threshold=threshold,
        )
        heuristic.decision, heuristic.policy_reason = self._apply_decision_policy(
            score=heuristic.score,
            confidence=heuristic.confidence,
            threshold=threshold,
            error_type=heuristic.error_type,
            retrieval_quality=heuristic.retrieval_quality,
            relevance_score=heuristic.relevance_score,
            grounding_score=heuristic.grounding_score,
            evidence_coverage=heuristic.evidence_coverage,
            missing_topics=heuristic.missing_topics,
            follow_up_queries=heuristic.follow_up_queries,
        )

        if (
            heuristic.error_type == "none"
            and heuristic.score >= min(1.0, threshold + self.EARLY_ACCEPT_MARGIN)
        ):
            heuristic.rationale = (
                f"{heuristic.rationale} Early-accept by heuristic policy "
                f"(threshold={threshold:.2f})."
            )
            return heuristic

        if (
            heuristic.error_type in {"retrieval_issue", "query_issue"}
            and heuristic.score <= max(0.0, threshold - self.EARLY_REJECT_MARGIN)
        ):
            heuristic.rationale = (
                f"{heuristic.rationale} Early-stop before LLM evaluator because "
                f"{heuristic.error_type} is already clear."
            )
            return heuristic

        llm_review = self._llm_evaluation(
            question=question,
            answer=answer,
            source_documents=source_documents,
            llm_model=llm_model,
        )
        if llm_review is None:
            return heuristic

        llm_score = clamp(llm_review.get("score", heuristic.score))
        llm_confidence = clamp(llm_review.get("confidence", llm_score))
        final_score = clamp(0.65 * llm_score + 0.35 * heuristic.score)
        final_confidence = clamp(0.50 * llm_confidence + 0.30 * final_score + 0.20 * heuristic.confidence)

        error_type = self._normalize_error_type(llm_review.get("error_type")) or heuristic.error_type
        reason = str(llm_review.get("reason") or heuristic.rationale).strip()
        missing_topics = unique_preserve_order(
            [str(item).strip() for item in llm_review.get("missing_topics", []) if str(item).strip()]
        )
        follow_up_queries = unique_preserve_order(
            [str(item).strip() for item in llm_review.get("follow_up_queries", []) if str(item).strip()]
        )

        if not missing_topics:
            missing_topics = heuristic.missing_topics
        if not follow_up_queries:
            follow_up_queries = heuristic.follow_up_queries

        decision, policy_reason = self._apply_decision_policy(
            score=final_score,
            confidence=final_confidence,
            threshold=threshold,
            error_type=error_type,
            retrieval_quality=heuristic.retrieval_quality,
            relevance_score=heuristic.relevance_score,
            grounding_score=heuristic.grounding_score,
            evidence_coverage=heuristic.evidence_coverage,
            missing_topics=missing_topics,
            follow_up_queries=follow_up_queries,
            suggested_decision=self._normalize_decision(llm_review.get("decision"), error_type=error_type),
        )

        return SelfRAGEvaluation(
            score=final_score,
            confidence=final_confidence,
            decision=decision,
            error_type=error_type,
            rationale=reason,
            missing_topics=missing_topics,
            follow_up_queries=follow_up_queries,
            retrieval_quality=heuristic.retrieval_quality,
            relevance_score=heuristic.relevance_score,
            grounding_score=heuristic.grounding_score,
            evidence_coverage=heuristic.evidence_coverage,
            used_llm_review=True,
            policy_reason=policy_reason,
        )

    def _llm_evaluation(
        self,
        *,
        question: str,
        answer: str,
        source_documents: List[Document],
        llm_model: str | None,
    ) -> dict | None:
        prompt = (
    "Bạn đang đánh giá câu trả lời của hệ thống Self-RAG.\n"
    "Chỉ dựa vào bằng chứng được cung cấp để nhận xét câu trả lời.\n"
    "Trả về JSON thuần tuý theo đúng schema sau, không thêm bất kỳ nội dung nào khác:\n"
    "{\n"
    '  "score": 0.0,\n'
    '  "confidence": 0.0,\n'
    '  "error_type": "none|retrieval_issue|query_issue|missing_info|reasoning_issue",\n'
    '  "decision": "accept|rewrite|multi_hop|regenerate",\n'
    '  "reason": "giải thích ngắn gọn",\n'
    '  "missing_topics": ["chủ đề còn thiếu"],\n'
    '  "follow_up_queries": ["truy vấn tiếp theo"]\n'
    "}\n\n"
    "Hướng dẫn phân loại lỗi (bắt buộc tuân thủ):\n"
    "- retrieval_issue : bằng chứng thu được quá yếu hoặc không liên quan.\n"
    "- query_issue     : truy vấn quá rộng, mơ hồ hoặc chưa đủ tập trung.\n"
    "- missing_info    : cần thêm bằng chứng từ phần khác của tài liệu.\n"
    "- reasoning_issue : bằng chứng đủ nhưng câu trả lời chưa khai thác tốt.\n\n"
    f"Câu hỏi:\n{question}\n\n"
    f"Câu trả lời:\n{answer}\n\n"
    f"Bằng chứng:\n{self._build_context_excerpt(source_documents)}"
)
        try:
            llm = get_llm(temperature=0.0, model=llm_model)
            raw = llm.invoke(prompt)
            parsed = parse_json_object(raw)
            return parsed or None
        except Exception as exc:
            LOG.warning("[SelfRAGEvaluator] LLM evaluation failed: %s", exc)
            return None

    def _heuristic_evaluation(
        self,
        *,
        question: str,
        answer: str,
        source_documents: List[Document],
        threshold: float,
    ) -> SelfRAGEvaluation:
        metrics = self._compute_metrics(
            question=question,
            answer=answer,
            source_documents=source_documents,
        )
        relevance = metrics["relevance_score"]
        grounding = metrics["grounding_score"]
        evidence_coverage = metrics["evidence_coverage"]
        retrieval_quality = metrics["retrieval_quality"]
        question_tokens = metrics["question_tokens"]
        context_tokens = metrics["context_tokens"]
        missing_signal = any(pattern in answer.lower() for pattern in self.MISSING_PATTERNS)

        heuristic_score = clamp(
            0.40 * grounding + 0.25 * relevance + 0.20 * evidence_coverage + 0.15 * retrieval_quality
        )
        heuristic_confidence = clamp(0.50 * heuristic_score + 0.30 * grounding + 0.20 * evidence_coverage)

        missing_topics = list(question_tokens - context_tokens)[:2]
        follow_up_queries = [f"{question.strip().rstrip('?.!')} {topic}".strip() for topic in missing_topics]

        if retrieval_quality < 0.12:
            error_type = "retrieval_issue"
        elif source_documents and relevance < 0.18:
            error_type = "query_issue"
        elif missing_signal or evidence_coverage < 0.35:
            error_type = "missing_info"
        elif grounding < 0.22:
            error_type = "reasoning_issue"
        else:
            error_type = "none"

        if heuristic_score >= threshold:
            decision = "accept"
            error_type = "none"
        else:
            decision = self._decision_from_error_type(error_type)

        reason = (
            f"Heuristic review: relevance={relevance:.2f}, grounding={grounding:.2f}, "
            f"evidence={evidence_coverage:.2f}, retrieval={retrieval_quality:.2f}."
        )

        return SelfRAGEvaluation(
            score=heuristic_score,
            confidence=heuristic_confidence,
            decision=decision,
            error_type=error_type,
            rationale=reason,
            missing_topics=unique_preserve_order(missing_topics),
            follow_up_queries=unique_preserve_order(follow_up_queries),
            retrieval_quality=retrieval_quality,
            relevance_score=relevance,
            grounding_score=grounding,
            evidence_coverage=evidence_coverage,
            policy_reason="heuristic_only",
        )

    def _compute_metrics(
        self,
        *,
        question: str,
        answer: str,
        source_documents: List[Document],
    ) -> dict:
        context_text = "\n".join(doc.page_content for doc in source_documents)
        context_tokens = set(tokenize_text(context_text))
        question_tokens = set(tokenize_text(question))
        answer_tokens = set(tokenize_text(answer))

        relevance = self._overlap_ratio(question_tokens, context_tokens)
        grounding = self._overlap_ratio(answer_tokens, context_tokens) if answer_tokens else 0.0
        evidence_coverage = clamp(len(source_documents) / 4.0)

        support_values = []
        for doc in source_documents:
            metadata = doc.metadata or {}
            try:
                support_values.append(float(metadata.get("selfrag_support_count", 1.0)))
            except (TypeError, ValueError):
                support_values.append(1.0)
        average_support = sum(support_values) / len(support_values) if support_values else 0.0
        support_density = clamp(average_support / 2.0)
        retrieval_quality = clamp(0.55 * relevance + 0.25 * evidence_coverage + 0.20 * support_density)

        return {
            "question_tokens": question_tokens,
            "context_tokens": context_tokens,
            "answer_tokens": answer_tokens,
            "relevance_score": relevance,
            "grounding_score": grounding,
            "evidence_coverage": evidence_coverage,
            "retrieval_quality": retrieval_quality,
        }

    def _build_context_excerpt(self, source_documents: List[Document]) -> str:
        items: List[str] = []
        for index, doc in enumerate(source_documents[:4], start=1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source", "unknown")
            snippet = " ".join(doc.page_content.split())[:260]
            items.append(f"[{index}] {source_name}: {snippet}")
        return "\n".join(items)

    def _overlap_ratio(self, lhs: set[str], rhs: set[str]) -> float:
        if not lhs:
            return 0.0
        return len(lhs & rhs) / len(lhs)

    def _normalize_error_type(self, error_type: str | None) -> str | None:
        normalized = str(error_type or "").strip().lower()
        if normalized in {"none", "retrieval_issue", "query_issue", "missing_info", "reasoning_issue"}:
            return normalized
        return None

    def _normalize_decision(self, decision: str | None, *, error_type: str) -> str | None:
        normalized = str(decision or "").strip().lower()
        if normalized in {"accept", "rewrite", "multi_hop", "regenerate", "proceed"}:
            return normalized
        return self._decision_from_error_type(error_type)

    def _decision_from_error_type(self, error_type: str) -> str:
        if error_type in {"retrieval_issue", "query_issue"}:
            return "rewrite"
        if error_type == "missing_info":
            return "multi_hop"
        if error_type == "reasoning_issue":
            return "regenerate"
        return "accept"

    def _apply_decision_policy(
        self,
        *,
        score: float,
        confidence: float,
        threshold: float,
        error_type: str,
        retrieval_quality: float,
        relevance_score: float,
        grounding_score: float,
        evidence_coverage: float,
        missing_topics: List[str],
        follow_up_queries: List[str],
        suggested_decision: str | None = None,
    ) -> tuple[str, str]:
        if score >= threshold or confidence >= min(1.0, threshold + 0.03):
            return "accept", "threshold_reached"

        if error_type in {"retrieval_issue", "query_issue"}:
            return "rewrite", f"rewrite_due_to_{error_type}"

        if error_type == "missing_info":
            if missing_topics or follow_up_queries:
                return "multi_hop", "missing_info_requires_new_evidence"
            if retrieval_quality < 0.18 or relevance_score < 0.18:
                return "rewrite", "missing_info_but_retrieval_weak"
            return "regenerate", "missing_info_without_clear_followup"

        if error_type == "reasoning_issue":
            if grounding_score >= 0.22 and retrieval_quality >= 0.20:
                return "regenerate", "reasoning_issue_with_sufficient_evidence"
            if missing_topics or follow_up_queries:
                return "multi_hop", "reasoning_issue_but_missing_topics_detected"
            return "rewrite", "reasoning_issue_but_retrieval_insufficient"

        if suggested_decision in {"rewrite", "multi_hop", "regenerate"}:
            return suggested_decision, f"llm_suggested_{suggested_decision}"

        if evidence_coverage >= 0.45 and grounding_score >= 0.28 and score >= max(0.0, threshold - 0.05):
            return "accept", "near_threshold_with_grounded_answer"

        return "regenerate", "default_regenerate_policy"
