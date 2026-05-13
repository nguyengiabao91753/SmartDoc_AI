from __future__ import annotations

from typing import Dict, List

from app.core.logger import LOG
from app.rag.base import BaseRAGModeEngine
from app.rag.models import RAGEngineResult, RAGQueryRequest

from .evaluator import SelfRAGEvaluation, SelfRAGEvaluator
from .planner import SelfRAGPlan, SelfRAGPlanner
from .responder import SelfRAGResponder
from .retriever import SelfRAGRetriever
from .rewriter import SelfRAGRewriter
from .utils import normalize_text


class SelfRAGEngine(BaseRAGModeEngine):
    mode = "selfrag"
    display_name = "Self-RAG"

    def __init__(self, store, embedding_service):
        super().__init__(store=store, embedding_service=embedding_service)
        self.planner = SelfRAGPlanner()
        self.retriever = SelfRAGRetriever(store=store, embedding_service=embedding_service)
        self.responder = SelfRAGResponder()
        self.evaluator = SelfRAGEvaluator()
        self.rewriter = SelfRAGRewriter()

    def query(self, request: RAGQueryRequest) -> RAGEngineResult:
        plan = self.planner.plan(request)
        try:
            return self._run_selfrag(plan)
        except Exception as exc:
            LOG.exception("[SelfRAGEngine] Self-RAG orchestration failed: %s", exc)
            fallback_docs = self.retriever.retrieve(plan=plan, query=plan.question, top_k=plan.retrieval_top_k)
            fallback_answer = self.responder.answer(
                question=plan.question,
                source_documents=fallback_docs,
                llm_model=plan.llm_model,
                active_query=plan.question,
                attempt=1,
            )
            fallback_confidence = 0.35 if fallback_docs else 0.10
            return RAGEngineResult(
                answer=self._append_confidence_footer(fallback_answer, fallback_confidence, accepted=False),
                source_documents=self.retriever.select_top_documents(
                    fallback_docs,
                    limit=max(plan.top_k, min(plan.merge_top_k, plan.top_k + 2)),
                ),
                metadata={
                    "mode": self.mode,
                    "accepted": False,
                    "attempts": 1,
                    "final_query": plan.question,
                    "confidence_score": fallback_confidence,
                    "evaluation_score": fallback_confidence,
                    "trace": [
                        {
                            "attempt": 1,
                            "query": plan.question,
                            "doc_count": len(fallback_docs),
                            "draft_answer": fallback_answer,
                            "score": fallback_confidence,
                            "confidence": fallback_confidence,
                            "decision": "fallback",
                            "reason": "Self-RAG fallback path was used after orchestration failure.",
                            "next_action": "fallback",
                        }
                    ],
                },
            )

    def _run_selfrag(self, plan: SelfRAGPlan) -> RAGEngineResult:
        active_query = plan.question
        query_history = [active_query]
        seen_queries = {normalize_text(active_query)}
        current_docs = []
        score_history: List[float] = []
        critique = None
        rewrites_used = 0
        multi_hops_used = 0
        accepted = False
        last_answer = ""
        last_evaluation = SelfRAGEvaluation(
            score=0.10,
            confidence=0.10,
            decision="rewrite",
            error_type="retrieval_issue",
            rationale="No attempt executed yet.",
            policy_reason="not_started",
        )
        trace: List[Dict[str, object]] = []
        needs_retrieval = True

        for attempt in range(1, plan.max_attempts + 1):
            effective_threshold = self._effective_threshold(plan, attempt)
            attempt_trace: Dict[str, object] = {
                "attempt": attempt,
                "query": active_query,
                "effective_threshold": round(effective_threshold, 4),
            }

            if needs_retrieval:
                retrieved_docs = self.retriever.retrieve(
                    plan=plan,
                    query=active_query,
                    top_k=plan.retrieval_top_k,
                )
                current_docs = self.retriever.merge_documents(
                    current_docs,
                    retrieved_docs,
                    limit=plan.merge_top_k,
                )
                needs_retrieval = False

                retrieval_eval = self.evaluator.assess_retrieval(
                    question=plan.question,
                    source_documents=current_docs,
                )
                attempt_trace.update(
                    {
                        "doc_count": len(current_docs),
                        "retrieval_quality": round(retrieval_eval.retrieval_quality, 4),
                        "relevance_score": round(retrieval_eval.relevance_score, 4),
                        "retrieval_decision": retrieval_eval.decision,
                        "retrieval_error_type": retrieval_eval.error_type,
                        "retrieval_reason": retrieval_eval.rationale,
                        "retrieval_policy": retrieval_eval.policy_reason,
                    }
                )

                if (
                    retrieval_eval.decision == "rewrite"
                    and retrieval_eval.error_type in {"retrieval_issue", "query_issue"}
                    and rewrites_used < plan.max_rewrites
                ):
                    rewritten_query = self.rewriter.rewrite(
                        question=plan.question,
                        evaluation=retrieval_eval,
                        source_documents=current_docs,
                        llm_model=plan.llm_model,
                        previous_queries=query_history,
                    )
                    normalized_rewrite = normalize_text(rewritten_query)
                    if normalized_rewrite and normalized_rewrite not in seen_queries:
                        rewrites_used += 1
                        active_query = rewritten_query
                        query_history.append(rewritten_query)
                        seen_queries.add(normalized_rewrite)
                        current_docs = []
                        critique = retrieval_eval.rationale
                        needs_retrieval = True
                        last_evaluation = retrieval_eval
                        attempt_trace["next_action"] = "rewrite_before_generate"
                        attempt_trace["rewritten_query"] = rewritten_query
                        trace.append(attempt_trace)
                        continue

                if not current_docs:
                    last_evaluation = retrieval_eval
                    last_answer = "Không tìm thấy bằng chứng đủ mạnh để trả lời từ tài liệu hiện tại."
                    attempt_trace["next_action"] = "early_stop_no_evidence"
                    trace.append(attempt_trace)
                    break

            last_answer = self.responder.answer(
                question=plan.question,
                source_documents=current_docs,
                llm_model=plan.llm_model,
                active_query=active_query,
                critique=critique,
                attempt=attempt,
            )
            last_evaluation = self.evaluator.evaluate(
                question=plan.question,
                answer=last_answer,
                source_documents=current_docs,
                llm_model=plan.llm_model,
                threshold=effective_threshold,
            )

            attempt_trace.update(
                {
                    "doc_count": len(current_docs),
                    "draft_answer": last_answer,
                    "score": round(last_evaluation.score, 4),
                    "confidence": round(last_evaluation.confidence, 4),
                    "decision": last_evaluation.decision,
                    "error_type": last_evaluation.error_type,
                    "reason": last_evaluation.rationale,
                    "policy_reason": last_evaluation.policy_reason,
                    "used_llm_review": last_evaluation.used_llm_review,
                }
            )
            score_history.append(last_evaluation.score)

            if last_evaluation.score >= effective_threshold:
                accepted = True
                attempt_trace["next_action"] = "accept"
                trace.append(attempt_trace)
                break

            attempts_left = attempt < plan.max_attempts
            if self._should_stop_early(
                score_history=score_history,
                plan=plan,
                attempts_left=attempts_left,
                current_decision=last_evaluation.decision,
            ):
                attempt_trace["next_action"] = "early_stop_stagnation"
                trace.append(attempt_trace)
                break

            critique = last_evaluation.rationale

            if (
                attempts_left
                and last_evaluation.decision == "rewrite"
                and last_evaluation.error_type in {"retrieval_issue", "query_issue"}
                and rewrites_used < plan.max_rewrites
            ):
                rewritten_query = self.rewriter.rewrite(
                    question=plan.question,
                    evaluation=last_evaluation,
                    source_documents=current_docs,
                    llm_model=plan.llm_model,
                    previous_queries=query_history,
                )
                normalized_rewrite = normalize_text(rewritten_query)
                if normalized_rewrite and normalized_rewrite not in seen_queries:
                    rewrites_used += 1
                    active_query = rewritten_query
                    query_history.append(rewritten_query)
                    seen_queries.add(normalized_rewrite)
                    current_docs = []
                    needs_retrieval = True
                    attempt_trace["next_action"] = "rewrite"
                    attempt_trace["rewritten_query"] = rewritten_query
                    trace.append(attempt_trace)
                    continue

            if (
                attempts_left
                and last_evaluation.decision == "multi_hop"
                and last_evaluation.error_type == "missing_info"
                and multi_hops_used < plan.max_multi_hops
            ):
                follow_up_queries = self.planner.plan_follow_up_queries(
                    question=plan.question,
                    draft_answer=last_answer,
                    evaluation_reason=last_evaluation.rationale,
                    missing_topics=last_evaluation.missing_topics,
                    suggested_queries=last_evaluation.follow_up_queries,
                    source_documents=current_docs,
                    llm_model=plan.llm_model,
                    limit=1,
                )
                next_query = next(
                    (
                        query
                        for query in follow_up_queries
                        if normalize_text(query) not in seen_queries
                    ),
                    "",
                )
                if next_query:
                    additional_docs = self.retriever.retrieve(
                        plan=plan,
                        query=next_query,
                        top_k=plan.hop_top_k,
                    )
                    merged_docs = self.retriever.merge_documents(
                        current_docs,
                        additional_docs,
                        limit=plan.merge_top_k,
                    )
                    if self._has_new_evidence(current_docs, merged_docs):
                        multi_hops_used += 1
                        active_query = next_query
                        query_history.append(next_query)
                        seen_queries.add(normalize_text(next_query))
                        current_docs = merged_docs
                        critique = (
                            f"{last_evaluation.rationale} Use the newly retrieved evidence "
                            "to fill the missing information from the partial answer."
                        )
                        attempt_trace["next_action"] = "multi_hop"
                        attempt_trace["partial_answer"] = self._truncate(last_answer, 220)
                        attempt_trace["follow_up_query"] = next_query
                        attempt_trace["extra_doc_count"] = len(additional_docs)
                        trace.append(attempt_trace)
                        continue

                attempt_trace["next_action"] = "early_stop_no_new_evidence"
                trace.append(attempt_trace)
                break

            if attempts_left and last_evaluation.decision == "regenerate":
                attempt_trace["next_action"] = "regenerate"
                trace.append(attempt_trace)
                continue

            attempt_trace["next_action"] = "return_last_answer"
            trace.append(attempt_trace)
            break

        final_source_limit = max(plan.top_k, min(plan.merge_top_k, plan.top_k + 2))
        final_docs = self.retriever.select_top_documents(current_docs, limit=final_source_limit)
        final_answer = self._append_confidence_footer(
            last_answer or "Không tìm thấy nội dung phù hợp trong tài liệu hiện tại.",
            last_evaluation.confidence,
            accepted=accepted,
        )

        metadata = {
            "mode": self.mode,
            "accepted": accepted,
            "attempts": len(trace),
            "final_query": active_query,
            "query_history": query_history,
            "rewrites_used": rewrites_used,
            "multi_hops_used": multi_hops_used,
            "confidence_score": last_evaluation.confidence,
            "evaluation_score": last_evaluation.score,
            "confidence_threshold": plan.confidence_threshold,
            "min_confidence_threshold": plan.min_confidence_threshold,
            "error_type": last_evaluation.error_type,
            "evaluation_reason": last_evaluation.rationale,
            "policy_reason": last_evaluation.policy_reason,
            "trace": trace,
        }
        return RAGEngineResult(
            answer=final_answer,
            source_documents=final_docs,
            metadata=metadata,
        )

    def _effective_threshold(self, plan: SelfRAGPlan, attempt: int) -> float:
        return max(
            plan.min_confidence_threshold,
            plan.confidence_threshold - 0.05 * attempt,
        )

    def _should_stop_early(
        self,
        *,
        score_history: List[float],
        plan: SelfRAGPlan,
        attempts_left: bool,
        current_decision: str,
    ) -> bool:
        if not attempts_left or len(score_history) < 3:
            return False
        if current_decision == "rewrite":
            return False
        previous_score = score_history[-2]
        current_score = score_history[-1]
        return current_score <= previous_score + plan.stagnation_tolerance

    def _has_new_evidence(self, previous_docs, merged_docs) -> bool:
        previous_keys = {self._doc_signature(doc) for doc in previous_docs}
        merged_keys = {self._doc_signature(doc) for doc in merged_docs}
        return len(merged_keys - previous_keys) > 0

    def _doc_signature(self, doc) -> str:
        metadata = doc.metadata or {}
        return "||".join(
            [
                str(metadata.get("document_id", "")),
                str(metadata.get("chunk", "")),
                str(metadata.get("page_start", "")),
                doc.page_content.strip()[:200],
            ]
        )

    def _truncate(self, text: str, limit: int) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip(" ,;") + "..."

    def _append_confidence_footer(self, answer: str, confidence: float, *, accepted: bool) -> str:
        base_answer = answer.strip() or "Không tìm thấy nội dung phù hợp trong tài liệu hiện tại."
        confidence_pct = int(round(max(0.0, min(1.0, confidence)) * 100))
        note = f"Độ tin cậy ước lượng: {confidence_pct}%"
        if not accepted and confidence_pct < 60:
            note += " (nên xác minh thêm)"
        return f"{base_answer}\n\n{note}"
