from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from app.ai.llm import get_llm
from app.rag.models import RAGQueryRequest


@dataclass
class CoRAGPlan:
    question: str
    sub_queries: List[str]
    search_type: str
    top_k: int
    document_id: int | None = None
    llm_model: str | None = None


class CoRAGPlanner:
    """
    Planner cho CoRAG.
    Dùng LLM để phân tách câu hỏi gốc thành các sub-queries độc lập.
    Mỗi sub-query nhắm vào 1 khía cạnh cụ thể để retrieve hiệu quả hơn.
    """

    MAX_SUBQUERIES = 3

    def plan(self, request: RAGQueryRequest) -> CoRAGPlan:
        sub_queries = self._decompose(request.question, request.llm_model)
        return CoRAGPlan(
            question=request.question,
            sub_queries=sub_queries,
            search_type=request.search_type,
            top_k=request.top_k,
            document_id=request.document_id,
            llm_model=request.llm_model,
        )

    def _decompose(self, question: str, llm_model: str | None = None) -> List[str]:
        """
        Tách câu hỏi thành sub-queries KHÁC NHAU và KHÔNG trùng nhau.
        Nếu câu hỏi đơn giản chỉ có 1 khía cạnh → trả về đúng 1 sub-query.
        Fallback về câu hỏi gốc nếu LLM thất bại.
        """
        prompt = (
            "Nhiệm vụ: Phân tích câu hỏi và tách thành các sub-questions để tìm kiếm tài liệu.\n\n"
            "NGUYÊN TẮC:\n"
            "- Mỗi sub-question phải tìm kiếm một KHÍA CẠNH KHÁC NHAU, không trùng lặp.\n"
            "- Nếu câu hỏi chỉ có 1 ý → chỉ tạo 1 sub-question.\n"
            "- Nếu câu hỏi có nhiều ý → tạo 2-3 sub-questions, mỗi cái 1 ý riêng.\n"
            "- Sub-question phải ngắn gọn, cụ thể, dễ tìm kiếm.\n\n"
            "CHỈ trả về JSON array thuần, không giải thích, không markdown, không ```\n"
            "Ví dụ: [\"câu hỏi con 1\", \"câu hỏi con 2\"]\n\n"
            f"Câu hỏi: {question}\n\n"
            "JSON:"
        )

        try:
            llm = get_llm(temperature=0.0, model=llm_model)
            raw = llm.invoke(prompt).strip()
            sub_queries = self._parse_subqueries(raw, question)
        except Exception:
            sub_queries = [question]

        # Loại bỏ trùng lặp, giữ thứ tự
        seen = set()
        unique = []
        for q in sub_queries:
            q_norm = q.strip().lower()
            if q_norm not in seen:
                seen.add(q_norm)
                unique.append(q.strip())

        return unique[:self.MAX_SUBQUERIES] if unique else [question]

    def _parse_subqueries(self, raw: str, fallback: str) -> List[str]:
        """Parse JSON array từ LLM output với nhiều fallback."""
        # Thử parse trực tiếp
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                result = [str(q).strip() for q in parsed if str(q).strip()]
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Tìm JSON array trong text
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    result = [str(q).strip() for q in parsed if str(q).strip()]
                    if result:
                        return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Parse dạng numbered list
        lines = raw.split('\n')
        sub_queries = []
        for line in lines:
            line = re.sub(r'^[\d\.\-\*\s"]+', '', line).strip().strip('"\'\' ')
            if len(line) > 8:
                sub_queries.append(line)

        if sub_queries:
            return sub_queries[:self.MAX_SUBQUERIES]

        return [fallback]