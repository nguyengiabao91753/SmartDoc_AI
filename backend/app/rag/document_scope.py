from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class DocumentScope:
    document_ids: List[int] = field(default_factory=list)
    session_id: int | None = None

    @property
    def primary_document_id(self) -> int | None:
        return self.document_ids[0] if self.document_ids else None

    def to_metadata_filters(self) -> Dict[str, Any] | None:
        if self.document_ids:
            if len(self.document_ids) == 1:
                return {"document_id": self.document_ids[0]}
            return {"document_id": {"$in": self.document_ids}}

        if self.session_id is not None:
            return {"session_id": self.session_id}

        return None

    def to_graph_document_ids(self) -> List[str]:
        return [str(document_id) for document_id in self.document_ids]


def resolve_document_scope(
    document_id: int | None = None,
    document_ids: List[int] | None = None,
    session_id: int | None = None,
) -> DocumentScope:
    normalized_ids: List[int] = []
    seen_ids: set[int] = set()

    for raw_value in document_ids or []:
        try:
            resolved_id = int(raw_value)
        except (TypeError, ValueError):
            continue
        if resolved_id in seen_ids:
            continue
        seen_ids.add(resolved_id)
        normalized_ids.append(resolved_id)

    if not normalized_ids and document_id is not None:
        try:
            normalized_ids.append(int(document_id))
        except (TypeError, ValueError):
            pass

    normalized_session_id = None
    if session_id is not None:
        try:
            normalized_session_id = int(session_id)
        except (TypeError, ValueError):
            normalized_session_id = None

    return DocumentScope(
        document_ids=normalized_ids,
        session_id=normalized_session_id,
    )
