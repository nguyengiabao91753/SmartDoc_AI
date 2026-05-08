from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_conversation_history_context(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    max_messages: int = 6,
) -> str:
    """Format recent chat turns into a compact prompt block."""
    if not messages:
        return ""

    recent_messages = list(messages)[-max_messages:]
    lines: list[str] = []

    for message in recent_messages:
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            speaker = "Người dùng"
        elif role == "assistant":
            speaker = "Trợ lý"
        else:
            speaker = role.title() if role else "Khác"

        lines.append(f"- {speaker}: {content}")

    return "\n".join(lines).strip()
