from typing import Any, Dict, List

from app.core.logger import LOG
from app.database.sqlite_db import get_conn, init_db


class DatabaseService:
    """Service layer for SQLite-backed documents, chat sessions, and message history."""

    def __init__(self):
        init_db()

    def create_document(self, filename: str, filepath: str) -> int:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, filepath) VALUES (?, ?)",
            (filename, filepath),
        )
        conn.commit()
        document_id = cur.lastrowid
        conn.close()
        LOG.info("Added document '%s' to DB with ID: %s", filename, document_id)
        return document_id

    def add_document(self, filename: str, filepath: str) -> int:
        """Backward-compatible alias used by older callers."""
        return self.create_document(filename, filepath)

    def delete_document(self, document_id: int):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        conn.close()
        LOG.info("Deleted document %s from DB", document_id)

    def create_chat_session(self, title: str, document_id: int | None = None) -> int:
        session_title = (title or "New chat").strip() or "New chat"
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_sessions (title, document_id)
            VALUES (?, ?)
            """,
            (session_title, document_id),
        )
        conn.commit()
        session_id = cur.lastrowid
        conn.close()
        LOG.info("Created chat session '%s' with ID: %s", session_title, session_id)
        return session_id

    def rename_chat_session(self, session_id: int, title: str):
        session_title = (title or "").strip() or "New chat"
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE chat_sessions
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_title, session_id),
        )
        conn.commit()
        conn.close()
        LOG.info("Renamed chat session %s to '%s'", session_id, session_title)

    def delete_chat_session(self, session_id: int):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        LOG.info("Deleted chat session %s", session_id)

    def attach_document_to_session(self, session_id: int, document_id: int, title: str | None = None):
        conn = get_conn()
        cur = conn.cursor()
        if title:
            cur.execute(
                """
                UPDATE chat_sessions
                SET document_id = ?, title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (document_id, title.strip() or "New chat", session_id),
            )
        else:
            cur.execute(
                """
                UPDATE chat_sessions
                SET document_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (document_id, session_id),
            )
        conn.commit()
        conn.close()
        LOG.info("Attached document %s to chat session %s", document_id, session_id)

    def get_chat_sessions(self) -> List[Dict[str, Any]]:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.id,
                s.title,
                s.document_id,
                s.created_at,
                s.updated_at,
                d.filename AS document_name,
                d.filepath AS document_path,
                COUNT(h.id) AS exchange_count
            FROM chat_sessions s
            LEFT JOIN documents d ON d.id = s.document_id
            LEFT JOIN chat_history h ON h.session_id = s.id
            GROUP BY
                s.id, s.title, s.document_id, s.created_at, s.updated_at,
                d.filename, d.filepath
            ORDER BY s.updated_at DESC, s.id DESC
            """
        )
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_chat_session(self, session_id: int) -> Dict[str, Any] | None:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.id,
                s.title,
                s.document_id,
                s.created_at,
                s.updated_at,
                d.filename AS document_name,
                d.filepath AS document_path
            FROM chat_sessions s
            LEFT JOIN documents d ON d.id = s.document_id
            WHERE s.id = ?
            """,
            (session_id,),
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def get_chat_history(self, session_id: int) -> List[Dict[str, Any]]:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT question, answer, created_at, search_type, rag_mode
            FROM chat_history
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        conn.close()

        history: List[Dict[str, Any]] = []
        for row in rows:
            if row["question"]:
                history.append(
                    {
                        "role": "user",
                        "content": row["question"],
                        "created_at": row["created_at"],
                    }
                )
            if row["answer"]:
                history.append(
                    {
                        "role": "assistant",
                        "content": row["answer"],
                        "created_at": row["created_at"],
                        "search_type": row["search_type"],
                        "rag_mode": row["rag_mode"],
                    }
                )
        return history

    def add_chat_history(
        self,
        session_id: int,
        question: str,
        answer: str,
        document_id: int | None = None,
        search_type: str = "vector",
        rag_mode: str = "rag",
    ):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_history (session_id, question, answer, document_id, search_type, rag_mode)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, question, answer, document_id, search_type, rag_mode),
        )
        cur.execute(
            """
            UPDATE chat_sessions
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )
        conn.commit()
        conn.close()
        LOG.info(
            "Added chat history entry for session %s (document=%s, search_type=%s, rag_mode=%s)",
            session_id,
            document_id,
            search_type,
            rag_mode,
        )

    def replace_document_chunks(self, document_id: int, chunks: List[Dict[str, Any]]):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, page, text_excerpt, vector_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    chunk.get("chunk_index"),
                    chunk.get("page"),
                    chunk.get("text_excerpt"),
                    chunk.get("vector_id"),
                )
                for chunk in chunks
            ],
        )
        conn.commit()
        conn.close()
        LOG.info("Stored %d chunk rows for document %s", len(chunks), document_id)


db_service = DatabaseService()
