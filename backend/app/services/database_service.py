from __future__ import annotations

from typing import Any, Dict, List

from app.core.logger import LOG
from app.database.sqlite_db import get_conn, init_db


class DatabaseService:
    """Service layer for SQLite-backed documents, sessions, and chat history."""

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
        return self.create_document(filename, filepath)

    def delete_document(self, document_id: int):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        cur.execute("DELETE FROM session_documents WHERE document_id = ?", (document_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        cur.execute(
            """
            UPDATE chat_sessions
            SET document_id = (
                SELECT sd.document_id
                FROM session_documents sd
                WHERE sd.session_id = chat_sessions.id
                ORDER BY sd.is_primary DESC, sd.added_at ASC
                LIMIT 1
            )
            WHERE document_id = ?
            """,
            (document_id,),
        )
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
        session_id = cur.lastrowid
        if document_id is not None:
            cur.execute(
                """
                INSERT OR IGNORE INTO session_documents (session_id, document_id, is_primary)
                VALUES (?, ?, 1)
                """,
                (session_id, document_id),
            )
        conn.commit()
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
        cur.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        LOG.info("Deleted chat session %s", session_id)

    def attach_document_to_session(self, session_id: int, document_id: int, title: str | None = None):
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT OR IGNORE INTO session_documents (session_id, document_id, is_primary)
            VALUES (?, ?, 0)
            """,
            (session_id, document_id),
        )

        cur.execute("SELECT document_id FROM chat_sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        current_primary_id = row["document_id"] if row else None

        if current_primary_id is None:
            cur.execute(
                """
                UPDATE chat_sessions
                SET document_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (document_id, session_id),
            )
            cur.execute(
                """
                UPDATE session_documents
                SET is_primary = CASE WHEN document_id = ? THEN 1 ELSE 0 END
                WHERE session_id = ?
                """,
                (document_id, session_id),
            )
        else:
            cur.execute(
                """
                UPDATE chat_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (session_id,),
            )

        if title:
            safe_title = title.strip() or "New chat"
            cur.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (safe_title, session_id),
            )

        conn.commit()
        conn.close()
        LOG.info("Attached document %s to chat session %s", document_id, session_id)

    def get_session_documents(self, session_id: int) -> List[Dict[str, Any]]:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.id, d.filename, d.filepath, sd.is_primary, sd.added_at
            FROM session_documents sd
            JOIN documents d ON d.id = sd.document_id
            WHERE sd.session_id = ?
            ORDER BY sd.is_primary DESC, sd.added_at ASC, d.id ASC
            """,
            (session_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows

    def _hydrate_session_documents(self, session_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not session_rows:
            return session_rows

        session_ids = [row["id"] for row in session_rows]
        placeholders = ",".join(["?"] * len(session_ids))

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT sd.session_id, d.id AS document_id, d.filename, d.filepath, sd.is_primary, sd.added_at
            FROM session_documents sd
            JOIN documents d ON d.id = sd.document_id
            WHERE sd.session_id IN ({placeholders})
            ORDER BY sd.session_id ASC, sd.is_primary DESC, sd.added_at ASC, d.id ASC
            """,
            tuple(session_ids),
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()

        by_session: Dict[int, List[Dict[str, Any]]] = {session_id: [] for session_id in session_ids}
        for row in rows:
            by_session.setdefault(row["session_id"], []).append(
                {
                    "id": row["document_id"],
                    "filename": row["filename"],
                    "filepath": row["filepath"],
                    "is_primary": int(row.get("is_primary") or 0),
                    "added_at": row.get("added_at"),
                }
            )

        for session in session_rows:
            documents = by_session.get(session["id"], [])
            session["documents"] = documents
            session["document_ids"] = [doc["id"] for doc in documents]
            if documents:
                primary_doc = next((doc for doc in documents if doc.get("is_primary") == 1), documents[0])
                session["document_id"] = primary_doc["id"]
                session["document_name"] = primary_doc["filename"]
                session["document_path"] = primary_doc["filepath"]
            else:
                session["document_id"] = None
                session["document_name"] = None
                session["document_path"] = None

        return session_rows

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
                COUNT(h.id) AS exchange_count
            FROM chat_sessions s
            LEFT JOIN chat_history h ON h.session_id = s.id
            GROUP BY s.id, s.title, s.document_id, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC, s.id DESC
            """
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return self._hydrate_session_documents(rows)

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
                (
                    SELECT COUNT(*)
                    FROM chat_history h
                    WHERE h.session_id = s.id
                ) AS exchange_count
            FROM chat_sessions s
            WHERE s.id = ?
            """,
            (session_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None

        session = dict(row)
        return self._hydrate_session_documents([session])[0]

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

    def get_document_by_filepath(self, filepath: str) -> Dict[str, Any] | None:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, filename, filepath FROM documents WHERE filepath = ?", (filepath,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None


db_service = DatabaseService()
