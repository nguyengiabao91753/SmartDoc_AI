import os
import sqlite3

from app.core.config import settings
from app.core.logger import LOG


def get_conn():
    db_path = settings.SQLITE_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table_name})")
    return {row["name"] for row in cur.fetchall()}


def _ensure_column(cur: sqlite3.Cursor, table_name: str, column_name: str, column_sql: str):
    existing_columns = _get_columns(cur, table_name)
    if column_name not in existing_columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


def _backfill_sessions(cur: sqlite3.Cursor):
    cur.execute(
        """
        SELECT d.id, d.filename, d.uploaded_at
        FROM documents d
        LEFT JOIN chat_sessions s ON s.document_id = d.id
        WHERE s.id IS NULL
        ORDER BY d.uploaded_at ASC, d.id ASC
        """
    )
    missing_sessions = cur.fetchall()
    for row in missing_sessions:
        cur.execute(
            """
            INSERT INTO chat_sessions (title, document_id, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (row["filename"], row["id"], row["uploaded_at"], row["uploaded_at"]),
        )

    cur.execute("SELECT id, document_id FROM chat_sessions WHERE document_id IS NOT NULL")
    session_map = {row["document_id"]: row["id"] for row in cur.fetchall()}
    for document_id, session_id in session_map.items():
        cur.execute(
            """
            UPDATE chat_history
            SET session_id = ?
            WHERE document_id = ? AND session_id IS NULL
            """,
            (session_id, document_id),
        )

    cur.execute(
        """
        SELECT COUNT(*) AS total
        FROM chat_history
        WHERE session_id IS NULL AND (question IS NOT NULL OR answer IS NOT NULL)
        """
    )
    orphan_row = cur.fetchone()
    orphan_count = int(orphan_row["total"]) if orphan_row is not None else 0
    if orphan_count:
        cur.execute(
            """
            INSERT INTO chat_sessions (title)
            VALUES (?)
            """,
            ("Imported chat",),
        )
        orphan_session_id = cur.lastrowid
        cur.execute(
            """
            UPDATE chat_history
            SET session_id = ?
            WHERE session_id IS NULL
            """,
            (orphan_session_id,),
        )


def _backfill_session_documents(cur: sqlite3.Cursor):
    # Ensure every existing session-document relation is represented in the mapping table.
    cur.execute(
        """
        INSERT OR IGNORE INTO session_documents (session_id, document_id, is_primary)
        SELECT id AS session_id, document_id, 1
        FROM chat_sessions
        WHERE document_id IS NOT NULL
        """
    )


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            document_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS session_documents (
            session_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            is_primary INTEGER NOT NULL DEFAULT 0,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, document_id),
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            chunk_index INTEGER,
            page INTEGER,
            text_excerpt TEXT,
            vector_id INTEGER,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            document_id INTEGER,
            search_type TEXT DEFAULT 'vector',
            rag_mode TEXT DEFAULT 'rag',
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id),
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
        """
    )

    _ensure_column(cur, "chat_history", "session_id", "session_id INTEGER")
    _ensure_column(cur, "chat_history", "search_type", "search_type TEXT DEFAULT 'vector'")
    _ensure_column(cur, "chat_history", "rag_mode", "rag_mode TEXT DEFAULT 'rag'")

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at
        ON chat_sessions(updated_at DESC)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chat_history_session_id
        ON chat_history(session_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks(document_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_documents_session_id
        ON session_documents(session_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_documents_document_id
        ON session_documents(document_id)
        """
    )

    _backfill_sessions(cur)
    _backfill_session_documents(cur)

    conn.commit()
    conn.close()
    LOG.info("SQLite DB initialized at %s", settings.SQLITE_PATH)
