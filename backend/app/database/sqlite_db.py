import sqlite3
from pathlib import Path
from app.core.config import settings
from app.core.logger import LOG
import os

def get_conn():
    db_path = settings.SQLITE_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Documents table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # Chunks metadata (optional)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_index INTEGER,
        page INTEGER,
        text_excerpt TEXT,
        vector_id INTEGER,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    """)
    # Chat history
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        document_id INTEGER
    );
    """)
    conn.commit()
    conn.close()
    LOG.info("SQLite DB initialized at %s", settings.SQLITE_PATH)