from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..core.models import Document, DocumentChunk


class Repository:
    def __init__(self, database_url: str = "sqlite:///data/knowledge.db"):
        path = database_url.removeprefix("sqlite:///")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    heading TEXT,
                    source_type TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            """)

    def document_by_hash(self, content_hash: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
        return self._document(row) if row else None

    def save_document(self, document: Document, chunks: list[DocumentChunk]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document.document_id,))
            connection.execute("""INSERT OR REPLACE INTO documents(document_id, filename, source_type, content_hash, metadata)
                VALUES (?, ?, ?, ?, ?)""", (document.document_id, document.filename, document.source_type, document.content_hash, json.dumps(document.metadata)))
            connection.executemany("""INSERT INTO chunks(chunk_id, document_id, content, page, section, heading, source_type, token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", [(chunk.chunk_id, chunk.document_id, chunk.content, chunk.page, chunk.section, chunk.heading, chunk.source_type, chunk.token_count, json.dumps(chunk.metadata)) for chunk in chunks])

    def list_documents(self) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._document(row) for row in rows]

    def all_chunks(self) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM chunks ORDER BY document_id, rowid").fetchall()
        return [DocumentChunk(row["chunk_id"], row["document_id"], row["content"], row["page"], row["section"], row["heading"], row["source_type"], row["token_count"], json.loads(row["metadata"])) for row in rows]

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        return Document(row["document_id"], row["filename"], row["source_type"], row["content_hash"], json.loads(row["metadata"]))