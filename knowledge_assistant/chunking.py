from __future__ import annotations

import hashlib
import re
from typing import Any

from .core.models import Document, DocumentChunk


def chunk_records(document: Document, records: list[dict[str, Any]], chunk_size: int = 800, overlap: int = 120) -> list[DocumentChunk]:
    if chunk_size < 50 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be >= 50 and overlap must be smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    for record in records:
        content = str(record.get("content", "")).strip()
        if not content:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", content) if part.strip()] or [content]
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 1 > chunk_size:
                chunks.append(_make_chunk(document, current, record, len(chunks)))
                current = " ".join(current.split()[-max(1, overlap // 5):])
            current = f"{current} {paragraph}".strip()
            if len(current) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", current)
                current = ""
                for sentence in sentences:
                    if current and len(current) + len(sentence) + 1 > chunk_size:
                        chunks.append(_make_chunk(document, current, record, len(chunks)))
                        current = " ".join(current.split()[-max(1, overlap // 5):])
                    current = f"{current} {sentence}".strip()
        if current:
            chunks.append(_make_chunk(document, current, record, len(chunks)))
    return chunks


def _make_chunk(document: Document, content: str, record: dict[str, Any], index: int) -> DocumentChunk:
    chunk_id = hashlib.sha1(f"{document.document_id}:{index}:{content}".encode()).hexdigest()[:20]
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        content=content,
        page=record.get("page"),
        section=record.get("section"),
        heading=record.get("heading"),
        source_type=document.source_type,
        token_count=len(content.split()),
        metadata=document.metadata.copy(),
    )