from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    filename: str
    source_type: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    page: int | None = None
    section: str | None = None
    heading: str | None = None
    source_type: str = "text"
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]
    evidence_confidence: float
    intent: str = "FACT"
    direct_answer: str = ""
    explanation: str = ""
    recommendations: list[str] = field(default_factory=list)
    evidence_table: list[dict[str, str]] = field(default_factory=list)