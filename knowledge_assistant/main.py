from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core.config import Settings
from .services.knowledge_service import KnowledgeService


settings = Settings.from_env()
service = KnowledgeService(settings)
app = FastAPI(title=settings.app_name, version="2.0.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    document_id: str | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/api/documents")
def documents() -> list[dict]:
    return [document.__dict__ for document in service.repository.list_documents()]


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "upload.txt").name
    if not re.fullmatch(r"[\w .()-]+", filename, flags=re.ASCII):
        raise HTTPException(status_code=400, detail="Unsafe filename")
    suffix = Path(filename).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(await file.read(settings.max_upload_mb * 1024 * 1024 + 1))
            temporary_path = Path(temporary.name)
        document, chunks, created = service.index_file(temporary_path)
        return {"document": document.__dict__, "chunks": chunks, "created": created}
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    results = service.search(request.query, request.top_k, request.document_id)
    return {"results": [{"score": result.score, "lexical_score": result.lexical_score, "semantic_score": result.semantic_score, "chunk": result.chunk.to_dict()} for result in results]}


@app.post("/api/chat")
def chat(request: SearchRequest) -> dict:
    answer = service.answer(request.query, request.top_k, request.document_id)
    return {
        "text": answer.text,
        "direct_answer": answer.direct_answer,
        "explanation": answer.explanation,
        "recommendations": answer.recommendations,
        "evidence_table": answer.evidence_table,
        "intent": answer.intent,
        "evidence_confidence": answer.evidence_confidence,
        "sources": [{"score": source.score, "chunk": source.chunk.to_dict()} for source in answer.sources],
    }