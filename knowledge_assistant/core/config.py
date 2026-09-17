from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_name: str = "Knowledge Assistant 2.0"
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/knowledge.db"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 20
    rerank_top_k: int = 8
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    embedding_provider: str = "none"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "none"
    llm_model: str = ""
    openai_api_key: str = ""
    max_upload_mb: int = 50

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = root or Path.cwd()
        load_dotenv(root / ".env")
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", str(root / "data"))),
            database_url=os.getenv("DATABASE_URL", "sqlite:///data/knowledge.db"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("TOP_K", "20")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "8")),
            semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.7")),
            lexical_weight=float(os.getenv("LEXICAL_WEIGHT", "0.3")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "none"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_provider=os.getenv("LLM_PROVIDER", "none"),
            llm_model=os.getenv("LLM_MODEL", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        )