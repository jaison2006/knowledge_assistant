from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .core.models import DocumentChunk, SearchResult


class EmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self.model.encode([query], normalize_embeddings=True))[0]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [re.findall(r"[a-z0-9]+", text.lower()) for text in texts]
        self.average_length = sum(len(document) for document in self.documents) / max(1, len(self.documents))
        self.document_frequency: dict[str, int] = {}
        for document in self.documents:
            for term in set(document):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

    def scores(self, query: str) -> np.ndarray:
        query_terms = re.findall(r"[a-z0-9]+", query.lower())
        scores = np.zeros(len(self.documents))
        total_documents = len(self.documents)
        for index, document in enumerate(self.documents):
            counts = {term: document.count(term) for term in set(query_terms)}
            for term, frequency in counts.items():
                if not frequency:
                    continue
                inverse_frequency = np.log(1 + (total_documents - self.document_frequency.get(term, 0) + 0.5) / (self.document_frequency.get(term, 0) + 0.5))
                length_factor = 1 - self.b + self.b * len(document) / max(1, self.average_length)
                scores[index] += inverse_frequency * frequency * (self.k1 + 1) / (frequency + self.k1 * length_factor)
        return scores / max(1e-9, float(scores.max())) if scores.max() else scores


class HybridRetriever:
    def __init__(self, chunks: list[DocumentChunk], semantic_weight: float = 0.7, lexical_weight: float = 0.3, embedding_provider: EmbeddingProvider | None = None):
        self.chunks = chunks
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk.content for chunk in chunks]) if chunks else None
        self.bm25 = BM25([chunk.content for chunk in chunks]) if chunks else None
        self.embedding_provider = embedding_provider
        self.embeddings = embedding_provider.embed_documents([chunk.content for chunk in chunks]) if embedding_provider and chunks else None

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        if not self.chunks or self.matrix is None:
            return []
        lexical = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        bm25 = self.bm25.scores(query) if self.bm25 is not None else lexical
        lexical = (lexical + bm25) / 2
        semantic = lexical
        if self.embedding_provider is not None and self.embeddings is not None:
            semantic = self.embeddings @ self.embedding_provider.embed_query(query)
        scores = self.semantic_weight * semantic + self.lexical_weight * lexical
        ranked = np.argsort(scores)[::-1]
        return [SearchResult(self.chunks[index], float(scores[index]), float(lexical[index]), float(semantic[index])) for index in ranked[:top_k] if scores[index] > 0]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRetriever":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def create_embedding_provider(provider: str, model: str) -> EmbeddingProvider | None:
    if provider.lower() in {"local", "sentence-transformers", "sentence_transformers"}:
        try:
            return SentenceTransformerProvider(model)
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to enable local semantic embeddings") from exc
    return None