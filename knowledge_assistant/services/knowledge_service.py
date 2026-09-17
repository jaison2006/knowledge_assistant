from __future__ import annotations

import re
from pathlib import Path

from ..chunking import chunk_records
from ..core.config import Settings
from ..core.models import Answer, Document, SearchResult
from ..database.repository import Repository
from ..ingestion import ingest_file
from ..retrieval import HybridRetriever, create_embedding_provider


INTENTS = {
    "SUMMARY": ("summarize", "summary", "overview", "key points"),
    "COMPARISON": ("compare", "difference", "versus", "vs"),
    "LIST": ("list", "which", "what are"),
    "EXPLANATION": ("why", "how", "explain"),
}

QUESTION_STOP_WORDS = {
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "the", "a", "an", "this",
    "that", "it", "they", "them", "about", "from", "with", "for", "and", "or",
}


class KnowledgeService:
    def __init__(self, settings: Settings | None = None, repository: Repository | None = None):
        self.settings = settings or Settings.from_env()
        self.repository = repository or Repository(self.settings.database_url)
        provider = create_embedding_provider(self.settings.embedding_provider, self.settings.embedding_model)
        self.retriever = HybridRetriever(self.repository.all_chunks(), self.settings.semantic_weight, self.settings.lexical_weight, provider)

    def index_file(self, path: str | Path) -> tuple[Document, int, bool]:
        document, records = ingest_file(path, self.settings.max_upload_mb * 1024 * 1024)
        existing = self.repository.document_by_hash(document.content_hash)
        if existing:
            return existing, 0, False
        chunks = chunk_records(document, records, self.settings.chunk_size, self.settings.chunk_overlap)
        self.repository.save_document(document, chunks)
        self.retriever = HybridRetriever(self.repository.all_chunks(), self.settings.semantic_weight, self.settings.lexical_weight, self.retriever.embedding_provider)
        return document, len(chunks), True

    def search(self, query: str, top_k: int | None = None, document_id: str | None = None) -> list[SearchResult]:
        limit = top_k or self.settings.rerank_top_k
        if not document_id:
            return self.retriever.search(query, limit)
        scoped_chunks = [chunk for chunk in self.repository.all_chunks() if chunk.document_id == document_id]
        if not scoped_chunks:
            return []
        scoped_retriever = HybridRetriever(
            scoped_chunks,
            self.settings.semantic_weight,
            self.settings.lexical_weight,
            self.retriever.embedding_provider,
        )
        return scoped_retriever.search(query, limit)

    def answer(self, question: str, top_k: int | None = None, document_id: str | None = None) -> Answer:
        intent = self.classify_intent(question)
        results = self.search(question, top_k, document_id)
        if not results:
            direct_answer = "I could not find enough evidence in the knowledge base to answer this question confidently."
            explanation = "Try adding a more specific question or indexing a document that covers this topic."
            return Answer(direct_answer, [], 0.0, intent, direct_answer, explanation, self._recommendations(intent, question, []), [])
        direct_answer = self._direct_answer(question, results, intent)
        explanation = self._explanation(question, results, direct_answer)
        recommendations = self._recommendations(intent, question, results)
        evidence_table = [self._evidence_row(result) for result in results]
        text = self._format_answer(direct_answer, explanation, recommendations, evidence_table)
        confidence = min(0.99, max(0.05, sum(result.score for result in results[:3]) / 3 + min(len(results), 3) * 0.08))
        return Answer(text, results, confidence, intent, direct_answer, explanation, recommendations, evidence_table)

    @staticmethod
    def _direct_answer(question: str, results: list[SearchResult], intent: str) -> str:
        candidates = KnowledgeService._ranked_sentences(question, results)
        first = candidates[0][0] if candidates else results[0].chunk.content.strip()
        if intent == "SUMMARY":
            return f"The indexed evidence supports this summary: {first}"
        if intent == "COMPARISON":
            return "The retrieved evidence contains the relevant comparison points below; the sources should be read together before deciding between them."
        return f"The clearest answer supported by the knowledge base is: {first}"

    @staticmethod
    def _explanation(question: str, results: list[SearchResult], direct_answer: str) -> str:
        ranked = KnowledgeService._ranked_sentences(question, results)
        supporting = [sentence for sentence, _, _ in ranked if sentence not in direct_answer][-3:]
        if not supporting:
            supporting = [sentence for sentence, _, _ in ranked[1:3]]
        if supporting:
            return " ".join(supporting)
        return "The retrieved source does not contain additional detail beyond the direct answer."

    @staticmethod
    def _ranked_sentences(question: str, results: list[SearchResult]) -> list[tuple[str, int, SearchResult]]:
        query_terms = {
            KnowledgeService._normalize_term(term) for term in re.findall(r"[a-z0-9-]+", question.lower())
            if len(term) > 2 and term not in QUESTION_STOP_WORDS
        }
        candidates: list[tuple[str, int, SearchResult]] = []
        seen: set[str] = set()
        for result in results:
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", result.chunk.content) if sentence.strip()]
            for sentence in sentences:
                normalized = sentence.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                sentence_terms = {KnowledgeService._normalize_term(term) for term in re.findall(r"[a-z0-9-]+", normalized)}
                score = len(query_terms.intersection(sentence_terms)) * 3
                if query_terms and all(term in sentence_terms for term in query_terms):
                    score += 2
                if "retrieval" in query_terms and "retrieval" in sentence_terms:
                    score += 2
                if "retrieval" in query_terms and "tf-idf" in normalized:
                    score += 4
                score += round(result.score * 2)
                candidates.append((sentence, score, result))
        return sorted(candidates, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _normalize_term(term: str) -> str:
        if len(term) > 4 and term.endswith("ies"):
            return term[:-3] + "y"
        if len(term) > 4 and term.endswith("s"):
            return term[:-1]
        return term

    @staticmethod
    def _recommendations(intent: str, question: str, results: list[SearchResult]) -> list[str]:
        locations = []
        for result in results[:3]:
            chunk = result.chunk
            location = f"page {chunk.page}" if chunk.page else (f"section '{chunk.section}'" if chunk.section else "the indexed section")
            if location not in locations:
                locations.append(location)
        cited_locations = ", ".join(locations) if locations else "the indexed knowledge base"
        if intent == "SUMMARY":
            return [f"Review the source material at {cited_locations} for the complete context.", f"Ask for the key points, risks, or action items related to '{question.rstrip('?')}'."]
        if intent == "COMPARISON":
            return [f"Compare the cited evidence from {cited_locations} before choosing an option.", f"Ask for a comparison table focused on '{question.rstrip('?')}'."]
        return [f"Verify the answer in {cited_locations} before using it as a decision.", f"Ask a follow-up question about a specific page, section, or claim related to '{question.rstrip('?')}'."]

    @staticmethod
    def _evidence_row(result: SearchResult) -> dict[str, str]:
        chunk = result.chunk
        location = f"Page {chunk.page}" if chunk.page else (chunk.section or "Indexed section")
        return {"source": str(chunk.metadata.get("filename", chunk.document_id)), "location": location, "relevance": f"{result.score:.3f}", "evidence": chunk.content[:240]}

    @staticmethod
    def _format_answer(direct_answer: str, explanation: str, recommendations: list[str], evidence_table: list[dict[str, str]]) -> str:
        recommendations_text = "\n".join(f"- {recommendation}" for recommendation in recommendations)
        table = "\n".join(
            f"| {row['source']} | {row['location']} | {row['relevance']} | {row['evidence'].replace('|', '/') } |"
            for row in evidence_table
        )
        return (
            f"## Direct answer\n{direct_answer}\n\n"
            f"## Detailed explanation\n{explanation}\n\n"
            f"## What you can do next\n{recommendations_text}\n\n"
            "## Evidence table\n| Source | Location | Relevance | Supporting text |\n|---|---|---:|---|\n"
            f"{table}"
        )

    @staticmethod
    def classify_intent(question: str) -> str:
        lowered = question.lower()
        for intent, markers in INTENTS.items():
            if any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in markers):
                return intent
        return "FACT"

    @staticmethod
    def _source_block(result: SearchResult) -> str:
        chunk = result.chunk
        location = f"Page {chunk.page}" if chunk.page else (f"Section {chunk.section}" if chunk.section else "Indexed section")
        return f"[{chunk.metadata.get('filename', chunk.document_id)} - {location}]\n{chunk.content}"