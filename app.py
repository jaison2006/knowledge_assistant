from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "with",
    "about", "from", "into", "over", "under", "that", "this", "these", "those",
    "there", "their", "them", "they", "it", "its", "is", "are", "was", "were",
    "be", "been", "being", "as", "at", "by", "on", "in", "of", "to", "your",
    "you", "we", "our", "us", "have", "has", "had", "can", "could", "should",
    "would", "may", "might", "will", "also", "must", "not", "more", "most",
    "such", "than", "through", "using", "used", "each", "every", "including",
    "results", "document", "pdf", "page", "pages"
}


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    feature_keywords: List[str]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(file_like) -> str:
    text_parts: List[str] = []
    reader = PdfReader(file_like)
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"\n[Page {page_number}]\n{page_text}")
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) <= chunk_size:
            current = f"{current} {paragraph}".strip()
        else:
            if current:
                chunks.append(current.strip())
            if len(paragraph) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                segment = ""
                for sentence in sentences:
                    if len(segment) + len(sentence) <= chunk_size:
                        segment = f"{segment} {sentence}".strip()
                    else:
                        if segment:
                            chunks.append(segment)
                        segment = sentence
                if segment:
                    current = segment
            else:
                current = paragraph

    if current:
        chunks.append(current.strip())

    if len(chunks) <= 1:
        return [clean_text(chunk) for chunk in chunks if clean_text(chunk)]

    overlapped: List[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue
        previous = overlapped[-1]
        previous_words = previous.split()
        current_words = chunk.split()
        overlap_words = previous_words[-max(1, overlap // 5):]
        merged = " ".join(overlap_words + current_words)
        if len(merged) > chunk_size * 2:
            merged = merged[: chunk_size * 2]
        overlapped.append(merged)
    return [clean_text(part) for part in overlapped if clean_text(part)]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    freq = Counter(word for word in words if word not in STOP_WORDS)
    return [word for word, _ in freq.most_common(top_n)]


def split_document_to_chunks(raw_text: str, source_name: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for page_index, chunk_text_value in enumerate(chunk_text(raw_text), start=1):
        features = extract_keywords(chunk_text_value)
        chunks.append(
            DocumentChunk(
                text=chunk_text_value,
                source=source_name,
                page=page_index,
                feature_keywords=features,
            )
        )
    return chunks


def build_index(chunks: Iterable[DocumentChunk]):
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(query: str, chunks: List[DocumentChunk], top_k: int = 5):
    if not chunks:
        return []

    vectorizer, matrix = build_index(chunks)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).flatten()
    ranking = np.argsort(similarities)[::-1]

    results = []
    for idx in ranking[:top_k]:
        score = float(similarities[idx])
        if score <= 0:
            continue
        results.append((chunks[idx], score))
    return results


def build_prompt(question: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
    context = "\n\n".join(
        f"[Source: {chunk.source} - Page {chunk.page}]\n{chunk.text}"
        for chunk, _ in context_chunks
    )
    return (
        "Use only the context below to answer the question. "
        "If the evidence is not in the document, say that clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def generate_answer(question: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": build_prompt(question, context_chunks)}],
            )
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text.strip()
        except Exception:
            pass

    base = "\n\n".join(
        f"From {chunk.source}, page {chunk.page}: {chunk.text}"
        for chunk, _ in context_chunks
    )
    return "Based on the retrieved sections, here is the answer:\n\n" + base


def run_query(question: str, chunks: List[DocumentChunk], top_k: int = 5) -> tuple[str, List[Tuple[DocumentChunk, float]]]:
    context = retrieve(question, chunks, top_k=top_k)
    answer = generate_answer(question, context)
    return answer, context


def build_streamlit_app():
    if st is None:
        raise RuntimeError("streamlit is not installed")

    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    st.title("PDF RAG Assistant")
    st.caption("Upload a large PDF, split it into context-aware chunks, and find the top 5 relevant features.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        uploaded_file.seek(0)
        raw_text = extract_pdf_text(uploaded_file)
        chunks = split_document_to_chunks(raw_text, uploaded_file.name)
        st.session_state["document_chunks"] = chunks
        st.success(f"{len(chunks)} chunks created from {uploaded_file.name}")

    if "document_chunks" not in st.session_state:
        st.warning("Please upload a PDF to start.")
        return

    question = st.text_area("Ask a question about the uploaded PDF", height=120)
    if st.button("Find the best matches") and question:
        answer, ranked_chunks = run_query(question, st.session_state["document_chunks"], top_k=5)
        st.session_state["answer"] = answer
        st.session_state["ranked_chunks"] = ranked_chunks

    if "answer" in st.session_state and "ranked_chunks" in st.session_state:
        st.subheader("Answer")
        st.write(st.session_state["answer"])

        all_top_keywords = []
        for chunk, _ in st.session_state["ranked_chunks"]:
            all_top_keywords.extend(chunk.feature_keywords)
        unique_keywords = list(dict.fromkeys(all_top_keywords))[:5]
        st.subheader("Top 5 features")
        st.write(", ".join(unique_keywords) if unique_keywords else "No strong feature keywords found")

        st.subheader("Top 5 relevant sections")
        for index, (chunk, score) in enumerate(st.session_state["ranked_chunks"], start=1):
            with st.container():
                st.markdown(f"### {index}. {chunk.source} — Page {chunk.page} (score: {score:.3f})")
                st.write("Features:", ", ".join(chunk.feature_keywords) or "No strong keywords found")
                st.write(chunk.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PDF RAG app.")
    parser.add_argument("--question", type=str, help="Question to ask for the uploaded document.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of relevant chunks to return.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.question:
        sample_path = Path(__file__).parent / "sample_data" / "knowledge.txt"
        if sample_path.exists():
            raw_text = sample_path.read_text(encoding="utf-8")
            chunks = split_document_to_chunks(raw_text, sample_path.name)
            answer, context = run_query(args.question, chunks, top_k=args.top_k)
            print("ANSWER:")
            print(answer)
            print("\nTOP MATCHES:")
            for idx, (chunk, score) in enumerate(context, start=1):
                print(f"{idx}. {chunk.source} | page {chunk.page} | score={score:.3f}")
                print("Features:", ", ".join(chunk.feature_keywords) or "No keyword match")
                print(chunk.text[:400])
                print("-" * 60)
            return

    if st is None:
        raise RuntimeError("streamlit is required to run the UI.")
    build_streamlit_app()


if __name__ == "__main__":
    main()
