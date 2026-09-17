from __future__ import annotations

import argparse
from pathlib import Path

from knowledge_assistant.core.config import Settings
from knowledge_assistant.services.knowledge_service import KnowledgeService


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Assistant 2.0")
    parser.add_argument("--question", help="Question to ask against the local knowledge base")
    parser.add_argument("--file", action="append", help="File to index before answering")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    service = KnowledgeService(Settings.from_env(Path(__file__).parent))
    for file_path in args.file or []:
        document, count, created = service.index_file(file_path)
        print(f"{'Indexed' if created else 'Already indexed'} {document.filename} ({count} chunks)")
    if args.question:
        if not args.file and not service.repository.list_documents():
            sample_path = Path(__file__).parent / "sample_data" / "knowledge.txt"
            if sample_path.exists():
                service.index_file(sample_path)
        answer = service.answer(args.question, args.top_k)
        print("ANSWER:\n" + answer.text)
        print(f"\nEvidence confidence: {answer.evidence_confidence:.0%}")
        for index, source in enumerate(answer.sources, 1):
            print(f"[{index}] {source.chunk.document_id} | page={source.chunk.page} | score={source.score:.3f}")
        return
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install streamlit or pass --question") from exc
    st.set_page_config(page_title="Knowledge Assistant 2.0", layout="wide")
    st.title("Knowledge Assistant 2.0")
    st.caption("A grounded local knowledge workspace with source-aware retrieval.")
    uploaded = st.file_uploader("Add a source", type=["pdf", "docx", "pptx", "txt", "md", "html", "csv", "json"])
    if uploaded is not None and st.button("Index source"):
        upload_dir = service.settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        document, count, created = service.index_file(target)
        st.session_state["active_document_id"] = document.document_id
        st.success(f"{'Indexed' if created else 'Already indexed'} {document.filename}: {count} chunks")
    st.metric("Documents", len(service.repository.list_documents()))
    documents = service.repository.list_documents()
    document_options = ["all"] + [document.document_id for document in documents]
    active_document_id = st.session_state.get("active_document_id", "all")
    selected_index = document_options.index(active_document_id) if active_document_id in document_options else 0
    selected_document_id = st.selectbox(
        "Answer from",
        document_options,
        index=selected_index,
        format_func=lambda document_id: "All indexed documents" if document_id == "all" else next(
            document.filename for document in documents if document.document_id == document_id
        ),
    )
    question = st.text_area("Ask your knowledge base")
    if st.button("Search and answer") and question:
        answer = service.answer(question, document_id=None if selected_document_id == "all" else selected_document_id)
        st.subheader("Direct answer")
        st.write(answer.direct_answer)
        st.subheader("Detailed explanation")
        st.write(answer.explanation)
        st.subheader("What you can do next")
        for recommendation in answer.recommendations:
            st.write(f"- {recommendation}")
        st.subheader("Evidence table")
        st.dataframe(answer.evidence_table, use_container_width=True, hide_index=True)
        st.caption(f"Intent: {answer.intent} | Evidence confidence: {answer.evidence_confidence:.0%}")
        st.subheader("Sources")
        for source in answer.sources:
            st.write(f"{source.chunk.document_id} | page {source.chunk.page or 'n/a'} | {source.score:.3f}")


if __name__ == "__main__":
    main()
