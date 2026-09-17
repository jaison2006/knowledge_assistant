from pathlib import Path

from knowledge_assistant.chunking import chunk_records
from knowledge_assistant.core.config import Settings
from knowledge_assistant.database.repository import Repository
from knowledge_assistant.ingestion import ingest_file
from knowledge_assistant.services.knowledge_service import KnowledgeService


def test_text_ingestion_and_hybrid_retrieval(tmp_path: Path):
    document, records = ingest_file(Path("sample_data/knowledge.txt"))
    chunks = chunk_records(document, records, chunk_size=800, overlap=120)
    repository = Repository(f"sqlite:///{tmp_path / 'knowledge.db'}")
    service = KnowledgeService(Settings(database_url=f"sqlite:///{tmp_path / 'knowledge.db'}"), repository)
    repository.save_document(document, chunks)
    service.retriever = service.retriever.__class__(repository.all_chunks())
    results = service.search("TF-IDF retrieval")
    assert results
    assert any("TF-IDF" in result.chunk.content for result in results)


def test_duplicate_documents_are_not_reprocessed(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'knowledge.db'}"
    service = KnowledgeService(Settings(database_url=database_url), Repository(database_url))
    first, count, created = service.index_file("sample_data/knowledge.txt")
    second, duplicate_count, duplicate_created = service.index_file("sample_data/knowledge.txt")
    assert first.document_id == second.document_id
    assert count > 0 and created
    assert duplicate_count == 0 and not duplicate_created


def test_unsupported_files_are_rejected(tmp_path: Path):
    path = tmp_path / "secrets.exe"
    path.write_bytes(b"not a document")
    try:
        ingest_file(path)
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("unsupported file was accepted")


def test_answer_is_structured_for_readability(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'knowledge.db'}"
    service = KnowledgeService(Settings(database_url=database_url), Repository(database_url))
    service.index_file("sample_data/knowledge.txt")
    answer = service.answer("What retrieval method does the project use?")
    assert answer.direct_answer
    assert answer.explanation
    assert answer.recommendations
    assert answer.evidence_table
    assert "TF-IDF" in answer.direct_answer
    assert "## Direct answer" in answer.text
    assert "## Evidence table" in answer.text


def test_answer_can_be_scoped_to_one_document(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'knowledge.db'}"
    service = KnowledgeService(Settings(database_url=database_url), Repository(database_url))
    document, _, _ = service.index_file("sample_data/knowledge.txt")
    answer = service.answer("What retrieval method does the project use?", document_id=document.document_id)
    assert answer.sources
    assert all(source.chunk.document_id == document.document_id for source in answer.sources)
    assert "page" in answer.recommendations[0].lower() or "indexed" in answer.recommendations[0].lower()