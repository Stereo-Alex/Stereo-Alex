import numpy as np
import pytest
from pathlib import Path
from pocket_gm.retrieval.store import Store, sourcebook_table, notes_table, sessions_table


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "lancedb")


def make_embedding(n: int = 384) -> np.ndarray:
    v = np.random.rand(n).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_and_query(store):
    table = sourcebook_table("kingmaker")
    chunks = [{"text": "The Stag Lord is a bandit warlord.", "source_type": "pdf",
                "filename": "kb.pdf", "page": 42, "heading": "Chapter 3", "chunk_index": 0}]
    emb = np.array([make_embedding()])
    store.add_documents(table, chunks, emb, "kingmaker")

    query_vec = make_embedding()
    results = store.query(table, query_vec, top_k=1)
    assert len(results) == 1
    assert results[0].text == "The Stag Lord is a bandit warlord."
    assert results[0].campaign_id == "kingmaker"
    assert 0.0 <= results[0].score <= 1.0


def test_query_empty_table_returns_empty(store):
    results = store.query(sourcebook_table("nonexistent"), make_embedding(), top_k=5)
    assert results == []


def test_table_exists(store):
    table = notes_table("kingmaker")
    assert not store.table_exists(table)
    chunks = [{"text": "My notes.", "source_type": "markdown", "filename": "notes.md",
                "page": 0, "heading": "", "chunk_index": 0}]
    store.add_documents(table, chunks, np.array([make_embedding()]), "kingmaker")
    assert store.table_exists(table)


def test_drop_table(store):
    table = sessions_table("kingmaker")
    chunks = [{"text": "Session text.", "source_type": "transcript", "filename": "s1.txt",
                "page": 0, "heading": "", "chunk_index": 0, "session_number": 1,
                "session_date": "2025-01-01", "timestamp_start": 0.0}]
    store.add_documents(table, chunks, np.array([make_embedding()]), "kingmaker")
    store.drop_table(table)
    assert not store.table_exists(table)


def test_add_appends_to_existing_table(store):
    table = sourcebook_table("test")
    emb = np.array([make_embedding()])
    chunks1 = [{"text": "First chunk.", "source_type": "pdf", "filename": "a.pdf", "page": 1, "heading": "", "chunk_index": 0}]
    chunks2 = [{"text": "Second chunk.", "source_type": "pdf", "filename": "a.pdf", "page": 2, "heading": "", "chunk_index": 1}]
    store.add_documents(table, chunks1, emb, "test")
    store.add_documents(table, chunks2, emb, "test")
    results = store.query(table, make_embedding(), top_k=10)
    assert len(results) == 2


def test_delete_by_filename_removes_only_that_file(store):
    table = sourcebook_table("test")
    emb = np.array([make_embedding()])
    store.add_documents(table, [{"text": "From A.", "source_type": "pdf", "filename": "a.pdf",
                                 "page": 1, "heading": "", "chunk_index": 0}], emb, "test")
    store.add_documents(table, [{"text": "From B.", "source_type": "pdf", "filename": "b.pdf",
                                 "page": 1, "heading": "", "chunk_index": 0}], emb, "test")
    assert store.count(table) == 2

    removed = store.delete_by_filename(table, "a.pdf")
    assert removed == 1
    assert store.count(table) == 1
    # The surviving row is b.pdf, and it is still queryable.
    results = store.query(table, make_embedding(), top_k=10)
    assert len(results) == 1
    assert results[0].filename == "b.pdf"


def test_delete_by_filename_missing_table_is_noop(store):
    assert store.delete_by_filename(sourcebook_table("ghost"), "x.pdf") == 0


def test_delete_by_filename_unknown_file_is_noop(store):
    table = notes_table("test")
    store.add_documents(table, [{"text": "note", "source_type": "markdown", "filename": "n.md",
                                 "page": 0, "heading": "", "chunk_index": 0}],
                        np.array([make_embedding()]), "test")
    assert store.delete_by_filename(table, "absent.md") == 0
    assert store.count(table) == 1
