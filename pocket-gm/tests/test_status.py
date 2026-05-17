from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pocket_gm.retrieval.store import Store, notes_table, sourcebook_table, sessions_table
from pocket_gm.cli.commands.status import (
    _get_chunk_counts,
    _get_ingested_files,
    _get_last_query,
)
from pocket_gm.core.ingest_registry import get_registry_path, mark_ingested


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "lancedb")


def make_embedding(n: int = 384) -> np.ndarray:
    v = np.random.rand(n).astype(np.float32)
    return v / np.linalg.norm(v)


def add_chunks(store: Store, table_name: str, campaign_id: str, n: int = 3) -> None:
    chunks = [
        {
            "text": f"Chunk {i}",
            "source_type": "pdf",
            "filename": "book.pdf",
            "page": i,
            "heading": "",
            "chunk_index": i,
        }
        for i in range(n)
    ]
    embeddings = np.array([make_embedding() for _ in range(n)])
    store.add_documents(table_name, chunks, embeddings, campaign_id)


# ── count() ───────────────────────────────────────────────────────────────────

def test_count_returns_zero_for_nonexistent_table(store):
    assert store.count(sourcebook_table("no_such_campaign")) == 0


def test_count_returns_correct_count(store):
    table = sourcebook_table("kingmaker")
    add_chunks(store, table, "kingmaker", n=5)
    assert store.count(table) == 5


def test_count_accumulates_across_adds(store):
    table = notes_table("kingmaker")
    add_chunks(store, table, "kingmaker", n=3)
    add_chunks(store, table, "kingmaker", n=2)
    assert store.count(table) == 5


def test_count_zero_after_drop(store):
    table = sessions_table("test")
    add_chunks(store, table, "test", n=4)
    assert store.count(table) == 4
    store.drop_table(table)
    assert store.count(table) == 0


# ── _get_chunk_counts ─────────────────────────────────────────────────────────

def test_get_chunk_counts_all_zero_when_empty(store):
    counts = _get_chunk_counts(store, "empty_campaign")
    assert counts == {"Sourcebook": 0, "GM Notes": 0, "Sessions": 0}


def test_get_chunk_counts_reflects_added_docs(store):
    campaign_id = "kingmaker"
    add_chunks(store, sourcebook_table(campaign_id), campaign_id, n=10)
    add_chunks(store, notes_table(campaign_id), campaign_id, n=4)

    counts = _get_chunk_counts(store, campaign_id)
    assert counts["Sourcebook"] == 10
    assert counts["GM Notes"] == 4
    assert counts["Sessions"] == 0


# ── _get_ingested_files ───────────────────────────────────────────────────────

def test_get_ingested_files_empty_when_no_registry(tmp_path):
    result = _get_ingested_files(tmp_path / "campaigns", "kingmaker")
    assert result == {}


def test_get_ingested_files_groups_by_extension(tmp_path):
    campaigns_dir = tmp_path / "campaigns"
    rp = get_registry_path(campaigns_dir, "kingmaker")
    mark_ingested(rp, "hash1", "sourcebook.pdf", 100)
    mark_ingested(rp, "hash2", "notes.md", 50)
    mark_ingested(rp, "hash3", "worldbuilding.txt", 30)

    result = _get_ingested_files(campaigns_dir, "kingmaker")
    assert "sourcebook" in result
    assert "sourcebook.pdf" in result["sourcebook"]
    assert "notes" in result
    assert set(result["notes"]) == {"notes.md", "worldbuilding.txt"}


# ── _get_last_query ───────────────────────────────────────────────────────────

def test_get_last_query_returns_none_when_no_log(tmp_path):
    result = _get_last_query(tmp_path / "logs", "kingmaker")
    assert result is None


def test_get_last_query_returns_none_when_no_matching_campaign(tmp_path):
    logs_path = tmp_path / "logs"
    logs_path.mkdir()
    log_file = logs_path / "queries.ndjson"
    record = {
        "timestamp": "2025-03-15T19:30:00+00:00",
        "campaign_id": "other_campaign",
        "question": "Who is the Stag Lord?",
    }
    log_file.write_text(json.dumps(record) + "\n")

    result = _get_last_query(logs_path, "kingmaker")
    assert result is None


def test_get_last_query_returns_last_matching_entry(tmp_path):
    logs_path = tmp_path / "logs"
    logs_path.mkdir()
    log_file = logs_path / "queries.ndjson"

    records = [
        {
            "timestamp": "2025-03-10T10:00:00+00:00",
            "campaign_id": "kingmaker",
            "question": "First question",
        },
        {
            "timestamp": "2025-03-15T19:30:00+00:00",
            "campaign_id": "kingmaker",
            "question": "Who is the Stag Lord?",
        },
        {
            "timestamp": "2025-03-16T08:00:00+00:00",
            "campaign_id": "other",
            "question": "Unrelated",
        },
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    result = _get_last_query(logs_path, "kingmaker")
    assert result is not None
    assert "Who is the Stag Lord?" in result
    assert "2025-03-15" in result


# ── status command smoke test ─────────────────────────────────────────────────

def test_status_command_no_crash_when_empty(tmp_path):
    """Invoke the status command helpers against an empty store without crashing."""
    store = Store(tmp_path / "lancedb")
    campaigns_dir = tmp_path / "campaigns"
    logs_path = tmp_path / "logs"

    counts = _get_chunk_counts(store, "kingmaker")
    ingested = _get_ingested_files(campaigns_dir, "kingmaker")
    last = _get_last_query(logs_path, "kingmaker")

    assert all(v == 0 for v in counts.values())
    assert ingested == {}
    assert last is None
