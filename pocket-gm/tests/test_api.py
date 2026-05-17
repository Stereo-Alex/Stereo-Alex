from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def _tmp_cfg(tmp_path):
    from pocket_gm.core.config import Config
    cfg = Config(
        campaigns_dir=tmp_path / "campaigns",
        lancedb_path=tmp_path / "vectors.db",
        logs_path=tmp_path / "logs",
    )
    cfg.campaigns_dir.mkdir(parents=True)
    return cfg


@pytest.fixture()
def _campaign(_tmp_cfg):
    from pocket_gm.core.campaign import Campaign, add_campaign
    c = Campaign(id="test", name="Test Campaign", created_at="2024-01-01")
    add_campaign(_tmp_cfg.campaigns_dir, c)
    return c


@pytest.fixture()
def client(_tmp_cfg, _campaign):
    from pocket_gm.api import app as app_mod
    from unittest.mock import MagicMock

    app_mod._cfg = None
    app_mod._embedder = None
    app_mod._store = None

    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = np.zeros(384)

    with patch("pocket_gm.api.app.load_config", return_value=_tmp_cfg), \
         patch("pocket_gm.api.app.Embedder", return_value=mock_embedder):
        yield TestClient(app_mod.app)

    app_mod._cfg = None
    app_mod._embedder = None
    app_mod._store = None


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ollama_available(client):
    with patch("pocket_gm.api.app.OllamaClient") as mock_cls:
        mock_cls.return_value.is_available.return_value = True
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["ollama_available"] is True


def test_health_ollama_unavailable(client):
    with patch("pocket_gm.api.app.OllamaClient") as mock_cls:
        mock_cls.return_value.is_available.return_value = False
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ollama_available"] is False


# ── /campaigns ────────────────────────────────────────────────────────────────

def test_list_campaigns(client, _campaign):
    resp = client.get("/campaigns")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "test"
    assert data[0]["name"] == "Test Campaign"


def test_list_campaigns_empty(_tmp_cfg):
    from unittest.mock import MagicMock
    from pocket_gm.api import app as app_mod
    app_mod._cfg = None
    app_mod._embedder = None
    app_mod._store = None

    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = np.zeros(384)

    with patch("pocket_gm.api.app.load_config", return_value=_tmp_cfg), \
         patch("pocket_gm.api.app.Embedder", return_value=mock_embedder):
        c = TestClient(app_mod.app)
        resp = c.get("/campaigns")

    assert resp.status_code == 200
    assert resp.json() == []


# ── /campaigns/{id}/status ────────────────────────────────────────────────────

def test_status_zero_chunks(client):
    resp = client.get("/campaigns/test/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["campaign_id"] == "test"
    assert body["sourcebook_chunks"] == 0
    assert body["notes_chunks"] == 0
    assert body["sessions_chunks"] == 0


def test_status_unknown_campaign(client):
    resp = client.get("/campaigns/nope/status")
    assert resp.status_code == 404


# ── /campaigns/{id}/sessions ──────────────────────────────────────────────────

def test_sessions_empty(client):
    resp = client.get("/campaigns/test/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_sessions_unknown_campaign(client):
    resp = client.get("/campaigns/nope/sessions")
    assert resp.status_code == 404


def test_sessions_with_data(client, _tmp_cfg):
    from pocket_gm.retrieval.store import Store, sessions_table

    store = Store(_tmp_cfg.lancedb_path)
    table = sessions_table("test")
    chunks = [
        {"text": "We fought the Stag Lord", "source_type": "transcript",
         "filename": "session_01.txt", "page": 0, "heading": "", "chunk_index": 0,
         "session_number": 1, "session_date": "2024-01-15", "timestamp_start": 0.0},
        {"text": "The party rested at Oleg's", "source_type": "transcript",
         "filename": "session_02.txt", "page": 0, "heading": "", "chunk_index": 0,
         "session_number": 2, "session_date": "2024-02-10", "timestamp_start": 60.0},
    ]
    embeddings = np.random.rand(2, 384).astype(np.float32)
    store.add_documents(table, chunks, embeddings, "test")

    resp = client.get("/campaigns/test/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["session_number"] == 1
    assert data[0]["session_date"] == "2024-01-15"
    assert data[1]["session_number"] == 2


# ── POST /query ───────────────────────────────────────────────────────────────

def _make_chunk(text="Stag Lord is a bandit lord", score=0.8, source_type="pdf",
                filename="kingmaker.pdf", page=42, heading="Chapter 1",
                session_number=0, session_date="", timestamp_start=0.0):
    from pocket_gm.retrieval.store import RetrievedChunk
    return RetrievedChunk(
        text=text, score=score, source_type=source_type,
        filename=filename, page=page, heading=heading,
        chunk_index=0, campaign_id="test",
        session_number=session_number, session_date=session_date,
        timestamp_start=timestamp_start,
    )


def test_query_no_relevant_chunks(client):
    with patch("pocket_gm.api.app.query_all_sync") as mock_q, \
         patch("pocket_gm.api.app.Embedder") as mock_emb:
        mock_emb.return_value.embed_one.return_value = np.zeros(384)
        from pocket_gm.retrieval.router import QueryResult
        mock_q.return_value = QueryResult(sourcebook=[], notes=[], sessions=[])

        resp = client.post("/query", json={"question": "Who is nobody?", "campaign_id": "test"})

    assert resp.status_code == 200
    body = resp.json()
    assert "No relevant information" in body["answer"]
    assert body["citations"] == []


def test_query_with_answer(client):
    chunk = _make_chunk()

    with patch("pocket_gm.api.app.query_all_sync") as mock_q, \
         patch("pocket_gm.api.app.Embedder") as mock_emb, \
         patch("pocket_gm.api.app.OllamaClient") as mock_ollama, \
         patch("pocket_gm.api.app.build_prompt") as mock_prompt:

        mock_emb.return_value.embed_one.return_value = np.zeros(384)
        from pocket_gm.retrieval.router import QueryResult
        mock_q.return_value = QueryResult(sourcebook=[chunk], notes=[], sessions=[])

        index_map = [(1, chunk)]
        mock_prompt.return_value = ("fake prompt", index_map)
        mock_ollama.return_value.is_available.return_value = True
        mock_ollama.return_value.generate.return_value = "The Stag Lord is a bandit lord [1]."

        resp = client.post("/query", json={"question": "Who is the Stag Lord?", "campaign_id": "test"})

    assert resp.status_code == 200
    body = resp.json()
    assert "Stag Lord" in body["answer"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["ref"] == 1
    assert body["citations"][0]["filename"] == "kingmaker.pdf"
    assert body["citations"][0]["page"] == 42


def test_query_ollama_unavailable(client):
    chunk = _make_chunk()

    with patch("pocket_gm.api.app.query_all_sync") as mock_q, \
         patch("pocket_gm.api.app.Embedder") as mock_emb, \
         patch("pocket_gm.api.app.OllamaClient") as mock_ollama, \
         patch("pocket_gm.api.app.build_prompt") as mock_prompt:

        mock_emb.return_value.embed_one.return_value = np.zeros(384)
        from pocket_gm.retrieval.router import QueryResult
        mock_q.return_value = QueryResult(sourcebook=[chunk], notes=[], sessions=[])
        mock_prompt.return_value = ("fake prompt", [(1, chunk)])
        mock_ollama.return_value.is_available.return_value = False

        resp = client.post("/query", json={"question": "Who?", "campaign_id": "test"})

    assert resp.status_code == 503


def test_query_unknown_campaign(client):
    resp = client.post("/query", json={"question": "Who?", "campaign_id": "nope"})
    assert resp.status_code == 404


def test_query_invalid_empty_question(client):
    resp = client.post("/query", json={"question": "", "campaign_id": "test"})
    assert resp.status_code == 422


def test_query_top_k_passed_through(client):
    chunk = _make_chunk()

    with patch("pocket_gm.api.app.query_all_sync") as mock_q, \
         patch("pocket_gm.api.app.Embedder") as mock_emb, \
         patch("pocket_gm.api.app.OllamaClient") as mock_ollama, \
         patch("pocket_gm.api.app.build_prompt") as mock_prompt:

        mock_emb.return_value.embed_one.return_value = np.zeros(384)
        from pocket_gm.retrieval.router import QueryResult
        mock_q.return_value = QueryResult(sourcebook=[chunk], notes=[], sessions=[])
        mock_prompt.return_value = ("fake prompt", [(1, chunk)])
        mock_ollama.return_value.is_available.return_value = True
        mock_ollama.return_value.generate.return_value = "Answer [1]."

        client.post("/query", json={"question": "Who?", "campaign_id": "test", "top_k": 10})

    args = mock_q.call_args
    # top_k is either the 4th positional arg or a kwarg
    top_k_val = args[1].get("top_k") if args[1] else args[0][3]
    assert top_k_val == 10


def test_query_fully_grounded_flag(client):
    chunk = _make_chunk()

    with patch("pocket_gm.api.app.query_all_sync") as mock_q, \
         patch("pocket_gm.api.app.Embedder") as mock_emb, \
         patch("pocket_gm.api.app.OllamaClient") as mock_ollama, \
         patch("pocket_gm.api.app.build_prompt") as mock_prompt:

        mock_emb.return_value.embed_one.return_value = np.zeros(384)
        from pocket_gm.retrieval.router import QueryResult
        mock_q.return_value = QueryResult(sourcebook=[chunk], notes=[], sessions=[])
        mock_prompt.return_value = ("fake prompt", [(1, chunk)])
        mock_ollama.return_value.is_available.return_value = True
        # Every sentence has a citation → fully grounded
        mock_ollama.return_value.generate.return_value = "The Stag Lord is a bandit [1]."

        resp = client.post("/query", json={"question": "Who?", "campaign_id": "test"})

    assert resp.json()["fully_grounded"] is True
    assert resp.json()["uncited_count"] == 0


# ── store.list_sessions ───────────────────────────────────────────────────────

def test_list_sessions_empty_table(tmp_path):
    from pocket_gm.retrieval.store import Store, sessions_table
    store = Store(tmp_path / "v.db")
    assert store.list_sessions(sessions_table("x")) == []


def test_list_sessions_ordered(tmp_path):
    from pocket_gm.retrieval.store import Store, sessions_table

    store = Store(tmp_path / "v.db")
    table = sessions_table("camp")
    chunks = [
        {"text": "foo", "source_type": "transcript", "filename": "s2.txt",
         "page": 0, "heading": "", "chunk_index": 0,
         "session_number": 2, "session_date": "2024-02-01", "timestamp_start": 0.0},
        {"text": "bar", "source_type": "transcript", "filename": "s1.txt",
         "page": 0, "heading": "", "chunk_index": 0,
         "session_number": 1, "session_date": "2024-01-01", "timestamp_start": 0.0},
        {"text": "baz", "source_type": "transcript", "filename": "s1.txt",
         "page": 0, "heading": "", "chunk_index": 1,
         "session_number": 1, "session_date": "2024-01-01", "timestamp_start": 30.0},
    ]
    embeddings = np.random.rand(3, 384).astype(np.float32)
    store.add_documents(table, chunks, embeddings, "camp")

    sessions = store.list_sessions(table)
    assert len(sessions) == 2
    assert sessions[0]["session_number"] == 1
    assert sessions[0]["chunk_count"] == 2
    assert sessions[1]["session_number"] == 2
    assert sessions[1]["chunk_count"] == 1
