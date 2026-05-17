"""
Shared fixtures for integration tests.

Provides:
- fake_embedder: deterministic BOW embedder (no model download needed)
- sample_pdf_path: generates a test PDF with PyMuPDF
- sample_vault_path: creates a mini Obsidian vault
- sample_transcript_json_path: creates a JSON session transcript
- populated_store: Store with all three tables pre-loaded with real chunks
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Deterministic BOW embedder
# ---------------------------------------------------------------------------

def _fake_embed(text: str, dim: int = 384) -> np.ndarray:
    """Produce a deterministic 384-dim embedding via word hashing.

    Words that appear in both query and document will hash to the same
    bucket, so cosine similarity is proportional to word overlap — making
    retrieval tests meaningful without any real model.
    """
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class FakeEmbedder:
    """Drop-in replacement for pocket_gm.ingestion.embedder.Embedder."""

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.stack([_fake_embed(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return _fake_embed(text)


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# PDF fixture
# ---------------------------------------------------------------------------

def _make_sample_pdf(path: Path) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()

    # Page 1: title + body
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (50, 60),
        "KINGMAKER — THE STOLEN LANDS",
        fontsize=18,
        color=(0, 0, 0),
    )
    page.insert_text(
        (50, 100),
        (
            "The Stolen Lands are a vast wilderness region south of Brevoy. "
            "The region is ruled by bandits, monsters, and worse. "
            "The players are tasked with exploring and civilising this territory."
        ),
        fontsize=10,
        color=(0, 0, 0),
    )

    # Page 2: Stag Lord
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (50, 60),
        "THE STAG LORD",
        fontsize=16,
        color=(0, 0, 0),
    )
    page.insert_text(
        (50, 95),
        (
            "The Stag Lord is the most powerful bandit lord in the Stolen Lands. "
            "He rules from a fortified keep on the banks of the Tuskwater lake. "
            "His greatest weakness is his severe alcoholism, which makes him "
            "unpredictable and dangerous even to his own men. "
            "He wears an enchanted stag skull helm that grants him magical abilities."
        ),
        fontsize=10,
        color=(0, 0, 0),
    )

    # Page 3: Oleg's Trading Post
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (50, 60),
        "OLEG'S TRADING POST",
        fontsize=14,
        color=(0, 0, 0),
    )
    page.insert_text(
        (50, 95),
        (
            "Oleg Leveton runs a trading post on the northern border of the Stolen Lands. "
            "He and his wife Svetlana are being extorted by a band of bandits. "
            "Oleg has asked the players for help."
        ),
        fontsize=10,
        color=(0, 0, 0),
    )

    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fixtures_dir / "sample.pdf"
    if not pdf_path.exists():
        _make_sample_pdf(pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Obsidian vault fixture
# ---------------------------------------------------------------------------

_STAG_LORD_MD = """\
---
tags: [npc, villain, bandit]
aliases: [The Stag Lord, Stag Lord]
---
# Stag Lord
The Stag Lord is the main villain of the Kingmaker campaign.
He controls [[Tuskwater Keep]] from the south.
See also: [[Akiros Ismort]]
"""

_AKIROS_MD = """\
# Akiros Ismort
Akiros was the Stag Lord's lieutenant who secretly despised his master.
He surrendered to the players during the final assault.
"""

_TUSKWATER_MD = """\
# Tuskwater Keep
The Stag Lord's fortress on the banks of the Tuskwater lake.
It houses twenty bandits and a dungeon.
"""


@pytest.fixture(scope="session")
def sample_vault_path(tmp_path_factory) -> Path:
    fixtures_dir = Path(__file__).parent / "fixtures"
    vault_dir = fixtures_dir / "sample_vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    (vault_dir / "Stag Lord.md").write_text(_STAG_LORD_MD, encoding="utf-8")
    (vault_dir / "Akiros Ismort.md").write_text(_AKIROS_MD, encoding="utf-8")
    (vault_dir / "Tuskwater Keep.md").write_text(_TUSKWATER_MD, encoding="utf-8")

    return vault_dir


# ---------------------------------------------------------------------------
# Transcript JSON fixture
# ---------------------------------------------------------------------------

_TRANSCRIPT_SEGMENTS = [
    {"start": 0.0,  "end": 10.5, "text": "The party arrived at the Stag Lord's fort.",                          "speaker": "DM"},
    {"start": 10.5, "end": 25.0, "text": "We sneak around the eastern wall to avoid the guards.",              "speaker": "Player"},
    {"start": 25.0, "end": 45.0, "text": "Akiros Ismort opens the gate for the party and surrenders immediately.", "speaker": "DM"},
    {"start": 45.0, "end": 60.0, "text": "The Stag Lord confronts the party on the battlements. He is clearly drunk.", "speaker": "DM"},
]


@pytest.fixture(scope="session")
def sample_transcript_json_path(tmp_path_factory) -> Path:
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = fixtures_dir / "sample_transcript.json"
    transcript_path.write_text(json.dumps(_TRANSCRIPT_SEGMENTS, indent=2), encoding="utf-8")
    return transcript_path


# ---------------------------------------------------------------------------
# Populated store fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def populated_store(
    tmp_path_factory,
    fake_embedder,
    sample_pdf_path,
    sample_vault_path,
    sample_transcript_json_path,
):
    """A Store with sourcebook, notes, and sessions tables populated from fixture data."""
    from pocket_gm.ingestion.chunker import (
        chunk_obsidian_note,
        chunk_pdf_pages,
        chunk_transcript,
    )
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
    from pocket_gm.ingestion.pdf_loader import load_pdf
    from pocket_gm.retrieval.store import (
        Store,
        notes_table,
        sessions_table,
        sourcebook_table,
    )

    db_dir = tmp_path_factory.mktemp("store")
    store = Store(db_dir / "vectors.db")
    campaign_id = "kingmaker_test"

    # --- sourcebook (PDF) ---
    pdf_pages = load_pdf(sample_pdf_path)
    pdf_chunks = chunk_pdf_pages(pdf_pages, chunk_size=128, chunk_overlap=16)
    pdf_dicts = [
        {
            "text": c.text,
            "source_type": c.source_type,
            "filename": c.filename,
            "page": c.page,
            "heading": c.heading,
            "chunk_index": c.chunk_index,
            "session_number": 0,
            "session_date": "",
            "timestamp_start": 0.0,
        }
        for c in pdf_chunks
    ]
    pdf_embeddings = fake_embedder.embed([d["text"] for d in pdf_dicts])
    store.add_documents(sourcebook_table(campaign_id), pdf_dicts, pdf_embeddings, campaign_id)

    # --- notes (Obsidian vault) ---
    notes = load_obsidian_vault(sample_vault_path)
    notes_chunks_all = []
    for note in notes:
        notes_chunks_all.extend(chunk_obsidian_note(note, chunk_size=128, chunk_overlap=16))

    notes_dicts = [
        {
            "text": c.text,
            "source_type": c.source_type,
            "filename": c.filename,
            "page": 0,
            "heading": c.heading,
            "chunk_index": c.chunk_index,
            "session_number": 0,
            "session_date": "",
            "timestamp_start": 0.0,
        }
        for c in notes_chunks_all
    ]
    notes_embeddings = fake_embedder.embed([d["text"] for d in notes_dicts])
    store.add_documents(notes_table(campaign_id), notes_dicts, notes_embeddings, campaign_id)

    # --- sessions (transcript JSON) ---
    raw_segments = json.loads(sample_transcript_json_path.read_text())
    full_text = " ".join(seg["text"] for seg in raw_segments)
    session_chunks = chunk_transcript(
        full_text,
        filename="sample_transcript.json",
        session_number=1,
        session_date="2025-01-01",
        chunk_size=64,
        chunk_overlap=8,
        timestamps=raw_segments,
    )
    # Add a second session so list_sessions can be tested
    session2_chunks = chunk_transcript(
        "The party levelled up after defeating the Stag Lord.",
        filename="session2.json",
        session_number=2,
        session_date="2025-01-08",
        chunk_size=64,
        chunk_overlap=8,
    )
    all_session_chunks = session_chunks + session2_chunks
    session_embeddings = fake_embedder.embed([d["text"] for d in all_session_chunks])
    store.add_documents(sessions_table(campaign_id), all_session_chunks, session_embeddings, campaign_id)

    return store, campaign_id
