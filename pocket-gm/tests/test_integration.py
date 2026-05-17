"""
Integration tests for the pocket-gm RAG pipeline.

Uses real code paths with fixture data (PDFs, Obsidian vault, transcript JSON)
and a deterministic fake embedder — no mocks except where the LLM is required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# PDF loading tests
# ---------------------------------------------------------------------------


def test_pdf_loads_all_pages(sample_pdf_path):
    from pocket_gm.ingestion.pdf_loader import load_pdf

    pages = load_pdf(sample_pdf_path)
    assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
    for i, page in enumerate(pages, start=1):
        assert hasattr(page, "text"), "RawChunk should have a 'text' field"
        assert page.page == i, f"Expected page_num {i}, got {page.page}"
        assert page.text.strip(), f"Page {i} text should not be empty"


def test_pdf_heading_detection(sample_pdf_path):
    from pocket_gm.ingestion.pdf_loader import load_pdf

    pages = load_pdf(sample_pdf_path)
    # Page 2 is the Stag Lord page
    stag_lord_page = pages[1]
    # The heading should have been detected on this page
    assert "STAG LORD" in stag_lord_page.heading.upper(), (
        f"Expected 'STAG LORD' in heading, got: {stag_lord_page.heading!r}"
    )


def test_pdf_chunks_have_metadata(sample_pdf_path):
    from pocket_gm.ingestion.chunker import chunk_pdf_pages
    from pocket_gm.ingestion.pdf_loader import load_pdf

    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pdf_pages(pages)

    assert chunks, "chunk_pdf_pages should return at least one chunk"
    for chunk in chunks:
        assert chunk.filename == sample_pdf_path.name, (
            f"Expected filename {sample_pdf_path.name!r}, got {chunk.filename!r}"
        )
        assert chunk.page in (1, 2, 3), f"Unexpected page number: {chunk.page}"
        assert chunk.source_type == "pdf", f"Expected source_type='pdf', got {chunk.source_type!r}"
        assert isinstance(chunk.chunk_index, int), "chunk_index should be an int"
        # heading is a string (may be empty for pages before first heading)
        assert isinstance(chunk.heading, str)


def test_pdf_heading_prefix_injected(sample_pdf_path):
    from pocket_gm.ingestion.chunker import chunk_pdf_pages
    from pocket_gm.ingestion.pdf_loader import load_pdf

    pages = load_pdf(sample_pdf_path)
    chunks = chunk_pdf_pages(pages)

    # Find chunks from page 2 (Stag Lord page)
    stag_lord_chunks = [c for c in chunks if c.page == 2]
    assert stag_lord_chunks, "Should have chunks from page 2 (Stag Lord)"

    # The heading prefix should be injected into chunk text
    for chunk in stag_lord_chunks:
        assert "STAG LORD" in chunk.text.upper(), (
            f"Expected 'STAG LORD' heading prefix in chunk text, got: {chunk.text[:80]!r}"
        )


# ---------------------------------------------------------------------------
# Markdown / notes tests
# ---------------------------------------------------------------------------


def test_markdown_chunks_have_headings():
    from pocket_gm.ingestion.chunker import chunk_markdown

    md_text = """# Introduction
This is the introduction section about the Stag Lord.

## Background
The Stag Lord came to power in the Stolen Lands.

## Abilities
He has many magical abilities from his helm.
"""
    chunks = chunk_markdown(md_text, filename="test_note.md")
    assert chunks, "chunk_markdown should produce chunks"

    # Each chunk should have a heading (section it belongs to)
    headings_found = {c.heading for c in chunks}
    assert "Introduction" in headings_found or any("Introduction" in h for h in headings_found), (
        f"Expected 'Introduction' heading among chunks. Found: {headings_found}"
    )
    assert "Background" in headings_found or any("Background" in h for h in headings_found), (
        f"Expected 'Background' heading among chunks. Found: {headings_found}"
    )


def test_markdown_chunks_source_type():
    from pocket_gm.ingestion.chunker import chunk_markdown

    md_text = "# A heading\nSome content about something important.\n"
    chunks = chunk_markdown(md_text, filename="test.md")
    assert chunks, "Should produce at least one chunk"
    for chunk in chunks:
        assert chunk.source_type == "markdown", (
            f"Expected source_type='markdown', got {chunk.source_type!r}"
        )


# ---------------------------------------------------------------------------
# Obsidian vault tests
# ---------------------------------------------------------------------------


def test_obsidian_vault_loads_all_notes(sample_vault_path):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault

    notes = load_obsidian_vault(sample_vault_path)
    assert len(notes) == 3, f"Expected 3 notes, got {len(notes)}: {[n.title for n in notes]}"


def test_obsidian_wikilinks_resolved(sample_vault_path):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault

    notes = load_obsidian_vault(sample_vault_path)
    stag_lord_note = next((n for n in notes if "Stag Lord" in n.title), None)
    assert stag_lord_note is not None, "Could not find Stag Lord note"

    # [[Tuskwater Keep]] should be resolved to its display name in the body
    assert "Tuskwater Keep" in stag_lord_note.body, (
        f"Expected 'Tuskwater Keep' in body after wikilink resolution. "
        f"Body preview: {stag_lord_note.body[:200]!r}"
    )


def test_obsidian_frontmatter_tags(sample_vault_path):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault

    notes = load_obsidian_vault(sample_vault_path)
    stag_lord_note = next((n for n in notes if "Stag Lord" in n.title), None)
    assert stag_lord_note is not None, "Could not find Stag Lord note"

    assert stag_lord_note.tags == ["npc", "villain", "bandit"], (
        f"Expected tags ['npc', 'villain', 'bandit'], got {stag_lord_note.tags}"
    )


def test_obsidian_chunk_includes_title(sample_vault_path):
    from pocket_gm.ingestion.chunker import chunk_obsidian_note
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault

    notes = load_obsidian_vault(sample_vault_path)
    stag_lord_note = next((n for n in notes if "Stag Lord" in n.title), None)
    assert stag_lord_note is not None, "Could not find Stag Lord note"

    chunks = chunk_obsidian_note(stag_lord_note)
    assert chunks, "chunk_obsidian_note should produce chunks"

    # The title "Stag Lord" should appear in at least one chunk's text
    has_title = any("Stag Lord" in c.text for c in chunks)
    assert has_title, (
        f"Expected 'Stag Lord' in at least one chunk. "
        f"Chunk texts: {[c.text[:60] for c in chunks]}"
    )


# ---------------------------------------------------------------------------
# Transcript tests
# ---------------------------------------------------------------------------


def test_transcript_chunks_have_timestamps(sample_transcript_json_path):
    from pocket_gm.ingestion.chunker import chunk_transcript

    segments = json.loads(sample_transcript_json_path.read_text())
    full_text = " ".join(seg["text"] for seg in segments)

    chunks = chunk_transcript(
        full_text,
        filename="sample_transcript.json",
        session_number=1,
        session_date="2025-01-01",
        chunk_size=32,
        chunk_overlap=4,
        timestamps=segments,
    )

    assert chunks, "chunk_transcript should produce chunks"
    for chunk in chunks:
        assert "timestamp_start" in chunk, f"Missing 'timestamp_start' key in chunk: {chunk.keys()}"
        assert "session_number" in chunk, f"Missing 'session_number' key in chunk"
        assert "session_date" in chunk, f"Missing 'session_date' key in chunk"
        assert chunk["session_number"] == 1
        assert chunk["session_date"] == "2025-01-01"
        assert isinstance(chunk["timestamp_start"], float)


# ---------------------------------------------------------------------------
# Store (sqlite-vec) tests
# ---------------------------------------------------------------------------


def test_store_add_and_count(fake_embedder, tmp_path):
    from pocket_gm.retrieval.store import Store, sourcebook_table

    store = Store(tmp_path / "vectors.db")
    table = sourcebook_table("test_campaign")

    chunks = [
        {"text": "The Stag Lord is a bandit.", "source_type": "pdf",
         "filename": "book.pdf", "page": 1, "heading": "Chapter 1", "chunk_index": 0,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0},
        {"text": "He rules the Stolen Lands.", "source_type": "pdf",
         "filename": "book.pdf", "page": 2, "heading": "Chapter 1", "chunk_index": 1,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0},
    ]
    embeddings = fake_embedder.embed([c["text"] for c in chunks])
    store.add_documents(table, chunks, embeddings, "test_campaign")

    assert store.count(table) == 2, f"Expected 2 chunks, got {store.count(table)}"


def test_store_query_returns_relevant_chunk(fake_embedder, tmp_path):
    from pocket_gm.retrieval.store import Store, sourcebook_table

    store = Store(tmp_path / "vectors.db")
    table = sourcebook_table("test_campaign")

    chunks = [
        {"text": "The Stag Lord is the most powerful bandit lord in the Stolen Lands.",
         "source_type": "pdf", "filename": "book.pdf", "page": 1,
         "heading": "THE STAG LORD", "chunk_index": 0,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0},
        {"text": "Oleg runs a trading post on the northern border.",
         "source_type": "pdf", "filename": "book.pdf", "page": 3,
         "heading": "TRADING POST", "chunk_index": 1,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0},
    ]
    embeddings = fake_embedder.embed([c["text"] for c in chunks])
    store.add_documents(table, chunks, embeddings, "test_campaign")

    query_vec = fake_embedder.embed_one("stag lord bandit")
    results = store.query(table, query_vec, top_k=5)

    assert results, "Query should return at least one result"
    # The Stag Lord chunk should appear in the top results
    texts = [r.text for r in results]
    assert any("Stag Lord" in t or "bandit lord" in t for t in texts), (
        f"Expected Stag Lord chunk in top results. Got: {texts}"
    )


def test_store_query_irrelevant_returns_low_score(fake_embedder, tmp_path):
    from pocket_gm.retrieval.store import Store, sourcebook_table

    store = Store(tmp_path / "vectors.db")
    table = sourcebook_table("test_campaign")

    chunks = [
        {"text": "The Stag Lord is the most powerful bandit lord.",
         "source_type": "pdf", "filename": "book.pdf", "page": 1,
         "heading": "THE STAG LORD", "chunk_index": 0,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0},
    ]
    embeddings = fake_embedder.embed([c["text"] for c in chunks])
    store.add_documents(table, chunks, embeddings, "test_campaign")

    # A very unrelated query should yield a low similarity score
    query_vec = fake_embedder.embed_one("french cuisine restaurants paris croissant wine")
    results = store.query(table, query_vec, top_k=5)

    if results:
        best_score = max(r.score for r in results)
        assert best_score < 0.3, (
            f"Expected low score (<0.3) for irrelevant query, got {best_score:.4f}"
        )
    # Empty result is also acceptable for an irrelevant query


def test_store_drop_table(fake_embedder, tmp_path):
    from pocket_gm.retrieval.store import Store, sourcebook_table

    store = Store(tmp_path / "vectors.db")
    table = sourcebook_table("test_campaign")

    chunks = [
        {"text": "Some text.", "source_type": "pdf", "filename": "f.pdf",
         "page": 1, "heading": "", "chunk_index": 0,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0},
    ]
    embeddings = fake_embedder.embed([c["text"] for c in chunks])
    store.add_documents(table, chunks, embeddings, "test_campaign")
    assert store.count(table) == 1

    store.drop_table(table)
    assert store.count(table) == 0, "After drop_table, count should be 0"


def test_store_list_sessions(populated_store):
    from pocket_gm.retrieval.store import sessions_table

    store, campaign_id = populated_store
    sessions = store.list_sessions(sessions_table(campaign_id))

    session_numbers = {s["session_number"] for s in sessions}
    assert 1 in session_numbers, f"Session 1 not found. Sessions: {sessions}"
    assert 2 in session_numbers, f"Session 2 not found. Sessions: {sessions}"


# ---------------------------------------------------------------------------
# Router tests (parallel query)
# ---------------------------------------------------------------------------


def test_router_queries_all_stores(populated_store, fake_embedder):
    from pocket_gm.retrieval.router import QueryResult, query_all_sync

    store, campaign_id = populated_store
    query_vec = fake_embedder.embed_one("Stag Lord bandit")

    result = query_all_sync(store, campaign_id, query_vec, top_k=5)

    assert isinstance(result, QueryResult)
    assert isinstance(result.sourcebook, list)
    assert isinstance(result.notes, list)
    assert isinstance(result.sessions, list)

    # All three tables are populated — each should return results
    assert result.sourcebook, "sourcebook should have results"
    assert result.notes, "notes should have results"
    assert result.sessions, "sessions should have results"


def test_router_is_empty_above_threshold(populated_store, fake_embedder):
    from pocket_gm.retrieval.router import query_all_sync

    store, campaign_id = populated_store
    query_vec = fake_embedder.embed_one("Stag Lord bandit")

    result = query_all_sync(store, campaign_id, query_vec, top_k=5)

    # With a very low threshold, we should find something
    assert not result.is_empty(threshold=0.05), (
        "Expected is_empty to return False when relevant content exists and threshold is low"
    )


def test_router_is_empty_below_threshold(populated_store, fake_embedder):
    from pocket_gm.retrieval.router import query_all_sync

    store, campaign_id = populated_store
    # A nonsense query with no word overlap with the content
    query_vec = fake_embedder.embed_one("zxqvwbpfk irrelevant gibberish xyzzyx")

    result = query_all_sync(store, campaign_id, query_vec, top_k=5)

    # With a high threshold, nothing should pass
    assert result.is_empty(threshold=0.5), (
        "Expected is_empty to return True for nonsense query with high threshold"
    )


# ---------------------------------------------------------------------------
# Prompt builder tests (real chunks)
# ---------------------------------------------------------------------------


def _make_real_chunk(text: str, score: float = 0.8) -> "RetrievedChunk":
    from pocket_gm.retrieval.store import RetrievedChunk

    return RetrievedChunk(
        text=text,
        score=score,
        source_type="pdf",
        filename="sample.pdf",
        page=2,
        heading="THE STAG LORD",
        chunk_index=0,
        campaign_id="kingmaker_test",
    )


def test_prompt_builder_contains_chunk_text():
    from pocket_gm.synthesis.prompt_builder import build_prompt

    chunk_text = "The Stag Lord is the most powerful bandit lord in the Stolen Lands."
    chunk = _make_real_chunk(chunk_text, score=0.9)

    prompt, index_map = build_prompt(
        "Who is the Stag Lord?",
        sourcebook_chunks=[chunk],
        notes_chunks=[],
        session_chunks=[],
        threshold=0.5,
    )

    assert chunk_text in prompt, (
        f"Expected chunk text in prompt. Prompt preview: {prompt[:300]!r}"
    )


def test_prompt_builder_citation_markers():
    from pocket_gm.synthesis.prompt_builder import build_prompt

    chunk = _make_real_chunk("The Stag Lord is a villain.", score=0.9)

    prompt, index_map = build_prompt(
        "Who is the Stag Lord?",
        sourcebook_chunks=[chunk],
        notes_chunks=[],
        session_chunks=[],
        threshold=0.5,
    )

    assert "[1]" in prompt, f"Expected '[1]' citation marker in prompt. Prompt: {prompt[:300]!r}"


def test_prompt_builder_returns_index_map():
    from pocket_gm.retrieval.store import RetrievedChunk
    from pocket_gm.synthesis.prompt_builder import build_prompt

    chunk1 = _make_real_chunk("The Stag Lord is a villain.", score=0.9)
    chunk2 = RetrievedChunk(
        text="Akiros was the lieutenant.",
        score=0.85,
        source_type="obsidian",
        filename="Akiros Ismort.md",
        page=0,
        heading="Akiros Ismort",
        chunk_index=0,
        campaign_id="kingmaker_test",
    )

    _, index_map = build_prompt(
        "Who is the Stag Lord?",
        sourcebook_chunks=[chunk1],
        notes_chunks=[chunk2],
        session_chunks=[],
        threshold=0.5,
    )

    assert len(index_map) == 2, f"Expected 2 entries in index_map, got {len(index_map)}"
    # index_map entries should be (int, RetrievedChunk)
    for num, chunk in index_map:
        assert isinstance(num, int), f"Citation number should be int, got {type(num)}"
        assert isinstance(chunk, RetrievedChunk), f"Expected RetrievedChunk, got {type(chunk)}"


# ---------------------------------------------------------------------------
# Grounding tests (real answer text)
# ---------------------------------------------------------------------------


def test_grounding_detects_uncited_sentence():
    from pocket_gm.retrieval.store import RetrievedChunk
    from pocket_gm.synthesis.grounding import validate_citations

    chunk = RetrievedChunk(
        text="The Stag Lord is evil.",
        score=0.9,
        source_type="pdf",
        filename="book.pdf",
        page=1,
        heading="",
        chunk_index=0,
        campaign_id="kingmaker_test",
    )
    index_map = [(1, chunk)]

    answer = "The Stag Lord is evil. He lives in the forest."
    result = validate_citations(answer, index_map)

    assert not result.is_fully_grounded, "Should NOT be fully grounded (uncited sentence present)"
    assert any("forest" in s for s in result.uncited_sentences), (
        f"Expected uncited sentence containing 'forest'. Uncited: {result.uncited_sentences}"
    )


def test_grounding_fully_cited_answer():
    from pocket_gm.retrieval.store import RetrievedChunk
    from pocket_gm.synthesis.grounding import validate_citations

    chunk = RetrievedChunk(
        text="The Stag Lord is evil. He lives by the lake.",
        score=0.9,
        source_type="pdf",
        filename="book.pdf",
        page=1,
        heading="",
        chunk_index=0,
        campaign_id="kingmaker_test",
    )
    index_map = [(1, chunk)]

    answer = "The Stag Lord is evil [1]. He lives by the lake [1]."
    result = validate_citations(answer, index_map)

    assert result.citations_used == [1], f"Expected citations_used=[1], got {result.citations_used}"
    assert result.is_fully_grounded, (
        f"Expected fully grounded answer. Uncited: {result.uncited_sentences}"
    )


def test_grounding_json_mode():
    from pocket_gm.retrieval.store import RetrievedChunk
    from pocket_gm.synthesis.grounding import parse_json_answer

    chunk = RetrievedChunk(
        text="The Stag Lord is evil.",
        score=0.9,
        source_type="pdf",
        filename="book.pdf",
        page=1,
        heading="",
        chunk_index=0,
        campaign_id="kingmaker_test",
    )
    index_map = [(1, chunk)]

    raw_json = '{"answer": "The Stag Lord is evil.", "citations": [1]}'
    result = parse_json_answer(raw_json, index_map)

    assert result.text == "The Stag Lord is evil.", f"Unexpected answer text: {result.text!r}"
    assert result.citations_used == [1], f"Expected citations_used=[1], got {result.citations_used}"


# ---------------------------------------------------------------------------
# Full end-to-end pipeline test
# ---------------------------------------------------------------------------


def test_full_rag_pipeline(populated_store, fake_embedder):
    from pocket_gm.retrieval.router import query_all_sync
    from pocket_gm.synthesis.grounding import validate_citations
    from pocket_gm.synthesis.prompt_builder import build_prompt

    store, campaign_id = populated_store

    # Step 1: Embed query
    query = "Who is the Stag Lord?"
    query_vec = fake_embedder.embed_one(query)
    assert query_vec.shape == (384,), f"Expected 384-dim vector, got shape {query_vec.shape}"

    # Step 2: Query all stores
    result = query_all_sync(store, campaign_id, query_vec, top_k=5)

    # Step 3: Assert result is not empty with a low threshold (fake embedder is coarser)
    assert not result.is_empty(threshold=0.05), (
        "Expected results above threshold 0.05 for 'Who is the Stag Lord?' query"
    )

    # Step 4: Build prompt
    prompt, index_map = build_prompt(
        query,
        sourcebook_chunks=result.sourcebook,
        notes_chunks=result.notes,
        session_chunks=result.sessions,
        threshold=0.05,
    )

    assert prompt, "build_prompt should return a non-empty prompt when there are results"
    assert index_map, "index_map should not be empty when results exist"

    # Verify prompt contains chunk text from the retrieval results
    # At least one chunk text should appear in the prompt
    all_chunk_texts = [c.text for c in result.sourcebook + result.notes + result.sessions]
    found_in_prompt = any(
        # Check if a significant portion of the chunk text appears
        chunk_text[:30] in prompt
        for chunk_text in all_chunk_texts
        if chunk_text
    )
    assert found_in_prompt, "Expected at least one chunk's text to appear in the prompt"

    # Step 5: Validate citations with a mock LLM answer
    # Use the first citation number from index_map
    first_idx = index_map[0][0]
    mock_answer = f"The Stag Lord is a powerful bandit lord [{first_idx}]. He rules the Stolen Lands [{first_idx}]."
    grounded = validate_citations(mock_answer, index_map)

    assert isinstance(grounded.text, str)
    assert first_idx in grounded.citations_used, (
        f"Expected citation [{first_idx}] to be used. citations_used={grounded.citations_used}"
    )


# ---------------------------------------------------------------------------
# Eval harness test
# ---------------------------------------------------------------------------


def test_eval_harness_with_real_log(tmp_path):
    from pocket_gm.eval.harness import run_eval

    campaign_id = "kingmaker_test"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Write a single NDJSON query log entry
    log_entry = {
        "campaign_id": campaign_id,
        "question": "Who is the Stag Lord?",
        "retrieved_chunks": [
            {
                "filename": "sample.pdf",
                "text": "THE STAG LORD > The Stag Lord is the most powerful bandit lord in the Stolen Lands.",
                "score": 0.85,
                "source_type": "pdf",
            }
        ],
        "answer": "The Stag Lord is the most powerful bandit lord [1].",
    }
    log_file = logs_dir / "queries.ndjson"
    log_file.write_text(json.dumps(log_entry) + "\n", encoding="utf-8")

    # Write a minimal eval set that matches the log entry
    eval_set = [
        {
            "question": "Who is the Stag Lord?",
            "expected_filename": "sample.pdf",
            "expected_text_fragment": "bandit lord",
        }
    ]
    eval_set_path = tmp_path / "eval_set.json"
    eval_set_path.write_text(json.dumps(eval_set), encoding="utf-8")

    # Run eval
    metrics = run_eval(logs_dir, campaign_id, eval_set_path)

    assert metrics.total_queries == 1, f"Expected 1 query, got {metrics.total_queries}"
    assert metrics.recall_at_k == 1.0, (
        f"Expected recall_at_k=1.0 (chunk contains 'bandit lord'), got {metrics.recall_at_k}"
    )
    assert metrics.citation_coverage > 0.0, (
        f"Expected citation_coverage > 0, got {metrics.citation_coverage}"
    )
