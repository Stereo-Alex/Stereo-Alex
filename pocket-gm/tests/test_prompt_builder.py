import pytest
from pocket_gm.synthesis.prompt_builder import build_prompt
from pocket_gm.retrieval.store import RetrievedChunk


def make_chunk(source_type: str, score: float = 0.9, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text="The Stag Lord rules the Greenbelt.",
        score=score,
        source_type=source_type,
        filename="kb.pdf" if source_type == "pdf" else "notes.md",
        page=page,
        heading="Chapter 3",
        chunk_index=0,
        campaign_id="kingmaker",
        session_number=1 if source_type == "transcript" else 0,
        session_date="2025-01-01" if source_type == "transcript" else "",
        timestamp_start=3620.0 if source_type == "transcript" else 0.0,
    )


def test_prompt_includes_question():
    prompt, _ = build_prompt("Who is the Stag Lord?", [make_chunk("pdf")], [], [], threshold=0.5)
    assert "Who is the Stag Lord?" in prompt


def test_prompt_builds_index_map():
    _, index_map = build_prompt("test?", [make_chunk("pdf"), make_chunk("markdown")], [], [], threshold=0.5)
    assert len(index_map) == 2
    assert index_map[0][0] == 1
    assert index_map[1][0] == 2


def test_below_threshold_excluded():
    low = make_chunk("pdf", score=0.1)
    prompt, index_map = build_prompt("test?", [low], [], [], threshold=0.5)
    assert prompt == ""
    assert index_map == []


def test_session_timestamp_formatted():
    session_chunk = make_chunk("transcript")
    prompt, _ = build_prompt("test?", [], [], [session_chunk], threshold=0.5)
    assert "1:00:20" in prompt  # 3620 seconds = 1:00:20


def test_empty_sources_returns_empty():
    prompt, index_map = build_prompt("anything?", [], [], [], threshold=0.5)
    assert prompt == ""
    assert index_map == []
