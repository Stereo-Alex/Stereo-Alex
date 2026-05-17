import pytest
from pocket_gm.synthesis.grounding import validate_citations, parse_json_answer
from pocket_gm.retrieval.store import RetrievedChunk


def make_chunk(idx: int) -> tuple[int, RetrievedChunk]:
    return (idx, RetrievedChunk(
        text="some text",
        score=0.9,
        source_type="pdf",
        filename="book.pdf",
        page=1,
        heading="",
        chunk_index=idx,
        campaign_id="test",
    ))


def test_fully_cited_answer():
    index_map = [make_chunk(1), make_chunk(2)]
    answer = "The Stag Lord is a bandit. [1] He lives in a fort. [2]"
    result = validate_citations(answer, index_map)
    assert result.citations_used == [1, 2]
    assert result.uncited_sentences == []
    assert result.is_fully_grounded


def test_uncited_sentence_detected():
    index_map = [make_chunk(1)]
    answer = "The Stag Lord is a bandit. [1] He has many followers."
    result = validate_citations(answer, index_map)
    assert "He has many followers." in result.uncited_sentences
    assert not result.is_fully_grounded


def test_invalid_citation_id_ignored():
    index_map = [make_chunk(1)]
    answer = "Some fact. [1] Another fact. [99]"
    result = validate_citations(answer, index_map)
    assert 99 not in result.citations_used
    assert 1 in result.citations_used


def test_empty_answer():
    result = validate_citations("", [])
    assert result.citations_used == []
    assert result.uncited_sentences == []
    assert result.is_fully_grounded


def test_not_found_response():
    result = validate_citations("Not found in available sources.", [])
    assert result.is_fully_grounded or "Not found" in result.text


# ---------------------------------------------------------------------------
# R6 — parse_json_answer tests
# ---------------------------------------------------------------------------

def test_parse_json_answer_success():
    """Valid JSON response is parsed; citations array is authoritative."""
    index_map = [make_chunk(1), make_chunk(2), make_chunk(3)]
    raw = '{"answer": "The Stag Lord is a bandit.", "citations": [1, 3]}'
    result = parse_json_answer(raw, index_map)
    assert result.text == "The Stag Lord is a bandit."
    assert result.citations_used == [1, 3]
    assert result.uncited_sentences == []


def test_parse_json_answer_filters_invalid_citation_ids():
    """Citation IDs not in index_map are silently dropped."""
    index_map = [make_chunk(1)]
    raw = '{"answer": "Some fact.", "citations": [1, 99, 100]}'
    result = parse_json_answer(raw, index_map)
    assert result.citations_used == [1]
    assert 99 not in result.citations_used


def test_parse_json_answer_empty_citations():
    """Empty citations array in JSON is valid."""
    index_map = [make_chunk(1)]
    raw = '{"answer": "Not found in available sources.", "citations": []}'
    result = parse_json_answer(raw, index_map)
    assert result.text == "Not found in available sources."
    assert result.citations_used == []


def test_parse_json_answer_fallback_to_text_mode():
    """When JSON parsing fails, falls back to validate_citations on raw text."""
    index_map = [make_chunk(1), make_chunk(2)]
    # Not valid JSON — model responded in inline citation format instead
    raw = "The Stag Lord is a bandit. [1] He has many followers. [2]"
    result = parse_json_answer(raw, index_map)
    # Fallback must return a GroundedAnswer with the inline citations parsed
    assert result.citations_used == [1, 2]
    assert result.text == raw


def test_parse_json_answer_malformed_json_fallback():
    """Truncated/malformed JSON falls back gracefully."""
    index_map = [make_chunk(1)]
    raw = '{"answer": "Half-formed answer", "citations": [1'  # truncated
    result = parse_json_answer(raw, index_map)
    # Falls back to validate_citations; raw text has no [N] markers
    assert result.text == raw


def test_parse_json_answer_strips_markdown_fences():
    """Markdown code fences around JSON are stripped before parsing."""
    index_map = [make_chunk(1), make_chunk(2)]
    raw = '```json\n{"answer": "Answer here.", "citations": [1, 2]}\n```'
    result = parse_json_answer(raw, index_map)
    assert result.text == "Answer here."
    assert result.citations_used == [1, 2]


def test_parse_json_answer_citations_sorted():
    """Citations are returned in ascending order regardless of JSON order."""
    index_map = [make_chunk(1), make_chunk(2), make_chunk(3)]
    raw = '{"answer": "Multi-cite.", "citations": [3, 1, 2]}'
    result = parse_json_answer(raw, index_map)
    assert result.citations_used == [1, 2, 3]


def test_parse_json_answer_no_index_map():
    """Works with an empty index_map — all citations filtered out."""
    raw = '{"answer": "Something.", "citations": [1, 2]}'
    result = parse_json_answer(raw, [])
    assert result.citations_used == []
    assert result.text == "Something."
