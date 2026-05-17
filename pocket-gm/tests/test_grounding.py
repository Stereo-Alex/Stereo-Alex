import pytest
from pocket_gm.synthesis.grounding import validate_citations
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
