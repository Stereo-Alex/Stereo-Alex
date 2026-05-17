import pytest
from pocket_gm.ingestion.chunker import _split_words, chunk_markdown, chunk_transcript
from pocket_gm.ingestion.pdf_loader import RawChunk
from pocket_gm.ingestion.chunker import chunk_pdf_pages


def test_split_words_basic():
    words = list(range(10))
    text = " ".join(str(w) for w in words)
    windows = _split_words(text, size=4, overlap=1)
    assert windows[0] == "0 1 2 3"
    assert windows[1] == "3 4 5 6"


def test_split_words_shorter_than_chunk():
    result = _split_words("hello world", size=100, overlap=10)
    assert result == ["hello world"]


def test_split_words_empty():
    assert _split_words("", 10, 2) == []


def test_chunk_pdf_pages_heading_prefix():
    pages = [RawChunk(text="The Stag Lord rules the Greenbelt", page=1, heading="Chapter 3", filename="kb.pdf")]
    chunks = chunk_pdf_pages(pages, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Chapter 3 > ")
    assert chunks[0].page == 1
    assert chunks[0].source_type == "pdf"


def test_chunk_pdf_pages_no_heading():
    pages = [RawChunk(text="Some text here", page=2, heading="", filename="kb.pdf")]
    chunks = chunk_pdf_pages(pages, chunk_size=100, chunk_overlap=10)
    assert chunks[0].text == "Some text here"


def test_chunk_markdown_headings():
    md = "# The Stag Lord\nA bandit warlord.\n## His Fort\nLocated on the Tuskwater."
    chunks = chunk_markdown(md, "notes.md", chunk_size=100, chunk_overlap=10)
    assert any("The Stag Lord" in c.heading for c in chunks)
    assert all(c.source_type == "markdown" for c in chunks)


def test_chunk_markdown_no_heading():
    md = "Just some plain text without headings."
    chunks = chunk_markdown(md, "plain.md", chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 1


def test_chunk_transcript_basic():
    text = "The party enters the fort. They fight the guards. The Stag Lord appears."
    results = chunk_transcript(text, "session_01.txt", session_number=1, session_date="2025-01-01", chunk_size=8, chunk_overlap=2)
    assert len(results) > 1
    assert results[0]["source_type"] == "transcript"
    assert results[0]["session_number"] == 1
    assert "timestamp_start" in results[0]
