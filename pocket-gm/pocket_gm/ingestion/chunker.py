from __future__ import annotations

from dataclasses import dataclass, field

from pocket_gm.ingestion.pdf_loader import RawChunk


@dataclass
class Chunk:
    text: str          # chunk text, prefixed with heading context
    page: int
    heading: str
    filename: str
    chunk_index: int
    source_type: str   # "pdf" | "markdown" | "obsidian" | "transcript"
    tags: list[str] = field(default_factory=list)
    title: str = ""


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    windows: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return windows


def chunk_pdf_pages(pages: list[RawChunk], chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        prefix = f"{page.heading} > " if page.heading else ""
        windows = _split_words(page.text, chunk_size, chunk_overlap)
        for window in windows:
            chunks.append(Chunk(
                text=prefix + window,
                page=page.page,
                heading=page.heading,
                filename=page.filename,
                chunk_index=idx,
                source_type="pdf",
            ))
            idx += 1
    return chunks


def chunk_markdown(text: str, filename: str, chunk_size: int = 256, chunk_overlap: int = 32) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_heading = ""
    idx = 0
    current_block: list[str] = []

    def flush(heading: str) -> None:
        nonlocal idx
        if not current_block:
            return
        block_text = " ".join(current_block)
        prefix = f"{heading} > " if heading else ""
        for window in _split_words(block_text, chunk_size, chunk_overlap):
            chunks.append(Chunk(
                text=prefix + window,
                page=0,
                heading=heading,
                filename=filename,
                chunk_index=idx,
                source_type="markdown",
            ))
            idx += 1
        current_block.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush(current_heading)
            current_heading = stripped.lstrip("#").strip()
        elif stripped:
            current_block.append(stripped)

    flush(current_heading)
    return chunks


def chunk_obsidian_note(note, chunk_size: int = 256, chunk_overlap: int = 32) -> list[Chunk]:
    """
    Chunk a pre-cleaned ObsidianNote. The note's title is injected as the top-level
    heading prefix so every chunk carries note identity even without surrounding context.
    Tags and aliases from frontmatter are appended to the first chunk so they are
    searchable (e.g. querying a tag finds the note).
    """
    # Prepend title as a top-level heading so the markdown chunker picks it up
    body = f"# {note.title}\n{note.body}"

    # Append tags + aliases as a discoverable footer on the note text
    extras: list[str] = []
    if note.tags:
        extras.append("Tags: " + ", ".join(note.tags))
    if note.aliases:
        extras.append("Also known as: " + ", ".join(note.aliases))
    if extras:
        body += "\n" + " | ".join(extras)

    raw_chunks = chunk_markdown(body, note.path.name, chunk_size, chunk_overlap)

    # Upgrade source_type and carry note metadata
    result: list[Chunk] = []
    for c in raw_chunks:
        result.append(Chunk(
            text=c.text,
            page=0,
            heading=c.heading or note.title,
            filename=note.path.name,
            chunk_index=c.chunk_index,
            source_type="obsidian",
            tags=note.tags,
            title=note.title,
        ))
    return result


def chunk_transcript(
    text: str,
    filename: str,
    session_number: int,
    session_date: str,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    timestamps: list[dict] | None = None,
) -> list[dict]:
    """Returns raw dicts (not Chunk) so timestamp metadata can be included."""
    results: list[dict] = []
    words = text.split()
    if not words:
        return results

    # Build cumulative char offsets per segment so we can map word positions to timestamps.
    seg_offsets: list[tuple[int, float]] = []  # (char_start_of_segment, seg["start"])
    if timestamps:
        offset = 0
        for seg in timestamps:
            seg_offsets.append((offset, float(seg.get("start", 0.0))))
            offset += len(seg.get("text", "")) + 1  # +1 for the joining space

    idx = 0
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        # Approximate timestamp: find the last segment whose char offset <= chunk's char position.
        timestamp_start = 0.0
        if seg_offsets:
            char_pos = len(" ".join(words[:start])) + (1 if start > 0 else 0)
            for seg_char_start, seg_time in seg_offsets:
                if seg_char_start <= char_pos:
                    timestamp_start = seg_time
                else:
                    break

        results.append({
            "text": chunk_text,
            "chunk_index": idx,
            "source_type": "transcript",
            "filename": filename,
            "session_number": session_number,
            "session_date": session_date,
            "timestamp_start": timestamp_start,
        })
        idx += 1
        if end == len(words):
            break
        start += chunk_size - chunk_overlap

    return results
