from __future__ import annotations

from pocket_gm.retrieval.store import RetrievedChunk


def _format_source_label(chunk: RetrievedChunk, idx: int) -> str:
    if chunk.source_type == "transcript":
        ts = int(chunk.timestamp_start)
        h, m, s = ts // 3600, (ts % 3600) // 60, ts % 60
        ts_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        return f"[{idx}] Session {chunk.session_number} ({chunk.session_date}, {ts_str})"
    elif chunk.source_type == "markdown":
        heading = f" — {chunk.heading}" if chunk.heading else ""
        return f"[{idx}] {chunk.filename}{heading}"
    else:
        page = f", p.{chunk.page}" if chunk.page else ""
        heading = f" — {chunk.heading}" if chunk.heading else ""
        return f"[{idx}] {chunk.filename}{page}{heading}"


def build_prompt(
    question: str,
    sourcebook_chunks: list[RetrievedChunk],
    notes_chunks: list[RetrievedChunk],
    session_chunks: list[RetrievedChunk],
    threshold: float = 0.35,
    json_mode: bool = False,
) -> tuple[str, list[tuple[int, RetrievedChunk]]]:
    """
    Returns (prompt_text, index_map) where index_map is [(citation_number, chunk), ...].
    Only includes chunks above threshold.
    """
    index_map: list[tuple[int, RetrievedChunk]] = []
    sections: list[str] = []
    idx = 1

    def add_section(title: str, chunks: list[RetrievedChunk]) -> None:
        nonlocal idx
        relevant = [c for c in chunks if c.score >= threshold]
        if not relevant:
            return
        lines = [f"SOURCE: {title}"]
        for chunk in relevant:
            label = _format_source_label(chunk, idx)
            lines.append(f"{label}\n{chunk.text}")
            index_map.append((idx, chunk))
            idx += 1
        sections.append("\n".join(lines))

    add_section("CAMPAIGN SOURCEBOOK", sourcebook_chunks)
    add_section("GM NOTES", notes_chunks)
    add_section("SESSION TRANSCRIPTS", session_chunks)

    if not sections:
        return "", []

    body = "\n\n---\n\n".join(sections)

    if json_mode:
        instructions = """INSTRUCTIONS:
- Answer using ONLY the sources above.
- If information is not present in the sources, set answer to "Not found in available sources." and citations to [].
- Do NOT invent, speculate, or add information not in the sources above.

Respond ONLY with valid JSON in this exact format:
{"answer": "...", "citations": [1, 2, 3]}"""
    else:
        instructions = """INSTRUCTIONS:
- Answer using ONLY the sources above.
- Cite every fact with its citation number like [1] or [2][3].
- If information is not present in the sources, say exactly: "Not found in available sources."
- Do NOT invent, speculate, or add information not in the sources above.
- Keep the answer factual and concise."""

    prompt = f"""{body}

---

QUESTION: {question}

{instructions}"""

    return prompt, index_map
