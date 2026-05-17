from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pocket_gm.retrieval.store import RetrievedChunk


@dataclass
class GroundedAnswer:
    text: str
    citations_used: list[int]
    uncited_sentences: list[str]
    index_map: list[tuple[int, RetrievedChunk]]

    @property
    def is_fully_grounded(self) -> bool:
        return len(self.uncited_sentences) == 0


_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+")


def validate_citations(
    answer_text: str,
    index_map: list[tuple[int, RetrievedChunk]],
) -> GroundedAnswer:
    valid_ids = {idx for idx, _ in index_map}

    cited = {int(m) for m in _CITATION_RE.findall(answer_text)}
    citations_used = sorted(cited & valid_ids)

    # Split text on citation markers; each text segment between/after markers
    # represents prose that either was or wasn't followed by a citation.
    # Segments: [text, [N], text, [N], trailing_text]
    parts = re.split(r"(\[\d+\])", answer_text)
    uncited: list[str] = []

    for i, part in enumerate(parts):
        # Even indices are text segments; odd indices are citation markers
        if i % 2 == 1:
            continue
        text = part.strip()
        if not text:
            continue
        # This text segment is cited if the next part is a citation marker
        has_following_citation = (i + 1 < len(parts) and bool(_CITATION_RE.match(parts[i + 1].strip())))
        if not has_following_citation:
            # Extract individual sentences from this uncited segment
            sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
            uncited.extend(sentences)

    return GroundedAnswer(
        text=answer_text,
        citations_used=citations_used,
        uncited_sentences=uncited,
        index_map=index_map,
    )


def parse_json_answer(
    raw: str,
    index_map: list[tuple[int, RetrievedChunk]],
) -> GroundedAnswer:
    """
    Parse a JSON-mode LLM response into a GroundedAnswer.

    Expected JSON shape: {"answer": "...", "citations": [1, 2, 3]}

    If JSON parsing fails, falls back to ``validate_citations`` on the raw
    text so the caller always gets a usable result.

    The ``citations`` array is treated as the authoritative source of citation
    IDs; only IDs that exist in *index_map* are kept.
    """
    valid_ids = {idx for idx, _ in index_map}

    # Try to extract JSON from the response.  Some models wrap it in markdown
    # code fences; strip those before parsing.
    cleaned = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        payload = json.loads(cleaned)
        answer_text = str(payload.get("answer", ""))
        raw_citations = payload.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []
        citations_used = sorted(
            int(c) for c in raw_citations
            if isinstance(c, (int, float)) and int(c) in valid_ids
        )
        return GroundedAnswer(
            text=answer_text,
            citations_used=citations_used,
            uncited_sentences=[],  # JSON mode — citations array is authoritative
            index_map=index_map,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        # Graceful fallback: treat raw text as a normal citation-inline answer
        return validate_citations(raw, index_map)
