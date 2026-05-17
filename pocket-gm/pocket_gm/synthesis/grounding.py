from __future__ import annotations

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
