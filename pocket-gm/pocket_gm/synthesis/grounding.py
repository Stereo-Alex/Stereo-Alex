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


def _clean_uncited(text: str) -> str:
    """Strip citation markers from a reported uncited sentence and tidy the
    whitespace the removal leaves behind (e.g. "fact [9]." -> "fact.")."""
    cleaned = _CITATION_RE.sub("", text)
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def validate_citations(
    answer_text: str,
    index_map: list[tuple[int, RetrievedChunk]],
) -> GroundedAnswer:
    valid_ids = {idx for idx, _ in index_map}

    cited = {int(m) for m in _CITATION_RE.findall(answer_text)}
    citations_used = sorted(cited & valid_ids)

    # The canonical "not found" response makes no factual claims, so it is
    # grounded by definition. This mirrors the eval harness (metrics.py) and
    # avoids showing a spurious "uncited sentence" warning for a non-answer.
    if answer_text.strip().lower().startswith("not found"):
        return GroundedAnswer(
            text=answer_text,
            citations_used=citations_used,
            uncited_sentences=[],
            index_map=index_map,
        )

    # Split the answer on *valid* citation markers only. A prose segment is
    # grounded iff it is immediately followed by one of these markers.
    #
    # Splitting on valid ids (rather than any "[n]") is what makes the check
    # robust on two fronts:
    #   - Hallucinated ids — "300 HP [7]." when only [1]..[3] exist — are not
    #     split points, so "[7]" stays inline as text and provides no grounding;
    #     the segment is flagged. The model can't fabricate both a fact and a
    #     citation number and slip past.
    #   - Periods inside numbers/abbreviations ("3.5 damage", "vs.") are never
    #     split points, so they don't fracture a correctly-cited sentence.
    if valid_ids:
        split_re = re.compile(r"(\[(?:" + "|".join(str(i) for i in sorted(valid_ids)) + r")\])")
        parts = split_re.split(answer_text)
    else:
        parts = [answer_text]

    uncited: list[str] = []
    for i, part in enumerate(parts):
        # Even indices are prose segments; odd indices are valid markers.
        if i % 2 == 1:
            continue
        if not part.strip():
            continue
        # Grounded iff a valid marker immediately follows this segment.
        if i + 1 < len(parts):
            continue
        # Trailing/unfollowed segment — flag each sentence it contains.
        for sentence in _SENTENCE_RE.findall(part):
            cleaned = _clean_uncited(sentence)
            if cleaned and any(ch.isalnum() for ch in cleaned):
                uncited.append(cleaned)
        # Any residual text with no terminal punctuation is still a claim.
        residual = _clean_uncited(_SENTENCE_RE.sub("", part))
        if residual and any(ch.isalnum() for ch in residual):
            uncited.append(residual)

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
