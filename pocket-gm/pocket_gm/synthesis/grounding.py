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
# A sentence body up to its terminal punctuation, plus any citation markers
# that immediately trail it. This captures both citation styles:
#   "fact [1]."   -> marker inside the body (before the period)
#   "fact. [1]"   -> marker in the trailing group (after the period)
_SENTENCE_WITH_CITES_RE = re.compile(r"([^.!?]*[.!?]+)(\s*(?:\[\d+\]\s*)*)")


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

    # Sentence-first validation. Each sentence (its body up to terminal
    # punctuation, plus any markers trailing it) is grounded only if it carries
    # a citation marker whose id was actually retrieved.
    #
    # This is deliberately strict about hallucinated ids: "The dragon has 300
    # HP [7]." when only [1]..[3] exist is NOT grounded, because the model can
    # otherwise fabricate both the fact and a plausible-looking citation number
    # and slip past the check.
    uncited: list[str] = []
    last_end = 0
    for m in _SENTENCE_WITH_CITES_RE.finditer(answer_text):
        last_end = m.end()
        unit = m.group(0)
        if not unit.strip():
            continue
        marker_ids = {int(c) for c in _CITATION_RE.findall(unit)}
        if not (marker_ids & valid_ids):
            uncited.append(_clean_uncited(unit))

    # Any trailing prose past the last terminal punctuation is still a claim
    # (e.g. an answer with no closing period) and must be checked too.
    trailing = answer_text[last_end:]
    if trailing.strip():
        marker_ids = {int(c) for c in _CITATION_RE.findall(trailing)}
        if not (marker_ids & valid_ids):
            uncited.append(_clean_uncited(trailing))

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
