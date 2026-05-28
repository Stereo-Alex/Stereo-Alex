from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass
class RawChunk:
    text: str
    page: int
    heading: str
    filename: str


class SpanInfo(TypedDict):
    """Minimal span dict extracted from a PyMuPDF text block."""
    text: str
    size: float
    flags: int  # PyMuPDF bitmask; bit 4 (0x10) = bold


def _is_bold(flags: int) -> bool:
    """Return True when the PyMuPDF font-flags bitmask has the bold bit set."""
    return bool(flags & (2 ** 4))


def _collect_body_median(spans: list[SpanInfo]) -> float:
    """Return the median font size across all non-empty spans on a page.

    Falls back to 10.0 when there are no spans so downstream comparisons
    always have a sensible denominator.
    """
    sizes = [s["size"] for s in spans if s["text"].strip()]
    if not sizes:
        return 10.0
    return statistics.median(sizes)


def detect_headings(
    spans: list[SpanInfo],
    body_median: float | None = None,
    max_heading_chars: int = 100,
) -> list[bool]:
    """Return a parallel list of booleans indicating heading status.

    Multi-signal heuristic (2+ signals → heading):

    Signal 1 – *relative size*: span font size > body_median * 1.15
    Signal 2 – *bold + above-body*: bold AND font size >= body_median
    Signal 3 – *ALL-CAPS short line*: text is all-caps and <= max_heading_chars

    A candidate is only considered when the text is non-empty and
    shorter than max_heading_chars characters (long lines are body text).

    Parameters
    ----------
    spans:
        Sequence of span dicts (keys: ``text``, ``size``, ``flags``).
    body_median:
        Pre-computed median body font size.  If *None* it is computed from
        *spans* automatically.
    max_heading_chars:
        Lines longer than this are never headings regardless of other signals.
    """
    if body_median is None:
        body_median = _collect_body_median(spans)

    results: list[bool] = []
    for span in spans:
        text = span["text"].strip()
        size = span.get("size", 0)
        flags = span.get("flags", 0)

        # Long lines or empty spans are never headings.
        if not text or len(text) > max_heading_chars:
            results.append(False)
            continue

        bold = _is_bold(flags)

        signal_relative_size = size > body_median * 1.15
        signal_bold_above_body = bold and size >= body_median
        signal_allcaps = text == text.upper() and any(c.isalpha() for c in text)

        signals = sum([signal_relative_size, signal_bold_above_body, signal_allcaps])
        results.append(signals >= 2)

    return results


def load_pdf(path: Path) -> list[RawChunk]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pymupdf is required: pip install pymupdf")

    doc = fitz.open(str(path))
    try:
        return _extract_pages(doc, path)
    finally:
        doc.close()


def _extract_pages(doc, path: Path) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    current_heading = ""
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        # First pass: collect all spans to compute the page's body-font median.
        all_spans: list[SpanInfo] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    all_spans.append(
                        SpanInfo(
                            text=span.get("text", ""),
                            size=span.get("size", 0),
                            flags=span.get("flags", 0),
                        )
                    )

        body_median = _collect_body_median(all_spans)
        heading_flags = detect_headings(all_spans, body_median=body_median)

        # Second pass: build page text and track the last heading seen.
        page_lines: list[str] = []
        for span_info, is_heading in zip(all_spans, heading_flags):
            text = span_info["text"].strip()
            if not text:
                continue
            if is_heading:
                current_heading = text
            page_lines.append(text)

        if page_lines:
            chunks.append(RawChunk(
                text=" ".join(page_lines),
                page=page_num,
                heading=current_heading,
                filename=path.name,
            ))

    return chunks
