"""Tests for the heading-detection logic in pdf_loader.

All tests use synthetic span data so no real PDF file is required.
"""
from __future__ import annotations

import pytest

from pocket_gm.ingestion.pdf_loader import (
    SpanInfo,
    _collect_body_median,
    _is_bold,
    detect_headings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD_FLAG = 2 ** 4  # bit 4 set → bold in PyMuPDF


def make_span(text: str, size: float, bold: bool = False) -> SpanInfo:
    return SpanInfo(text=text, size=size, flags=BOLD_FLAG if bold else 0)


# ---------------------------------------------------------------------------
# _is_bold
# ---------------------------------------------------------------------------

class TestIsBold:
    def test_bold_flag_set(self):
        assert _is_bold(BOLD_FLAG) is True

    def test_bold_flag_not_set(self):
        assert _is_bold(0) is False

    def test_other_flags_do_not_trigger_bold(self):
        # flags = 1 (bit 0) – italic in some renderers, never bold
        assert _is_bold(1) is False

    def test_bold_combined_with_other_flags(self):
        assert _is_bold(BOLD_FLAG | 0b111) is True


# ---------------------------------------------------------------------------
# _collect_body_median
# ---------------------------------------------------------------------------

class TestCollectBodyMedian:
    def test_uniform_size(self):
        spans = [make_span("word", 10.0) for _ in range(5)]
        assert _collect_body_median(spans) == 10.0

    def test_mixed_sizes(self):
        spans = [
            make_span("a", 8.0),
            make_span("b", 10.0),
            make_span("c", 10.0),
            make_span("d", 12.0),
            make_span("e", 24.0),  # heading outlier
        ]
        # sorted: 8, 10, 10, 12, 24 → median = 10
        assert _collect_body_median(spans) == 10.0

    def test_empty_spans_returns_fallback(self):
        assert _collect_body_median([]) == 10.0

    def test_blank_text_spans_ignored(self):
        spans = [make_span("", 20.0), make_span("  ", 20.0), make_span("word", 8.0)]
        assert _collect_body_median(spans) == 8.0


# ---------------------------------------------------------------------------
# detect_headings – signal coverage
# ---------------------------------------------------------------------------

class TestDetectHeadings:
    """Each test verifies that the right combination of signals fires."""

    # --- relative-size + bold (two signals, no all-caps) ---
    def test_large_bold_text_is_heading(self):
        body = 10.0
        spans = [make_span("Chapter One", size=14.0, bold=True)]
        result = detect_headings(spans, body_median=body)
        assert result == [True]

    # --- relative-size alone is NOT enough (only 1 signal) ---
    def test_large_not_bold_not_allcaps_is_not_heading(self):
        body = 10.0
        # 12 > 10 * 1.15 = 11.5, so signal_relative_size fires
        # not bold, not all-caps → only 1 signal → not a heading
        spans = [make_span("Flavour text right here", size=12.0, bold=False)]
        result = detect_headings(spans, body_median=body)
        assert result == [False]

    # --- all-caps + bold (two signals, below relative-size threshold) ---
    def test_allcaps_bold_body_size_is_heading(self):
        body = 10.0
        # size = 10 = body_median: signal_relative_size is False (not > 11.5)
        # bold + size >= body → signal_bold_above_body True
        # ALL-CAPS → signal_allcaps True
        spans = [make_span("STR DEX CON", size=10.0, bold=True)]
        result = detect_headings(spans, body_median=body)
        assert result == [True]

    # --- all-caps + large (two signals, not bold) ---
    def test_allcaps_large_not_bold_is_heading(self):
        body = 10.0
        # size 12 > 11.5 → signal_relative_size True
        # not bold → signal_bold_above_body False
        # ALL-CAPS → signal_allcaps True
        spans = [make_span("GOBLIN WARRIOR", size=12.0, bold=False)]
        result = detect_headings(spans, body_median=body)
        assert result == [True]

    # --- only all-caps alone is NOT enough ---
    def test_allcaps_body_size_not_bold_not_heading(self):
        body = 10.0
        # size = 10 = body_median: signal_relative_size False
        # not bold: signal_bold_above_body False
        # ALL-CAPS: signal_allcaps True → only 1 signal
        spans = [make_span("AC HP SPEED", size=10.0, bold=False)]
        result = detect_headings(spans, body_median=body)
        assert result == [False]

    # --- all three signals fire ---
    def test_all_three_signals_is_heading(self):
        body = 10.0
        spans = [make_span("DRAGON", size=16.0, bold=True)]
        result = detect_headings(spans, body_median=body)
        assert result == [True]

    # --- normal body text ---
    def test_body_text_not_heading(self):
        body = 10.0
        spans = [make_span("The dragon attacks the party.", size=10.0, bold=False)]
        result = detect_headings(spans, body_median=body)
        assert result == [False]

    # --- long lines are never headings ---
    def test_long_line_not_heading_even_if_bold_and_large(self):
        body = 10.0
        long_text = "A" * 101  # > 100 chars
        spans = [make_span(long_text, size=20.0, bold=True)]
        result = detect_headings(spans, body_median=body)
        assert result == [False]

    # --- exactly 100 chars is still a candidate ---
    def test_exactly_100_chars_can_be_heading(self):
        body = 10.0
        text_100 = "A" * 100  # ALL-CAPS, exactly 100 chars
        spans = [make_span(text_100, size=14.0, bold=True)]
        result = detect_headings(spans, body_median=body)
        assert result == [True]

    # --- empty text is never a heading ---
    def test_empty_span_not_heading(self):
        spans = [make_span("", size=20.0, bold=True)]
        result = detect_headings(spans, body_median=10.0)
        assert result == [False]

    # --- whitespace-only span not a heading ---
    def test_whitespace_span_not_heading(self):
        spans = [make_span("   ", size=20.0, bold=True)]
        result = detect_headings(spans, body_median=10.0)
        assert result == [False]

    # --- mixed list of spans ---
    def test_mixed_page_spans(self):
        body = 10.0
        spans = [
            make_span("CHAPTER ONE", size=14.0, bold=True),   # heading: all 3 signals
            make_span("The party travels north.", size=10.0, bold=False),  # body
            make_span("ENCOUNTER TABLE", size=12.0, bold=False),  # heading: size + allcaps
            make_span("Roll on a d6 to determine the encounter.", size=10.0),  # body
        ]
        result = detect_headings(spans, body_median=body)
        assert result == [True, False, True, False]

    # --- body_median computed automatically when not supplied ---
    def test_auto_computes_body_median(self):
        # All spans same size → median = 10 → no relative-size signal
        spans = [
            make_span("STAT BLOCK", size=10.0, bold=True),   # bold+allcaps → heading
            make_span("The troll hits for 2d6.", size=10.0, bold=False),  # body
        ]
        result = detect_headings(spans)  # no body_median kwarg
        assert result == [True, False]

    # --- custom max_heading_chars ---
    def test_custom_max_heading_chars(self):
        body = 10.0
        # 60-char ALL-CAPS bold large text
        text = "A" * 60
        spans = [make_span(text, size=14.0, bold=True)]

        # With the default limit (100) this should be a heading.
        assert detect_headings(spans, body_median=body)[0] is True

        # With a smaller limit (50) the same span should NOT be a heading.
        assert detect_headings(spans, body_median=body, max_heading_chars=50)[0] is False

    # --- stat-block numbers mixed with letters are not forced to heading ---
    def test_mixed_case_number_line_not_heading(self):
        body = 10.0
        # Lowercase + numbers → not all-caps; body size; not bold → 0 signals
        spans = [make_span("Hit Points 42", size=10.0, bold=False)]
        result = detect_headings(spans, body_median=body)
        assert result == [False]
