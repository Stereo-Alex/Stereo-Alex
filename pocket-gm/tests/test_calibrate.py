from __future__ import annotations

import json
from pathlib import Path

import pytest

from pocket_gm.eval.calibrate import CalibrationResult, calibrate_threshold, _percentile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_log(tmp_path: Path, entries: list[dict]) -> Path:
    log_file = tmp_path / "queries.ndjson"
    with open(log_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------
# _percentile unit tests
# ---------------------------------------------------------------------------

def test_percentile_single_element():
    assert _percentile([0.5], 25) == 0.5
    assert _percentile([0.5], 75) == 0.5


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_two_elements():
    result = _percentile([0.0, 1.0], 50)
    assert abs(result - 0.5) < 1e-9


def test_percentile_known_distribution():
    # For [0, 1, 2, 3, 4] p50 should be 2.0
    values = sorted([0.0, 1.0, 2.0, 3.0, 4.0])
    assert _percentile(values, 50) == pytest.approx(2.0)
    assert _percentile(values, 0) == pytest.approx(0.0)
    assert _percentile(values, 100) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# calibrate_threshold — empty log
# ---------------------------------------------------------------------------

def test_empty_log_no_file(tmp_path):
    """No queries.ndjson at all → all zeros, no crash."""
    result = calibrate_threshold(tmp_path, "my_campaign")
    assert isinstance(result, CalibrationResult)
    assert result.total_scores == 0
    assert result.suggested_threshold == 0.0
    assert "No score data" in result.reasoning


def test_empty_log_file(tmp_path):
    """queries.ndjson exists but is empty."""
    (tmp_path / "queries.ndjson").write_text("")
    result = calibrate_threshold(tmp_path, "my_campaign")
    assert result.total_scores == 0


def test_log_with_no_matching_campaign(tmp_path):
    """Entries exist but belong to a different campaign."""
    write_log(tmp_path, [
        {
            "campaign_id": "other",
            "question": "q",
            "retrieved_chunks": [{"score": 0.8}],
        }
    ])
    result = calibrate_threshold(tmp_path, "my_campaign")
    assert result.total_scores == 0


def test_log_entries_without_score_field(tmp_path):
    """Chunks without a 'score' key are skipped gracefully."""
    write_log(tmp_path, [
        {
            "campaign_id": "c1",
            "question": "q",
            "retrieved_chunks": [{"text": "no score here"}],
        }
    ])
    result = calibrate_threshold(tmp_path, "c1")
    assert result.total_scores == 0


# ---------------------------------------------------------------------------
# calibrate_threshold — normal distribution
# ---------------------------------------------------------------------------

def _make_entry(campaign_id: str, scores: list[float]) -> dict:
    return {
        "campaign_id": campaign_id,
        "question": "some question",
        "retrieved_chunks": [{"score": s} for s in scores],
    }


def test_single_score(tmp_path):
    write_log(tmp_path, [_make_entry("c1", [0.7])])
    result = calibrate_threshold(tmp_path, "c1")
    assert result.total_scores == 1
    assert result.p10 == pytest.approx(0.7)
    assert result.p25 == pytest.approx(0.7)
    assert result.p50 == pytest.approx(0.7)
    assert result.p75 == pytest.approx(0.7)
    assert result.p90 == pytest.approx(0.7)
    assert result.suggested_threshold == pytest.approx(0.7, abs=1e-4)


def test_multiple_entries_aggregated(tmp_path):
    """Scores from multiple log entries for the same campaign are pooled."""
    write_log(tmp_path, [
        _make_entry("c1", [0.2, 0.4]),
        _make_entry("c1", [0.6, 0.8]),
    ])
    result = calibrate_threshold(tmp_path, "c1")
    assert result.total_scores == 4


def test_normal_distribution_percentiles(tmp_path):
    """Sorted scores [0.1, 0.2, 0.3, 0.4, 0.5] — check p50 = 0.3."""
    write_log(tmp_path, [_make_entry("c1", [0.5, 0.1, 0.4, 0.2, 0.3])])
    result = calibrate_threshold(tmp_path, "c1")
    assert result.total_scores == 5
    assert result.p50 == pytest.approx(0.3, abs=1e-6)
    assert result.p10 < result.p25 < result.p50 < result.p75 < result.p90


# ---------------------------------------------------------------------------
# calibrate_threshold — suggested threshold calculation
# ---------------------------------------------------------------------------

def test_suggested_threshold_is_p25(tmp_path):
    """The suggested threshold must equal p25 (rounded to 4 dp)."""
    write_log(tmp_path, [_make_entry("c1", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])])
    result = calibrate_threshold(tmp_path, "c1")
    expected_p25 = round(result.p25, 4)
    assert result.suggested_threshold == pytest.approx(expected_p25, abs=1e-6)


def test_suggested_threshold_in_reasoning(tmp_path):
    """The reasoning string references the suggested threshold value."""
    write_log(tmp_path, [_make_entry("c1", [0.3, 0.5, 0.7, 0.9])])
    result = calibrate_threshold(tmp_path, "c1")
    assert str(result.suggested_threshold) in result.reasoning or \
        f"{result.suggested_threshold:.4f}" in result.reasoning


def test_calibration_result_str(tmp_path):
    """CalibrationResult.__str__ should include all percentile labels."""
    write_log(tmp_path, [_make_entry("c1", [0.2, 0.4, 0.6, 0.8, 1.0])])
    result = calibrate_threshold(tmp_path, "c1")
    s = str(result)
    for label in ("p10", "p25", "p50", "p75", "p90", "Suggested threshold"):
        assert label in s


def test_malformed_json_line_skipped(tmp_path):
    """A malformed NDJSON line is silently skipped."""
    log_file = tmp_path / "queries.ndjson"
    with open(log_file, "w") as f:
        f.write("not json at all\n")
        f.write(json.dumps(_make_entry("c1", [0.5])) + "\n")
    result = calibrate_threshold(tmp_path, "c1")
    assert result.total_scores == 1
