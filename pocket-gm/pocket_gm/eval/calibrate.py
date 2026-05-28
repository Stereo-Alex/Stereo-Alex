from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrationResult:
    """Distribution of cosine scores from retrieved chunks in the query log."""
    total_scores: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    suggested_threshold: float
    reasoning: str

    def __str__(self) -> str:
        return (
            f"Score distribution ({self.total_scores} scores):\n"
            f"  p10: {self.p10:.4f}\n"
            f"  p25: {self.p25:.4f}\n"
            f"  p50: {self.p50:.4f}\n"
            f"  p75: {self.p75:.4f}\n"
            f"  p90: {self.p90:.4f}\n"
            f"\nSuggested threshold: {self.suggested_threshold:.4f}\n"
            f"Reasoning: {self.reasoning}"
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a sorted list using linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    # Using the nearest-rank / linear interpolation method
    rank = pct / 100.0 * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_values[-1]
    frac = rank - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])


def calibrate_threshold(logs_path: Path, campaign_id: str) -> CalibrationResult:
    """
    Load the query log for *campaign_id* from *logs_path*/queries.ndjson,
    extract all retrieved-chunk cosine scores, and suggest a relevance threshold.

    The suggested threshold is the p25 of observed scores (low-tail cutoff):
    it removes the bottom quarter of retrieved chunks while keeping the vast
    majority of likely-relevant results.
    """
    log_file = logs_path / "queries.ndjson"

    scores: list[float] = []

    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("campaign_id") != campaign_id:
                    continue
                # Prefer the full score distribution (includes below-threshold
                # results) when present; fall back to the above-threshold chunks
                # for older log records that predate "all_scores".
                if isinstance(record.get("all_scores"), list):
                    for score in record["all_scores"]:
                        if isinstance(score, (int, float)):
                            scores.append(float(score))
                else:
                    for chunk in record.get("retrieved_chunks", []):
                        score = chunk.get("score")
                        if isinstance(score, (int, float)):
                            scores.append(float(score))

    if not scores:
        return CalibrationResult(
            total_scores=0,
            p10=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            p90=0.0,
            suggested_threshold=0.0,
            reasoning=(
                "No score data found in the query log. "
                "Run some queries first, then re-calibrate."
            ),
        )

    scores.sort()
    p10 = _percentile(scores, 10)
    p25 = _percentile(scores, 25)
    p50 = _percentile(scores, 50)
    p75 = _percentile(scores, 75)
    p90 = _percentile(scores, 90)

    suggested = round(p25, 4)
    reasoning = (
        f"Threshold set at the 25th percentile ({suggested:.4f}) of {len(scores)} observed "
        f"cosine scores. This removes the lowest-scoring quarter of retrieved chunks "
        f"(likely noise) while retaining ≥75% of results. "
        f"If recall is too low, lower the threshold toward p10 ({p10:.4f}); "
        f"if too many irrelevant chunks appear, raise it toward p50 ({p50:.4f})."
    )

    return CalibrationResult(
        total_scores=len(scores),
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        suggested_threshold=suggested,
        reasoning=reasoning,
    )
