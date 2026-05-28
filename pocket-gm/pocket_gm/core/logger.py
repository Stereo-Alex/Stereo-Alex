from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pocket_gm.retrieval.store import RetrievedChunk
from pocket_gm.synthesis.grounding import GroundedAnswer


def log_query(
    logs_path: Path,
    campaign_id: str,
    question: str,
    result: GroundedAnswer,
    llm_model: str,
    all_retrieved: list[RetrievedChunk] | None = None,
) -> None:
    """Append a query record to queries.ndjson.

    ``all_retrieved`` should be the *full* set of retrieved chunks (every
    top-k result from every store, before the relevance gate). Logging the
    complete score distribution — not just the above-threshold chunks that
    made it into the answer — is what lets ``eval calibrate`` recommend
    *lowering* the threshold; otherwise calibration only ever sees scores that
    already passed the gate and can only push it higher.
    """
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "queries.ndjson"

    def chunk_dict(c: RetrievedChunk) -> dict:
        return {
            "text": c.text[:200],
            "score": round(c.score, 4),
            "source_type": c.source_type,
            "filename": c.filename,
            "page": c.page,
            "session_number": c.session_number,
            "timestamp_start": c.timestamp_start,
        }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "question": question,
        "llm_model": llm_model,
        "answer": result.text,
        "citations_used": result.citations_used,
        "uncited_sentences": result.uncited_sentences,
        "retrieved_chunks": [chunk_dict(c) for _, c in result.index_map],
    }

    if all_retrieved is not None:
        record["all_scores"] = [round(c.score, 4) for c in all_retrieved]

    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
