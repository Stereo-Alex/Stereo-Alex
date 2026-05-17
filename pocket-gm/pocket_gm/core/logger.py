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
) -> None:
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

    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
