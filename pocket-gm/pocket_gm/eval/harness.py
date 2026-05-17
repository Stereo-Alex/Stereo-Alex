from __future__ import annotations

import json
from pathlib import Path

from pocket_gm.eval.metrics import EvalMetrics, compute_metrics


def load_query_log(logs_path: Path, campaign_id: str) -> list[dict]:
    log_file = logs_path / "queries.ndjson"
    if not log_file.exists():
        return []
    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("campaign_id") == campaign_id:
                    entries.append(record)
            except json.JSONDecodeError:
                continue
    return entries


def load_eval_set(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_eval(logs_path: Path, campaign_id: str, eval_set_path: Path) -> EvalMetrics:
    log_entries = load_query_log(logs_path, campaign_id)
    eval_set = load_eval_set(eval_set_path)
    return compute_metrics(log_entries, eval_set)
