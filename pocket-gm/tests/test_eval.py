import json
import pytest
from pathlib import Path
from pocket_gm.eval.metrics import compute_metrics, EvalMetrics
from pocket_gm.eval.harness import load_query_log, run_eval


def write_log(tmp_path: Path, entries: list[dict]) -> Path:
    log_file = tmp_path / "queries.ndjson"
    with open(log_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return tmp_path


def write_eval_set(tmp_path: Path, items: list[dict]) -> Path:
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(items))
    return path


def test_recall_hit(tmp_path):
    logs = write_log(tmp_path, [{
        "question": "Who is the Stag Lord?",
        "campaign_id": "km",
        "answer": "A bandit warlord. [1]",
        "retrieved_chunks": [{"filename": "kb.pdf", "text": "the stag lord rules the greenbelt", "score": 0.9}],
    }])
    es = write_eval_set(tmp_path, [{"question": "Who is the Stag Lord?", "expected_filename": "kb.pdf", "expected_text_fragment": "stag lord"}])
    metrics = run_eval(tmp_path, "km", es)
    assert metrics.recall_at_k == 1.0


def test_recall_miss(tmp_path):
    logs = write_log(tmp_path, [{
        "question": "Who is the Stag Lord?",
        "campaign_id": "km",
        "answer": "A bandit. [1]",
        "retrieved_chunks": [{"filename": "other.pdf", "text": "unrelated content", "score": 0.9}],
    }])
    es = write_eval_set(tmp_path, [{"question": "Who is the Stag Lord?", "expected_filename": "kb.pdf", "expected_text_fragment": "stag lord"}])
    metrics = run_eval(tmp_path, "km", es)
    assert metrics.recall_at_k == 0.0


def test_citation_coverage_full(tmp_path):
    metrics = compute_metrics(
        [{"question": "q1", "campaign_id": "km", "answer": "Some fact. [1]", "retrieved_chunks": []}],
        [{"question": "q1", "expected_filename": "", "expected_text_fragment": ""}],
    )
    assert metrics.citation_coverage == 1.0


def test_no_result_counted(tmp_path):
    metrics = compute_metrics(
        [{"question": "q1", "campaign_id": "km", "answer": "Not found in available sources.", "retrieved_chunks": []}],
        [{"question": "q1", "expected_filename": "", "expected_text_fragment": ""}],
    )
    assert metrics.no_result_rate == 1.0


def test_empty_log(tmp_path):
    es = write_eval_set(tmp_path, [{"question": "Who?", "expected_filename": "", "expected_text_fragment": ""}])
    # No queries.ndjson → no matches → total_queries reflects eval_set size but recall is 0
    metrics = run_eval(tmp_path, "km", es)
    assert metrics.recall_at_k == 0.0


def test_load_query_log_filters_by_campaign(tmp_path):
    write_log(tmp_path, [
        {"question": "q1", "campaign_id": "km", "answer": "", "retrieved_chunks": []},
        {"question": "q2", "campaign_id": "other", "answer": "", "retrieved_chunks": []},
    ])
    entries = load_query_log(tmp_path, "km")
    assert len(entries) == 1
    assert entries[0]["question"] == "q1"
