from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalMetrics:
    total_queries: int
    recall_at_k: float          # fraction of queries where a correct chunk appeared in top-k
    citation_coverage: float    # avg fraction of answer sentences that have citations
    no_result_rate: float       # fraction of queries that hit the retrieval gate

    def __str__(self) -> str:
        return (
            f"Queries:          {self.total_queries}\n"
            f"Recall@k:         {self.recall_at_k:.1%}\n"
            f"Citation coverage:{self.citation_coverage:.1%}\n"
            f"No-result rate:   {self.no_result_rate:.1%}"
        )


def compute_metrics(log_entries: list[dict], eval_set: list[dict]) -> EvalMetrics:
    """
    log_entries: records from queries.ndjson
    eval_set: list of {"question": str, "expected_filename": str, "expected_text_fragment": str}
    """
    if not eval_set:
        return EvalMetrics(0, 0.0, 0.0, 0.0)

    # Build index: question -> log entry
    log_by_question = {e["question"]: e for e in log_entries}

    recall_hits = 0
    coverage_scores: list[float] = []
    no_results = 0

    for item in eval_set:
        q = item["question"]
        entry = log_by_question.get(q)
        if not entry:
            continue

        retrieved = entry.get("retrieved_chunks", [])
        answer = entry.get("answer", "")

        # Recall@k: did any retrieved chunk come from the expected file
        # and contain the expected text fragment?
        expected_file = item.get("expected_filename", "")
        expected_fragment = item.get("expected_text_fragment", "").lower()
        hit = any(
            (not expected_file or c.get("filename", "") == expected_file)
            and (not expected_fragment or expected_fragment in c.get("text", "").lower())
            for c in retrieved
        )
        if hit:
            recall_hits += 1

        # Citation coverage: fraction of sentences that have a citation
        if not answer or answer.strip().startswith("Not found"):
            no_results += 1
            coverage_scores.append(1.0)  # no-result answers are fully grounded by definition
        else:
            import re
            # Mirror grounding.py: split on citation markers, uncited segments are trailing text
            parts = re.split(r"(\[\d+\])", answer)
            total_sentences, cited_sentences = 0, 0
            for i, part in enumerate(parts):
                if i % 2 == 1:  # citation marker
                    continue
                text = part.strip()
                if not text:
                    continue
                has_following_citation = i + 1 < len(parts) and bool(re.match(r"\[\d+\]", parts[i + 1].strip()))
                sub = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+", text) if s.strip()]
                total_sentences += max(len(sub), 1)
                if has_following_citation:
                    cited_sentences += max(len(sub), 1)
            cited = cited_sentences
            coverage_scores.append(cited / total_sentences if total_sentences else 1.0)

    n = len(eval_set)
    return EvalMetrics(
        total_queries=n,
        recall_at_k=recall_hits / n if n else 0.0,
        citation_coverage=sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0,
        no_result_rate=no_results / n if n else 0.0,
    )
