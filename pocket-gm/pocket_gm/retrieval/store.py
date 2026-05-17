from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Table name helpers
def sourcebook_table(campaign_id: str) -> str:
    return f"sourcebook_{campaign_id}"

def notes_table(campaign_id: str) -> str:
    return f"notes_{campaign_id}"

def sessions_table(campaign_id: str) -> str:
    return f"sessions_{campaign_id}"


@dataclass
class RetrievedChunk:
    text: str
    score: float        # cosine similarity (higher = more relevant)
    source_type: str
    filename: str
    page: int
    heading: str
    chunk_index: int
    campaign_id: str
    # Session-only fields (empty for static sources)
    session_number: int = 0
    session_date: str = ""
    timestamp_start: float = 0.0


class Store:
    def __init__(self, lancedb_path: Path):
        import lancedb
        self._db = lancedb.connect(str(lancedb_path))

    def add_documents(
        self,
        table_name: str,
        chunks: list[dict],
        embeddings: np.ndarray,
        campaign_id: str,
    ) -> None:
        import pyarrow as pa

        records = []
        for chunk, vec in zip(chunks, embeddings):
            records.append({
                "vector": vec.tolist(),
                "text": chunk.get("text", ""),
                "source_type": chunk.get("source_type", ""),
                "filename": chunk.get("filename", ""),
                "page": chunk.get("page", 0),
                "heading": chunk.get("heading", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "campaign_id": campaign_id,
                "session_number": chunk.get("session_number", 0),
                "session_date": chunk.get("session_date", ""),
                "timestamp_start": float(chunk.get("timestamp_start", 0.0)),
            })

        if table_name in self._db.list_tables().tables:
            tbl = self._db.open_table(table_name)
            tbl.add(records)
        else:
            self._db.create_table(table_name, records)

    def query(
        self,
        table_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if table_name not in self._db.list_tables().tables:
            return []

        tbl = self._db.open_table(table_name)
        results = (
            tbl.search(query_vector.tolist())
            .limit(top_k)
            .to_list()
        )

        chunks = []
        for r in results:
            # LanceDB returns _distance (L2) when using cosine on normalized vecs
            # Since embeddings are normalized, L2 distance d maps to cosine sim: 1 - d/2
            raw_dist = r.get("_distance", 0.0)
            score = max(0.0, 1.0 - raw_dist / 2.0)
            chunks.append(RetrievedChunk(
                text=r["text"],
                score=score,
                source_type=r["source_type"],
                filename=r["filename"],
                page=r.get("page", 0),
                heading=r.get("heading", ""),
                chunk_index=r.get("chunk_index", 0),
                campaign_id=r.get("campaign_id", ""),
                session_number=r.get("session_number", 0),
                session_date=r.get("session_date", ""),
                timestamp_start=r.get("timestamp_start", 0.0),
            ))
        return chunks

    def table_exists(self, table_name: str) -> bool:
        return table_name in self._db.list_tables().tables

    def drop_table(self, table_name: str) -> None:
        if self.table_exists(table_name):
            self._db.drop_table(table_name)
