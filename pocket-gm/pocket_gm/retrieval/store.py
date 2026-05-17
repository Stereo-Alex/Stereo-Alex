from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec


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
        # Parameter name kept as lancedb_path to avoid breaking config.py.
        # It now points to a SQLite .db file (directory is accepted too; the
        # .db file is created alongside it if a bare directory path is given).
        db_path = Path(lancedb_path)
        if db_path.suffix == "":
            # Treat as a directory path (legacy behaviour); put the db file inside.
            db_path.mkdir(parents=True, exist_ok=True)
            db_path = db_path / "vectors.db"
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        sqlite_vec.load(self._conn)
        self._conn.execute("PRAGMA journal_mode=WAL")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_tables(self, name: str, dim: int = 384) -> None:
        """Create the vec0 virtual table and its metadata companion if absent."""
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} "
            f"USING vec0(embedding float[{dim}])"
        )
        self._conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {name}_meta (
                rowid         INTEGER PRIMARY KEY,
                text          TEXT,
                source_type   TEXT,
                filename      TEXT,
                page          INTEGER,
                heading       TEXT,
                chunk_index   INTEGER,
                campaign_id   TEXT,
                session_number INTEGER,
                session_date  TEXT,
                timestamp_start REAL
            )"""
        )
        self._conn.commit()

    def _table_exists_raw(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (f"{name}_meta",),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_documents(
        self,
        table_name: str,
        chunks: list[dict],
        embeddings: np.ndarray,
        campaign_id: str,
    ) -> None:
        if not chunks:
            return

        # Infer dimension from first embedding
        dim = embeddings.shape[1] if embeddings.ndim == 2 else embeddings.shape[0]
        self._ensure_tables(table_name, dim)

        for chunk, vec in zip(chunks, embeddings):
            blob = sqlite_vec.serialize_float32(vec.astype(np.float32))
            cur = self._conn.execute(
                f"INSERT INTO {table_name}(embedding) VALUES (?)",
                (blob,),
            )
            row_id = cur.lastrowid
            self._conn.execute(
                f"""INSERT INTO {table_name}_meta
                    (rowid, text, source_type, filename, page, heading,
                     chunk_index, campaign_id, session_number, session_date, timestamp_start)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row_id,
                    chunk.get("text", ""),
                    chunk.get("source_type", ""),
                    chunk.get("filename", ""),
                    chunk.get("page", 0),
                    chunk.get("heading", ""),
                    chunk.get("chunk_index", 0),
                    campaign_id,
                    chunk.get("session_number", 0),
                    chunk.get("session_date", ""),
                    float(chunk.get("timestamp_start", 0.0)),
                ),
            )
        self._conn.commit()

    def query(
        self,
        table_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not self._table_exists_raw(table_name):
            return []

        blob = sqlite_vec.serialize_float32(query_vector.astype(np.float32))
        rows = self._conn.execute(
            f"SELECT rowid, distance FROM {table_name} "
            f"WHERE embedding MATCH ? AND k = ?",
            (blob, top_k),
        ).fetchall()

        if not rows:
            return []

        rowids = [r["rowid"] for r in rows]
        distances = {r["rowid"]: r["distance"] for r in rows}

        placeholders = ",".join("?" * len(rowids))
        meta_rows = self._conn.execute(
            f"SELECT * FROM {table_name}_meta WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()

        # Build map for ordering
        meta_map = {m["rowid"]: m for m in meta_rows}

        chunks: list[RetrievedChunk] = []
        for rid in rowids:
            if rid not in meta_map:
                continue
            m = meta_map[rid]
            dist = distances[rid]
            # sqlite-vec returns L2 distance for normalized vectors.
            # cosine similarity = 1 - L2^2/2, but for small distances the
            # approximation score = max(0, 1 - dist/2) matches LanceDB behaviour.
            score = max(0.0, 1.0 - dist / 2.0)
            chunks.append(RetrievedChunk(
                text=m["text"],
                score=score,
                source_type=m["source_type"],
                filename=m["filename"],
                page=m["page"] or 0,
                heading=m["heading"] or "",
                chunk_index=m["chunk_index"] or 0,
                campaign_id=m["campaign_id"] or "",
                session_number=m["session_number"] or 0,
                session_date=m["session_date"] or "",
                timestamp_start=m["timestamp_start"] or 0.0,
            ))
        return chunks

    def count(self, table_name: str) -> int:
        """Return the number of rows in a table, or 0 if the table does not exist."""
        if not self._table_exists_raw(table_name):
            return 0
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table_name}_meta"
        ).fetchone()
        return row["n"] if row else 0

    def table_exists(self, table_name: str) -> bool:
        return self._table_exists_raw(table_name)

    def drop_table(self, table_name: str) -> None:
        if not self._table_exists_raw(table_name):
            return
        self._conn.execute(f"DROP TABLE IF EXISTS {table_name}_meta")
        self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self._conn.commit()
