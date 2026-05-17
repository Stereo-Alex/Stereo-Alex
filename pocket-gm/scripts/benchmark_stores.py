"""
benchmark_stores.py
-------------------
Research item R1: LanceDB vs sqlite-vec performance benchmark.

Measures insert time, single-vector query time (top-5), and on-disk size
at 1K, 5K, and 20K synthetic 384-dim normalised float32 vectors that
match the output shape of all-MiniLM-L6-v2.

Run with:
    python scripts/benchmark_stores.py
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import struct
import tempfile
import time
from typing import NamedTuple

import numpy as np

# ─── constants ────────────────────────────────────────────────────────────────
DIM = 384
SCALES = [1_000, 5_000, 20_000]
TOP_K = 5
QUERY_REPEATS = 20          # average over N independent queries for stability
RNG_SEED = 42

# ─── helpers ──────────────────────────────────────────────────────────────────

def make_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n normalised float32 vectors of dimension DIM."""
    vecs = rng.standard_normal((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def dir_size_bytes(path: str) -> int:
    """Recursive size of a directory (or single file)."""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class Result(NamedTuple):
    backend: str
    scale: int
    insert_s: float
    query_avg_ms: float
    disk: str


# ─── LanceDB benchmark ────────────────────────────────────────────────────────

def bench_lancedb(scale: int, vecs: np.ndarray, rng: np.random.Generator) -> Result:
    import lancedb

    tmpdir = tempfile.mkdtemp(prefix="bench_lancedb_")
    try:
        db = lancedb.connect(tmpdir)

        # Build records
        records = [
            {"vector": vecs[i].tolist(), "id": i, "text": f"doc_{i}"}
            for i in range(scale)
        ]

        # Insert
        t0 = time.perf_counter()
        db.create_table("bench", records)
        insert_s = time.perf_counter() - t0

        # Query
        tbl = db.open_table("bench")
        query_times: list[float] = []
        for _ in range(QUERY_REPEATS):
            q = make_vectors(1, rng)[0]
            t0 = time.perf_counter()
            tbl.search(q.tolist()).limit(TOP_K).to_list()
            query_times.append((time.perf_counter() - t0) * 1_000)

        disk = dir_size_bytes(tmpdir)
        return Result(
            backend="LanceDB",
            scale=scale,
            insert_s=insert_s,
            query_avg_ms=float(np.mean(query_times)),
            disk=human_bytes(disk),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── sqlite-vec benchmark ─────────────────────────────────────────────────────

def bench_sqlite_vec(scale: int, vecs: np.ndarray, rng: np.random.Generator) -> Result:
    import sqlite_vec

    tmp_db = tempfile.mktemp(prefix="bench_sqlite_vec_", suffix=".db")
    try:
        con = sqlite3.connect(tmp_db)
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)

        con.execute(
            f"CREATE VIRTUAL TABLE bench USING vec0(embedding float[{DIM}])"
        )
        con.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, text TEXT)")

        # Insert – sqlite_vec uses rowid; we batch via executemany
        def vec_blob(v: np.ndarray) -> bytes:
            return struct.pack(f"{DIM}f", *v.tolist())

        t0 = time.perf_counter()
        con.executemany(
            "INSERT INTO bench(rowid, embedding) VALUES (?, ?)",
            ((i, vec_blob(vecs[i])) for i in range(scale)),
        )
        con.commit()
        insert_s = time.perf_counter() - t0

        # Query
        query_times: list[float] = []
        for _ in range(QUERY_REPEATS):
            q = make_vectors(1, rng)[0]
            q_blob = vec_blob(q)
            t0 = time.perf_counter()
            con.execute(
                """
                SELECT rowid, distance
                FROM bench
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (q_blob, TOP_K),
            ).fetchall()
            query_times.append((time.perf_counter() - t0) * 1_000)

        con.close()
        disk = dir_size_bytes(tmp_db)
        return Result(
            backend="sqlite-vec",
            scale=scale,
            insert_s=insert_s,
            query_avg_ms=float(np.mean(query_times)),
            disk=human_bytes(disk),
        )
    finally:
        try:
            os.unlink(tmp_db)
        except OSError:
            pass


# ─── main ─────────────────────────────────────────────────────────────────────

def print_table(results: list[Result]) -> None:
    header = (
        f"{'Backend':<12} {'Scale':>6}  {'Insert (s)':>12}  "
        f"{'Query avg (ms)':>16}  {'Disk size':>12}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.backend:<12} {r.scale:>6,}  {r.insert_s:>12.3f}  "
            f"{r.query_avg_ms:>16.3f}  {r.disk:>12}"
        )
    print(sep)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Pre-generate all vectors so generation time is excluded from timings
    all_vecs = {scale: make_vectors(scale, rng) for scale in SCALES}

    # Check sqlite-vec
    sqlite_vec_available = False
    sqlite_vec_reason = ""
    try:
        import sqlite_vec  # noqa: F401
        sqlite_vec_available = True
    except ImportError as exc:
        sqlite_vec_reason = str(exc)

    print("\n=== pocket-gm: LanceDB vs sqlite-vec benchmark ===")
    print(f"Dimensions : {DIM}  |  top-k : {TOP_K}  |  query repeats : {QUERY_REPEATS}")
    if not sqlite_vec_available:
        print(f"sqlite-vec : NOT available — {sqlite_vec_reason}")
    else:
        print("sqlite-vec : available (0.1.9)")
    print()

    results: list[Result] = []

    for scale in SCALES:
        vecs = all_vecs[scale]
        print(f"Running scale={scale:,} …", flush=True)

        res_lance = bench_lancedb(scale, vecs, np.random.default_rng(RNG_SEED + scale))
        results.append(res_lance)
        print(f"  LanceDB  : insert={res_lance.insert_s:.3f}s  query={res_lance.query_avg_ms:.3f}ms  disk={res_lance.disk}")

        if sqlite_vec_available:
            res_sv = bench_sqlite_vec(scale, vecs, np.random.default_rng(RNG_SEED + scale))
            results.append(res_sv)
            print(f"  sqlite-vec: insert={res_sv.insert_s:.3f}s  query={res_sv.query_avg_ms:.3f}ms  disk={res_sv.disk}")

    print()
    print_table(results)

    # ── Summary analysis ──────────────────────────────────────────────────────
    if sqlite_vec_available:
        lance_results  = [r for r in results if r.backend == "LanceDB"]
        sv_results     = [r for r in results if r.backend == "sqlite-vec"]

        lance_insert_avg = np.mean([r.insert_s for r in lance_results])
        sv_insert_avg    = np.mean([r.insert_s for r in sv_results])
        lance_query_avg  = np.mean([r.query_avg_ms for r in lance_results])
        sv_query_avg     = np.mean([r.query_avg_ms for r in sv_results])

        insert_winner = "LanceDB" if lance_insert_avg < sv_insert_avg else "sqlite-vec"
        query_winner  = "LanceDB" if lance_query_avg  < sv_query_avg  else "sqlite-vec"

        print(
            f"\nInsert winner  : {insert_winner}"
            f"  (LanceDB {lance_insert_avg:.3f}s avg vs sqlite-vec {sv_insert_avg:.3f}s avg)"
        )
        print(
            f"Query winner   : {query_winner}"
            f"  (LanceDB {lance_query_avg:.3f}ms avg vs sqlite-vec {sv_query_avg:.3f}ms avg)"
        )

    print(
        "\nPortability    : sqlite-vec → single .db file (easy backup/share)."
        "\n                 LanceDB   → directory of Lance/Parquet files (richer schema, ANN index)."
    )

    if sqlite_vec_available:
        if query_winner == "LanceDB":
            rec = (
                "Keep LanceDB. It wins on query latency (the hot path for every user "
                "request) and offers ANN indexing that will matter as the corpus grows. "
                "sqlite-vec's single-file portability is attractive but not a blocker "
                "given pocket-gm's local-first design."
            )
        else:
            rec = (
                "Consider switching to sqlite-vec. It is faster for both insert and "
                "query at these scales, ships as a single portable .db file, and has "
                "no heavyweight Rust/Arrow dependency chain. Revisit if the corpus "
                "exceeds 100K documents where LanceDB ANN indexing may tip the balance."
            )
        print(f"\nRecommendation : {rec}")
    else:
        print(
            "\nRecommendation : sqlite-vec could not be installed in this environment. "
            "Keep LanceDB for now; re-evaluate when sqlite-vec wheels are available."
        )

    print()


if __name__ == "__main__":
    main()
