from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pocket_gm.retrieval.store import RetrievedChunk, Store, notes_table, sessions_table, sourcebook_table


@dataclass
class QueryResult:
    sourcebook: list[RetrievedChunk]
    notes: list[RetrievedChunk]
    sessions: list[RetrievedChunk]

    def all_above_threshold(self, threshold: float) -> list[RetrievedChunk]:
        combined = self.sourcebook + self.notes + self.sessions
        return [c for c in combined if c.score >= threshold]

    def is_empty(self, threshold: float) -> bool:
        return len(self.all_above_threshold(threshold)) == 0


async def _query_table(store: Store, table_name: str, query_vec, top_k: int) -> list[RetrievedChunk]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, store.query, table_name, query_vec, top_k)


async def query_all(
    store: Store,
    campaign_id: str,
    query_vector,
    top_k: int = 5,
) -> QueryResult:
    sb_table = sourcebook_table(campaign_id)
    nt_table = notes_table(campaign_id)
    ss_table = sessions_table(campaign_id)

    sb, nt, ss = await asyncio.gather(
        _query_table(store, sb_table, query_vector, top_k),
        _query_table(store, nt_table, query_vector, top_k),
        _query_table(store, ss_table, query_vector, top_k),
    )

    return QueryResult(sourcebook=sb, notes=nt, sessions=ss)


def query_all_sync(
    store: Store,
    campaign_id: str,
    query_vector,
    top_k: int = 5,
) -> QueryResult:
    return asyncio.run(query_all(store, campaign_id, query_vector, top_k))
