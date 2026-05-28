from __future__ import annotations

from fastapi import FastAPI, HTTPException

from pocket_gm.api.models import (
    CampaignOut,
    CitationOut,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionOut,
    StatusOut,
)
from pocket_gm.core.campaign import get_campaign, list_campaigns
from pocket_gm.core.config import load_config
from pocket_gm.core.logger import log_query
from pocket_gm.ingestion.embedder import Embedder
from pocket_gm.retrieval.router import query_all_sync
from pocket_gm.retrieval.store import Store, notes_table, sessions_table, sourcebook_table
from pocket_gm.synthesis.grounding import parse_json_answer, validate_citations
from pocket_gm.synthesis.llm import build_llm_client
from pocket_gm.synthesis.ollama_client import OllamaClient
from pocket_gm.synthesis.prompt_builder import build_prompt

app = FastAPI(
    title="Pocket GM API",
    description="Grounded Q&A for tabletop RPG Game Masters.",
    version="0.1.0",
)


# Module-level singletons — loaded once on startup so the embedding model
# is not re-initialized on every request.
_cfg = None
_embedder = None
_store = None


def _get_singletons():
    global _cfg, _embedder, _store
    if _cfg is None:
        _cfg = load_config()
        _embedder = Embedder(_cfg.embedding.model)
        _store = Store(_cfg.lancedb_path)
    return _cfg, _embedder, _store


@app.get("/health", response_model=HealthResponse)
def health():
    cfg, _, _ = _get_singletons()
    ollama = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
    return HealthResponse(status="ok", ollama_available=ollama.is_available())


@app.get("/campaigns", response_model=list[CampaignOut])
def get_campaigns():
    cfg, _, _ = _get_singletons()
    return [CampaignOut(id=c.id, name=c.name, created_at=c.created_at) for c in list_campaigns(cfg.campaigns_dir)]


@app.get("/campaigns/{campaign_id}/sessions", response_model=list[SessionOut])
def get_sessions(campaign_id: str):
    cfg, _, store = _get_singletons()
    if not get_campaign(cfg.campaigns_dir, campaign_id):
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")
    sessions = store.list_sessions(sessions_table(campaign_id))
    return [SessionOut(**s) for s in sessions]


@app.get("/campaigns/{campaign_id}/status", response_model=StatusOut)
def get_status(campaign_id: str):
    cfg, _, store = _get_singletons()
    if not get_campaign(cfg.campaigns_dir, campaign_id):
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")
    return StatusOut(
        campaign_id=campaign_id,
        sourcebook_chunks=store.count(sourcebook_table(campaign_id)),
        notes_chunks=store.count(notes_table(campaign_id)),
        sessions_chunks=store.count(sessions_table(campaign_id)),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    cfg, embedder, store = _get_singletons()

    if not get_campaign(cfg.campaigns_dir, req.campaign_id):
        raise HTTPException(status_code=404, detail=f"Campaign '{req.campaign_id}' not found")

    query_vec = embedder.embed_one(req.question)
    result = query_all_sync(store, req.campaign_id, query_vec, top_k=req.top_k)
    all_retrieved = result.sourcebook + result.notes + result.sessions

    if result.is_empty(cfg.retrieval.relevance_threshold):
        from pocket_gm.synthesis.grounding import GroundedAnswer
        try:
            log_query(
                cfg.logs_path, req.campaign_id, req.question,
                GroundedAnswer(text="", citations_used=[], uncited_sentences=[], index_map=[]),
                cfg.llm.model, all_retrieved=all_retrieved,
            )
        except Exception:
            pass
        return QueryResponse(
            answer="No relevant information found in your campaign materials.",
            citations=[],
            fully_grounded=True,
            uncited_count=0,
        )

    prompt, index_map = build_prompt(
        req.question,
        result.sourcebook,
        result.notes,
        result.sessions,
        threshold=cfg.retrieval.relevance_threshold,
        json_mode=cfg.llm.json_citations,
    )

    try:
        llm = build_llm_client(cfg.llm)
    except (RuntimeError, ImportError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not llm.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider '{cfg.llm.provider}' is not available.",
        )

    answer_text = llm.generate(prompt, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)

    grounded = (
        parse_json_answer(answer_text, index_map)
        if cfg.llm.json_citations
        else validate_citations(answer_text, index_map)
    )

    # Log the query so /query traffic feeds the eval harness and calibration,
    # exactly like the CLI's `ask` command does.
    try:
        log_query(cfg.logs_path, req.campaign_id, req.question, grounded, cfg.llm.model, all_retrieved=all_retrieved)
    except Exception:
        # Logging must never break a query response.
        pass

    citations = [
        CitationOut(
            ref=idx,
            source_type=chunk.source_type,
            filename=chunk.filename,
            page=chunk.page,
            heading=chunk.heading,
            session_number=chunk.session_number,
            session_date=chunk.session_date,
            timestamp_start=chunk.timestamp_start,
            score=round(chunk.score, 4),
        )
        for idx, chunk in index_map
    ]

    return QueryResponse(
        answer=grounded.text,
        citations=citations,
        fully_grounded=grounded.is_fully_grounded,
        uncited_count=len(grounded.uncited_sentences),
    )
