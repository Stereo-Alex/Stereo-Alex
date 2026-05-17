from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    campaign_id: str = Field(..., description="Campaign ID")
    top_k: int = Field(5, ge=1, le=20, description="Chunks to retrieve per store")


class CitationOut(BaseModel):
    ref: int
    source_type: str
    filename: str
    page: int
    heading: str
    session_number: int
    session_date: str
    timestamp_start: float
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    fully_grounded: bool
    uncited_count: int


class CampaignOut(BaseModel):
    id: str
    name: str
    created_at: str


class SessionOut(BaseModel):
    session_number: int
    session_date: str
    chunk_count: int


class StatusOut(BaseModel):
    campaign_id: str
    sourcebook_chunks: int
    notes_chunks: int
    sessions_chunks: int


class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_available: bool
