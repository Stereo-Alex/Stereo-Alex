from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64


@dataclass
class RetrievalConfig:
    top_k: int = 5
    relevance_threshold: float = 0.35
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "phi3:mini"
    temperature: float = 0.1
    max_tokens: int = 512
    json_citations: bool = False


@dataclass
class WhisperConfig:
    model_size: str = "base"
    language: str = "en"
    device: str = "cpu"


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class ChunkingConfig:
    sourcebook: ChunkConfig = field(default_factory=lambda: ChunkConfig(512, 64))
    notes: ChunkConfig = field(default_factory=lambda: ChunkConfig(256, 32))
    sessions: ChunkConfig = field(default_factory=lambda: ChunkConfig(256, 32))


@dataclass
class Config:
    campaigns_dir: Path = field(default_factory=lambda: Path("~/.pocket-gm/campaigns").expanduser())
    lancedb_path: Path = field(default_factory=lambda: Path("~/.pocket-gm/lancedb").expanduser())
    logs_path: Path = field(default_factory=lambda: Path("~/.pocket-gm/logs").expanduser())
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


_DEFAULT_CONFIG_PATH = Path("~/.pocket-gm/config.yaml").expanduser()


def load_config(path: Path | None = None) -> Config:
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    cfg = Config()
    if "campaigns_dir" in data:
        cfg.campaigns_dir = Path(data["campaigns_dir"]).expanduser()
    if "lancedb_path" in data:
        cfg.lancedb_path = Path(data["lancedb_path"]).expanduser()
    if "logs_path" in data:
        cfg.logs_path = Path(data["logs_path"]).expanduser()

    if emb := data.get("embedding"):
        cfg.embedding = EmbeddingConfig(**emb)
    if ret := data.get("retrieval"):
        cfg.retrieval = RetrievalConfig(**ret)
    if llm := data.get("llm"):
        cfg.llm = LLMConfig(**llm)
    if wh := data.get("whisper"):
        cfg.whisper = WhisperConfig(**wh)
    if ch := data.get("chunking"):
        cfg.chunking = ChunkingConfig(
            sourcebook=ChunkConfig(**ch["sourcebook"]) if "sourcebook" in ch else ChunkConfig(512, 64),
            notes=ChunkConfig(**ch["notes"]) if "notes" in ch else ChunkConfig(256, 32),
            sessions=ChunkConfig(**ch["sessions"]) if "sessions" in ch else ChunkConfig(256, 32),
        )

    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    config_path = path or _DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "campaigns_dir": str(cfg.campaigns_dir),
        "lancedb_path": str(cfg.lancedb_path),
        "logs_path": str(cfg.logs_path),
        "embedding": {"model": cfg.embedding.model, "batch_size": cfg.embedding.batch_size},
        "retrieval": {
            "top_k": cfg.retrieval.top_k,
            "relevance_threshold": cfg.retrieval.relevance_threshold,
            "reranker_enabled": cfg.retrieval.reranker_enabled,
            "reranker_model": cfg.retrieval.reranker_model,
        },
        "llm": {
            "provider": cfg.llm.provider,
            "base_url": cfg.llm.base_url,
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
            "json_citations": cfg.llm.json_citations,
        },
        "whisper": {
            "model_size": cfg.whisper.model_size,
            "language": cfg.whisper.language,
            "device": cfg.whisper.device,
        },
        "chunking": {
            "sourcebook": {"chunk_size": cfg.chunking.sourcebook.chunk_size, "chunk_overlap": cfg.chunking.sourcebook.chunk_overlap},
            "notes": {"chunk_size": cfg.chunking.notes.chunk_size, "chunk_overlap": cfg.chunking.notes.chunk_overlap},
            "sessions": {"chunk_size": cfg.chunking.sessions.chunk_size, "chunk_overlap": cfg.chunking.sessions.chunk_overlap},
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
