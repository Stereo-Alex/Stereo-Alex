from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def get_registry_path(campaigns_dir: Path, campaign_id: str) -> Path:
    """Return the path to ingested.json for a given campaign."""
    campaign_dir = campaigns_dir / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    return campaign_dir / "ingested.json"


def _load(registry_path: Path) -> dict:
    if not registry_path.exists():
        return {}
    with open(registry_path) as f:
        return json.load(f)


def _save(registry_path: Path, data: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(data, f, indent=2)


def is_ingested(registry_path: Path, file_hash: str) -> bool:
    """Return True if the file with the given SHA256 hash has already been ingested."""
    return file_hash in _load(registry_path)


def mark_ingested(
    registry_path: Path,
    file_hash: str,
    filename: str,
    chunk_count: int,
) -> None:
    """Record that a file has been ingested."""
    data = _load(registry_path)
    data[file_hash] = {
        "filename": filename,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
    }
    _save(registry_path, data)


def hash_file(file_path: Path, chunk_size: int = 65536) -> str:
    """Return a SHA256 hex digest of the file's full contents.

    Hashing the whole file (streamed in *chunk_size* blocks) avoids false
    "already ingested" collisions between different files that happen to share
    a common header — common for exported PDFs or recordings from the same rig.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
