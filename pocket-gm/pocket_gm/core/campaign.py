from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Campaign:
    id: str
    name: str
    created_at: str


def _registry_path(campaigns_dir: Path) -> Path:
    campaigns_dir.mkdir(parents=True, exist_ok=True)
    return campaigns_dir / "campaigns.json"


def list_campaigns(campaigns_dir: Path) -> list[Campaign]:
    path = _registry_path(campaigns_dir)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [Campaign(**c) for c in data]


def get_campaign(campaigns_dir: Path, campaign_id: str) -> Campaign | None:
    for c in list_campaigns(campaigns_dir):
        if c.id == campaign_id:
            return c
    return None


def add_campaign(campaigns_dir: Path, campaign: Campaign) -> None:
    campaigns = list_campaigns(campaigns_dir)
    if any(c.id == campaign.id for c in campaigns):
        raise ValueError(f"Campaign '{campaign.id}' already exists")
    campaigns.append(campaign)
    _save(campaigns_dir, campaigns)


def delete_campaign(campaigns_dir: Path, campaign_id: str) -> None:
    campaigns = [c for c in list_campaigns(campaigns_dir) if c.id != campaign_id]
    _save(campaigns_dir, campaigns)


def _save(campaigns_dir: Path, campaigns: list[Campaign]) -> None:
    path = _registry_path(campaigns_dir)
    with open(path, "w") as f:
        json.dump([asdict(c) for c in campaigns], f, indent=2)
