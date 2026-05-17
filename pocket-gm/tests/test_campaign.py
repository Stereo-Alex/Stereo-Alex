import json
import pytest
from pathlib import Path
from pocket_gm.core.campaign import Campaign, add_campaign, delete_campaign, get_campaign, list_campaigns


@pytest.fixture
def campaigns_dir(tmp_path):
    return tmp_path / "campaigns"


def test_new_campaign_creates_registry(campaigns_dir):
    c = Campaign(id="kingmaker", name="Kingmaker", created_at="2025-01-01T00:00:00Z")
    add_campaign(campaigns_dir, c)
    assert (campaigns_dir / "campaigns.json").exists()
    result = list_campaigns(campaigns_dir)
    assert len(result) == 1
    assert result[0].id == "kingmaker"


def test_duplicate_campaign_raises(campaigns_dir):
    c = Campaign(id="kingmaker", name="Kingmaker", created_at="2025-01-01T00:00:00Z")
    add_campaign(campaigns_dir, c)
    with pytest.raises(ValueError):
        add_campaign(campaigns_dir, c)


def test_get_campaign(campaigns_dir):
    c = Campaign(id="pf2e", name="PF2E", created_at="2025-01-01T00:00:00Z")
    add_campaign(campaigns_dir, c)
    assert get_campaign(campaigns_dir, "pf2e") == c
    assert get_campaign(campaigns_dir, "missing") is None


def test_delete_campaign(campaigns_dir):
    c = Campaign(id="test", name="Test", created_at="2025-01-01T00:00:00Z")
    add_campaign(campaigns_dir, c)
    delete_campaign(campaigns_dir, "test")
    assert list_campaigns(campaigns_dir) == []


def test_empty_registry(campaigns_dir):
    assert list_campaigns(campaigns_dir) == []
