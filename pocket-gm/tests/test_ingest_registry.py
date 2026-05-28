from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pocket_gm.core.ingest_registry import (
    get_registry_path,
    hash_file,
    is_ingested,
    mark_ingested,
)


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "campaigns" / "test_campaign" / "ingested.json"


# ── is_ingested ───────────────────────────────────────────────────────────────

def test_is_ingested_returns_false_for_unknown_hash(registry_path):
    assert is_ingested(registry_path, "deadbeef") is False


def test_is_ingested_returns_false_when_registry_missing(registry_path):
    # File does not exist yet
    assert not registry_path.exists()
    assert is_ingested(registry_path, "abc123") is False


# ── mark_ingested + is_ingested ───────────────────────────────────────────────

def test_mark_then_is_ingested_returns_true(registry_path):
    mark_ingested(registry_path, "aabbcc", "sourcebook.pdf", 42)
    assert is_ingested(registry_path, "aabbcc") is True


def test_mark_ingested_creates_registry_file(registry_path):
    assert not registry_path.exists()
    mark_ingested(registry_path, "hash1", "file.pdf", 10)
    assert registry_path.exists()


# ── Persistence across calls ──────────────────────────────────────────────────

def test_registry_persists_across_calls(tmp_path):
    rp = tmp_path / "campaigns" / "camp" / "ingested.json"
    mark_ingested(rp, "hash_persist", "notes.md", 5)

    # Simulate a separate "call" by loading via is_ingested using the same path
    assert is_ingested(rp, "hash_persist") is True


# ── Independent hashes ────────────────────────────────────────────────────────

def test_different_hashes_are_tracked_independently(registry_path):
    mark_ingested(registry_path, "hash_a", "file_a.pdf", 100)
    mark_ingested(registry_path, "hash_b", "file_b.pdf", 200)

    assert is_ingested(registry_path, "hash_a") is True
    assert is_ingested(registry_path, "hash_b") is True
    assert is_ingested(registry_path, "hash_c") is False


def test_unknown_hash_returns_false_after_other_marks(registry_path):
    mark_ingested(registry_path, "known", "known.pdf", 1)
    assert is_ingested(registry_path, "unknown") is False


# ── hash_file consistency ─────────────────────────────────────────────────────

def test_hash_file_consistent_for_same_content(tmp_path):
    content = b"Hello, Golarion!" * 1000
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"
    f1.write_bytes(content)
    f2.write_bytes(content)

    assert hash_file(f1) == hash_file(f2)


def test_hash_file_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"content one")
    f2.write_bytes(b"content two")

    assert hash_file(f1) != hash_file(f2)


def test_hash_file_matches_sha256(tmp_path):
    content = b"Stag Lord chapter"
    f = tmp_path / "sourcebook.pdf"
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert hash_file(f) == expected


def test_hash_file_distinguishes_files_sharing_a_large_header(tmp_path):
    """Two files with an identical 128 KB header but different tails must hash
    differently — guards against the old first-64 KB-only behaviour that caused
    false 'already ingested' skips."""
    shared_header = b"X" * (128 * 1024)
    f1 = tmp_path / "book_a.pdf"
    f2 = tmp_path / "book_b.pdf"
    f1.write_bytes(shared_header + b"unique tail A")
    f2.write_bytes(shared_header + b"unique tail B")

    assert hash_file(f1) != hash_file(f2)


# ── get_registry_path ─────────────────────────────────────────────────────────

def test_get_registry_path_creates_campaign_dir(tmp_path):
    campaigns_dir = tmp_path / "campaigns"
    rp = get_registry_path(campaigns_dir, "kingmaker")
    assert rp == campaigns_dir / "kingmaker" / "ingested.json"
    assert (campaigns_dir / "kingmaker").is_dir()
