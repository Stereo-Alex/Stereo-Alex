"""
End-to-end functional tests that drive the real CLI commands through Typer's
CliRunner — exercising the command modules that unit tests never touch (and
where the `ingest notes` crash, the empty-slug bug, and the delete-orphan bug
all lived).

Everything is real except the embedding model: the Embedder is patched to a
deterministic word-hashing fake so no network/model download is needed. The
sqlite-vec store, PDF parsing, Obsidian loader, registry, and config are all
genuine.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from pocket_gm.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Deterministic fake embedder (word-hashing → meaningful overlap similarity)
# ---------------------------------------------------------------------------
def _fake_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class _FakeEmbedder:
    def __init__(self, *a, **k):
        pass

    def embed(self, texts, batch_size: int = 64):
        return np.stack([_fake_embed(t) for t in texts])

    def embed_one(self, text):
        return _fake_embed(text)


@pytest.fixture
def gm_env(tmp_path, monkeypatch):
    """Redirect ~/.pocket-gm to a temp HOME and patch the embedder everywhere."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Config dataclass captures HOME at import via default_factory(expanduser),
    # so also patch the module-level default paths to the temp home.
    from pocket_gm.core import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_DEFAULT_CONFIG_PATH", tmp_path / ".pocket-gm" / "config.yaml")

    # Make every command construct the fake embedder.
    monkeypatch.setattr("pocket_gm.cli.commands.ingest.Embedder", _FakeEmbedder)
    monkeypatch.setattr("pocket_gm.cli.commands.query.Embedder", _FakeEmbedder)
    monkeypatch.setattr("pocket_gm.cli.commands.sessions.Embedder", _FakeEmbedder)
    return tmp_path


def _write_sample_pdf(path: Path) -> None:
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 60), "THE STAG LORD", fontsize=16)
    page.insert_text((50, 100),
                     "The Stag Lord rules from a keep on the Tuskwater. "
                     "His weakness is severe alcoholism.", fontsize=10)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# campaign lifecycle
# ---------------------------------------------------------------------------
def test_campaign_new_and_list(gm_env):
    r = runner.invoke(app, ["campaign", "new", "Kingmaker"])
    assert r.exit_code == 0, r.output
    assert "kingmaker" in r.output

    r = runner.invoke(app, ["campaign", "list"])
    assert r.exit_code == 0
    assert "kingmaker" in r.output


def test_campaign_new_duplicate_rejected(gm_env):
    runner.invoke(app, ["campaign", "new", "Kingmaker"])
    r = runner.invoke(app, ["campaign", "new", "Kingmaker"])
    assert r.exit_code == 1
    assert "already exists" in r.output


def test_campaign_new_empty_slug_rejected(gm_env):
    """L3: a name that slugifies to empty must be rejected, not stored blank."""
    r = runner.invoke(app, ["campaign", "new", "!!!"])
    assert r.exit_code == 1
    assert "empty campaign id" in r.output


# ---------------------------------------------------------------------------
# ingest pdf
# ---------------------------------------------------------------------------
def test_ingest_pdf_then_status(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)

    r = runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output
    assert "Ingested" in r.output

    r = runner.invoke(app, ["status", "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output
    assert "Sourcebook" in r.output


def test_ingest_pdf_idempotent(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])

    r = runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])
    assert r.exit_code == 0
    assert "Already ingested" in r.output


def test_ingest_pdf_force_replaces_not_duplicates(gm_env, tmp_path):
    """--force must replace a file's chunks, not append a second copy."""
    from pocket_gm.core.config import load_config
    from pocket_gm.retrieval.store import Store, sourcebook_table

    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])

    cfg = load_config()
    n1 = Store(cfg.lancedb_path).count(sourcebook_table("kingmaker"))

    r = runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker", "--force"])
    assert r.exit_code == 0, r.output
    n2 = Store(cfg.lancedb_path).count(sourcebook_table("kingmaker"))
    assert n2 == n1, f"force re-ingest duplicated chunks: {n1} -> {n2}"


# ---------------------------------------------------------------------------
# ingest notes — the path that used to crash (C2)
# ---------------------------------------------------------------------------
def test_ingest_notes_directory(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "stag.md").write_text("# Stag Lord\nHe is a bandit lord.\n")
    (notes_dir / "oleg.md").write_text("# Oleg\nRuns a trading post.\n")

    r = runner.invoke(app, ["ingest", "notes", str(notes_dir), "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output
    assert "Ingested" in r.output

    # Re-running must skip both (idempotent) without crashing.
    r2 = runner.invoke(app, ["ingest", "notes", str(notes_dir), "--campaign", "kingmaker"])
    assert r2.exit_code == 0, r2.output
    assert "already ingested" in r2.output.lower()


def test_ingest_notes_single_file(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    f = tmp_path / "lore.md"
    f.write_text("# Lore\nThe Stolen Lands are wild.\n")
    r = runner.invoke(app, ["ingest", "notes", str(f), "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output


# ---------------------------------------------------------------------------
# ingest obsidian
# ---------------------------------------------------------------------------
def test_ingest_obsidian_vault(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Stag Lord.md").write_text(
        "---\ntags: [villain]\n---\n# Stag Lord\nControls [[Tuskwater Keep]].\n"
    )
    (vault / "Tuskwater Keep.md").write_text("# Tuskwater Keep\nA fortress.\n")

    r = runner.invoke(app, ["ingest", "obsidian", str(vault), "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output
    assert "notes found" in r.output


# ---------------------------------------------------------------------------
# missing campaign / missing file error paths
# ---------------------------------------------------------------------------
def test_ingest_unknown_campaign(gm_env, tmp_path):
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    r = runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "ghost"])
    assert r.exit_code == 1
    assert "not found" in r.output.lower()


def test_ingest_missing_file(gm_env):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    r = runner.invoke(app, ["ingest", "pdf", "/no/such/file.pdf", "--campaign", "kingmaker"])
    assert r.exit_code == 1
    assert "not found" in r.output.lower()


# ---------------------------------------------------------------------------
# campaign delete cleanup (L2)
# ---------------------------------------------------------------------------
def test_campaign_delete_removes_data(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])

    # campaign dir + registry should now exist
    campaign_dir = tmp_path / ".pocket-gm" / "campaigns" / "kingmaker"
    assert campaign_dir.exists()

    r = runner.invoke(app, ["campaign", "delete", "kingmaker"], input="y\n")
    assert r.exit_code == 0, r.output
    assert "Deleted" in r.output

    # registry/transcripts dir removed; vector tables dropped
    assert not campaign_dir.exists()
    from pocket_gm.core.config import load_config
    from pocket_gm.retrieval.store import Store, sourcebook_table
    cfg = load_config()
    store = Store(cfg.lancedb_path)
    assert store.count(sourcebook_table("kingmaker")) == 0


def test_campaign_delete_unknown(gm_env):
    r = runner.invoke(app, ["campaign", "delete", "ghost"])
    assert r.exit_code == 1
    assert "not found" in r.output.lower()


# ---------------------------------------------------------------------------
# full ask pipeline with a stubbed LLM (real retrieval + grounding)
# ---------------------------------------------------------------------------
def test_ask_end_to_end_with_stub_llm(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])

    class _StubLLM:
        def is_available(self):
            return True

        def generate(self, prompt, temperature=0.1, max_tokens=512):
            return "The Stag Lord's weakness is alcoholism [1]."

    with patch("pocket_gm.cli.commands.query.build_llm_client", return_value=_StubLLM()):
        r = runner.invoke(app, ["ask", "What is the Stag Lord's weakness?", "--campaign", "kingmaker"])

    assert r.exit_code == 0, r.output
    assert "alcoholism" in r.output

    # A query log should now exist with all_scores recorded (M3/M4).
    import json
    log_file = tmp_path / ".pocket-gm" / "logs" / "queries.ndjson"
    assert log_file.exists()
    record = json.loads(log_file.read_text().splitlines()[-1])
    assert "all_scores" in record
    assert record["campaign_id"] == "kingmaker"


def _fake_transcript():
    from pocket_gm.ingestion.audio_transcriber import Transcript, TranscriptSegment
    return Transcript(
        segments=[
            TranscriptSegment(start=0.0, end=10.0, text="The party stormed the Stag Lord's fort."),
            TranscriptSegment(start=10.0, end=20.0, text="Akiros opened the gate and surrendered."),
        ],
        language="en",
    )


def test_session_add_is_idempotent(gm_env, tmp_path):
    """M2: re-adding the same audio file must be skipped, not re-ingested."""
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    audio = tmp_path / "session04.wav"
    audio.write_bytes(b"RIFF....fake wav bytes....")

    with patch("pocket_gm.cli.commands.sessions.transcribe", return_value=_fake_transcript()):
        r1 = runner.invoke(app, [
            "session", "add", str(audio),
            "--campaign", "kingmaker", "--date", "2025-03-10", "--number", "4",
        ])
        assert r1.exit_code == 0, r1.output
        assert "ingested" in r1.output.lower()

        # Second call: transcribe must NOT be invoked (early-return on registry).
        with patch("pocket_gm.cli.commands.sessions.transcribe") as mock_t:
            r2 = runner.invoke(app, [
                "session", "add", str(audio),
                "--campaign", "kingmaker", "--date", "2025-03-10", "--number", "4",
            ])
            assert r2.exit_code == 0, r2.output
            assert "already ingested" in r2.output.lower()
            mock_t.assert_not_called()


def test_session_add_force_reingests(gm_env, tmp_path):
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    audio = tmp_path / "session04.wav"
    audio.write_bytes(b"RIFF....fake wav bytes....")

    with patch("pocket_gm.cli.commands.sessions.transcribe", return_value=_fake_transcript()):
        runner.invoke(app, ["session", "add", str(audio), "--campaign", "kingmaker",
                            "--date", "2025-03-10", "--number", "4"])
        # --force bypasses the "already ingested" gate and re-ingests. The
        # existing transcript JSON is reused (re-transcription is expensive and
        # deterministic), so the command must NOT report a skip.
        r = runner.invoke(app, ["session", "add", str(audio), "--campaign", "kingmaker",
                                "--date", "2025-03-10", "--number", "4", "--force"])
        assert r.exit_code == 0, r.output
        assert "already ingested" not in r.output.lower()
        assert "ingested" in r.output.lower()


def test_eval_calibrate_after_query(gm_env, tmp_path):
    """eval calibrate should read the logged score distribution (M3)."""
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])

    class _StubLLM:
        def is_available(self): return True
        def generate(self, prompt, temperature=0.1, max_tokens=512):
            return "Alcoholism is his weakness [1]."

    with patch("pocket_gm.cli.commands.query.build_llm_client", return_value=_StubLLM()):
        runner.invoke(app, ["ask", "Stag Lord weakness", "--campaign", "kingmaker"])

    r = runner.invoke(app, ["eval", "calibrate", "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output
    assert "threshold" in r.output.lower()


def test_ask_no_results_logs_scores(gm_env, tmp_path):
    """A below-threshold query should still log its scores for calibration."""
    runner.invoke(app, ["campaign", "new", "kingmaker"])
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    runner.invoke(app, ["ingest", "pdf", str(pdf), "--campaign", "kingmaker"])

    # A query with zero word overlap → all similarities 0 → below threshold.
    r = runner.invoke(app, ["ask", "zzzqqq xyzzy plugh", "--campaign", "kingmaker"])
    assert r.exit_code == 0, r.output
    assert "No relevant information" in r.output

    import json
    log_file = tmp_path / ".pocket-gm" / "logs" / "queries.ndjson"
    assert log_file.exists()
    record = json.loads(log_file.read_text().splitlines()[-1])
    assert "all_scores" in record
