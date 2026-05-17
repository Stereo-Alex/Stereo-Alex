import pytest
from pathlib import Path
from pocket_gm.ingestion.obsidian_loader import (
    _parse_frontmatter,
    _clean_body,
    load_obsidian_vault,
    load_obsidian_file,
)
from pocket_gm.ingestion.chunker import chunk_obsidian_note


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_note(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


# ── Frontmatter ───────────────────────────────────────────────────────────────

def test_frontmatter_extracted():
    text = "---\ntitle: Stag Lord\ntags: [npc, villain]\n---\nHe is a bandit."
    meta, body = _parse_frontmatter(text)
    assert meta["title"] == "Stag Lord"
    assert "npc" in meta["tags"]
    assert body.strip() == "He is a bandit."


def test_no_frontmatter():
    text = "# Just a heading\nSome text."
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert "Just a heading" in body


def test_malformed_frontmatter_does_not_crash():
    text = "---\n: bad yaml [\n---\nBody text."
    meta, body = _parse_frontmatter(text)
    assert isinstance(meta, dict)


# ── Wikilinks ─────────────────────────────────────────────────────────────────

def test_wikilink_replaced_with_note_name():
    body = "The [[Stag Lord]] controls the Greenbelt."
    result = _clean_body(body, {}, set())
    assert "[[" not in result
    assert "Stag Lord" in result


def test_wikilink_with_alias_uses_alias():
    body = "He is known as [[Stag Lord|the Stag Lord]]."
    result = _clean_body(body, {}, set())
    assert "the Stag Lord" in result
    assert "[[" not in result


def test_image_embed_stripped():
    body = "See this image: ![[map.png]] for reference."
    result = _clean_body(body, {}, set())
    assert "![[" not in result
    assert "map.png" not in result


def test_note_embed_resolved(tmp_path):
    linked = write_note(tmp_path, "Stag Fort", "The fort sits on the Tuskwater lake.")
    vault_index = {"stag fort": linked}
    body = "The party approached ![[Stag Fort]] from the north."
    result = _clean_body(body, vault_index, set())
    assert "Tuskwater" in result
    assert "![[" not in result


def test_embed_cycle_does_not_recurse_forever(tmp_path):
    # Note A embeds B, B embeds A — should not infinite loop
    a = write_note(tmp_path, "NoteA", "Content of A, see ![[NoteB]]")
    b = write_note(tmp_path, "NoteB", "Content of B, see ![[NoteA]]")
    vault_index = {"notea": a, "noteb": b}
    result = _clean_body(a.read_text(), vault_index, {"notea"})
    assert isinstance(result, str)  # just no crash / infinite loop


# ── Callouts ─────────────────────────────────────────────────────────────────

def test_callout_header_stripped():
    body = "> [!NOTE] Remember this\n> The Stag Lord fears his father."
    result = _clean_body(body, {}, set())
    assert "[!NOTE]" not in result
    assert "Stag Lord fears his father" in result


def test_callout_blockquote_markers_stripped():
    body = "> [!WARNING]\n> This is dangerous.\n> Be careful."
    result = _clean_body(body, {}, set())
    assert ">" not in result
    assert "dangerous" in result


# ── Dataview ─────────────────────────────────────────────────────────────────

def test_dataview_block_stripped():
    body = "Some text.\n```dataview\nLIST FROM #npc\n```\nMore text."
    result = _clean_body(body, {}, set())
    assert "dataview" not in result.lower()
    assert "LIST FROM" not in result
    assert "Some text" in result
    assert "More text" in result


def test_dataview_inline_stripped():
    body = "The value is `= this.rating` for reference."
    result = _clean_body(body, {}, set())
    assert "`=" not in result


# ── Tags ─────────────────────────────────────────────────────────────────────

def test_inline_tags_converted():
    body = "The #Stag-Lord is a #villain in the Stolen Lands."
    result = _clean_body(body, {}, set())
    assert "#" not in result
    assert "Stag-Lord" in result or "Stag Lord" in result


# ── Full vault load ───────────────────────────────────────────────────────────

def test_load_vault_finds_notes(tmp_path):
    write_note(tmp_path, "Stag Lord", "---\ntags: [npc]\n---\n# Stag Lord\nA bandit warlord.")
    write_note(tmp_path, "Stolen Lands", "# Stolen Lands\nA wild frontier region.")
    notes = load_obsidian_vault(tmp_path)
    assert len(notes) == 2
    titles = [n.title for n in notes]
    assert "Stag Lord" in titles


def test_load_vault_skips_dot_folders(tmp_path):
    obsidian_dir = tmp_path / ".obsidian"
    obsidian_dir.mkdir()
    (obsidian_dir / "config.md").write_text("internal")
    write_note(tmp_path, "Real Note", "Actual content.")
    notes = load_obsidian_vault(tmp_path)
    assert len(notes) == 1
    assert notes[0].title == "Real Note"


def test_load_vault_tags_extracted(tmp_path):
    write_note(tmp_path, "Villain", "---\ntags: [npc, villain]\n---\nEvil character.")
    notes = load_obsidian_vault(tmp_path)
    assert "npc" in notes[0].tags
    assert "villain" in notes[0].tags


def test_load_vault_resolves_wikilinks(tmp_path):
    write_note(tmp_path, "Fort", "The fort is on the Tuskwater.")
    write_note(tmp_path, "Stag Lord", "He lives in [[Fort]].")
    notes = load_obsidian_vault(tmp_path)
    stag = next(n for n in notes if n.title == "Stag Lord")
    assert "[[" not in stag.body
    assert "Fort" in stag.body


def test_empty_note_skipped(tmp_path):
    write_note(tmp_path, "Empty", "---\ntags: []\n---\n")
    notes = load_obsidian_vault(tmp_path)
    assert len(notes) == 0


# ── Chunking ─────────────────────────────────────────────────────────────────

def test_chunk_obsidian_note_source_type(tmp_path):
    write_note(tmp_path, "Stag Lord", "# Stag Lord\nA bandit warlord controlling the Greenbelt.")
    notes = load_obsidian_vault(tmp_path)
    chunks = chunk_obsidian_note(notes[0], chunk_size=100, chunk_overlap=10)
    assert all(c.source_type == "obsidian" for c in chunks)
    assert len(chunks) >= 1


def test_chunk_carries_tags(tmp_path):
    write_note(tmp_path, "NPC", "---\ntags: [villain, npc]\n---\n# NPC\nSome villain.")
    notes = load_obsidian_vault(tmp_path)
    chunks = chunk_obsidian_note(notes[0])
    assert any("villain" in c.tags for c in chunks)


def test_chunk_title_in_text(tmp_path):
    write_note(tmp_path, "Stag Lord", "A bandit warlord.")
    notes = load_obsidian_vault(tmp_path)
    chunks = chunk_obsidian_note(notes[0], chunk_size=100, chunk_overlap=10)
    # Title is injected as heading prefix so it appears in chunk text
    assert any("Stag Lord" in c.text for c in chunks)
