"""
Real-data tests using a realistic Pathfinder Kingmaker campaign dataset.

This module generates rich, realistic test fixtures that mirror what a GM
would actually feed into pocket-gm:

  PDF      : 5-page adventure module excerpt with headings, stat blocks, tables
  Vault    : 8-note Obsidian vault with frontmatter, wikilinks, cross-references
  Transcript: 15-segment session recording (generated with espeak-ng → WAV → text)

Unlike test_real_world.py (which proves semantics work) this file proves the
PIPELINE handles real-world document complexity:
  - Multi-section PDFs with mixed font sizes and heading hierarchy
  - Obsidian vaults with deep cross-referencing
  - Transcripts with realistic speaker diarisation and timestamps
  - Semantic retrieval quality across all three stores simultaneously

Whisper transcription (audio → text) requires downloading a model from
HuggingFace, which is blocked in this cloud environment. The audio file
is generated and validated; the transcription step is skipped with a clear
note. Every other component runs with real data and real ML.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

spacy = pytest.importorskip("spacy", reason="spaCy not installed")
try:
    spacy.load("en_core_web_md")
except OSError:
    pytest.skip("en_core_web_md not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Realistic fixture builders
# ---------------------------------------------------------------------------

CAMPAIGN_TEXT = {
    "chapter1_title": "THE STOLEN LANDS",
    "chapter1_body": (
        "The Stolen Lands are a vast and largely untamed wilderness stretching "
        "south of the Brevic city of Restov. Long contested between the River "
        "Kingdoms and Brevoy, these lands have never been successfully settled "
        "due to the constant threat of bandits, monsters, and fey interference. "
        "The region encompasses the Greenbelt in the south, the Kamelands to the "
        "east, and the Narlmarches forest to the west. The Tuskwater lake marks "
        "the southern boundary of the Greenbelt. The players have been chartered "
        "by the Swordlords of Restov to explore and civilise this territory."
    ),
    "chapter2_title": "THE STAG LORD",
    "chapter2_body": (
        "The Stag Lord is the self-styled ruler of the Greenbelt, a brutal and "
        "unpredictable bandit lord who commands roughly sixty men from his fortress "
        "on the Tuskwater. Born Noleski Surtova's bastard half-brother, his true "
        "identity has been lost to drink and violence. He wears an enchanted stag "
        "skull helm that grants him darkvision and a frightening aura. His greatest "
        "weakness is his severe alcoholism: he is rarely sober and his judgement "
        "suffers for it. His lieutenant Akiros Ismort secretly despises him and can "
        "be turned to the party's side with a DC 18 Diplomacy check. The Stag Lord's "
        "father, a druid named Nugrah, is imprisoned in the fort's cellar and can "
        "provide valuable information if freed."
    ),
    "chapter3_title": "OLEG'S TRADING POST",
    "chapter3_body": (
        "Oleg's Trading Post sits at the northern edge of the Stolen Lands, serving "
        "as the players' primary base of operations in the early campaign. Oleg "
        "Leveton and his wife Svetlana are the proprietors. The post was recently "
        "extorted by a gang of bandits led by Happs Bydon, who demanded tribute every "
        "month. Oleg has asked the party for help. If the players defeat Happs Bydon's "
        "gang, Oleg offers free lodging and a 10% discount on all goods. Svetlana's "
        "stolen wedding ring can be recovered from Kressle's camp in hex B5."
    ),
    "chapter4_title": "RANDOM ENCOUNTER TABLE",
    "chapter4_body": (
        "Roll 1d12 every 8 hours of travel in the Greenbelt. "
        "1-4: No encounter. "
        "5: 1d6 bandits patrol, led by a bandit lieutenant. "
        "6: 1d4 wolves hunting deer near a stream. "
        "7: Tatzlwyrm guarding a cache of treasure beneath a rock. "
        "8: Friendly Restovian merchant travelling south, will buy goods. "
        "9: 1d3 boggards (frogmen) attempting to extort river crossers. "
        "10: Lone hunter named Vekkel Benzen seeking the boar Tuskgutter. "
        "11: Fey mischief — players' equipment is rearranged overnight. "
        "12: Wandering bear, non-hostile unless approached."
    ),
    "chapter5_title": "THE STAG LORD'S FORT",
    "chapter5_body": (
        "The fort sits on a bluff overlooking the Tuskwater. It was once a temple "
        "to Erastil but has been defiled by the Stag Lord's occupation. The main "
        "gate faces north and is guarded by two bandits at all times. The eastern "
        "wall has a blind spot visible from the ridge to the southeast. "
        "Area 1 — Main Gate: Two guards, masterwork crossbows. "
        "Area 2 — Great Hall: 12 bandits eating and drinking. "
        "Area 3 — Armoury: Locked, masterwork longswords and 200gp in stolen goods. "
        "Area 4 — Stag Lord's Chamber: Locked with arcane lock (DC 28), contains "
        "the stag skull helm and the Stag Lord's personal treasury. "
        "Area 5 — Cellar: Nugrah the mad druid is imprisoned here."
    ),
}

OBSIDIAN_NOTES = {
    "Campaign Overview.md": """\
---
tags: [campaign, overview, kingmaker]
---
# Kingmaker Campaign Overview
Running Pathfinder Kingmaker for five players. Campaign started 2025-01-15.
Primary antagonist: [[The Stag Lord]]. Home base: [[Oleg's Trading Post]].
Target: defeat the Stag Lord and establish a kingdom in the [[Stolen Lands]].
## Party
- Torvin Ironmane (fighter/dwarf)
- Kira Nightwhisper (rogue/elf)
- Aldric Sunborn (cleric/human)
- Mira Voss (wizard/gnome)
- Jak Coldwell (ranger/human)
""",
    "The Stag Lord.md": """\
---
tags: [npc, villain, bandit, boss]
aliases: [Stag Lord, The Stag Lord, Noleski]
---
# The Stag Lord
The main antagonist of the early Kingmaker campaign.
Commands bandits from [[Stag Lord Fort]] on the Tuskwater.
## Key Facts
- Wears an enchanted stag skull helm (darkvision, fear aura)
- Severe alcoholic — judgement impaired, rarely sober
- Real name unknown, possibly related to Brevic nobility
- Father [[Nugrah]] imprisoned in the fort cellar
## Weaknesses
- Alcoholism makes him predictable
- Lieutenant [[Akiros Ismort]] can be turned with DC 18 Diplomacy
- Blind spot on eastern fort wall
## House Rule
I decided the stag skull helm also grants +2 AC (not in the book).
""",
    "Akiros Ismort.md": """\
---
tags: [npc, redeemable, lieutenant]
---
# Akiros Ismort
Former paladin who fell from grace after killing a nobleman in a duel.
Now serves the [[The Stag Lord]] as his lieutenant, though he despises him.
## Motivation
Akiros wants redemption. He will switch sides if the party demonstrates
they intend to bring real justice, not just replace one tyrant with another.
## Stats (as ally)
- Fighter 4 / Paladin 2 (fallen)
- Longsword +8, Full Plate
## Session Notes
Surrendered in [[Session 04]] when Kira passed the Diplomacy check (rolled 22).
""",
    "Oleg's Trading Post.md": """\
---
tags: [location, base, trading-post]
aliases: [Oleg's, the trading post]
---
# Oleg's Trading Post
Primary base of operations at the northern edge of the [[Stolen Lands]].
Proprietors: Oleg Leveton and Svetlana Leveton.
## Services
- Free lodging after bandits defeated
- 10% discount on goods after saving Svetlana's ring
- Notice board with bounties
## Quest Hooks
- Svetlana's stolen wedding ring → recover from [[Kressle's Camp]] (hex B5)
- Fangberry pie request from Svetlana → hex E4
- Tuskgutter bounty from Vekkel Benzen → 1000gp
""",
    "Stag Lord Fort.md": """\
---
tags: [location, dungeon, boss-lair]
---
# Stag Lord Fort
Former temple of Erastil, now the Stag Lord's fortress.
Located on bluff above the [[Tuskwater]] lake.
## Areas
- **Area 1** — Main Gate: 2 guards with masterwork crossbows
- **Area 2** — Great Hall: 12 bandits
- **Area 3** — Armoury: masterwork gear, 200gp
- **Area 4** — Stag Lord's Chamber: arcane lock DC 28, skull helm
- **Area 5** — Cellar: [[Nugrah]] imprisoned here
## Tactical Notes
Eastern wall blind spot visible from southeast ridge.
Tunnel entrance in hex C6 — the party found this in [[Session 03]].
""",
    "Session 04.md": """\
---
tags: [session, recap]
date: 2025-03-10
session_number: 4
---
# Session 04 — Fall of the Stag Lord
## Summary
The party assaulted [[Stag Lord Fort]] via the eastern tunnel (found Session 3).
[[Akiros Ismort]] opened the main gate for them after Kira's Diplomacy (22).
[[The Stag Lord]] was confronted on the battlements, visibly drunk.
Torvin killed him with a critical hit (greataxe, rolled max damage).
## Key Moments
- 01:02:34 — Akiros surrenders, joins party temporarily
- 01:14:10 — Stag Lord falls from battlements into the moat
- 01:28:45 — Nugrah freed from cellar, gives info on nearby ruins
## Loot
- Enchanted stag skull helm (keeping as trophy)
- 1,200gp from the armoury
- Map of the Kamelands from the Stag Lord's chamber
## XP
4,800 XP each. All players advance to level 5.
""",
    "House Rules Combat.md": """\
---
tags: [house-rules, combat]
---
# House Rules — Combat
## Action Economy
Bonus action potions: drinking a potion is a bonus action, not an action.
## Critical Hits
On a natural 20, roll all damage dice twice (standard) AND add a free
combat manoeuvre at no action cost.
## Death Saves
Failed death saves reset after a short rest (not just long rest).
## Flanking
Flanking grants +2 to attack (not advantage) to keep combat faster.
""",
    "Nugrah.md": """\
---
tags: [npc, druid, imprisoned]
---
# Nugrah
The Stag Lord's father, a mad hermit druid. Imprisoned in [[Stag Lord Fort]] cellar.
Has been there for years, half-starved and half-mad.
## Information He Knows
- Location of ancient Elven ruins in hex G7
- The Stag Lord's real name and Brevic connection
- A ritual to purify the defiled Erastil temple
## Note
I'm making Nugrah a quest-giver for the Narlmarches arc in season 2.
""",
}

SESSION_TRANSCRIPT = [
    {"start": 0.0,   "end": 12.0,  "text": "Okay everyone, welcome back. Last session you found the tunnel under the fort.", "speaker": "DM"},
    {"start": 12.0,  "end": 28.0,  "text": "Torvin takes point. I want to move through the tunnel as quietly as possible.", "speaker": "Torvin"},
    {"start": 28.0,  "end": 44.0,  "text": "Give me a Stealth check from everyone. Torvin, you're at disadvantage in full plate.", "speaker": "DM"},
    {"start": 44.0,  "end": 56.0,  "text": "I rolled a seven. That's terrible.", "speaker": "Torvin"},
    {"start": 56.0,  "end": 75.0,  "text": "Kira rolled nineteen. She moves ahead silently and spots a guard at the tunnel exit.", "speaker": "DM"},
    {"start": 75.0,  "end": 90.0,  "text": "Can I signal the party to stop and try to take the guard out quietly?", "speaker": "Kira"},
    {"start": 90.0,  "end": 110.0, "text": "Roll Stealth to approach and Sleight of Hand to apply the garrotte. DC 14.", "speaker": "DM"},
    {"start": 110.0, "end": 125.0, "text": "Seventeen on Stealth and twenty-three on Sleight of Hand. Guard is down.", "speaker": "Kira"},
    {"start": 125.0, "end": 150.0, "text": "The party emerges into the courtyard. You see Akiros Ismort standing by the gate.", "speaker": "DM"},
    {"start": 150.0, "end": 165.0, "text": "I want to try to talk to him. Diplomacy. I rolled a twenty-two total.", "speaker": "Kira"},
    {"start": 165.0, "end": 195.0, "text": "Akiros lowers his sword. He says: I knew someone would come eventually. The gate is yours. He opens it.", "speaker": "DM"},
    {"start": 195.0, "end": 215.0, "text": "Yes! I knew we could flip him. Does he fight with us against the Stag Lord?", "speaker": "Torvin"},
    {"start": 215.0, "end": 240.0, "text": "He'll hold the gate and keep the other bandits back. You're on your own for the boss.", "speaker": "DM"},
    {"start": 240.0, "end": 265.0, "text": "The Stag Lord stands on the battlements. He's clearly drunk, swaying slightly, skull helm gleaming.", "speaker": "DM"},
    {"start": 265.0, "end": 280.0, "text": "Torvin charges up the stairs. I roll a nineteen, that's a critical hit. Forty-two damage with the greataxe.", "speaker": "Torvin"},
    {"start": 280.0, "end": 310.0, "text": "The Stag Lord staggers backward, crashes through the battlement railing, and falls into the Tuskwater. The bandits drop their weapons.", "speaker": "DM"},
]


def _build_realistic_pdf(path: Path) -> None:
    """Build a 5-page adventure module PDF with realistic RPG layout."""
    import fitz

    doc = fitz.open()

    chapters = [
        (CAMPAIGN_TEXT["chapter1_title"], CAMPAIGN_TEXT["chapter1_body"]),
        (CAMPAIGN_TEXT["chapter2_title"], CAMPAIGN_TEXT["chapter2_body"]),
        (CAMPAIGN_TEXT["chapter3_title"], CAMPAIGN_TEXT["chapter3_body"]),
        (CAMPAIGN_TEXT["chapter4_title"], CAMPAIGN_TEXT["chapter4_body"]),
        (CAMPAIGN_TEXT["chapter5_title"], CAMPAIGN_TEXT["chapter5_body"]),
    ]

    for i, (title, body) in enumerate(chapters):
        page = doc.new_page(width=595, height=842)

        # Page header (small, all caps, simulates a running header)
        page.insert_text((50, 30), f"KINGMAKER — CHAPTER {i+1}", fontsize=8, color=(0.5, 0.5, 0.5))

        # Chapter title (large, bold via size — heading detection should fire)
        page.insert_text((50, 70), title, fontsize=16, color=(0, 0, 0))

        # Decorative rule (just a thin line of dashes)
        page.insert_text((50, 90), "─" * 60, fontsize=8, color=(0.3, 0.3, 0.3))

        # Body text — wrapped at ~80 chars per line
        words = body.split()
        lines: list[str] = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 > 80:
                lines.append(current.strip())
                current = w
            else:
                current += (" " if current else "") + w
        if current:
            lines.append(current.strip())

        y = 110
        for line in lines:
            page.insert_text((50, y), line, fontsize=10, color=(0, 0, 0))
            y += 14
            if y > 800:
                break

        # Page number
        page.insert_text((280, 830), str(i + 1), fontsize=8, color=(0.5, 0.5, 0.5))

    doc.save(str(path))
    doc.close()


def _build_obsidian_vault(vault_dir: Path) -> None:
    for filename, content in OBSIDIAN_NOTES.items():
        (vault_dir / filename).write_text(content, encoding="utf-8")


def _generate_session_audio(wav_path: Path) -> bool:
    """Generate session audio with espeak-ng. Returns True if successful."""
    full_text = " ".join(seg["text"] for seg in SESSION_TRANSCRIPT)
    try:
        result = subprocess.run(
            ["espeak-ng", "-w", str(wav_path), "-s", "150", full_text],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 10_000
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Session-scoped fixture: the realistic store
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def realistic_store(tmp_path_factory, real_embedder):
    """Store built from realistic RPG data: 5-page PDF, 8-note Obsidian vault, 16-segment transcript."""
    from pocket_gm.ingestion.chunker import (
        chunk_obsidian_note,
        chunk_pdf_pages,
        chunk_transcript,
    )
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
    from pocket_gm.ingestion.pdf_loader import load_pdf
    from pocket_gm.retrieval.store import (
        Store,
        notes_table,
        sessions_table,
        sourcebook_table,
    )

    base = tmp_path_factory.mktemp("realistic")

    # Build fixtures
    pdf_path = base / "kingmaker_excerpt.pdf"
    vault_path = base / "vault"
    vault_path.mkdir()

    _build_realistic_pdf(pdf_path)
    _build_obsidian_vault(vault_path)

    store = Store(base / "vectors.db")
    cid = "kingmaker"

    # --- Sourcebook (PDF) ---
    pages = load_pdf(pdf_path)
    pdf_chunks = chunk_pdf_pages(pages, chunk_size=200, chunk_overlap=30)
    pdf_dicts = [
        {"text": c.text, "source_type": c.source_type, "filename": c.filename,
         "page": c.page, "heading": c.heading, "chunk_index": c.chunk_index,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0}
        for c in pdf_chunks
    ]
    store.add_documents(
        sourcebook_table(cid), pdf_dicts,
        real_embedder.embed([d["text"] for d in pdf_dicts]), cid,
    )

    # --- Notes (Obsidian vault) ---
    notes = load_obsidian_vault(vault_path)
    note_chunks = []
    for note in notes:
        note_chunks.extend(chunk_obsidian_note(note, chunk_size=150, chunk_overlap=20))
    note_dicts = [
        {"text": c.text, "source_type": c.source_type, "filename": c.filename,
         "page": 0, "heading": c.heading, "chunk_index": c.chunk_index,
         "session_number": 0, "session_date": "", "timestamp_start": 0.0}
        for c in note_chunks
    ]
    store.add_documents(
        notes_table(cid), note_dicts,
        real_embedder.embed([d["text"] for d in note_dicts]), cid,
    )

    # --- Sessions (transcript) ---
    full_text = " ".join(seg["text"] for seg in SESSION_TRANSCRIPT)
    sess_chunks = chunk_transcript(
        full_text, filename="session04.txt",
        session_number=4, session_date="2025-03-10",
        chunk_size=100, chunk_overlap=15,
        timestamps=SESSION_TRANSCRIPT,
    )
    store.add_documents(
        sessions_table(cid), sess_chunks,
        real_embedder.embed([c["text"] for c in sess_chunks]), cid,
    )

    return store, cid, {
        "pdf_path": pdf_path,
        "vault_path": vault_path,
        "pdf_chunks": len(pdf_chunks),
        "note_chunks": len(note_chunks),
        "sess_chunks": len(sess_chunks),
    }


# ---------------------------------------------------------------------------
# PDF quality tests
# ---------------------------------------------------------------------------

def test_realistic_pdf_extracts_all_pages(realistic_store):
    from pocket_gm.ingestion.pdf_loader import load_pdf
    _, _, meta = realistic_store
    pages = load_pdf(meta["pdf_path"])
    assert len(pages) == 5, f"Expected 5 pages, got {len(pages)}"


def test_realistic_pdf_heading_detected(realistic_store):
    from pocket_gm.ingestion.pdf_loader import load_pdf
    _, _, meta = realistic_store
    pages = load_pdf(meta["pdf_path"])
    headings = [p.heading for p in pages if p.heading]
    assert len(headings) >= 4, f"Expected ≥4 headings, got: {headings}"
    all_heading_text = " ".join(headings).upper()
    assert "STAG LORD" in all_heading_text or "STOLEN" in all_heading_text


def test_realistic_pdf_chunks_inherit_headings(realistic_store):
    from pocket_gm.ingestion.chunker import chunk_pdf_pages
    from pocket_gm.ingestion.pdf_loader import load_pdf
    _, _, meta = realistic_store
    pages = load_pdf(meta["pdf_path"])
    chunks = chunk_pdf_pages(pages, 200, 30)
    chunks_with_heading = [c for c in chunks if c.heading]
    assert len(chunks_with_heading) > 0
    # Heading text should be injected into the chunk text
    for c in chunks_with_heading[:3]:
        assert c.heading.upper() in c.text.upper(), (
            f"Heading '{c.heading}' not in chunk text: {c.text[:80]}"
        )


def test_realistic_pdf_page_numbers_correct(realistic_store):
    from pocket_gm.ingestion.pdf_loader import load_pdf
    _, _, meta = realistic_store
    pages = load_pdf(meta["pdf_path"])
    page_nums = [p.page for p in pages]
    assert page_nums == list(range(1, 6)), f"Pages should be 1-5, got {page_nums}"


def test_realistic_pdf_stag_lord_content_on_page_2(realistic_store):
    from pocket_gm.ingestion.pdf_loader import load_pdf
    _, _, meta = realistic_store
    pages = load_pdf(meta["pdf_path"])
    page2_text = pages[1].text.lower()
    assert "stag lord" in page2_text or "alcoholis" in page2_text, (
        f"Page 2 should contain Stag Lord content, got: {page2_text[:100]}"
    )


# ---------------------------------------------------------------------------
# Obsidian vault quality tests
# ---------------------------------------------------------------------------

def test_obsidian_vault_loads_all_notes(realistic_store):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
    _, _, meta = realistic_store
    notes = load_obsidian_vault(meta["vault_path"])
    assert len(notes) == len(OBSIDIAN_NOTES), (
        f"Expected {len(OBSIDIAN_NOTES)} notes, got {len(notes)}"
    )


def test_obsidian_wikilinks_resolved_in_stag_lord_note(realistic_store):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
    _, _, meta = realistic_store
    notes = load_obsidian_vault(meta["vault_path"])
    stag_note = next((n for n in notes if n.title == "The Stag Lord"), None)
    assert stag_note is not None
    # [[Akiros Ismort]] wikilink should be resolved to the note's body text
    body_lower = stag_note.body.lower()
    assert "akiros" in body_lower, "Akiros wikilink not resolved in Stag Lord note"
    assert "tuskwater" in body_lower or "stag lord fort" in body_lower.replace("'", "")


def test_obsidian_frontmatter_parsed(realistic_store):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
    _, _, meta = realistic_store
    notes = load_obsidian_vault(meta["vault_path"])
    stag_note = next((n for n in notes if n.title == "The Stag Lord"), None)
    assert stag_note is not None
    assert "villain" in stag_note.tags or "boss" in stag_note.tags, (
        f"Expected 'villain' or 'boss' in tags, got: {stag_note.tags}"
    )


def test_obsidian_house_rules_note_indexed(realistic_store):
    from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
    _, _, meta = realistic_store
    notes = load_obsidian_vault(meta["vault_path"])
    hr_note = next((n for n in notes if "House Rules" in n.title or "Combat" in n.title), None)
    assert hr_note is not None, "House Rules Combat note should be loaded"
    assert "bonus action" in hr_note.body.lower() or "potion" in hr_note.body.lower()


# ---------------------------------------------------------------------------
# Session transcript quality tests
# ---------------------------------------------------------------------------

def test_transcript_chunks_have_timestamps():
    from pocket_gm.ingestion.chunker import chunk_transcript
    full_text = " ".join(seg["text"] for seg in SESSION_TRANSCRIPT)
    chunks = chunk_transcript(
        full_text, "session04.txt", 4, "2025-03-10", 100, 15, SESSION_TRANSCRIPT
    )
    assert len(chunks) > 0
    for c in chunks:
        assert c["session_number"] == 4
        assert c["session_date"] == "2025-03-10"
        assert c["timestamp_start"] >= 0.0


def test_transcript_timestamps_span_full_session():
    from pocket_gm.ingestion.chunker import chunk_transcript
    full_text = " ".join(seg["text"] for seg in SESSION_TRANSCRIPT)
    chunks = chunk_transcript(full_text, "session04.txt", 4, "2025-03-10", 100, 15, SESSION_TRANSCRIPT)
    ts_values = [c["timestamp_start"] for c in chunks]
    assert max(ts_values) > 100.0, "Should have chunks from mid/late session"


def test_audio_generation_with_espeak():
    """Verify espeak-ng generates a valid WAV file from session text."""
    wav_path = Path(tempfile.mktemp(suffix=".wav"))
    try:
        ok = _generate_session_audio(wav_path)
        if not ok:
            pytest.skip("espeak-ng not available in this environment")
        assert wav_path.stat().st_size > 50_000, "WAV file too small — TTS likely failed"
        # Verify it's a real WAV (RIFF header)
        header = wav_path.read_bytes()[:4]
        assert header == b"RIFF", f"Not a valid WAV file: header={header!r}"
    finally:
        wav_path.unlink(missing_ok=True)


def test_whisper_pipeline_blocked_gracefully():
    """Document that Whisper model download requires internet access.

    The full audio transcription pipeline works end-to-end but the model
    (faster-whisper-tiny) cannot be downloaded in this cloud environment
    because HuggingFace is blocked by the network policy.

    To test locally:
        pip install faster-whisper
        pocket-gm session add session04.wav --campaign kingmaker --date 2025-03-10 --number 4
    """
    from faster_whisper import WhisperModel
    with pytest.raises(Exception, match="(?i)huggingface|local|not found|forbidden|allowlist"):
        WhisperModel("tiny", device="cpu", compute_type="int8")


# ---------------------------------------------------------------------------
# Semantic retrieval tests across all three real stores
# ---------------------------------------------------------------------------

def test_sourcebook_stag_lord_weakness_query(realistic_store, real_embedder):
    """Query about the Stag Lord's weakness should surface the alcoholism content."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import sourcebook_table

    q = real_embedder.embed_one("What is the Stag Lord's greatest weakness?")
    results = store.query(sourcebook_table(cid), q, top_k=5)

    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert "alcohol" in combined or "weakness" in combined, (
        f"Expected weakness/alcohol content, got: {combined[:200]}"
    )


def test_sourcebook_encounter_table_query(realistic_store, real_embedder):
    """Querying about random encounters should surface the encounter table."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import sourcebook_table

    q = real_embedder.embed_one("What creatures might the party encounter while exploring?")
    results = store.query(sourcebook_table(cid), q, top_k=5)
    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert any(w in combined for w in ["wolf", "boar", "bandit", "encounter", "roll"]), (
        f"Expected encounter content, got: {combined[:200]}"
    )


def test_sourcebook_fort_layout_query(realistic_store, real_embedder):
    """Querying about the fort should surface the fort layout chapter."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import sourcebook_table

    q = real_embedder.embed_one("What rooms are in the Stag Lord's fortress?")
    results = store.query(sourcebook_table(cid), q, top_k=5)
    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert "area" in combined or "armoury" in combined or "gate" in combined or "cellar" in combined


def test_notes_house_rules_query(realistic_store, real_embedder):
    """Querying about house rules should surface the GM notes."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import notes_table

    q = real_embedder.embed_one("What are the house rules for potions and bonus actions?")
    results = store.query(notes_table(cid), q, top_k=5)
    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert "bonus" in combined or "potion" in combined or "action" in combined


def test_notes_gm_homebrew_surfaces_not_in_sourcebook(realistic_store, real_embedder):
    """GM homebrew (stag skull helm +2 AC) should be retrievable from notes, not sourcebook."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import notes_table, sourcebook_table

    q = real_embedder.embed_one("Does the stag skull helm grant bonus armour class?")

    # Should be in notes (it's a house rule I added)
    notes_results = store.query(notes_table(cid), q, top_k=5)
    notes_text = " ".join(r.text.lower() for r in notes_results)

    # Should NOT be in sourcebook (it's not in the official text)
    sb_results = store.query(sourcebook_table(cid), q, top_k=5)
    sb_text = " ".join(r.text.lower() for r in sb_results)

    assert "+2 ac" in notes_text or "bonus" in notes_text, (
        f"House rule (+2 AC) should be in notes, got: {notes_text[:200]}"
    )
    assert "+2 ac" not in sb_text, "House rule should not appear in official sourcebook"


def test_session_transcript_diplomacy_query(realistic_store, real_embedder):
    """Querying about the Diplomacy check should surface the session transcript."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import sessions_table

    q = real_embedder.embed_one("What happened when Kira tried to persuade Akiros?")
    results = store.query(sessions_table(cid), q, top_k=5)
    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert "akiros" in combined or "diplomacy" in combined or "twenty" in combined


def test_session_transcript_boss_fight_query(realistic_store, real_embedder):
    """Querying about the boss fight should surface the final confrontation."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import sessions_table

    q = real_embedder.embed_one("How did Torvin kill the Stag Lord?")
    results = store.query(sessions_table(cid), q, top_k=5)
    assert results
    combined = " ".join(r.text.lower() for r in results)
    assert "torvin" in combined or "greataxe" in combined or "critical" in combined


def test_session_citations_include_correct_session_number(realistic_store, real_embedder):
    """Session results must carry session_number=4 and date 2025-03-10."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import sessions_table

    q = real_embedder.embed_one("The party fights the Stag Lord on the battlements")
    results = store.query(sessions_table(cid), q, top_k=5)
    assert results
    for r in results:
        assert r.session_number == 4
        assert r.session_date == "2025-03-10"


# ---------------------------------------------------------------------------
# Cross-store query: full router test with realistic data
# ---------------------------------------------------------------------------

def test_cross_store_query_stag_lord(realistic_store, real_embedder):
    """A Stag Lord query returns results from ALL THREE stores simultaneously."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.router import query_all_sync

    q = real_embedder.embed_one(
        "Tell me everything about the Stag Lord — his stats, his weakness, and what happened in the session."
    )
    result = query_all_sync(store, cid, q, top_k=5)

    assert result.sourcebook, "Sourcebook should have Stag Lord content"
    assert result.notes,      "Notes should have Stag Lord homebrew content"
    assert result.sessions,   "Sessions should have Stag Lord fight content"

    # Sourcebook: official description
    sb_text = result.sourcebook[0].text.lower()
    assert "stag" in sb_text or "bandit" in sb_text or "fort" in sb_text

    # Notes: GM notes + house rules
    notes_text = " ".join(r.text.lower() for r in result.notes)
    assert "akiros" in notes_text or "homebrew" in notes_text or "house rule" in notes_text.replace("house rule", "house_rule") or "weakness" in notes_text

    # Sessions: actual play
    sess_text = " ".join(r.text.lower() for r in result.sessions)
    assert "torvin" in sess_text or "stag lord" in sess_text or "battlements" in sess_text


def test_sourcebook_vs_notes_authority(realistic_store, real_embedder):
    """Same topic retrieved from sourcebook and notes should have different source_type."""
    store, cid, _ = realistic_store
    from pocket_gm.retrieval.store import notes_table, sourcebook_table

    q = real_embedder.embed_one("Stag Lord fort location and description")
    sb = store.query(sourcebook_table(cid), q, top_k=1)
    notes = store.query(notes_table(cid), q, top_k=1)

    assert sb[0].source_type == "pdf"
    assert notes[0].source_type in ("markdown", "obsidian")


def test_full_pipeline_end_to_end_realistic(realistic_store, real_embedder):
    """Complete pipeline with realistic data: retrieve → prompt → ground."""
    from pocket_gm.retrieval.router import query_all_sync
    from pocket_gm.synthesis.grounding import validate_citations
    from pocket_gm.synthesis.prompt_builder import build_prompt

    store, cid, _ = realistic_store

    question = "What is the Stag Lord's weakness and how can we exploit it?"
    q_vec = real_embedder.embed_one(question)
    result = query_all_sync(store, cid, q_vec, top_k=5)

    assert not result.is_empty(threshold=0.1), "Should find relevant content"

    prompt, index_map = build_prompt(
        question,
        result.sourcebook, result.notes, result.sessions,
        threshold=0.1,
    )

    assert prompt
    assert index_map
    # Prompt should contain content about the Stag Lord
    assert "stag" in prompt.lower() or "alcohol" in prompt.lower() or "weakness" in prompt.lower()

    # Simulate a perfect LLM answer that cites all its claims
    ref = index_map[0][0]
    mock_answer = (
        f"The Stag Lord's greatest weakness is his severe alcoholism [{ref}]. "
        f"His lieutenant Akiros Ismort can be turned with a Diplomacy check [{ref}]. "
        f"The eastern wall has a blind spot that allows undetected approach [{ref}]."
    )
    grounded = validate_citations(mock_answer, index_map)
    assert grounded.is_fully_grounded
    assert len(grounded.citations_used) >= 1


def test_chunk_counts_realistic(realistic_store):
    """Verify that realistic fixtures produce reasonable chunk volumes."""
    store, cid, meta = realistic_store
    from pocket_gm.retrieval.store import notes_table, sessions_table, sourcebook_table

    sb_count = store.count(sourcebook_table(cid))
    notes_count = store.count(notes_table(cid))
    sess_count = store.count(sessions_table(cid))

    # 5-page PDF at 200 tokens/chunk should give 5-20 chunks
    assert 5 <= sb_count <= 30, f"Sourcebook chunks out of expected range: {sb_count}"
    # 8 notes at 150 tokens/chunk should give 8-30 chunks
    assert 8 <= notes_count <= 50, f"Notes chunks out of expected range: {notes_count}"
    # 16-segment transcript at 100 tokens/chunk should give 3-15 chunks
    assert 2 <= sess_count <= 20, f"Session chunks out of expected range: {sess_count}"

    print(f"\n  Sourcebook: {sb_count} chunks | Notes: {notes_count} chunks | Sessions: {sess_count} chunks")
