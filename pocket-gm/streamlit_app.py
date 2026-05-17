"""
Pocket GM — Streamlit web UI.

Runs entirely in the user's browser session. No data is persisted between
sessions and no queries are logged externally. The ANTHROPIC_API_KEY is
read from Streamlit secrets and never exposed to the browser.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pocket GM",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API key — from Streamlit secrets (cloud) or environment variable (local)
# Never falls back to a hardcoded value.
# ---------------------------------------------------------------------------
def _get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Session-scoped store (one per browser session, wiped on refresh)
# ---------------------------------------------------------------------------
def _get_store():
    if "store" not in st.session_state:
        import sqlite_vec
        import sqlite3
        # Use a temp dir scoped to this session
        tmp = tempfile.mkdtemp(prefix="pocket_gm_")
        from pocket_gm.retrieval.store import Store
        st.session_state["store"] = Store(Path(tmp) / "session.db")
        st.session_state["store_dir"] = tmp
        st.session_state["campaigns"] = {}   # campaign_id → display name
        st.session_state["ingested_files"] = []
    return st.session_state["store"]


# ---------------------------------------------------------------------------
# Demo data — the Kingmaker fixtures so the app is usable without uploading
# ---------------------------------------------------------------------------
_DEMO_CAMPAIGN = "demo_kingmaker"

_DEMO_PDF_CONTENT = [
    ("THE STOLEN LANDS",
     "The Stolen Lands are a vast wilderness south of Brevoy ruled by bandits and monsters. "
     "The players are tasked with exploring and civilising this territory."),
    ("THE STAG LORD",
     "The Stag Lord is the most powerful bandit lord in the Stolen Lands. "
     "He rules from a fortified keep on the banks of the Tuskwater lake. "
     "His greatest weakness is his severe alcoholism, which makes him unpredictable. "
     "He wears an enchanted stag skull helm that grants darkvision and a fear aura. "
     "His real name is unknown; rumours connect him to Brevic nobility."),
    ("OLEG'S TRADING POST",
     "Oleg Leveton runs a trading post on the northern border of the Stolen Lands. "
     "He and his wife Svetlana are being extorted by a band of bandits led by Kressel. "
     "Oleg has asked the players for help in exchange for free lodging."),
    ("AKIROS ISMORT",
     "Akiros Ismort is the Stag Lord's lieutenant and a former paladin who fell from grace. "
     "He killed a nobleman in a duel and fled north into banditry. "
     "He despises the Stag Lord and can be turned with a DC 18 Diplomacy check. "
     "If persuaded he will open the fort gate and surrender."),
    ("STAG LORD'S FORT",
     "The fort is a ruined temple of Erastil on a bluff above the Tuskwater. "
     "It houses approximately twenty bandits plus Dovan and Auchs as lieutenants. "
     "The eastern wall has a blind spot visible from the southeast ridge. "
     "The cellar holds Nugrah, the Stag Lord's father, imprisoned there."),
]

_DEMO_NOTES = [
    ("House Rules.md",
     "# Combat House Rules\nInitiative: players roll 2d20 keep highest.\n"
     "Stag skull helm: I added +2 AC (not in the sourcebook).\n"
     "Death saves: on third failure, player narrates their death scene.\n"
     "Flanking grants advantage instead of +2 as per RAW."),
    ("The Stag Lord.md",
     "# GM Notes: The Stag Lord\nI decided he recognises Aldric's holy symbol "
     "— he used to worship Erastil before his fall. Play up his self-loathing.\n"
     "His alcoholism: he drinks from a flask every 2 rounds in combat or takes -2 to all rolls.\n"
     "tags: npc, villain, boss"),
    ("Party Notes.md",
     "# Party\nTorvin Ironmane — dwarf fighter, wants revenge on bandits who burned his home.\n"
     "Kira Nightwhisper — elf rogue, secretly has contacts in the Restovic Swordlords.\n"
     "Aldric Sunborn — human cleric of Erastil, will be conflicted about the fort.\n"
     "Mira Voss — gnome wizard, hoarding Numerian tech she found in hex E4.\n"
     "Jak Coldwell — human ranger, tracking animal, knows the terrain best."),
]

_DEMO_TRANSCRIPT = [
    {"start": 0.0, "end": 45.0, "text": "The party approaches the Stag Lord's fort under cover of darkness.", "speaker": "DM"},
    {"start": 45.0, "end": 90.0, "text": "Kira scouts the eastern wall and spots the blind spot from the southeast ridge.", "speaker": "DM"},
    {"start": 90.0, "end": 140.0, "text": "Akiros Ismort opens the gate before the party reaches it and drops his sword.", "speaker": "DM"},
    {"start": 140.0, "end": 195.0, "text": "Kira rolled a 22 on Diplomacy — Akiros switches sides immediately.", "speaker": "Player"},
    {"start": 195.0, "end": 260.0, "text": "The Stag Lord confronts the party on the battlements. He is visibly drunk, swaying.", "speaker": "DM"},
    {"start": 260.0, "end": 310.0, "text": "Aldric addresses him by his old name and the Stag Lord hesitates. Dramatic pause.", "speaker": "DM"},
]


@st.cache_resource(show_spinner="Loading demo campaign…")
def _build_demo_store():
    """Build the demo store once per app process (cached across sessions)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from pocket_gm.retrieval.store import (
        Store, sourcebook_table, notes_table, sessions_table,
    )
    from pocket_gm.ingestion.chunker import chunk_transcript

    tmp = tempfile.mkdtemp(prefix="pocket_gm_demo_")
    store = Store(Path(tmp) / "demo.db")
    campaign_id = _DEMO_CAMPAIGN

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True)

    # Sourcebook (demo PDF content as plain chunks)
    sb_chunks = [
        {
            "text": f"{title}\n{body}",
            "source_type": "pdf",
            "filename": "KingmakerDemo.pdf",
            "page": i + 1,
            "heading": title,
            "chunk_index": i,
            "session_number": 0,
            "session_date": "",
            "timestamp_start": 0.0,
        }
        for i, (title, body) in enumerate(_DEMO_PDF_CONTENT)
    ]
    store.add_documents(sourcebook_table(campaign_id), sb_chunks,
                        embed([c["text"] for c in sb_chunks]), campaign_id)

    # Notes
    nt_chunks = [
        {
            "text": content,
            "source_type": "markdown",
            "filename": filename,
            "page": 0,
            "heading": filename.replace(".md", ""),
            "chunk_index": i,
            "session_number": 0,
            "session_date": "",
            "timestamp_start": 0.0,
        }
        for i, (filename, content) in enumerate(_DEMO_NOTES)
    ]
    store.add_documents(notes_table(campaign_id), nt_chunks,
                        embed([c["text"] for c in nt_chunks]), campaign_id)

    # Sessions
    full_text = " ".join(seg["text"] for seg in _DEMO_TRANSCRIPT)
    ss_chunks = chunk_transcript(
        full_text, "session04.json", 4, "2025-03-10",
        chunk_size=64, chunk_overlap=8, timestamps=_DEMO_TRANSCRIPT,
    )
    store.add_documents(sessions_table(campaign_id), ss_chunks,
                        embed([c["text"] for c in ss_chunks]), campaign_id)

    return store


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _run_query(store, campaign_id: str, question: str, api_key: str, top_k: int = 5):
    from sentence_transformers import SentenceTransformer
    from pocket_gm.retrieval.router import query_all_sync
    from pocket_gm.synthesis.prompt_builder import build_prompt
    from pocket_gm.synthesis.grounding import validate_citations
    from pocket_gm.synthesis.claude_client import ClaudeClient

    @st.cache_resource(show_spinner=False)
    def _embedder():
        return SentenceTransformer("all-MiniLM-L6-v2")

    embedder = _embedder()
    q_vec = embedder.encode([question], normalize_embeddings=True)[0]

    result = query_all_sync(store, campaign_id, q_vec, top_k=top_k)
    threshold = 0.35

    if result.is_empty(threshold):
        return None, None, None

    prompt, index_map = build_prompt(
        question,
        result.sourcebook,
        result.notes,
        result.sessions,
        threshold=threshold,
        json_mode=False,
    )

    client = ClaudeClient(api_key=api_key)
    answer_text = client.generate(prompt)
    grounded = validate_citations(answer_text, index_map)

    return grounded, result, index_map


# ---------------------------------------------------------------------------
# UI — sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎲 Pocket GM")
st.sidebar.caption("Grounded answers from your campaign sources. No hallucinations.")

api_key = _get_api_key()
if not api_key:
    st.sidebar.error("No API key found. Add ANTHROPIC_API_KEY to Streamlit secrets.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Campaign", ["Demo (Kingmaker)", "Upload your own"], index=0)

campaign_id = _DEMO_CAMPAIGN
active_store = None

if mode == "Demo (Kingmaker)":
    with st.sidebar:
        with st.spinner("Preparing demo data…"):
            active_store = _build_demo_store()
    st.sidebar.success("Demo ready — 5 sourcebook chunks, 3 GM notes, 6 transcript segments.")

else:
    st.sidebar.markdown("### Upload files")
    uploaded_pdf = st.sidebar.file_uploader("PDF sourcebook", type=["pdf"])
    uploaded_notes = st.sidebar.file_uploader("GM notes (.md or .txt)", type=["md", "txt"],
                                               accept_multiple_files=True)

    if uploaded_pdf or uploaded_notes:
        store = _get_store()
        campaign_id = "user_upload"

        from sentence_transformers import SentenceTransformer
        @st.cache_resource(show_spinner=False)
        def _upload_embedder():
            return SentenceTransformer("all-MiniLM-L6-v2")
        embedder = _upload_embedder()

        if uploaded_pdf:
            import fitz
            from pocket_gm.retrieval.store import sourcebook_table
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            chunks = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    chunks.append({
                        "text": text[:1000],
                        "source_type": "pdf",
                        "filename": uploaded_pdf.name,
                        "page": page_num,
                        "heading": "",
                        "chunk_index": page_num - 1,
                        "session_number": 0,
                        "session_date": "",
                        "timestamp_start": 0.0,
                    })
            if chunks:
                import numpy as np
                vecs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
                store.add_documents(sourcebook_table(campaign_id), chunks, vecs, campaign_id)
                st.sidebar.success(f"PDF: {len(chunks)} pages ingested")

        if uploaded_notes:
            from pocket_gm.retrieval.store import notes_table
            all_note_chunks = []
            for f in uploaded_notes:
                content = f.read().decode("utf-8", errors="replace")
                all_note_chunks.append({
                    "text": content[:2000],
                    "source_type": "markdown",
                    "filename": f.name,
                    "page": 0,
                    "heading": f.name,
                    "chunk_index": 0,
                    "session_number": 0,
                    "session_date": "",
                    "timestamp_start": 0.0,
                })
            if all_note_chunks:
                import numpy as np
                vecs = embedder.encode([c["text"] for c in all_note_chunks], normalize_embeddings=True)
                store.add_documents(notes_table(campaign_id), all_note_chunks, vecs, campaign_id)
                st.sidebar.success(f"Notes: {len(all_note_chunks)} files ingested")

        active_store = store
    else:
        st.sidebar.info("Upload a PDF or notes file to get started.")

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Chunks to retrieve per source", min_value=2, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.caption(
    "No queries are stored or logged. "
    "Your API key is held in Streamlit secrets and never sent to the browser."
)

# ---------------------------------------------------------------------------
# UI — main panel
# ---------------------------------------------------------------------------
st.title("Ask your campaign")

if active_store is None:
    st.info("Select Demo or upload files in the sidebar to get started.")
    st.stop()

question = st.text_input(
    "Your question",
    placeholder="Who is the Stag Lord? What happened with Akiros? What are the house rules for initiative?",
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([1, 5])
ask = col_ask.button("Ask", type="primary", use_container_width=True)
if col_clear.button("Clear", use_container_width=False):
    st.session_state.pop("last_result", None)
    st.rerun()

if ask and question.strip():
    if not api_key:
        st.error("ANTHROPIC_API_KEY is not set. Add it to Streamlit secrets.")
    else:
        with st.spinner("Retrieving and generating answer…"):
            grounded, result, index_map = _run_query(
                active_store, campaign_id, question.strip(), api_key, top_k=top_k
            )
        st.session_state["last_result"] = (question, grounded, result, index_map)

# Display last result (persists across re-runs)
if "last_result" in st.session_state:
    question_display, grounded, result, index_map = st.session_state["last_result"]

    st.markdown(f"**Q:** {question_display}")
    st.markdown("---")

    if grounded is None:
        st.warning("Nothing relevant found in the campaign sources for that question.")
    else:
        st.markdown("### Answer")
        st.markdown(grounded.text)

        if grounded.uncited_sentences:
            with st.expander(f"⚠ {len(grounded.uncited_sentences)} sentence(s) without citations"):
                for s in grounded.uncited_sentences:
                    st.markdown(f"- {s}")
        else:
            st.success("All sentences are cited.")

        if index_map:
            st.markdown("### Sources")
            rows = []
            for ref, chunk in index_map:
                if ref in grounded.citations_used:
                    if chunk.source_type == "transcript":
                        ts = int(chunk.timestamp_start)
                        h, m, s = ts // 3600, (ts % 3600) // 60, ts % 60
                        location = f"Session {chunk.session_number} @ {h}:{m:02d}:{s:02d}" if h else f"Session {chunk.session_number} @ {m}:{s:02d}"
                    elif chunk.source_type == "pdf":
                        location = f"p.{chunk.page}" if chunk.page else "—"
                    else:
                        location = chunk.heading or chunk.filename

                    rows.append({
                        "Ref": f"[{ref}]",
                        "Type": chunk.source_type,
                        "File": chunk.filename,
                        "Location": location,
                        "Score": f"{chunk.score:.2f}",
                        "Excerpt": chunk.text[:120] + ("…" if len(chunk.text) > 120 else ""),
                    })

            if rows:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ref": st.column_config.TextColumn(width="small"),
                        "Score": st.column_config.TextColumn(width="small"),
                        "Type": st.column_config.TextColumn(width="small"),
                    },
                )

        with st.expander("Raw retrieved chunks (all sources)"):
            tab_sb, tab_nt, tab_ss = st.tabs(["Sourcebook", "GM Notes", "Sessions"])
            for tab, chunks in [
                (tab_sb, result.sourcebook),
                (tab_nt, result.notes),
                (tab_ss, result.sessions),
            ]:
                with tab:
                    if not chunks:
                        st.caption("No chunks retrieved from this source.")
                    for c in chunks:
                        st.markdown(f"**Score {c.score:.3f}** — {c.filename}")
                        st.caption(c.text[:300])
