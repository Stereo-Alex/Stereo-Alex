# Pocket GM

> A grounded Q&A tool for tabletop RPG Game Masters.  
> Ask questions about your campaign. Get cited answers from sourcebooks and sessions.  
> No hallucinations. No internet required.

```
$ pocket-gm ask "Who is the Stag Lord?" --campaign kingmaker

According to the sourcebook [1][2], the Stag Lord is a ruthless bandit warlord who
controls the Greenbelt from his fort on the Tuskwater. His real weakness is his
alcoholism and his fear of his imprisoned father.

In your sessions [3], the Stag Lord was killed by the party in Session 4 (2025-03-10
at 1:02:34). Akiros Ismort surrendered immediately after.

Sources
  [1] Kingmaker Part 1, p.42
  [2] Kingmaker Part 1, p.45
  [3] Session 4 — 2025-03-10 at 1:02:34
```

---

## What it does

Pocket GM keeps three kinds of knowledge about your campaign and searches all of them when you ask a question:

- **Sourcebooks** — official adventure PDFs and rulebooks (high authority, static)
- **GM Notes** — your personal notes, homebrew rules, and Obsidian vaults (your authority)
- **Sessions** — transcripts of your actual play sessions, automatically built from audio recordings (grows each session)

Every answer cites its sources. If something isn't in the sources, it says so instead of guessing.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- Pull a model: `ollama pull phi3:mini`

### Install

```bash
pip install pocket-gm
```

Or for development:

```bash
git clone https://github.com/Stereo-Alex/pocket-gm
cd pocket-gm
pip install -e ".[dev]"
```

### Your first campaign

```bash
# Create a campaign
pocket-gm campaign new "Kingmaker"

# Add your sourcebook PDFs
pocket-gm ingest pdf kingmaker_part1.pdf --campaign kingmaker

# Add your GM notes
pocket-gm ingest notes ./my-notes/ --campaign kingmaker

# Ask a question
pocket-gm ask "Who controls the Stolen Lands?" --campaign kingmaker
```

---

## Commands Reference

### Campaign management

```bash
pocket-gm campaign new <name>       # create a campaign
pocket-gm campaign list             # list all campaigns
pocket-gm campaign delete <id>      # delete a campaign and its data
```

### Ingesting materials

```bash
pocket-gm ingest pdf <file.pdf> --campaign <id>
    # add a sourcebook PDF to the sourcebook store

pocket-gm ingest notes <path> --campaign <id>
    # add markdown or text files to the notes store (file or directory)

pocket-gm ingest obsidian <vault-dir> --campaign <id>
    # ingest an Obsidian vault, resolving wikilinks, frontmatter, and embeds

pocket-gm ingest gdrive <folder-url> --campaign <id> [--audio]
    # sync a Google Drive folder; PDFs → sourcebook, docs/text → notes
    # requires: pip install 'pocket-gm[gdrive]'
    # --audio also downloads and transcribes audio files into the sessions store
```

All ingest commands are **idempotent** — re-running skips files that haven't changed (SHA256 hash check). Use `--force` to re-index.

### Session recordings

```bash
pocket-gm session add <audio.mp3> --campaign <id> --date 2025-03-10 --number 4
    # transcribe audio with Whisper and ingest into the sessions store

pocket-gm session list --campaign <id>
    # list all ingested sessions
```

### Querying

```bash
pocket-gm ask "<question>" --campaign <id>
```

### Campaign status

```bash
pocket-gm status --campaign <id>
    # show chunk counts per store, ingested files, and last query
```

### Evaluation

```bash
pocket-gm eval run --campaign <id> [--eval-set path/to/eval_set.json]
    # recall@k and citation coverage against a labeled eval set

pocket-gm eval calibrate --campaign <id>
    # suggest a relevance threshold based on your query log score distribution
```

### REST API server

```bash
pocket-gm serve [--host 127.0.0.1] [--port 8080] [--reload]
    # start the REST API; requires: pip install 'pocket-gm[serve]'
    # endpoints: POST /query, GET /campaigns, GET /campaigns/{id}/sessions,
    #            GET /campaigns/{id}/status, GET /health
    # interactive docs: http://localhost:8080/docs
```

---

## How it works

Pocket GM uses **retrieval-augmented generation (RAG)** with three separate vector stores per campaign, each queried in parallel:

```
Your question
    │
    ├──► Sourcebook store  ──┐
    ├──► GM Notes store    ──►── Ranked, cited chunks
    └──► Sessions store    ──┘         │
                                       ▼
                             Relevance gate (score ≥ 0.35)
                                       │
                             LLM synthesises answer
                             (only from retrieved chunks)
```

If nothing scores above the relevance threshold, the LLM is never called.

**Vector store:** [sqlite-vec](https://github.com/asg017/sqlite-vec) — single portable `.db` file, no server process.  
**Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` — 384-dim, runs on CPU.  
**LLM:** Local Ollama models (`phi3:mini` default, `mistral:7b-instruct` for harder queries).  
**Audio transcription:** `faster-whisper` — local, open source, timestamped segments.

---

## Grounding & Anti-Hallucination

Four layers prevent invented answers:

1. **Retrieval gate** — if no chunk scores above the relevance threshold, the LLM is never called and the system returns "not found"
2. **Prompt-level instruction** — the LLM is told to cite every fact with `[1][2]` markers and say "not found" for anything absent from the sources
3. **Citation enforcement** — output is post-processed to verify every citation marker refers to an actual retrieved chunk; uncited sentences are flagged with a warning
4. **Query logging** — every query, retrieved chunks, scores, and answer are logged to `~/.pocket-gm/logs/queries.ndjson` for offline inspection and evaluation

---

## Supported Models

| Model | Size | RAM | Quality | Recommended for |
|-------|------|-----|---------|-----------------|
| `phi3:mini` | 3.8B | 4 GB | Good | Default, fast queries |
| `llama3.2:3b` | 3B | 4 GB | Good | Alternative to phi3 |
| `mistral:7b-instruct` | 7B | 8 GB | Better | Complex, multi-source queries |
| `gemma2:2b` | 2B | 3 GB | Acceptable | Very low-resource machines |

---

## Configuration

Copy `config.example.yaml` to `~/.pocket-gm/config.yaml` and adjust:

```yaml
retrieval:
  top_k: 5
  relevance_threshold: 0.35   # raise if getting irrelevant results; run eval calibrate to tune

llm:
  model: phi3:mini             # any model available in your local Ollama
  json_citations: false        # set true for structured JSON citation output

whisper:
  model_size: base             # tiny / base / medium — tradeoff speed vs. accuracy
  diarize: false               # speaker diarization (requires pyannote-audio)
```

Full reference: see `config.example.yaml`.

---

## Optional extras

```bash
pip install 'pocket-gm[gdrive]'   # Google Drive sync
pip install 'pocket-gm[serve]'    # REST API server
pip install 'pocket-gm[dev]'      # pytest, pytest-asyncio
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for design notes including:
- Why three separate stores instead of one
- sqlite-vec vs. alternatives (LanceDB, ChromaDB, FAISS)
- Chunking strategy differences between sourcebooks, notes, and transcripts
- The four-layer grounding pipeline in detail
- Obsidian vault preprocessing (wikilinks, embeds, frontmatter, callouts)

---

## Development

```bash
pip install -e ".[dev]"
pytest                                          # 182 tests
pocket-gm eval calibrate --campaign <id>       # tune threshold from real queries
pocket-gm eval run --campaign <id>             # recall@k + citation coverage
```

### Adding a new source type

1. Add a loader in `pocket_gm/ingestion/`
2. Add a table name helper in `pocket_gm/retrieval/store.py`
3. Add a retrieval branch in `pocket_gm/retrieval/router.py`
4. Add a prompt section in `pocket_gm/synthesis/prompt_builder.py`
5. Wire a CLI command in `pocket_gm/cli/commands/ingest.py`

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Scaffold | ✅ | Project structure, config, campaign management |
| 1 — Sourcebook ingestion | ✅ | PDF → sqlite-vec pipeline with heading detection |
| 2 — Static query | ✅ | Sourcebooks + GM notes, relevance gate, citation grounding |
| 3 — Audio + sessions | ✅ | Whisper transcription → timestamped session store |
| 4 — Eval harness | ✅ | Recall@k, citation coverage, threshold calibration |
| 5 — Google Drive | ✅ | OAuth2 sync with MD5 manifest cache |
| 6 — REST API | ✅ | FastAPI server, `pocket-gm serve` |

---

## License

MIT
