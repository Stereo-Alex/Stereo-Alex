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

Pocket GM keeps two kinds of knowledge about your campaign and searches both when you ask a question:

- **Sourcebooks** — official adventure PDFs and rulebooks (high authority, static)
- **GM Notes** — your personal notes, homebrew rules, and campaign prep (your authority)
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
pocket-gm campaign delete <id>      # delete a campaign
```

### Ingesting materials

```bash
pocket-gm ingest pdf <file.pdf> --campaign <id>       # add a sourcebook
pocket-gm ingest notes <path> --campaign <id>         # add markdown/text notes
pocket-gm ingest gdrive <folder-url> --campaign <id>  # pull from Google Drive (requires extras)
```

### Ingesting sessions

```bash
pocket-gm session add <audio.mp3> --campaign <id> --date 2025-03-10 --number 4
pocket-gm session list --campaign <id>
```

### Querying

```bash
pocket-gm ask "<question>" --campaign <id>
```

### Evaluation

```bash
pocket-gm eval run --campaign <id>   # recall@k and citation coverage metrics
```

---

## How it works

Pocket GM uses **retrieval-augmented generation (RAG)** with three separate knowledge stores per campaign:

```
Your question
    │
    ├──► Sourcebook DB   ──┐
    ├──► GM Notes DB    ──►├── Ranked, cited chunks
    └──► Sessions DB    ──┘         │
                                    ▼
                          LLM synthesises answer
                          (only from retrieved chunks)
```

All three stores are queried in parallel. Each source is cited separately in the answer. If nothing relevant is found, the LLM is never called.

**Vector store:** [LanceDB](https://lancedb.github.io/lancedb/) — serverless, embedded, no server process required.  
**Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` — runs on CPU.  
**LLM:** Local Ollama models (`phi3:mini` default, `mistral:7b-instruct` for harder queries).  
**Audio transcription:** `faster-whisper` (local, open source).

---

## Grounding & Anti-Hallucination

Pocket GM uses four layers to prevent invented answers:

1. **Retrieval gate** — if no chunk scores above the relevance threshold (default 0.35 cosine similarity), the LLM is never called and the system returns "not found"
2. **Prompt-level instruction** — the LLM is explicitly told to cite every fact with `[1][2]` markers and say "not found" for anything absent from the sources
3. **Citation enforcement** — the output is post-processed to verify every citation marker refers to an actual retrieved chunk; uncited sentences are flagged with a warning
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
  relevance_threshold: 0.35   # raise if getting irrelevant results

llm:
  model: phi3:mini             # any model available in your Ollama

whisper:
  model_size: base             # tiny / base / medium — tradeoff speed vs accuracy
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for full design notes including:
- Why three separate stores instead of one
- Why LanceDB was chosen
- Chunking strategy differences between sourcebooks and transcripts
- The grounding pipeline in detail

---

## Development

```bash
pip install -e ".[dev]"
pytest
pocket-gm eval run --campaign <id>
```

### Adding a new source type

1. Add a loader in `pocket_gm/ingestion/`
2. Add a table name constant in `pocket_gm/retrieval/store.py`
3. Add a retrieval branch in `pocket_gm/retrieval/router.py`
4. Add a section in `pocket_gm/synthesis/prompt_builder.py`

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Scaffold | ✅ | Project structure, config, campaign management |
| 1 — Sourcebook ingestion | 🔄 | PDF → LanceDB pipeline |
| 2 — Static query | ⬜ | Ask questions against sourcebooks + GM notes |
| 3 — Audio + sessions | ⬜ | Whisper transcription → session DB |
| 4 — Eval harness | ⬜ | Recall@k and citation coverage metrics |
| 5 — Google Drive | ⬜ | Ingest from Google Drive |
| 6 — Mobile backend | ⬜ | FastAPI server + phone app |

---

## License

MIT
