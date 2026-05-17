# Pocket GM — Architecture

## Overview

Pocket GM is a local-first retrieval-augmented generation (RAG) system. Given a question, it retrieves relevant passages from three separate vector stores, then asks a local LLM to synthesise a cited answer from only those passages.

The two hard constraints that drive every design decision:

1. **No hallucinations** — the LLM must not invent facts. It can only use what was retrieved.
2. **Local-only** — no API calls, no cloud services, no data leaving the machine.

---

## Three Stores Per Campaign

Each campaign maintains three independent sqlite-vec tables:

| Store | Content | Chunk size | Citation label |
|-------|---------|------------|----------------|
| `sourcebook_{id}` | Official PDFs, rulebooks | 512 tokens, 64 overlap | `filename, p.N` |
| `notes_{id}` | GM notes, Obsidian vaults, markdown | 256 tokens, 32 overlap | `filename — heading` |
| `sessions_{id}` | Session transcripts from audio | 256 tokens, 32 overlap | `Session N (date, timestamp)` |

### Why three stores instead of one?

A single merged store loses information that matters:

- **Authority** — "Kingmaker p.45 says X" and "your GM note says Y" carry different weight; they must be cited separately so the GM can judge
- **Chunk size** — sourcebook prose is dense (512 tokens); session transcripts are conversational and need smaller windows (256) to stay coherent
- **Score calibration** — sourcebook and session embeddings occupy different regions of the vector space; a merged score would be noisy
- **Query latency** — all three are queried with `asyncio.gather` so there is no speed penalty over querying fewer stores

### Why not ChromaDB, FAISS, or LanceDB?

| Option | Reason rejected |
|--------|----------------|
| ChromaDB | User preference |
| FAISS | No built-in metadata, no persistence, requires separate file management |
| LanceDB | Good fit initially, but sqlite-vec ships as a single `.db` file with no server process and is simpler to distribute |
| **sqlite-vec** | Single `.db` file, WAL mode, integrates with standard sqlite3, no extra process |

---

## Data Flow

### Ingestion

```
Source file (PDF / markdown / audio / Drive)
    │
    ├─► Loader        extracts raw text + metadata (page, heading, timestamp)
    ├─► Chunker       sliding window with heading prefix injection
    ├─► Embedder      all-MiniLM-L6-v2, batch=64, L2-normalised numpy array
    ├─► Store.add_documents()
    │       ├─► INSERT INTO {table}(embedding) → vec0 virtual table
    │       └─► INSERT INTO {table}_meta → companion metadata table
    └─► IngestRegistry  SHA256 hash → ingested.json (idempotency)
```

### Query

```
User question
    │
    ├─► Embedder.embed_one()       same model as ingestion
    │
    ├─► asyncio.gather [parallel]
    │       ├─► Store.query(sourcebook_table, vec, top_k)
    │       ├─► Store.query(notes_table,      vec, top_k)
    │       └─► Store.query(sessions_table,   vec, top_k)
    │
    ├─► Relevance gate
    │       if best score < threshold (default 0.35):
    │           return "not found" — LLM never called
    │
    ├─► build_prompt()
    │       numbered [1][2][3] citation markers, three labelled sections
    │       json_mode=True: {"answer": "...", "citations": [1,2,3]} format
    │
    ├─► OllamaClient.generate()    httpx POST to localhost:11434
    │
    ├─► validate_citations() / parse_json_answer()
    │       extract citation markers, verify each ID is in the retrieved set,
    │       flag uncited sentences with a warning
    │
    └─► log_query()    NDJSON record → ~/.pocket-gm/logs/queries.ndjson
```

---

## Embedding & Scoring

**Model:** `all-MiniLM-L6-v2` (384-dim, ~80 MB, CPU-compatible, MIT licence)

Embeddings are L2-normalised before storage. sqlite-vec returns L2 distance; cosine similarity is recovered as:

```python
score = max(0.0, 1.0 - distance / 2.0)
```

This matches cosine similarity for normalised vectors without an extra dot-product step.

**Relevance threshold:** 0.35 by default. Tune with:

```bash
pocket-gm eval calibrate --campaign <id>
```

This computes score percentiles across your actual query log and suggests the 25th percentile as the threshold (loose enough to surface results, tight enough to block noise).

---

## Anti-Hallucination Layers

### Layer 1 — Retrieval gate

The LLM is never called if no retrieved chunk scores above `relevance_threshold`. The CLI returns a "not found" panel; the REST API returns an answer of `"No relevant information found"` with an empty citations list.

### Layer 2 — Prompt grounding

The system prompt explicitly instructs the LLM:

- Cite every factual claim with `[N]` markers
- Say "I don't have information about that" if a topic is absent from the provided context
- Never add information not present in the numbered sources

### Layer 3 — Citation enforcement (post-processing)

`validate_citations()` splits the answer on citation markers using `re.split(r"(\[\d+\])", answer)`. Any text segment not immediately followed by a citation marker is flagged as an uncited sentence. The CLI displays these as warnings; the API returns `uncited_count`.

`parse_json_answer()` handles `json_citations: true` mode — the LLM returns `{"answer": "...", "citations": [1,3]}` which is then validated against the index map before falling back to `validate_citations` on any parse error.

### Layer 4 — Query logging

Every query is appended as a JSON line to `~/.pocket-gm/logs/queries.ndjson`:

```json
{
  "timestamp": "2025-03-15T19:30:00+00:00",
  "campaign_id": "kingmaker",
  "question": "Who is the Stag Lord?",
  "retrieved_chunks": [...],
  "answer": "...",
  "llm_model": "phi3:mini"
}
```

This log is the foundation of the eval harness (`pocket-gm eval run`) and threshold calibration (`pocket-gm eval calibrate`).

---

## Ingestion Sources

### PDF (`pdf_loader.py`)

Two-pass extraction with PyMuPDF:

1. Collect all text spans and compute the body font-size median
2. Classify headings by 2-of-3 vote: relative size >1.15× median, bold+above-body size, ALL-CAPS short line

Heading text is prepended to each chunk so retrieval can match on section names even when the heading is in a different chunk.

### Markdown / Text (`chunker.py`)

Heading-aware sliding window. `##` and `###` headings reset the active heading context; each chunk stores the last heading seen.

### Obsidian Vault (`obsidian_loader.py`)

Preprocessing pipeline before chunking:

- YAML frontmatter parsed for `tags`, `aliases`, `title`
- Dataview blocks removed (`` ```dataview ``` ``)
- Callouts and admonitions stripped to plain text
- `![[embedded_note]]` links resolved recursively (max depth 2, cycle-safe via `visited` set)
- `[[wikilinks]]` converted to plain text
- Inline `#tags` extracted and appended to chunk text

### Audio (`audio_transcriber.py`)

`faster-whisper` produces per-segment timestamps. Chunks snap to segment boundaries so `timestamp_start` always points to the start of a real segment, not mid-speech. Session citations include `Session N (date, HH:MM:SS)` for direct audio navigation.

Speaker diarization is a no-op stub by default. Enabling it requires `pyannote-audio` and a HuggingFace token — see `WhisperConfig.diarize`.

### Google Drive (`drive_loader.py`)

OAuth2 flow (first run opens a browser; token cached at `~/.pocket-gm/gdrive_token.json`). Subsequent syncs skip unchanged files by comparing MD5 checksums from the Drive API against a local manifest JSON. Google Docs are exported as `text/plain`. Audio files are transcribed locally after download when `--audio` is passed.

---

## REST API (`pocket_gm/api/`)

Built with FastAPI. Module-level singletons (config, embedder, store) are initialised on first request so the embedding model is loaded once per process.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check + Ollama availability |
| `GET` | `/campaigns` | List all campaigns |
| `GET` | `/campaigns/{id}/sessions` | Sessions with chunk counts, ordered by number |
| `GET` | `/campaigns/{id}/status` | Chunk counts per store |
| `POST` | `/query` | Full RAG pipeline, returns `answer`, `citations`, `fully_grounded`, `uncited_count` |

Start with `pocket-gm serve` (requires `pip install 'pocket-gm[serve]'`). Interactive docs at `http://localhost:8080/docs`.

---

## File Layout

```
pocket-gm/
  pocket_gm/
    api/
      app.py           # FastAPI app and route handlers
      models.py        # Pydantic request/response models
    cli/
      main.py          # Typer app, sub-app registration
      commands/
        campaigns.py   # campaign new / list / delete
        ingest.py      # ingest pdf / notes / obsidian / gdrive
        query.py       # ask
        sessions.py    # session add / list
        status.py      # status
        eval_cmd.py    # eval run / calibrate
        serve.py       # serve (uvicorn launcher)
    core/
      config.py        # Config dataclass, load_config / save_config
      campaign.py      # Campaign dataclass, JSON registry CRUD
      ingest_registry.py  # SHA256 dedup, ingested.json per campaign
      logger.py        # NDJSON query log
    ingestion/
      pdf_loader.py        # PyMuPDF, two-pass heading detection
      markdown_loader.py   # (unused directly; chunker handles markdown)
      chunker.py           # chunk_pdf_pages, chunk_markdown, chunk_obsidian_note, chunk_transcript
      embedder.py          # sentence-transformers wrapper, L2 normalisation
      audio_transcriber.py # faster-whisper, segment timestamps, diarization stub
      obsidian_loader.py   # wikilink/embed/frontmatter preprocessing
      drive_loader.py      # Google Drive OAuth2, MD5 manifest sync
    retrieval/
      store.py   # sqlite-vec wrapper: add_documents, query, count, list_sessions, drop_table
      router.py  # asyncio.gather over all three stores, QueryResult, relevance helpers
    synthesis/
      prompt_builder.py   # citation-numbered prompt assembly, json_mode
      ollama_client.py    # httpx POST to Ollama, is_available()
      grounding.py        # validate_citations, parse_json_answer, GroundedAnswer
    eval/
      harness.py    # load_query_log, load_eval_set, run_eval
      metrics.py    # EvalMetrics: recall@k, citation_coverage, no_result_rate
      calibrate.py  # score percentiles from query log, suggest threshold
  tests/
    fixtures/
      sample.pdf
      eval_set.json
    test_*.py      # 182 tests
  docs/
    architecture.md   # this file
  config.example.yaml
  pyproject.toml
  README.md
```

---

## Configuration Reference

```yaml
campaigns_dir: ~/.pocket-gm/campaigns   # campaign registry JSON files
lancedb_path:  ~/.pocket-gm/vectors.db  # sqlite-vec .db file (key kept for backwards compat)
logs_path:     ~/.pocket-gm/logs        # NDJSON query log directory

embedding:
  model: all-MiniLM-L6-v2  # any sentence-transformers model
  batch_size: 64

retrieval:
  top_k: 5                  # chunks retrieved per store
  relevance_threshold: 0.35 # cosine similarity gate; tune with eval calibrate
  reranker_enabled: false   # cross-encoder reranker (Phase 3+ future)
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

llm:
  provider: ollama
  base_url: http://localhost:11434
  model: phi3:mini
  temperature: 0.1
  max_tokens: 512
  json_citations: false     # true → LLM returns {"answer": "...", "citations": [1,2]}

whisper:
  model_size: base    # tiny / base / small / medium
  language: en
  device: cpu         # cuda if available
  diarize: false      # requires pyannote-audio + HuggingFace token

chunking:
  sourcebook:
    chunk_size: 512
    chunk_overlap: 64
  notes:
    chunk_size: 256
    chunk_overlap: 32
  sessions:
    chunk_size: 256
    chunk_overlap: 32
```
