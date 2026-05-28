from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pocket_gm.core.campaign import get_campaign
from pocket_gm.core.config import load_config
from pocket_gm.core.ingest_registry import get_registry_path, hash_file, is_ingested, mark_ingested
from pocket_gm.ingestion.chunker import Chunk, chunk_markdown, chunk_obsidian_note, chunk_pdf_pages
from pocket_gm.ingestion.embedder import Embedder
from pocket_gm.ingestion.obsidian_loader import load_obsidian_vault
from pocket_gm.ingestion.pdf_loader import load_pdf
from pocket_gm.retrieval.store import Store, notes_table, sourcebook_table, sessions_table

app = typer.Typer(help="Ingest campaign materials")
console = Console()


def _require_campaign(campaign_id: str):
    cfg = load_config()
    campaign = get_campaign(cfg.campaigns_dir, campaign_id)
    if not campaign:
        console.print(f"[red]Campaign '{campaign_id}' not found.[/red] Run [bold]pocket-gm campaign list[/bold].")
        raise typer.Exit(1)
    return cfg, campaign


def _ingest_chunks(chunks: list[Chunk] | list[dict], table_name: str, campaign_id: str, cfg, label: str, replace: bool = False) -> None:
    embedder = Embedder(cfg.embedding.model)
    store = Store(cfg.lancedb_path)

    texts = [c["text"] if isinstance(c, dict) else c.text for c in chunks]

    raw = [c if isinstance(c, dict) else {
        "text": c.text, "source_type": c.source_type, "filename": c.filename,
        "page": c.page, "heading": c.heading, "chunk_index": c.chunk_index,
    } for c in chunks]

    # On --force re-ingestion, remove the file's existing chunks first so they
    # are replaced rather than duplicated in the vector store.
    if replace:
        for fname in {r.get("filename", "") for r in raw if r.get("filename")}:
            store.delete_by_filename(table_name, fname)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.embed(texts, batch_size=cfg.embedding.batch_size)

    store.add_documents(table_name, raw, embeddings, campaign_id)
    console.print(f"[green]Ingested[/green] {len(chunks)} chunks from {label}")


@app.command("pdf")
def ingest_pdf(
    file: Path = typer.Argument(..., help="PDF file to ingest"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already indexed"),
):
    """Ingest a sourcebook PDF."""
    cfg, camp = _require_campaign(campaign)
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    registry_path = get_registry_path(cfg.campaigns_dir, campaign)
    file_hash = hash_file(file)

    if not force and is_ingested(registry_path, file_hash):
        console.print(f"[dim]Already ingested: {file.name} (use --force to re-ingest)[/dim]")
        return

    console.print(f"Loading [bold]{file.name}[/bold]...")
    pages = load_pdf(file)
    chunks = chunk_pdf_pages(pages, cfg.chunking.sourcebook.chunk_size, cfg.chunking.sourcebook.chunk_overlap)
    console.print(f"  {len(pages)} pages → {len(chunks)} chunks")
    _ingest_chunks(chunks, sourcebook_table(campaign), campaign, cfg, file.name, replace=force)
    mark_ingested(registry_path, file_hash, file.name, len(chunks))


@app.command("notes")
def ingest_notes(
    path: Path = typer.Argument(..., help="Markdown file or directory"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already indexed"),
):
    """Ingest GM notes (markdown or text files)."""
    cfg, camp = _require_campaign(campaign)

    files: list[Path] = []
    if path.is_dir():
        files = list(path.rglob("*.md")) + list(path.rglob("*.txt"))
    elif path.is_file():
        files = [path]
    else:
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(1)

    if not files:
        console.print("[yellow]No markdown or text files found.[/yellow]")
        return

    registry_path = get_registry_path(cfg.campaigns_dir, campaign)
    all_chunks: list[Chunk] = []
    # (file_hash, filename, chunk_count) for files staged this run — committed
    # to the registry only after the bulk embed/store succeeds.
    pending: list[tuple[str, str, int]] = []
    skipped = 0

    for f in files:
        file_hash = hash_file(f)
        if not force and is_ingested(registry_path, file_hash):
            console.print(f"[dim]Already ingested: {f.name} (use --force to re-ingest)[/dim]")
            skipped += 1
            continue

        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(text, f.name, cfg.chunking.notes.chunk_size, cfg.chunking.notes.chunk_overlap)
        all_chunks.extend(chunks)
        console.print(f"  {f.name}: {len(chunks)} chunks")
        pending.append((file_hash, f.name, len(chunks)))

    if not all_chunks:
        if skipped:
            console.print("[dim]All files already ingested.[/dim]")
        return

    _ingest_chunks(all_chunks, notes_table(campaign), campaign, cfg, f"{len(pending)} file(s)", replace=force)

    # Mark all successfully ingested files
    for file_hash, filename, chunk_count in pending:
        mark_ingested(registry_path, file_hash, filename, chunk_count)


@app.command("obsidian")
def ingest_obsidian(
    vault: Path = typer.Argument(..., help="Path to Obsidian vault directory"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already indexed"),
):
    """Ingest an Obsidian vault, resolving wikilinks and frontmatter."""
    cfg, camp = _require_campaign(campaign)

    if not vault.is_dir():
        console.print(f"[red]Vault directory not found:[/red] {vault}")
        raise typer.Exit(1)

    console.print(f"Scanning vault [bold]{vault}[/bold]...")
    notes = load_obsidian_vault(vault)

    if not notes:
        console.print("[yellow]No notes found in vault (or all were empty after cleaning).[/yellow]")
        return

    console.print(f"  {len(notes)} notes found")

    registry_path = get_registry_path(cfg.campaigns_dir, campaign)
    all_chunks: list[Chunk] = []
    note_chunk_map: list[tuple[str, str, int]] = []  # (hash, filename, chunk_count)
    skipped = 0

    for note in notes:
        file_hash = hash_file(note.path)
        if not force and is_ingested(registry_path, file_hash):
            skipped += 1
            continue

        chunks = chunk_obsidian_note(note, cfg.chunking.notes.chunk_size, cfg.chunking.notes.chunk_overlap)
        all_chunks.extend(chunks)
        note_chunk_map.append((file_hash, note.path.name, len(chunks)))

    if skipped:
        console.print(f"  [dim]{skipped} note(s) already ingested (use --force to re-ingest)[/dim]")

    console.print(f"  {len(all_chunks)} total chunks")

    if not all_chunks:
        if skipped:
            console.print("[dim]All notes already ingested.[/dim]")
        else:
            console.print("[yellow]No chunks produced — notes may be empty.[/yellow]")
        return

    _ingest_chunks(all_chunks, notes_table(campaign), campaign, cfg, f"{len(note_chunk_map)} Obsidian notes", replace=force)

    for file_hash, filename, chunk_count in note_chunk_map:
        mark_ingested(registry_path, file_hash, filename, chunk_count)


@app.command("gdrive")
def ingest_gdrive(
    folder_url: str = typer.Argument(..., help="Google Drive folder URL or ID"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
    credentials: Path = typer.Option(
        Path("credentials.json"),
        "--credentials",
        help="Path to Google OAuth2 credentials JSON",
    ),
    token: Path = typer.Option(
        Path("~/.pocket-gm/gdrive_token.json").expanduser(),
        "--token",
        help="Path to cached OAuth2 token (created automatically)",
    ),
    audio: bool = typer.Option(False, "--audio", help="Also download audio files"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest already indexed files"),
):
    """Sync a Google Drive folder and ingest supported files."""
    try:
        from pocket_gm.ingestion.drive_loader import (
            build_service,
            parse_drive_id,
            sync_folder,
        )
        from pocket_gm.ingestion.audio_transcriber import transcribe, save_transcript
        from pocket_gm.ingestion.chunker import chunk_transcript
    except ImportError:
        console.print(
            "[red]Google Drive support requires extra dependencies.[/red]\n"
            "Install with: [bold]pip install 'pocket-gm[gdrive]'[/bold]"
        )
        raise typer.Exit(1)

    cfg, camp = _require_campaign(campaign)

    folder_id = parse_drive_id(folder_url)
    cache_dir = cfg.campaigns_dir / campaign / "gdrive_cache"
    manifest_path = cfg.campaigns_dir / campaign / "gdrive_manifest.json"

    console.print(f"Connecting to Google Drive (folder [bold]{folder_id}[/bold])...")
    try:
        service = build_service(token_path=token, credentials_path=credentials)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    with console.status("Syncing files from Drive..."):
        result = sync_folder(
            service,
            folder_id,
            cache_dir=cache_dir,
            manifest_path=manifest_path,
            include_audio=audio,
        )

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")

    console.print(
        f"Drive sync: [green]{len(result.downloaded)} downloaded[/green], "
        f"[dim]{result.skipped} skipped[/dim]"
    )

    if not result.downloaded:
        return

    registry_path = get_registry_path(cfg.campaigns_dir, campaign)

    # Separate documents from audio
    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"}
    doc_paths = [p for p in result.downloaded if p.suffix.lower() not in audio_exts]
    audio_paths = [p for p in result.downloaded if p.suffix.lower() in audio_exts]

    # Ingest documents (PDFs, text, markdown)
    pdf_chunks: list[Chunk] = []
    notes_chunks: list[Chunk] = []
    pdf_registry: list[tuple[str, str, int]] = []
    notes_registry: list[tuple[str, str, int]] = []

    for doc in doc_paths:
        file_hash = hash_file(doc)
        if not force and is_ingested(registry_path, file_hash):
            console.print(f"[dim]Already ingested: {doc.name}[/dim]")
            continue

        if doc.suffix.lower() == ".pdf":
            console.print(f"  Loading PDF: [bold]{doc.name}[/bold]")
            pages = load_pdf(doc)
            chunks = chunk_pdf_pages(pages, cfg.chunking.sourcebook.chunk_size, cfg.chunking.sourcebook.chunk_overlap)
            pdf_chunks.extend(chunks)
            pdf_registry.append((file_hash, doc.name, len(chunks)))
            console.print(f"    {len(pages)} pages → {len(chunks)} chunks")
        else:
            text = doc.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_markdown(text, doc.name, cfg.chunking.notes.chunk_size, cfg.chunking.notes.chunk_overlap)
            notes_chunks.extend(chunks)
            notes_registry.append((file_hash, doc.name, len(chunks)))
            console.print(f"  {doc.name}: {len(chunks)} chunks")

    if pdf_chunks:
        _ingest_chunks(pdf_chunks, sourcebook_table(campaign), campaign, cfg, "Drive PDFs", replace=force)
        for file_hash, filename, count in pdf_registry:
            mark_ingested(registry_path, file_hash, filename, count)

    if notes_chunks:
        _ingest_chunks(notes_chunks, notes_table(campaign), campaign, cfg, "Drive documents", replace=force)
        for file_hash, filename, count in notes_registry:
            mark_ingested(registry_path, file_hash, filename, count)

    # Transcribe and ingest audio (if --audio flag was set)
    for audio_file in audio_paths:
        file_hash = hash_file(audio_file)
        if not force and is_ingested(registry_path, file_hash):
            console.print(f"[dim]Already ingested audio: {audio_file.name}[/dim]")
            continue

        console.print(f"  Transcribing [bold]{audio_file.name}[/bold] (this may take a while)...")
        try:
            transcript = transcribe(
                audio_file,
                model_size=cfg.whisper.model_size,
                language=cfg.whisper.language,
                device=cfg.whisper.device,
            )
            transcripts_dir = cfg.campaigns_dir / campaign / "transcripts"
            save_transcript(transcript, transcripts_dir, audio_file.stem)

            raw_chunks = chunk_transcript(
                transcript.full_text,
                filename=audio_file.name,
                session_number=0,
                session_date="",
                chunk_size=cfg.chunking.sessions.chunk_size,
                chunk_overlap=cfg.chunking.sessions.chunk_overlap,
            )
            embedder = Embedder(cfg.embedding.model)
            texts = [c["text"] for c in raw_chunks]
            embeddings = embedder.embed(texts, batch_size=cfg.embedding.batch_size)
            store = Store(cfg.lancedb_path)
            store.add_documents(sessions_table(campaign), raw_chunks, embeddings, campaign)
            mark_ingested(registry_path, file_hash, audio_file.name, len(raw_chunks))
            console.print(f"    [green]Transcribed and ingested[/green] {len(raw_chunks)} chunks")
        except Exception as e:
            console.print(f"[yellow]Warning: could not process audio {audio_file.name}: {e}[/yellow]")
