from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pocket_gm.core.campaign import get_campaign
from pocket_gm.core.config import load_config
from pocket_gm.core.ingest_registry import get_registry_path, hash_file, is_ingested, mark_ingested
from pocket_gm.ingestion.audio_transcriber import load_transcript_json, save_transcript, transcribe
from pocket_gm.ingestion.chunker import chunk_transcript
from pocket_gm.ingestion.embedder import Embedder
from pocket_gm.retrieval.store import Store, sessions_table

app = typer.Typer(help="Manage session recordings")
console = Console()


@app.command("add")
def add_session(
    audio: Path = typer.Argument(..., help="Audio file (mp3, wav, m4a, flac)"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
    date: str = typer.Option(..., "--date", "-d", help="Session date (YYYY-MM-DD)"),
    number: int = typer.Option(..., "--number", "-n", help="Session number"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already indexed"),
):
    """Transcribe and ingest a session audio recording."""
    cfg = load_config()

    if not get_campaign(cfg.campaigns_dir, campaign):
        console.print(f"[red]Campaign '{campaign}' not found.[/red]")
        raise typer.Exit(1)

    if not audio.exists():
        console.print(f"[red]File not found:[/red] {audio}")
        raise typer.Exit(1)

    # Idempotency: skip if this exact audio file was already ingested.
    registry_path = get_registry_path(cfg.campaigns_dir, campaign)
    file_hash = hash_file(audio)
    if not force and is_ingested(registry_path, file_hash):
        console.print(f"[dim]Already ingested: {audio.name} (use --force to re-ingest)[/dim]")
        return

    raw_dir = cfg.campaigns_dir.parent / "raw" / "transcripts" / campaign
    stem = f"session_{number:02d}_{date}"

    # Check if already transcribed
    json_path = raw_dir / f"{stem}.json"
    if json_path.exists():
        console.print(f"[dim]Transcript already exists, re-using:[/dim] {json_path.name}")
        transcript = load_transcript_json(json_path)
    else:
        console.print(f"Transcribing [bold]{audio.name}[/bold] (model: {cfg.whisper.model_size})...")
        console.print("[dim]This may take a few minutes for long recordings.[/dim]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
            p.add_task("Transcribing audio...")
            transcript = transcribe(audio, cfg.whisper.model_size, cfg.whisper.language, cfg.whisper.device)
        txt_path, json_path = save_transcript(transcript, raw_dir, stem)
        console.print(f"  Transcript saved → {txt_path.name}")

    # Chunk transcript
    seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in transcript.segments]
    chunks = chunk_transcript(
        transcript.full_text,
        filename=f"{stem}.txt",
        session_number=number,
        session_date=date,
        chunk_size=cfg.chunking.sessions.chunk_size,
        chunk_overlap=cfg.chunking.sessions.chunk_overlap,
        timestamps=seg_dicts,
    )
    console.print(f"  {len(chunks)} chunks")

    # Embed and store
    embedder = Embedder(cfg.embedding.model)
    store = Store(cfg.lancedb_path)
    texts = [c["text"] for c in chunks]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.embed(texts, batch_size=cfg.embedding.batch_size)

    store.add_documents(sessions_table(campaign), chunks, embeddings, campaign)
    mark_ingested(registry_path, file_hash, audio.name, len(chunks))
    console.print(f"[green]Session {number} ingested[/green] ({date})")
