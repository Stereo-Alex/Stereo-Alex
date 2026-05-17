from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
from pocket_gm.retrieval.store import Store, notes_table, sourcebook_table

app = typer.Typer(help="Ingest campaign materials")
console = Console()


def _require_campaign(campaign_id: str):
    cfg = load_config()
    campaign = get_campaign(cfg.campaigns_dir, campaign_id)
    if not campaign:
        console.print(f"[red]Campaign '{campaign_id}' not found.[/red] Run [bold]pocket-gm campaign list[/bold].")
        raise typer.Exit(1)
    return cfg, campaign


def _ingest_chunks(chunks: list[Chunk] | list[dict], table_name: str, campaign_id: str, cfg, label: str) -> None:
    embedder = Embedder(cfg.embedding.model)
    store = Store(cfg.lancedb_path)

    texts = [c["text"] if isinstance(c, dict) else c.text for c in chunks]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.embed(texts, batch_size=cfg.embedding.batch_size)

    raw = [c if isinstance(c, dict) else {
        "text": c.text, "source_type": c.source_type, "filename": c.filename,
        "page": c.page, "heading": c.heading, "chunk_index": c.chunk_index,
    } for c in chunks]

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
    _ingest_chunks(chunks, sourcebook_table(campaign), campaign, cfg, file.name)
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

        # Mark ingested after collecting chunks (we'll commit after bulk embed)
        # Store hash→chunks mapping so we can call mark_ingested after embed
        f._pending_hash = file_hash  # type: ignore[attr-defined]
        f._pending_chunks = len(chunks)  # type: ignore[attr-defined]

    if not all_chunks:
        if skipped:
            console.print("[dim]All files already ingested.[/dim]")
        return

    _ingest_chunks(all_chunks, notes_table(campaign), campaign, cfg, f"{len(files) - skipped} file(s)")

    # Mark all successfully ingested files
    for f in files:
        if hasattr(f, "_pending_hash"):
            mark_ingested(registry_path, f._pending_hash, f.name, f._pending_chunks)  # type: ignore[attr-defined]


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

    _ingest_chunks(all_chunks, notes_table(campaign), campaign, cfg, f"{len(note_chunk_map)} Obsidian notes")

    for file_hash, filename, chunk_count in note_chunk_map:
        mark_ingested(registry_path, file_hash, filename, chunk_count)
