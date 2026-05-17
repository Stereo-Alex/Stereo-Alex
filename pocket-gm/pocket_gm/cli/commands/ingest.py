from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pocket_gm.core.campaign import get_campaign
from pocket_gm.core.config import load_config
from pocket_gm.ingestion.chunker import Chunk, chunk_markdown, chunk_pdf_pages
from pocket_gm.ingestion.embedder import Embedder
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
):
    """Ingest a sourcebook PDF."""
    cfg, camp = _require_campaign(campaign)
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    console.print(f"Loading [bold]{file.name}[/bold]...")
    pages = load_pdf(file)
    chunks = chunk_pdf_pages(pages, cfg.chunking.sourcebook.chunk_size, cfg.chunking.sourcebook.chunk_overlap)
    console.print(f"  {len(pages)} pages → {len(chunks)} chunks")
    _ingest_chunks(chunks, sourcebook_table(campaign), campaign, cfg, file.name)


@app.command("notes")
def ingest_notes(
    path: Path = typer.Argument(..., help="Markdown file or directory"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
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

    all_chunks: list[Chunk] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(text, f.name, cfg.chunking.notes.chunk_size, cfg.chunking.notes.chunk_overlap)
        all_chunks.extend(chunks)
        console.print(f"  {f.name}: {len(chunks)} chunks")

    _ingest_chunks(all_chunks, notes_table(campaign), campaign, cfg, f"{len(files)} file(s)")
