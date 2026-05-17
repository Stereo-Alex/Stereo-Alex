from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pocket_gm.core.campaign import get_campaign
from pocket_gm.core.config import load_config
from pocket_gm.core.ingest_registry import get_registry_path
from pocket_gm.retrieval.store import Store, notes_table, sessions_table, sourcebook_table

app = typer.Typer(help="Show campaign status")
console = Console()


def _get_chunk_counts(store: Store, campaign_id: str) -> dict[str, int]:
    """Return chunk counts for each store type."""
    return {
        "Sourcebook": store.count(sourcebook_table(campaign_id)),
        "GM Notes": store.count(notes_table(campaign_id)),
        "Sessions": store.count(sessions_table(campaign_id)),
    }


def _get_ingested_files(campaigns_dir: Path, campaign_id: str) -> dict[str, list[str]]:
    """Return filenames grouped by a rough source type based on the registry."""
    registry_path = get_registry_path(campaigns_dir, campaign_id)
    if not registry_path.exists():
        return {}

    with open(registry_path) as f:
        data: dict = json.load(f)

    # Group by extension heuristic: .pdf → sourcebook, .md/.txt → notes
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in data.values():
        filename: str = entry.get("filename", "")
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            groups["sourcebook"].append(filename)
        else:
            groups["notes"].append(filename)

    return dict(groups)


def _get_last_query(logs_path: Path, campaign_id: str) -> str | None:
    """Return a formatted string for the last query for this campaign, or None."""
    log_file = logs_path / "queries.ndjson"
    if not log_file.exists():
        return None

    last_record = None
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("campaign_id") == campaign_id:
                last_record = record

    if last_record is None:
        return None

    ts_raw: str = last_record.get("timestamp", "")
    question: str = last_record.get("question", "")

    # Format timestamp: "2025-03-15T19:30:00+00:00" → "2025-03-15 19:30"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts_raw)
        ts_display = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts_display = ts_raw

    return f"{ts_display} — \"{question}\""


@app.command("status")
def status(
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
):
    """Show what has been indexed for a campaign."""
    cfg = load_config()
    camp = get_campaign(cfg.campaigns_dir, campaign)
    if not camp:
        console.print(f"[red]Campaign '{campaign}' not found.[/red] Run [bold]pocket-gm campaign list[/bold].")
        raise typer.Exit(1)

    store = Store(cfg.lancedb_path)
    counts = _get_chunk_counts(store, campaign)
    total = sum(counts.values())

    # Header
    console.print(f"\n[bold]Status — {camp.name}[/bold]")

    # Chunk counts table
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Store")
    tbl.add_column("Chunks", justify="right")
    for store_name, n in counts.items():
        tbl.add_row(store_name, f"{n:,}")
    console.print(tbl)

    non_zero_stores = sum(1 for n in counts.values() if n > 0)
    console.print(f"Total: {total:,} chunks across {non_zero_stores} store(s)\n")

    # Ingested files
    ingested = _get_ingested_files(cfg.campaigns_dir, campaign)
    if ingested:
        total_files = sum(len(v) for v in ingested.values())
        console.print(f"[bold]Ingested files ({total_files}):[/bold]")
        for group, filenames in ingested.items():
            names_preview = ", ".join(filenames[:2])
            extra = len(filenames) - 2
            if extra > 0:
                names_preview += f" ({extra} more)"
            console.print(f"  {group + ':':12s} {names_preview}")
    else:
        console.print("[dim]No files ingested yet.[/dim]")

    # Last query
    last = _get_last_query(cfg.logs_path, campaign)
    if last:
        console.print(f"\n[bold]Last query:[/bold] {last}")
    else:
        console.print("\n[dim]No queries logged yet.[/dim]")
