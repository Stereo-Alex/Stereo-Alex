from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table

from pocket_gm.core.campaign import Campaign, add_campaign, delete_campaign, get_campaign, list_campaigns
from pocket_gm.core.config import load_config
from pocket_gm.retrieval.store import Store, notes_table, sessions_table, sourcebook_table

app = typer.Typer(help="Manage campaigns")
console = Console()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@app.command("new")
def new_campaign(name: str = typer.Argument(..., help="Campaign name")):
    """Create a new campaign."""
    cfg = load_config()
    campaign_id = _slugify(name)
    if not campaign_id:
        console.print(
            f"[red]Error:[/red] '{name}' produces an empty campaign id. "
            "Use a name with letters or numbers."
        )
        raise typer.Exit(1)
    campaign = Campaign(
        id=campaign_id,
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        add_campaign(cfg.campaigns_dir, campaign)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Created campaign[/green] [bold]{name}[/bold] (id: {campaign_id})")


@app.command("list")
def list_campaigns_cmd():
    """List all campaigns."""
    cfg = load_config()
    campaigns = list_campaigns(cfg.campaigns_dir)
    if not campaigns:
        console.print("No campaigns yet. Run [bold]pocket-gm campaign new <name>[/bold] to create one.")
        return
    table = Table(title="Campaigns")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Created")
    for c in campaigns:
        table.add_row(c.id, c.name, c.created_at[:10])
    console.print(table)


@app.command("delete")
def delete_campaign_cmd(campaign_id: str = typer.Argument(..., help="Campaign ID")):
    """Delete a campaign."""
    cfg = load_config()
    if not get_campaign(cfg.campaigns_dir, campaign_id):
        console.print(f"[red]Campaign '{campaign_id}' not found.[/red]")
        raise typer.Exit(1)
    confirm = typer.confirm(f"Delete campaign '{campaign_id}' and all its indexed data?")
    if not confirm:
        raise typer.Exit()

    # Drop the three vector tables so no orphaned embeddings linger in the db.
    try:
        store = Store(cfg.lancedb_path)
        for table in (sourcebook_table(campaign_id), notes_table(campaign_id), sessions_table(campaign_id)):
            store.drop_table(table)
    except Exception as e:
        console.print(f"[yellow]Warning: could not drop vector tables: {e}[/yellow]")

    # Remove the per-campaign directory (ingest registry, transcripts, gdrive cache).
    campaign_dir = cfg.campaigns_dir / campaign_id
    if campaign_dir.exists():
        shutil.rmtree(campaign_dir, ignore_errors=True)

    delete_campaign(cfg.campaigns_dir, campaign_id)
    console.print(f"[yellow]Deleted[/yellow] campaign '{campaign_id}' and its indexed data")
