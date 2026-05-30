from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from pocket_gm.core.config import Config, LLMConfig, save_config
from pocket_gm.core.campaign import Campaign, add_campaign, list_campaigns
from pocket_gm.core.config import _DEFAULT_CONFIG_PATH, load_config

app = typer.Typer(help="First-time setup wizard")
console = Console()

_OLLAMA_MODELS = [
    ("phi3:mini",          "3.8 B  — fastest, ~2.5 GB RAM (recommended for most laptops)"),
    ("llama3.2:3b",        "3 B    — balanced quality/speed, ~2 GB RAM"),
    ("mistral:7b-instruct","7 B    — best quality, ~4.5 GB RAM"),
]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _pick(prompt: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """Number-menu picker. Returns the value of the chosen option."""
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, (val, desc) in enumerate(options, 1):
        marker = "[green]>[/green]" if i == default else " "
        console.print(f"  {marker} [cyan]{i}[/cyan]  {val}  [dim]{desc}[/dim]")
    while True:
        raw = typer.prompt(f"   Choice", default=str(default))
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        console.print(f"[yellow]Enter a number between 1 and {len(options)}.[/yellow]")


@app.command("init")
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
):
    """Interactive first-time setup — creates your config and first campaign."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to Pocket GM[/bold cyan]\n"
        "[dim]Grounded Q&A for tabletop RPG Game Masters.[/dim]\n\n"
        "This wizard sets up your config and first campaign.\n"
        "Takes about 60 seconds.",
        border_style="cyan",
    ))

    # ── Existing config guard ────────────────────────────────────────────────
    if _DEFAULT_CONFIG_PATH.exists() and not force:
        console.print(f"\n[yellow]Config already exists:[/yellow] {_DEFAULT_CONFIG_PATH}")
        if not typer.confirm("Re-run setup and overwrite it?", default=False):
            console.print("[dim]Nothing changed. Run with --force to overwrite.[/dim]")
            raise typer.Exit(0)

    # ── LLM provider ────────────────────────────────────────────────────────
    provider = _pick(
        "How do you want to run the AI?",
        [
            ("ollama",  "Local models — free, private, no internet needed (requires Ollama)"),
            ("claude",  "Anthropic Claude API — best quality, needs internet + API key"),
        ],
    )

    # ── Provider-specific config ─────────────────────────────────────────────
    if provider == "ollama":
        model = _pick(
            "Which Ollama model?",
            _OLLAMA_MODELS,
        )
        cfg = Config()
        cfg.llm = LLMConfig(provider="ollama", base_url="http://localhost:11434", model=model)

        # Check if Ollama is actually reachable right now (non-fatal)
        try:
            import httpx
            r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if r.status_code == 200:
                tags = [m["name"] for m in r.json().get("models", [])]
                if not any(model.split(":")[0] in t for t in tags):
                    console.print(f"\n[yellow]Ollama is running but '{model}' is not pulled yet.[/yellow]")
                    console.print(f"  Run this now (in a separate terminal):\n  [bold]ollama pull {model}[/bold]")
                else:
                    console.print(f"\n[green]✓[/green] Ollama is running and '{model}' is ready.")
        except Exception:
            console.print(
                f"\n[yellow]Ollama doesn't appear to be running.[/yellow]\n"
                "  Install from [bold]https://ollama.com[/bold], then:\n"
                f"  [bold]ollama pull {model}[/bold]\n"
                "  [bold]ollama serve[/bold]"
            )

    else:  # claude
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            console.print("\n[yellow]ANTHROPIC_API_KEY not set in environment.[/yellow]")
            api_key = typer.prompt("  Paste your Anthropic API key (sk-ant-...)", hide_input=True)
            if not api_key.startswith("sk-ant-"):
                console.print("[yellow]That doesn't look like a valid key — continuing anyway.[/yellow]")
            # Write to shell rc for persistence hint
            console.print(
                f"\n[dim]Tip: add this to your shell profile so it persists:[/dim]\n"
                f'[dim]  export ANTHROPIC_API_KEY="{api_key[:12]}..."[/dim]'
            )
            # Set in current process so the rest of init can use it
            os.environ["ANTHROPIC_API_KEY"] = api_key

        claude_model = _pick(
            "Which Claude model?",
            [
                ("claude-haiku-4-5-20251001", "Fastest and cheapest  (~$0.001/question)"),
                ("claude-sonnet-4-6",         "Best quality          (~$0.01/question)"),
            ],
        )
        cfg = Config()
        cfg.llm = LLMConfig(provider="claude", model=claude_model)

    # ── Campaign name ────────────────────────────────────────────────────────
    console.print(Rule())
    existing = list_campaigns(cfg.campaigns_dir)
    if existing:
        console.print("\n[dim]Existing campaigns:[/dim] " + ", ".join(c.id for c in existing))

    campaign_name = typer.prompt("\nCampaign name (e.g. Kingmaker, Curse of Strahd)", default="My Campaign")
    campaign_id = _slugify(campaign_name)
    if not campaign_id:
        console.print("[red]That name produces an empty ID. Use letters or numbers.[/red]")
        raise typer.Exit(1)

    # ── Write config + create campaign ───────────────────────────────────────
    save_config(cfg)
    console.print(f"\n[green]✓[/green] Config written → [dim]{_DEFAULT_CONFIG_PATH}[/dim]")

    try:
        campaign = Campaign(id=campaign_id, name=campaign_name, created_at=datetime.now(timezone.utc).isoformat())
        add_campaign(cfg.campaigns_dir, campaign)
        console.print(f"[green]✓[/green] Campaign created → [bold]{campaign_name}[/bold] [dim](id: {campaign_id})[/dim]")
    except ValueError:
        console.print(f"[dim]Campaign '{campaign_id}' already exists — skipping.[/dim]")

    # ── Next steps ───────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold green]You're ready![/bold green]\n\n"
        f"Add your sourcebook PDFs:\n"
        f"  [cyan]pocket-gm ingest pdf <file.pdf> --campaign {campaign_id}[/cyan]\n\n"
        f"Add your GM notes (markdown files or a folder):\n"
        f"  [cyan]pocket-gm ingest notes <file-or-folder> --campaign {campaign_id}[/cyan]\n\n"
        f"Add a session recording:\n"
        f"  [cyan]pocket-gm session add <audio.mp3> --campaign {campaign_id} --date YYYY-MM-DD --number 1[/cyan]\n\n"
        f"Ask a question:\n"
        f"  [cyan]pocket-gm ask \"Who is the Stag Lord?\" --campaign {campaign_id}[/cyan]\n\n"
        f"[dim]Check what's indexed: pocket-gm status --campaign {campaign_id}[/dim]",
        title="[bold]Next steps[/bold]",
        border_style="green",
    ))
