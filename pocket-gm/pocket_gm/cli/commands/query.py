from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pocket_gm.core.campaign import get_campaign
from pocket_gm.core.config import load_config
from pocket_gm.core.logger import log_query
from pocket_gm.ingestion.embedder import Embedder
from pocket_gm.retrieval.router import query_all_sync
from pocket_gm.retrieval.store import Store
from pocket_gm.synthesis.grounding import parse_json_answer, validate_citations
from pocket_gm.synthesis.ollama_client import OllamaClient
from pocket_gm.synthesis.prompt_builder import build_prompt

app = typer.Typer()
console = Console()


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
):
    """Ask a question about your campaign."""
    cfg = load_config()

    if not get_campaign(cfg.campaigns_dir, campaign):
        console.print(f"[red]Campaign '{campaign}' not found.[/red]")
        raise typer.Exit(1)

    # Embed query
    embedder = Embedder(cfg.embedding.model)
    query_vec = embedder.embed_one(question)

    # Retrieve from all three stores in parallel
    store = Store(cfg.lancedb_path)
    result = query_all_sync(store, campaign, query_vec, top_k=cfg.retrieval.top_k)

    # Retrieval gate
    if result.is_empty(cfg.retrieval.relevance_threshold):
        console.print(Panel(
            "[yellow]No relevant information found in your campaign materials or session transcripts.[/yellow]\n"
            f"Question: [italic]{question}[/italic]",
            title="Pocket GM — Not Found",
            border_style="yellow",
        ))
        return

    # Build prompt and call LLM
    prompt, index_map = build_prompt(
        question,
        result.sourcebook,
        result.notes,
        result.sessions,
        threshold=cfg.retrieval.relevance_threshold,
        json_mode=cfg.llm.json_citations,
    )

    ollama = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model)

    if not ollama.is_available():
        console.print(f"[red]Ollama is not running at {cfg.llm.base_url}.[/red] Start it with: [bold]ollama serve[/bold]")
        raise typer.Exit(1)

    with console.status("Thinking..."):
        answer_text = ollama.generate(prompt, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)

    if cfg.llm.json_citations:
        grounded = parse_json_answer(answer_text, index_map)
    else:
        grounded = validate_citations(answer_text, index_map)

    # Display answer
    answer_display = Text(grounded.text)
    console.print(Panel(answer_display, title=f"[bold cyan]{question}[/bold cyan]", border_style="cyan"))

    # Warn about uncited sentences
    if grounded.uncited_sentences:
        console.print(f"[yellow]⚠ {len(grounded.uncited_sentences)} sentence(s) without citations:[/yellow]")
        for s in grounded.uncited_sentences:
            console.print(f"  [dim]{s}[/dim]")

    # Sources table
    if index_map:
        table = Table(title="Sources", show_header=True, header_style="bold")
        table.add_column("Ref", style="cyan", width=5)
        table.add_column("Source", style="dim")
        table.add_column("Score", width=6)

        for idx, chunk in index_map:
            if chunk.source_type == "transcript":
                ts = int(chunk.timestamp_start)
                h, m, s = ts // 3600, (ts % 3600) // 60, ts % 60
                ts_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
                label = f"Session {chunk.session_number} ({chunk.session_date}, {ts_str})"
            elif chunk.source_type == "markdown":
                label = chunk.filename + (f" — {chunk.heading}" if chunk.heading else "")
            else:
                label = chunk.filename + (f", p.{chunk.page}" if chunk.page else "")
                if chunk.heading:
                    label += f" — {chunk.heading}"
            table.add_row(f"[{idx}]", label, f"{chunk.score:.2f}")

        console.print(table)

    # Log
    log_query(cfg.logs_path, campaign, question, grounded, cfg.llm.model)
