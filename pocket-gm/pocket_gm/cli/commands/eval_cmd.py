from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pocket_gm.core.config import load_config
from pocket_gm.eval.harness import run_eval

app = typer.Typer(help="Evaluate retrieval and grounding quality")
console = Console()


@app.command("run")
def eval_run(
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign ID"),
    eval_set: Path = typer.Option(
        Path("tests/fixtures/eval_set.json"),
        "--eval-set",
        "-e",
        help="Path to labeled eval set JSON",
    ),
):
    """Run retrieval quality evaluation against a labeled eval set."""
    cfg = load_config()

    if not eval_set.exists():
        console.print(f"[red]Eval set not found:[/red] {eval_set}")
        console.print("Create one at tests/fixtures/eval_set.json — see docs/eval-guide.md")
        raise typer.Exit(1)

    metrics = run_eval(cfg.logs_path, campaign, eval_set)

    table = Table(title=f"Eval Results — {campaign}", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total queries evaluated", str(metrics.total_queries))
    table.add_row("Recall@k", f"{metrics.recall_at_k:.1%}")
    table.add_row("Citation coverage", f"{metrics.citation_coverage:.1%}")
    table.add_row("No-result rate", f"{metrics.no_result_rate:.1%}")

    console.print(table)

    if metrics.recall_at_k < 0.6:
        console.print("[yellow]Tip:[/yellow] Low recall — try lowering chunk_size or reducing relevance_threshold in config.yaml")
    if metrics.citation_coverage < 0.8:
        console.print("[yellow]Tip:[/yellow] Low citation coverage — the LLM may need a stronger grounding instruction or a larger model")
