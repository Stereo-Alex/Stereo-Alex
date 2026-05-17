from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Run the Pocket GM REST API server")
console = Console()


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (development only)"),
):
    """Start the Pocket GM REST API server (requires pip install 'pocket-gm[serve]')."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Serve support requires extra dependencies.[/red]\n"
            "Install with: [bold]pip install 'pocket-gm[serve]'[/bold]"
        )
        raise typer.Exit(1)

    console.print(f"[bold cyan]Pocket GM API[/bold cyan] listening on [bold]http://{host}:{port}[/bold]")
    console.print("  [dim]POST /query[/dim]         — ask a question")
    console.print("  [dim]GET  /campaigns[/dim]     — list campaigns")
    console.print("  [dim]GET  /campaigns/{id}/sessions[/dim] — sessions for a campaign")
    console.print("  [dim]GET  /campaigns/{id}/status[/dim]   — chunk counts")
    console.print("  [dim]GET  /health[/dim]         — liveness + Ollama check")
    console.print("  [dim]GET  /docs[/dim]            — interactive API docs")

    uvicorn.run(
        "pocket_gm.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )
