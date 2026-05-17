import typer

from pocket_gm.cli.commands import campaigns, ingest, query, sessions

app = typer.Typer(
    name="pocket-gm",
    help="Grounded Q&A for tabletop RPG Game Masters.",
    no_args_is_help=True,
)

app.add_typer(campaigns.app, name="campaign")
app.add_typer(ingest.app, name="ingest")
app.add_typer(query.app, name="")
app.add_typer(sessions.app, name="session")


if __name__ == "__main__":
    app()
