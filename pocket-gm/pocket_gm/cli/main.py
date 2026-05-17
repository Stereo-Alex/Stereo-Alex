import typer

from pocket_gm.cli.commands import campaigns, ingest

app = typer.Typer(
    name="pocket-gm",
    help="Grounded Q&A for tabletop RPG Game Masters.",
    no_args_is_help=True,
)

app.add_typer(campaigns.app, name="campaign")
app.add_typer(ingest.app, name="ingest")


if __name__ == "__main__":
    app()
