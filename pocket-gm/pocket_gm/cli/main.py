import typer

from pocket_gm.cli.commands import campaigns, eval_cmd, ingest, init_cmd, query, serve, sessions, status

app = typer.Typer(
    name="pocket-gm",
    help="Grounded Q&A for tabletop RPG Game Masters.",
    no_args_is_help=True,
)

app.add_typer(init_cmd.app, name="")
app.add_typer(campaigns.app, name="campaign")
app.add_typer(ingest.app, name="ingest")
app.add_typer(query.app, name="")
app.add_typer(sessions.app, name="session")
app.add_typer(eval_cmd.app, name="eval")
app.add_typer(status.app, name="")
app.add_typer(serve.app, name="")


if __name__ == "__main__":
    app()
