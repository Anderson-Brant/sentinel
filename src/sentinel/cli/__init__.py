"""Typer-based command-line interface.

Entry point installed as the `sentinel` script (see pyproject.toml).

Example usage:

    sentinel ingest prices SPY --start 2015-01-01
    sentinel features build SPY
    sentinel train SPY --model logistic
    sentinel evaluate SPY
    sentinel predict SPY
    sentinel demo SPY       # end-to-end convenience wrapper

The package is split by domain - ingestion in :mod:`sentinel.cli.ingest`,
feature building in :mod:`sentinel.cli.features`, train/evaluate/predict/
explain in :mod:`sentinel.cli.modeling`, backtest/regimes/ablate in
:mod:`sentinel.cli.analysis`, scheduler commands in
:mod:`sentinel.cli.schedule`, and the end-to-end wrapper in
:mod:`sentinel.cli.demo`. Every command is *registered* here, so the full
CLI surface stays visible in one place.
"""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel import __version__
from sentinel.cli.analysis import ablate, backtest, regimes
from sentinel.cli.demo import demo
from sentinel.cli.features import features_app
from sentinel.cli.ingest import ingest_app, score_reddit_sentiment
from sentinel.cli.modeling import evaluate, explain, predict, train
from sentinel.cli.schedule import schedule_app
from sentinel.utils.logging import get_logger

app = typer.Typer(
    name="sentinel",
    help="Market intelligence & stock prediction platform.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(schedule_app, name="schedule")

console = Console()
log = get_logger(__name__)


@app.command()
def version() -> None:
    """Print the installed Sentinel version."""
    console.print(f"[bold]sentinel[/bold] {__version__}")


app.command("score-sentiment")(score_reddit_sentiment)
app.command()(train)
app.command()(evaluate)
app.command()(backtest)
app.command()(regimes)
app.command()(ablate)
app.command()(predict)
app.command()(explain)
app.command()(demo)


if __name__ == "__main__":
    app()
