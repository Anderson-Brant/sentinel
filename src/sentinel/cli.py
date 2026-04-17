"""Typer-based command-line interface.

Entry point installed as the `sentinel` script (see pyproject.toml).

Example usage:

    sentinel ingest prices SPY --start 2015-01-01
    sentinel features build SPY
    sentinel train SPY --model logistic
    sentinel evaluate SPY
    sentinel predict SPY
    sentinel demo SPY       # end-to-end convenience wrapper
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from sentinel import __version__
from sentinel.config import load_config
from sentinel.utils.logging import get_logger

app = typer.Typer(
    name="sentinel",
    help="Market intelligence & stock prediction platform.",
    no_args_is_help=True,
    add_completion=False,
)

ingest_app = typer.Typer(help="Data ingestion commands.", no_args_is_help=True)
features_app = typer.Typer(help="Feature engineering commands.", no_args_is_help=True)

app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")

console = Console()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed Sentinel version."""
    console.print(f"[bold]sentinel[/bold] {__version__}")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@ingest_app.command("prices")
def ingest_prices(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. SPY, AAPL, BTC-USD."),
    start: Optional[str] = typer.Option(None, help="ISO start date. Defaults to config."),
    end: Optional[str] = typer.Option(None, help="ISO end date. Defaults to today."),
    interval: Optional[str] = typer.Option(None, help="yfinance interval (e.g. 1d, 1h)."),
) -> None:
    """Download OHLCV history and persist it to DuckDB."""
    from sentinel.ingestion.market import ingest_prices as _ingest
    from sentinel.storage.duckdb_store import DuckDBStore

    cfg = load_config()
    start = start or cfg.ingestion.market.default_start
    interval = interval or cfg.ingestion.market.default_interval

    df = _ingest(symbol, start=start, end=end, interval=interval)
    store = DuckDBStore()
    n = store.write_prices(symbol, df)
    console.print(f"[green]✓[/green] Ingested [bold]{n}[/bold] rows for [cyan]{symbol}[/cyan]")


@ingest_app.command("reddit")
def ingest_reddit(
    symbol: str = typer.Argument(..., help="Ticker symbol to search for."),  # noqa: ARG001
) -> None:
    """Pull Reddit posts mentioning SYMBOL. (Not yet implemented.)"""
    console.print(
        "[yellow]reddit ingestion is on the roadmap — see README.[/yellow]\n"
        "Configure credentials in .env when ready."
    )
    raise typer.Exit(code=1)


@ingest_app.command("twitter")
def ingest_twitter(
    symbol: str = typer.Argument(..., help="Ticker symbol / cashtag to search for."),  # noqa: ARG001
) -> None:
    """Pull tweets mentioning SYMBOL. (Not yet implemented.)"""
    console.print(
        "[yellow]twitter ingestion is on the roadmap — see README.[/yellow]\n"
        "Configure credentials in .env when ready."
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@features_app.command("build")
def features_build(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
) -> None:
    """Compute technical features + target from stored prices and persist them."""
    from sentinel.features.pipeline import build_feature_table
    from sentinel.storage.duckdb_store import DuckDBStore

    store = DuckDBStore()
    prices = store.read_prices(symbol)
    if prices.empty:
        console.print(
            f"[red]No prices for {symbol}.[/red] Run `sentinel ingest prices {symbol}` first."
        )
        raise typer.Exit(code=1)

    features = build_feature_table(prices, cfg=load_config())
    n = store.write_features(symbol, features)
    console.print(
        f"[green]✓[/green] Built [bold]{n}[/bold] feature rows × "
        f"[bold]{len(features.columns)}[/bold] cols for [cyan]{symbol}[/cyan]"
    )


# ---------------------------------------------------------------------------
# Train / Evaluate / Predict
# ---------------------------------------------------------------------------


@app.command()
def train(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest."),
) -> None:
    """Train a baseline model on SYMBOL's features."""
    from sentinel.models.registry import save_model, train_model
    from sentinel.storage.duckdb_store import DuckDBStore

    store = DuckDBStore()
    features = store.read_features(symbol)
    if features.empty:
        console.print(
            f"[red]No features for {symbol}.[/red] Run `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    result = train_model(features, model_name=model, cfg=cfg)
    path = save_model(symbol, model, result)
    console.print(
        f"[green]✓[/green] Trained [bold]{model}[/bold] on [cyan]{symbol}[/cyan]. "
        f"Holdout accuracy = [bold]{result.holdout_accuracy:.3f}[/bold] "
        f"(baseline = {result.baseline_accuracy:.3f}). Saved to [dim]{path}[/dim]."
    )


@app.command()
def evaluate(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest."),
) -> None:
    """Walk-forward evaluation with full metric report."""
    from sentinel.evaluation.walk_forward import walk_forward_evaluate
    from sentinel.reporting.console import render_evaluation
    from sentinel.storage.duckdb_store import DuckDBStore

    store = DuckDBStore()
    features = store.read_features(symbol)
    if features.empty:
        console.print(
            f"[red]No features for {symbol}.[/red] Run `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    report = walk_forward_evaluate(features, model_name=model, cfg=load_config())
    render_evaluation(symbol, model, report)


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest."),
) -> None:
    """Generate the latest prediction using the saved model."""
    from sentinel.models.registry import load_model, predict_latest
    from sentinel.reporting.console import render_prediction
    from sentinel.storage.duckdb_store import DuckDBStore

    store = DuckDBStore()
    features = store.read_features(symbol)
    if features.empty:
        console.print(f"[red]No features for {symbol}.[/red]")
        raise typer.Exit(code=1)

    artifact = load_model(symbol, model)
    if artifact is None:
        console.print(
            f"[red]No saved {model} model for {symbol}.[/red] "
            f"Run `sentinel train {symbol} --model {model}` first."
        )
        raise typer.Exit(code=1)

    pred = predict_latest(artifact, features)
    render_prediction(symbol, model, pred)


# ---------------------------------------------------------------------------
# Demo — end-to-end convenience wrapper
# ---------------------------------------------------------------------------


@app.command()
def demo(
    symbol: str = typer.Argument("SPY", help="Ticker to demo on. Defaults to SPY."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest."),
) -> None:
    """Run the full MVP loop end-to-end: ingest → features → train → evaluate → predict."""
    console.rule(f"[bold]Sentinel demo — {symbol}[/bold]")
    ingest_prices(symbol=symbol, start=None, end=None, interval=None)
    features_build(symbol=symbol)
    train(symbol=symbol, model=model)
    evaluate(symbol=symbol, model=model)
    predict(symbol=symbol, model=model)


if __name__ == "__main__":
    app()
