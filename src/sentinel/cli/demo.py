"""End-to-end demo wrapper."""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel.cli.analysis import backtest
from sentinel.cli.features import features_build
from sentinel.cli.ingest import ingest_prices
from sentinel.cli.modeling import evaluate, predict, train

console = Console()


def demo(
    symbol: str = typer.Argument("SPY", help="Ticker to demo on. Defaults to SPY."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
) -> None:
    """Run the full MVP loop end-to-end: ingest → features → train → evaluate → backtest → predict.

    Note: these sub-commands are Typer-decorated. When we call them as plain
    Python functions (not via the CLI dispatcher), every ``typer.Option(...)``
    default would remain an unresolved ``OptionInfo`` sentinel - which then
    blows up downstream (e.g. MLflow trying to ``urlparse`` it). So we pass
    *every* option-defaulted parameter explicitly here, mirroring the CLI
    defaults.
    """
    console.rule(f"[bold]Sentinel demo - {symbol}[/bold]")
    ingest_prices(symbol=symbol, start=None, end=None, interval=None)
    features_build(symbol=symbol, with_sentiment=False)
    train(
        symbol=symbol,
        model=model,
        track=False,
        experiment="sentinel",
        mlflow_uri=None,
    )
    evaluate(symbol=symbol, model=model)
    backtest(
        symbol=symbol,
        model=model,
        long_threshold=0.55,
        short_threshold=0.45,
        cost_bps=2.0,
        allow_short=False,
        periods_per_year=252,
        vol_target=None,
        vol_lookback=20,
        max_leverage=1.0,
        track=False,
        experiment="sentinel",
        mlflow_uri=None,
    )
    predict(symbol=symbol, model=model)
