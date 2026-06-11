"""Feature engineering commands."""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel.config import load_config

features_app = typer.Typer(help="Feature engineering commands.", no_args_is_help=True)

console = Console()


@features_app.command("build")
def features_build(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    with_sentiment: bool = typer.Option(
        False,
        "--with-sentiment/--no-sentiment",
        help="Also join Reddit sentiment features onto the technical block.",
    ),
) -> None:
    """Compute technical features + target from stored prices and persist them."""
    from sentinel.features.pipeline import build_feature_table
    from sentinel.storage import get_store

    store = get_store()
    prices = store.read_prices(symbol)
    if prices.empty:
        console.print(
            f"[red]No prices for {symbol}.[/red] Run `sentinel ingest prices {symbol}` first."
        )
        raise typer.Exit(code=1)

    sentiment_df = None
    if with_sentiment:
        import pandas as pd

        from sentinel.features.sentiment import sentiment_features_for_symbol

        sentiment_df = sentiment_features_for_symbol(
            store, symbol, index=pd.to_datetime(prices.index)
        )
        from sentinel.features.sentiment import MENTION_COUNT_COLS

        total_mentions = (
            0
            if sentiment_df.empty
            else int(
                sum(
                    sentiment_df[c].sum()
                    for c in MENTION_COUNT_COLS
                    if c in sentiment_df.columns
                )
            )
        )
        if total_mentions == 0:
            console.print(
                f"[yellow]warning:[/yellow] no Reddit/Twitter mentions for {symbol}; "
                "sentiment columns will be all zeros. "
                f"Try `sentinel ingest reddit --whitelist {symbol}` or "
                f"`sentinel ingest twitter --whitelist {symbol}`."
            )

    features = build_feature_table(prices, cfg=load_config(), sentiment=sentiment_df)
    n = store.write_features(symbol, features)
    tag = " (+sentiment)" if with_sentiment else ""
    console.print(
        f"[green]✓[/green] Built [bold]{n}[/bold] feature rows × "
        f"[bold]{len(features.columns)}[/bold] cols for [cyan]{symbol}[/cyan]{tag}"
    )
