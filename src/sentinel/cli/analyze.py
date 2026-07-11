"""The `sentinel analyze` command: long-term scorecard for a ticker.

Registration on the top-level app happens in :mod:`sentinel.cli`.
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()

ANALYZE_DEFAULT_START = "2000-01-01"


def analyze(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    detail: str | None = typer.Option(
        None, "--detail", help="Expand one row into its underlying numbers: price | valuation."
    ),
    start: str = typer.Option(
        ANALYZE_DEFAULT_START,
        help="History start when prices need to be fetched. Earlier = longer CAGR windows.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Skip network fetches; use stored prices only (no valuation row).",
    ),
) -> None:
    """Long-term analysis scorecard: quality, valuation, price history, insiders, competitive.

    Grades come with one line of evidence each. This is structured information
    for a human decision, not a buy/sell signal.
    """
    from sentinel.analyze.analysis import build_analysis
    from sentinel.analyze.render import render_analysis
    from sentinel.storage import get_store

    if detail not in (None, "price", "valuation"):
        console.print(f"[red]Unknown --detail {detail!r}.[/red] Use price | valuation.")
        raise typer.Exit(code=1)

    store = get_store()
    prices = store.read_prices(symbol)
    if prices.empty and not offline:
        from sentinel.ingestion.market import ingest_prices

        console.print(f"[dim]No stored prices for {symbol.upper()}; fetching from {start}…[/dim]")
        try:
            prices = ingest_prices(symbol, start=start)
        except Exception as exc:
            console.print(f"[red]Price fetch failed for {symbol.upper()}: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        store.write_prices(symbol, prices)

    if prices.empty:
        console.print(
            f"[red]No prices for {symbol}.[/red] Run `sentinel ingest prices {symbol}` first."
        )
        raise typer.Exit(code=1)

    snapshot = None
    if not offline:
        from sentinel.fundamental.valuation import fetch_snapshot

        snapshot = fetch_snapshot(symbol)

    analysis = build_analysis(symbol, prices=prices, snapshot=snapshot)
    render_analysis(analysis, detail=detail)
