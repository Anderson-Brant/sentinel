"""Ingestion commands: prices, reddit, twitter, crypto + sentiment re-scoring.

``score-sentiment`` lives here (not in features) because the ingest commands
invoke it inline after a fetch - it is part of landing data, not building the
feature table.
"""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel.config import load_config

ingest_app = typer.Typer(help="Data ingestion commands.", no_args_is_help=True)

console = Console()


@ingest_app.command("prices")
def ingest_prices(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. SPY, AAPL, BTC-USD."),
    start: str | None = typer.Option(None, help="ISO start date. Defaults to config."),
    end: str | None = typer.Option(None, help="ISO end date. Defaults to today."),
    interval: str | None = typer.Option(None, help="yfinance interval (e.g. 1d, 1h)."),
) -> None:
    """Download OHLCV history and persist it to the configured store."""
    from sentinel.ingestion.market import ingest_prices as _ingest
    from sentinel.storage import get_store

    cfg = load_config()
    start = start or cfg.ingestion.market.default_start
    interval = interval or cfg.ingestion.market.default_interval

    df = _ingest(symbol, start=start, end=end, interval=interval)
    store = get_store()
    n = store.write_prices(symbol, df)
    console.print(f"[green]✓[/green] Ingested [bold]{n}[/bold] rows for [cyan]{symbol}[/cyan]")


@ingest_app.command("reddit")
def ingest_reddit(
    whitelist: str | None = typer.Option(
        None,
        "--whitelist",
        help="Comma-separated tickers to match without $ prefix (e.g. SPY,AAPL,TSLA).",
    ),
    subreddits: str | None = typer.Option(
        None,
        "--subreddits",
        help="Comma-separated subreddits. Overrides config.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Max posts per subreddit. Overrides config."
    ),
    score_sentiment: bool = typer.Option(
        True, help="Also run VADER sentiment scoring on newly fetched posts."
    ),
) -> None:
    """Pull Reddit posts, extract ticker mentions, optionally score sentiment."""
    from sentinel.ingestion.reddit import ingest_posts
    from sentinel.storage import get_store

    cfg = load_config()
    reddit_cfg = cfg.ingestion.reddit
    if subreddits:
        reddit_cfg = reddit_cfg.model_copy(
            update={"subreddits": [s.strip() for s in subreddits.split(",") if s.strip()]}
        )
    if limit:
        reddit_cfg = reddit_cfg.model_copy(update={"max_posts_per_run": int(limit)})

    wl = (
        {t.strip().upper() for t in whitelist.split(",") if t.strip()}
        if whitelist
        else None
    )

    store = get_store()
    counts = ingest_posts(store=store, reddit_cfg=reddit_cfg, symbol_whitelist=wl)
    console.print(
        f"[green]✓[/green] Reddit: fetched [bold]{counts['fetched']}[/bold], "
        f"wrote [bold]{counts['posts_written']}[/bold] posts, "
        f"[bold]{counts['mentions_written']}[/bold] mentions."
    )

    if score_sentiment and counts["posts_written"] > 0:
        score_reddit_sentiment()


def score_reddit_sentiment() -> None:
    """Re-score all Reddit posts AND tweets currently in storage with VADER."""
    from sentinel.features.sentiment import score_posts, score_tweets
    from sentinel.storage import get_store

    store = get_store()
    # Pull every row - backend-agnostic; tables are small and this is a batch op.
    posts = store.read_all_reddit_posts()
    tweets = store.read_all_tweets()

    if posts.empty and tweets.empty:
        console.print(
            "[yellow]No Reddit posts or tweets in storage - nothing to score.[/yellow]"
        )
        return

    n_posts = 0
    if not posts.empty:
        n_posts = store.update_reddit_sentiment(score_posts(posts))

    n_tweets = 0
    if not tweets.empty:
        n_tweets = store.update_tweet_sentiment(score_tweets(tweets))

    console.print(
        f"[green]✓[/green] Scored [bold]{n_posts}[/bold] posts + "
        f"[bold]{n_tweets}[/bold] tweets."
    )


@ingest_app.command("twitter")
def ingest_twitter(
    whitelist: str | None = typer.Option(
        None,
        "--whitelist",
        help="Comma-separated tickers. Used to build the cashtag query "
        "(e.g. SPY,AAPL,TSLA → '($SPY OR $AAPL OR $TSLA) -is:retweet lang:en') "
        "and to filter mention extraction.",
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        help="Raw v2 recent-search query. Overrides the one built from --whitelist.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Max tweets per run. Overrides config."
    ),
    score_sentiment_flag: bool = typer.Option(
        True,
        "--score-sentiment/--no-score-sentiment",
        help="Also run VADER sentiment scoring on newly fetched tweets.",
    ),
) -> None:
    """Pull tweets via Twitter/X v2 API, extract mentions, optionally score sentiment."""
    from sentinel.ingestion.twitter import ingest_tweets
    from sentinel.storage import get_store

    cfg = load_config()
    twitter_cfg = cfg.ingestion.twitter
    if limit:
        twitter_cfg = twitter_cfg.model_copy(update={"max_tweets_per_run": int(limit)})

    wl = (
        {t.strip().upper() for t in whitelist.split(",") if t.strip()}
        if whitelist
        else None
    )

    if query is None and not wl:
        console.print(
            "[red]Provide either --whitelist or --query.[/red] "
            "Cashtag queries need at least one ticker."
        )
        raise typer.Exit(code=1)

    store = get_store()
    try:
        counts = ingest_tweets(
            store=store,
            twitter_cfg=twitter_cfg,
            symbol_whitelist=wl,
            query=query,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    console.print(
        f"[green]✓[/green] Twitter: fetched [bold]{counts['fetched']}[/bold], "
        f"wrote [bold]{counts['tweets_written']}[/bold] tweets, "
        f"[bold]{counts['mentions_written']}[/bold] mentions."
    )

    if score_sentiment_flag and counts["tweets_written"] > 0:
        score_reddit_sentiment()


@ingest_app.command("crypto")
def ingest_crypto(
    symbol: str = typer.Argument(
        ..., help="Crypto symbol: yfinance-style BTC-USD or CCXT-style BTC/USDT."
    ),
    start: str | None = typer.Option(
        None, "--start", help="ISO start date. Defaults to config."
    ),
    end: str | None = typer.Option(
        None, "--end", help="ISO end date. Defaults to most recent."
    ),
    interval: str | None = typer.Option(
        None,
        "--interval",
        help="CCXT timeframe (1d, 1h, 5m, ...). Defaults to config (1d).",
    ),
    exchange: str | None = typer.Option(
        None,
        "--exchange",
        help="CCXT exchange id (binance, coinbase, kraken, ...). Defaults to config.",
    ),
    quote: str | None = typer.Option(
        None,
        "--quote",
        help="Exchange-side quote currency (USDT, USDC, ...). Defaults to config.",
    ),
) -> None:
    """Download crypto OHLCV history via CCXT and persist to the configured store.

    Crypto bars share the same `prices` table as equities - the symbol is
    stored as ``BTC-USD`` regardless of which stablecoin the exchange quotes
    in, so `sentinel features build BTC-USD` and the rest of the pipeline
    work identically for crypto and stocks.
    """
    from sentinel.ingestion.crypto import ingest_crypto_prices
    from sentinel.storage import get_store

    cfg = load_config()
    cc = cfg.ingestion.crypto
    start = start or cc.default_start
    interval = interval or cc.default_interval
    exchange = exchange or cc.default_exchange
    quote = quote or cc.default_quote

    try:
        df = ingest_crypto_prices(
            symbol,
            start=start,
            end=end,
            timeframe=interval,
            exchange=exchange,
            default_quote=quote,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    stored_symbol = str(df["symbol"].iloc[0]) if not df.empty else symbol.upper()
    store = get_store()
    n = store.write_prices(stored_symbol, df)
    console.print(
        f"[green]✓[/green] Ingested [bold]{n}[/bold] {interval} bars for "
        f"[cyan]{stored_symbol}[/cyan] from [cyan]{exchange}[/cyan]"
    )
