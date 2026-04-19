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
schedule_app = typer.Typer(
    help="Run + inspect scheduled ingestion / feature jobs.", no_args_is_help=True
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(schedule_app, name="schedule")

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


@app.command("score-sentiment")
def score_reddit_sentiment(
    scorer: str = typer.Option(
        "vader",
        "--scorer",
        help="Sentiment backend: 'vader' (rule-based, no deps) or 'finbert' "
        "(financial-news transformer; requires transformers + torch).",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        help="Override the HF model checkpoint (finbert only). "
        "Default: ProsusAI/finbert.",
    ),
    batch_size: int = typer.Option(
        16, "--batch-size", help="Batch size for finBERT inference.", min=1
    ),
) -> None:
    """Re-score all Reddit posts AND tweets currently in storage."""
    from sentinel.features.sentiment import score_posts, score_tweets
    from sentinel.storage import get_store

    store = get_store()
    # Pull every row — backend-agnostic; tables are small and this is a batch op.
    posts = store.read_all_reddit_posts()
    tweets = store.read_all_tweets()

    if posts.empty and tweets.empty:
        console.print(
            "[yellow]No Reddit posts or tweets in storage — nothing to score.[/yellow]"
        )
        return

    fb = None
    if scorer.lower() != "vader":
        from sentinel.features.finbert import DEFAULT_MODEL, FinBertScorer

        try:
            fb = FinBertScorer(
                model_name=model_name or DEFAULT_MODEL,
                batch_size=batch_size,
            )
        except ImportError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e
        console.print(
            f"[cyan]Using finBERT[/cyan] ({model_name or DEFAULT_MODEL}) "
            f"on [bold]{len(posts)}[/bold] posts + "
            f"[bold]{len(tweets)}[/bold] tweets (batch={batch_size})."
        )

    n_posts = 0
    if not posts.empty:
        scored_posts = score_posts(posts) if fb is None else score_posts(posts, scorer=fb)
        n_posts = store.update_reddit_sentiment(scored_posts)

    n_tweets = 0
    if not tweets.empty:
        scored_tweets = (
            score_tweets(tweets) if fb is None else score_tweets(tweets, scorer=fb)
        )
        n_tweets = store.update_tweet_sentiment(scored_tweets)

    console.print(
        f"[green]✓[/green] Scored [bold]{n_posts}[/bold] posts + "
        f"[bold]{n_tweets}[/bold] tweets using [cyan]{scorer.lower()}[/cyan]."
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

    Crypto bars share the same `prices` table as equities — the symbol is
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


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Train / Evaluate / Predict
# ---------------------------------------------------------------------------


@app.command()
def train(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    track: bool = typer.Option(
        False, "--track/--no-track", help="Log params/metrics/artifacts to MLflow."
    ),
    experiment: str = typer.Option(
        "sentinel", "--experiment", help="MLflow experiment name (only used with --track)."
    ),
    mlflow_uri: str | None = typer.Option(
        None, "--mlflow-uri", help="MLflow tracking URI (default: mlflow's built-in, ./mlruns)."
    ),
) -> None:
    """Train a baseline model on SYMBOL's features."""
    from sentinel.models.registry import save_model, train_model
    from sentinel.storage import get_store
    from sentinel.tracking import get_tracker

    store = get_store()
    features = store.read_features(symbol)
    if features.empty:
        console.print(
            f"[red]No features for {symbol}.[/red] Run `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    tracker = get_tracker(track=track, experiment=experiment, tracking_uri=mlflow_uri)
    with tracker.start_run(run_name=f"train__{symbol.upper()}__{model}"):
        tracker.set_tag("command", "train")
        tracker.set_tag("symbol", symbol.upper())
        tracker.set_tag("model", model)
        result = train_model(features, model_name=model, cfg=cfg)
        path = save_model(symbol, model, result)
        tracker.log_params(
            {
                "symbol": symbol.upper(),
                "model": model,
                "random_state": cfg.modeling.random_state,
                "test_fraction": cfg.modeling.test_fraction,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_features": len(result.feature_names),
            }
        )
        tracker.log_metrics(
            {
                "holdout_accuracy": result.holdout_accuracy,
                "holdout_f1": result.holdout_f1,
                "holdout_roc_auc": result.holdout_roc_auc,
                "baseline_accuracy": result.baseline_accuracy,
                "class_balance": float(result.metadata.get("class_balance", float("nan"))),
            }
        )
        tracker.log_artifact(path)

    console.print(
        f"[green]✓[/green] Trained [bold]{model}[/bold] on [cyan]{symbol}[/cyan]. "
        f"Holdout accuracy = [bold]{result.holdout_accuracy:.3f}[/bold] "
        f"(baseline = {result.baseline_accuracy:.3f}). Saved to [dim]{path}[/dim]."
    )


@app.command()
def evaluate(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
) -> None:
    """Walk-forward evaluation with full metric report."""
    from sentinel.evaluation.walk_forward import walk_forward_evaluate
    from sentinel.reporting.console import render_evaluation
    from sentinel.storage import get_store

    store = get_store()
    features = store.read_features(symbol)
    if features.empty:
        console.print(
            f"[red]No features for {symbol}.[/red] Run `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    report = walk_forward_evaluate(features, model_name=model, cfg=load_config())
    render_evaluation(symbol, model, report)


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    long_threshold: float = typer.Option(0.55, help="Enter long when P(up) exceeds this."),
    short_threshold: float = typer.Option(0.45, help="Enter short when P(up) falls below this (requires --allow-short)."),
    cost_bps: float = typer.Option(2.0, help="Transaction cost in basis points per unit of |Δposition|."),
    allow_short: bool = typer.Option(False, help="Permit short positions."),
    periods_per_year: int = typer.Option(
        252, help="Annualization factor. 252 for daily equities, 365 for crypto daily."
    ),
    vol_target: float | None = typer.Option(
        None,
        "--vol-target",
        help="Annualized portfolio vol target (e.g. 0.10 for 10%). "
        "When set, positions are sized so size * realized_vol ≈ target. "
        "Default: off (fixed-size {-1, 0, +1}).",
    ),
    vol_lookback: int = typer.Option(
        20, "--vol-lookback", help="Rolling window for the realized-vol estimate."
    ),
    max_leverage: float = typer.Option(
        1.0,
        "--max-leverage",
        help="Cap on position size when vol-targeting. 1.0 means never lever beyond fully invested.",
    ),
    track: bool = typer.Option(
        False, "--track/--no-track", help="Log params/metrics to MLflow."
    ),
    experiment: str = typer.Option(
        "sentinel", "--experiment", help="MLflow experiment name (only used with --track)."
    ),
    mlflow_uri: str | None = typer.Option(
        None, "--mlflow-uri", help="MLflow tracking URI."
    ),
) -> None:
    """Backtest a model's walk-forward probabilities as a trading strategy.

    Uses OOS probabilities from walk-forward splits (never a full-sample refit),
    shifts positions by one bar to avoid look-ahead, and charges transaction
    costs on every change in |position|. Compares against buy-and-hold.
    """
    from sentinel.backtest.engine import backtest as run_backtest
    from sentinel.evaluation.walk_forward import walk_forward_predictions
    from sentinel.reporting.console import render_backtest
    from sentinel.storage import get_store
    from sentinel.tracking import get_tracker

    store = get_store()
    features = store.read_features(symbol)
    prices = store.read_prices(symbol)
    if features.empty or prices.empty:
        console.print(
            f"[red]Missing data for {symbol}.[/red] "
            f"Run `sentinel ingest prices {symbol}` and `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    probs = walk_forward_predictions(features, model_name=model, cfg=cfg)

    tracker = get_tracker(track=track, experiment=experiment, tracking_uri=mlflow_uri)
    with tracker.start_run(run_name=f"backtest__{symbol.upper()}__{model}"):
        tracker.set_tag("command", "backtest")
        tracker.set_tag("symbol", symbol.upper())
        tracker.set_tag("model", model)
        tracker.log_params(
            {
                "symbol": symbol.upper(),
                "model": model,
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "cost_bps": cost_bps,
                "allow_short": allow_short,
                "periods_per_year": periods_per_year,
                "target_vol_annual": vol_target,
                "vol_lookback": vol_lookback,
                "max_leverage": max_leverage,
            }
        )
        report = run_backtest(
            prices=prices,
            probabilities=probs,
            symbol=symbol.upper(),
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            cost_bps=cost_bps,
            allow_short=allow_short,
            periods_per_year=periods_per_year,
            target_vol_annual=vol_target,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
        )
        tracker.log_metrics(
            {
                "total_return": report.total_return,
                "benchmark_total_return": report.benchmark_total_return,
                "excess_total_return": report.total_return - report.benchmark_total_return,
                "annualized_return": report.annualized_return,
                "benchmark_annualized_return": report.benchmark_annualized_return,
                "annualized_vol": report.annualized_vol,
                "sharpe": report.sharpe,
                "benchmark_sharpe": report.benchmark_sharpe,
                "max_drawdown": report.max_drawdown,
                "benchmark_max_drawdown": report.benchmark_max_drawdown,
                "win_rate": report.win_rate,
                "n_trades": report.n_trades,
                "exposure": report.exposure,
                "turnover": report.turnover,
                "n_oos_bars": report.n_oos_bars,
            }
        )
    render_backtest(report)


@app.command()
def regimes(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    long_threshold: float = typer.Option(0.55, help="Long entry threshold."),
    short_threshold: float = typer.Option(0.45, help="Short entry threshold."),
    cost_bps: float = typer.Option(2.0, help="Transaction cost in bps."),
    allow_short: bool = typer.Option(False, help="Permit short positions."),
    vol_window: int = typer.Option(20, help="Rolling window (bars) for vol-regime detection."),
    trend_fast: int = typer.Option(50, help="Fast SMA length for trend regime."),
    trend_slow: int = typer.Option(200, help="Slow SMA length for trend regime."),
    periods_per_year: int = typer.Option(252, help="Annualization factor."),
    vol_target: float | None = typer.Option(
        None,
        "--vol-target",
        help="Annualized vol target for position sizing in the underlying backtest.",
    ),
    vol_lookback: int = typer.Option(
        20, "--vol-lookback", help="Rolling window for the realized-vol estimate (sizing)."
    ),
    max_leverage: float = typer.Option(
        1.0, "--max-leverage", help="Cap on position size when --vol-target is set."
    ),
) -> None:
    """Backtest SYMBOL and slice performance by volatility + trend regime.

    Answers the question "when does this strategy actually work?" by
    computing strategy-vs-benchmark metrics within three volatility buckets
    (rolling-std terciles) and a bull/bear trend classifier (fast vs slow
    SMA crossover). No retraining — this is a post-hoc slice of the same
    OOS backtest output.
    """
    from sentinel.backtest.engine import backtest as run_backtest
    from sentinel.evaluation.regimes import analyze_regimes
    from sentinel.evaluation.walk_forward import walk_forward_predictions
    from sentinel.reporting.console import render_regime_analysis
    from sentinel.storage import get_store

    store = get_store()
    features = store.read_features(symbol)
    prices = store.read_prices(symbol)
    if features.empty or prices.empty:
        console.print(
            f"[red]Missing data for {symbol}.[/red] "
            f"Run `sentinel ingest prices {symbol}` and "
            f"`sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    probs = walk_forward_predictions(features, model_name=model, cfg=cfg)
    report = run_backtest(
        prices=prices,
        probabilities=probs,
        symbol=symbol.upper(),
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        cost_bps=cost_bps,
        allow_short=allow_short,
        periods_per_year=periods_per_year,
        target_vol_annual=vol_target,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
    )
    regime_reports = analyze_regimes(
        report,
        prices,
        vol_window=vol_window,
        trend_fast=trend_fast,
        trend_slow=trend_slow,
        periods_per_year=periods_per_year,
    )
    render_regime_analysis(regime_reports)


@app.command()
def ablate(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    backtest_variants: bool = typer.Option(
        True,
        "--backtest/--no-backtest",
        help="Also backtest each variant to compare Sharpe / total return.",
    ),
    long_threshold: float = typer.Option(0.55, help="Backtest long entry threshold."),
    short_threshold: float = typer.Option(0.45, help="Backtest short entry threshold."),
    cost_bps: float = typer.Option(2.0, help="Transaction cost in bps."),
    allow_short: bool = typer.Option(False, help="Permit short positions in backtests."),
) -> None:
    """Compare technical-only vs sentiment-only vs hybrid on the same walk-forward splits.

    Requires Reddit posts already ingested + sentiment scored for SYMBOL. The
    command rebuilds the hybrid feature table in-memory (so stored features
    don't need to include sentiment), runs walk-forward on the three variants,
    and reports whether sentiment actually adds out-of-sample value.
    """
    import pandas as pd

    from sentinel.evaluation.ablation import run_ablation
    from sentinel.features.pipeline import build_feature_table
    from sentinel.features.sentiment import (
        MENTION_COUNT_COLS,
        SENTIMENT_FEATURE_COLS,
        sentiment_features_for_symbol,
    )
    from sentinel.reporting.console import render_ablation
    from sentinel.storage import get_store

    store = get_store()
    prices = store.read_prices(symbol)
    if prices.empty:
        console.print(
            f"[red]No prices for {symbol}.[/red] Run `sentinel ingest prices {symbol}` first."
        )
        raise typer.Exit(code=1)

    sentiment_df = sentiment_features_for_symbol(
        store, symbol, index=pd.to_datetime(prices.index)
    )
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
            f"[red]No Reddit or Twitter mentions found for {symbol}.[/red] "
            f"Run `sentinel ingest reddit --whitelist {symbol}` or "
            f"`sentinel ingest twitter --whitelist {symbol}` first."
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    features = build_feature_table(prices, cfg=cfg, sentiment=sentiment_df)

    report = run_ablation(
        features,
        symbol=symbol.upper(),
        model_name=model,
        cfg=cfg,
        sentiment_columns=SENTIMENT_FEATURE_COLS,
        prices=prices if backtest_variants else None,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        cost_bps=cost_bps,
        allow_short=allow_short,
    )
    render_ablation(report)


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
) -> None:
    """Generate the latest prediction using the saved model."""
    from sentinel.models.registry import load_model, predict_latest
    from sentinel.reporting.console import render_prediction
    from sentinel.storage import get_store

    store = get_store()
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
# Explain — feature importance
# ---------------------------------------------------------------------------


@app.command()
def explain(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    method: str = typer.Option(
        "permutation", "--method", help="Importance method: permutation | shap."
    ),
    top: int = typer.Option(20, "--top", help="Rows to render in the table."),
    n_repeats: int = typer.Option(
        10, "--n-repeats", help="Shuffles per feature (permutation method only)."
    ),
    scoring: str = typer.Option(
        "accuracy", "--scoring", help="Score to permute against: accuracy | roc_auc."
    ),
    max_samples: int = typer.Option(
        500, "--max-samples", help="Subsample size for SHAP (ignored for permutation)."
    ),
) -> None:
    """Explain which features the saved model actually leans on.

    Uses the saved ``(symbol, model)`` pipeline. Run ``sentinel train`` first.
    """
    from sentinel.evaluation.importance import permutation_importance, shap_importance
    from sentinel.models.registry import load_model
    from sentinel.reporting.console import render_importance
    from sentinel.storage import get_store

    method = method.lower()
    if method not in {"permutation", "shap"}:
        console.print(f"[red]Unknown method {method!r}.[/red] Use 'permutation' or 'shap'.")
        raise typer.Exit(code=1)

    store = get_store()
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

    feat_cols = artifact.feature_names
    X = features[feat_cols].astype(float).to_numpy()
    y = features["target_direction"].astype(int).to_numpy()

    if method == "permutation":
        result = permutation_importance(
            artifact.pipeline,
            X,
            y,
            feat_cols,
            n_repeats=n_repeats,
            random_state=load_config().modeling.random_state,
            scoring=scoring,
        )
    else:
        try:
            result = shap_importance(
                artifact.pipeline,
                X,
                feat_cols,
                max_samples=max_samples,
                random_state=load_config().modeling.random_state,
            )
        except ImportError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e

    render_importance(symbol, model, result, top=top)


# ---------------------------------------------------------------------------
# Demo — end-to-end convenience wrapper
# ---------------------------------------------------------------------------


@app.command()
def demo(
    symbol: str = typer.Argument("SPY", help="Ticker to demo on. Defaults to SPY."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
) -> None:
    """Run the full MVP loop end-to-end: ingest → features → train → evaluate → backtest → predict.

    Note: these sub-commands are Typer-decorated. When we call them as plain
    Python functions (not via the CLI dispatcher), every ``typer.Option(...)``
    default would remain an unresolved ``OptionInfo`` sentinel — which then
    blows up downstream (e.g. MLflow trying to ``urlparse`` it). So we pass
    *every* option-defaulted parameter explicitly here, mirroring the CLI
    defaults.
    """
    console.rule(f"[bold]Sentinel demo — {symbol}[/bold]")
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


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@schedule_app.command("run")
def schedule_run(
    once: bool = typer.Option(
        False,
        "--once/--forever",
        help="Run one pass over due jobs and exit (default), or loop forever.",
    ),
    tick_seconds: int | None = typer.Option(
        None,
        "--tick-seconds",
        help="Seconds between ticks in --forever mode. Defaults to scheduler.tick_seconds.",
    ),
) -> None:
    """Run scheduled jobs defined under ``scheduler.jobs`` in config."""
    from sentinel.scheduling import Scheduler, load_jobs_from_config
    from sentinel.storage import get_store

    cfg = load_config()
    jobs = load_jobs_from_config(cfg)
    if not jobs:
        console.print(
            "[yellow]No scheduler.jobs configured.[/yellow] "
            "Add jobs to your YAML config — see README."
        )
        raise typer.Exit(code=0)

    store = get_store()
    sched = Scheduler(store, jobs)

    if once:
        runs = sched.run_once()
        if not runs:
            console.print("[dim]No jobs due.[/dim]")
            return
        for r in runs:
            colour = {"success": "green", "error": "red", "skipped": "yellow"}.get(
                r.status, "white"
            )
            console.print(
                f"[{colour}]{r.status}[/{colour}] [cyan]{r.job_name}[/cyan] "
                f"in {r.duration_seconds:.2f}s, rows={r.rows_written}"
                + (f" — {r.error}" if r.error else "")
            )
        return

    tick = tick_seconds if tick_seconds is not None else cfg.scheduler.tick_seconds
    console.print(
        f"[green]✓[/green] Scheduler running. Jobs: "
        f"{', '.join(j.name for j in jobs)}. Tick: {tick}s. Ctrl-C to stop."
    )
    sched.run_forever(tick_seconds=tick)
    console.print("[dim]Scheduler stopped.[/dim]")


@schedule_app.command("status")
def schedule_status() -> None:
    """Show every configured job and when it last ran / is next due."""
    from datetime import timedelta

    from rich.table import Table

    from sentinel.scheduling import Scheduler, load_jobs_from_config
    from sentinel.storage import get_store

    cfg = load_config()
    jobs = load_jobs_from_config(cfg)
    if not jobs:
        console.print("[yellow]No scheduler.jobs configured.[/yellow]")
        raise typer.Exit(code=0)

    store = get_store()
    sched = Scheduler(store, jobs)
    now = sched.clock()

    table = Table(title="Scheduled jobs")
    table.add_column("name", style="cyan", no_wrap=True)
    table.add_column("kind")
    table.add_column("interval", justify="right")
    table.add_column("enabled", justify="center")
    table.add_column("last run", style="dim")
    table.add_column("next due", style="dim")

    for spec in jobs:
        last = store.last_run_for(spec.name)
        next_due = sched.next_due_at(spec)
        last_str = last.isoformat(timespec="seconds") if last else "—"
        if not spec.enabled:
            next_str = "disabled"
        elif next_due is None:
            next_str = "next tick"
        elif next_due <= now:
            next_str = "due now"
        else:
            delta = next_due - now
            next_str = f"in {int(delta.total_seconds())}s"
        table.add_row(
            spec.name,
            spec.kind,
            f"{spec.interval_seconds}s",
            "✓" if spec.enabled else "✗",
            last_str,
            next_str,
        )
        _ = timedelta  # quiet unused import warning
    console.print(table)


@schedule_app.command("history")
def schedule_history(
    job: str | None = typer.Option(
        None, "--job", help="Filter to a single job name."
    ),
    limit: int = typer.Option(
        20, "--limit", help="Max rows to show, newest first.", min=1
    ),
) -> None:
    """Show the most recent rows of the ``job_runs`` log."""
    from rich.table import Table

    from sentinel.storage import get_store

    store = get_store()
    df = store.read_job_runs(job_name=job, limit=limit)
    if df.empty:
        console.print("[dim]No job runs recorded yet.[/dim]")
        return

    table = Table(title=f"Job history ({len(df)} row{'s' if len(df) != 1 else ''})")
    table.add_column("job", style="cyan")
    table.add_column("started", style="dim")
    table.add_column("dur", justify="right")
    table.add_column("status")
    table.add_column("rows", justify="right")
    table.add_column("error", style="red")

    for row in df.itertuples(index=False):
        status = str(row.status)
        colour = {"success": "green", "error": "red", "skipped": "yellow"}.get(
            status, "white"
        )
        duration = (row.finished_at - row.started_at).total_seconds()
        table.add_row(
            str(row.job_name),
            row.started_at.isoformat(timespec="seconds"),
            f"{duration:.2f}s",
            f"[{colour}]{status}[/{colour}]",
            str(int(row.rows_written) if row.rows_written is not None else 0),
            (row.error or "")[:80],
        )
    console.print(table)


if __name__ == "__main__":
    app()
