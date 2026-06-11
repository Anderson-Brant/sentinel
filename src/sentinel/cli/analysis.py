"""Strategy analysis commands: backtest, regimes, ablate.

These are plain functions; registration on the top-level app happens in
:mod:`sentinel.cli` so the full command surface stays in one place.
"""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel.config import load_config

console = Console()


def backtest(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    long_threshold: float = typer.Option(0.55, help="Enter long when P(up) exceeds this."),
    short_threshold: float = typer.Option(0.45, help="Enter short when P(up) falls below this (requires --allow-short)."),
    cost_bps: float = typer.Option(2.0, help="Transaction cost in basis points per unit of |Δposition| (per side)."),
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
    SMA crossover). No retraining - this is a post-hoc slice of the same
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
