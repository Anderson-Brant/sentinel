"""Regime detection and regime-sliced performance.

Why
---
A backtest that reports a single Sharpe ratio over 10 years of data hides
the question every reader actually has: *when does this strategy work?*
A long-biased model can post a great Sharpe simply because the test window
included a long bull run. Slicing performance by regime forces us to be
honest:

    - **Volatility regime** — bins each bar into low / mid / high realized
      vol (rolling std of asset returns), classified by quantile across the
      full window.
    - **Trend regime** — bull when a fast SMA is above a slow SMA on the
      asset itself, bear otherwise.

For each regime, we report strategy *and* benchmark metrics on the
within-regime subset of returns. The benchmark column is what makes this
honest: "long-only model has Sharpe 1.2 in bull regime" is meaningless until
you see that buy-and-hold did 1.5 in the same regime.

What we do NOT do
-----------------
- Hidden-Markov-model regime detection. That's a research project of its own;
  for a portfolio piece, transparent quantile + SMA-crossover rules are
  defensible and easy to audit.
- Per-regime backtests with retrained models. The point is to slice
  *existing* OOS performance, not to leak future regime info into training.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


Axis = Literal["volatility", "trend"]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_vol_regimes(
    returns: pd.Series,
    *,
    window: int = 20,
    labels: Iterable[str] = ("low", "mid", "high"),
) -> pd.Series:
    """Classify each bar into a vol regime by quantile of rolling std.

    The first ``window - 1`` bars are NaN (insufficient history) — slicing
    skips them automatically. Quantile cuts are computed on the *full* series
    of rolling-std values, so the bucket boundaries are calibrated to the
    test window. (Using only the warm-up period would let us claim "no
    look-ahead" but would also produce arbitrary boundaries on short tests.)

    The resulting categorical Series shares ``returns.index``.
    """
    labels = list(labels)
    if len(labels) < 2:
        raise ValueError("need at least 2 regime labels")
    if returns.empty:
        return pd.Series([], index=returns.index, dtype="object", name="vol_regime")

    rolling_vol = returns.rolling(window, min_periods=window).std()
    n_buckets = len(labels)

    # qcut handles equal-frequency binning; duplicates="drop" guards against
    # constant rolling-std (e.g. synthetic flat returns) collapsing buckets.
    try:
        regime = pd.qcut(rolling_vol, q=n_buckets, labels=labels, duplicates="drop")
    except ValueError:
        # All values identical → assign everything to the middle bucket.
        regime = pd.Series(
            pd.Categorical([labels[len(labels) // 2]] * len(returns), categories=labels),
            index=returns.index,
        )
    out = pd.Series(regime, index=returns.index, name="vol_regime")
    return out


def detect_trend_regimes(
    prices: pd.Series,
    *,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """Bull/bear classification from a moving-average crossover on the asset.

    ``bull`` when fast SMA > slow SMA, ``bear`` otherwise. The first
    ``slow - 1`` bars are NaN.
    """
    if fast >= slow:
        raise ValueError("fast window must be strictly less than slow window")
    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()
    diff = fast_ma - slow_ma
    regime = pd.Series(np.where(diff > 0, "bull", "bear"), index=prices.index)
    # Mask warm-up period so callers can drop it cleanly.
    regime = regime.where(slow_ma.notna() & fast_ma.notna(), other=np.nan)
    regime.name = "trend_regime"
    return regime


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RegimeMetrics:
    label: str
    n_bars: int
    fraction_of_time: float
    strategy_total_return: float
    strategy_annualized_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    benchmark_total_return: float
    benchmark_annualized_return: float
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    exposure: float  # fraction of bars in regime where strategy held a position


@dataclass
class RegimeReport:
    symbol: str
    axis: Axis
    description: str
    metrics: list[RegimeMetrics] = field(default_factory=list)

    def by_label(self) -> dict[str, RegimeMetrics]:
        return {m.label: m for m in self.metrics}


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running = equity.cummax()
    return float((equity / running - 1.0).min())


def _ann_sharpe(returns: pd.Series, periods_per_year: int) -> tuple[float, float, float]:
    """Return (total, annualized, sharpe) on a return series."""
    if returns.empty:
        return float("nan"), float("nan"), float("nan")
    total = float((1 + returns).prod() - 1)
    mean, std = float(returns.mean()), float(returns.std(ddof=0))
    ann = (1 + mean) ** periods_per_year - 1 if not np.isnan(mean) else float("nan")
    sharpe = (mean / std) * np.sqrt(periods_per_year) if std and std > 0 else float("nan")
    return total, float(ann), float(sharpe)


def slice_by_regime(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    positions: pd.Series,
    regime: pd.Series,
    periods_per_year: int = 252,
) -> list[RegimeMetrics]:
    """Compute per-regime metrics from the backtest output.

    All inputs must share an index. Bars where ``regime`` is NaN (warm-up)
    are excluded from the totals — the ``fraction_of_time`` denominator is
    the number of *valid* regime bars, not the full series length, so the
    fractions across regimes always sum to 1.0.
    """
    # Align everything on the strategy index.
    idx = strategy_returns.index
    bench = benchmark_returns.reindex(idx).fillna(0.0)
    pos = positions.reindex(idx).fillna(0)
    reg = regime.reindex(idx)

    valid = reg.notna()
    total_valid = int(valid.sum())
    if total_valid == 0:
        return []

    # Stable label order: keep first-seen ordering, but if the regime is
    # categorical, honor its category order so "low → mid → high" stays sorted.
    if isinstance(reg.dtype, pd.CategoricalDtype):
        labels = [str(c) for c in reg.cat.categories if c in set(reg.dropna().unique())]
    else:
        labels = list(dict.fromkeys(reg.dropna().tolist()))

    out: list[RegimeMetrics] = []
    for label in labels:
        mask = (reg == label) & valid
        n = int(mask.sum())
        if n == 0:
            continue
        s_ret = strategy_returns[mask]
        b_ret = bench[mask]
        s_pos = pos[mask]

        s_total, s_ann, s_sharpe = _ann_sharpe(s_ret, periods_per_year)
        b_total, b_ann, b_sharpe = _ann_sharpe(b_ret, periods_per_year)
        s_dd = _max_drawdown((1 + s_ret).cumprod())
        b_dd = _max_drawdown((1 + b_ret).cumprod())

        out.append(
            RegimeMetrics(
                label=str(label),
                n_bars=n,
                fraction_of_time=n / total_valid,
                strategy_total_return=s_total,
                strategy_annualized_return=s_ann,
                strategy_sharpe=s_sharpe,
                strategy_max_drawdown=s_dd,
                benchmark_total_return=b_total,
                benchmark_annualized_return=b_ann,
                benchmark_sharpe=b_sharpe,
                benchmark_max_drawdown=b_dd,
                exposure=float((s_pos != 0).mean()),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Convenience: build both regime reports from a BacktestReport
# ---------------------------------------------------------------------------


def analyze_regimes(
    backtest_report,
    prices: pd.DataFrame,
    *,
    vol_window: int = 20,
    vol_labels: Iterable[str] = ("low", "mid", "high"),
    trend_fast: int = 50,
    trend_slow: int = 200,
    periods_per_year: int = 252,
) -> list[RegimeReport]:
    """Run vol + trend regime analysis on a completed backtest.

    Returns a list of two :class:`RegimeReport`s. Empty list if the backtest
    is degenerate (no OOS bars).
    """
    if backtest_report.strategy_returns.empty:
        log.warning("Backtest has no returns — skipping regime analysis")
        return []

    asset_returns = prices["close"].pct_change().fillna(0.0)

    reports: list[RegimeReport] = []

    vol = detect_vol_regimes(asset_returns, window=vol_window, labels=vol_labels)
    vol_metrics = slice_by_regime(
        strategy_returns=backtest_report.strategy_returns,
        benchmark_returns=backtest_report.benchmark_returns,
        positions=backtest_report.positions,
        regime=vol,
        periods_per_year=periods_per_year,
    )
    reports.append(
        RegimeReport(
            symbol=backtest_report.symbol,
            axis="volatility",
            description=f"Realized-vol terciles ({vol_window}-bar rolling std)",
            metrics=vol_metrics,
        )
    )

    trend = detect_trend_regimes(prices["close"], fast=trend_fast, slow=trend_slow)
    trend_metrics = slice_by_regime(
        strategy_returns=backtest_report.strategy_returns,
        benchmark_returns=backtest_report.benchmark_returns,
        positions=backtest_report.positions,
        regime=trend,
        periods_per_year=periods_per_year,
    )
    reports.append(
        RegimeReport(
            symbol=backtest_report.symbol,
            axis="trend",
            description=f"SMA{trend_fast} vs SMA{trend_slow} crossover (bull/bear)",
            metrics=trend_metrics,
        )
    )

    return reports


__all__ = [
    "Axis",
    "RegimeMetrics",
    "RegimeReport",
    "detect_vol_regimes",
    "detect_trend_regimes",
    "slice_by_regime",
    "analyze_regimes",
]
