"""Backtest engine.

Takes prices + out-of-sample probabilities, converts them into positions,
applies transaction costs, and reports an equity curve with risk-adjusted
metrics.

Design rules (the things that separate an honest backtest from a bad one):

1. **No look-ahead.** ``probability_t`` is only available at *close of day t*.
   The earliest it can influence performance is the return realized on day
   ``t+1``. We enforce this by shifting positions forward by one bar before
   multiplying with returns.

2. **Transaction costs.** A cost of ``|Δposition| * cost_bps / 1e4`` is
   deducted from the strategy return on every bar where the position
   changes — including the first entry and the final flatten.

3. **Benchmarks matter.** Every report includes buy-and-hold on the same
   asset. If the model doesn't beat buy-and-hold after costs, that is the
   result — we don't hide it.

4. **Probabilities may be NaN.** Before enough history exists for
   walk-forward training, probabilities are NaN. Those rows are treated as
   "no signal → position 0". No trades, no costs, no P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.backtest.sizing import realized_volatility, vol_target_scale
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int           # +1 (long) or -1 (short)
    entry_price: float
    exit_price: float
    holding_days: int
    gross_return: float      # (exit/entry - 1) * direction
    cost: float              # total cost paid at entry + exit (fraction, e.g. 0.0004 = 4 bps)
    net_return: float        # gross_return - cost


@dataclass
class BacktestReport:
    symbol: str
    long_threshold: float
    short_threshold: float
    cost_bps: float
    allow_short: bool

    # time series, all aligned to the price index
    probabilities: pd.Series
    positions: pd.Series
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    equity_curve: pd.Series
    benchmark_equity: pd.Series

    # summary metrics
    total_return: float
    benchmark_total_return: float
    annualized_return: float
    benchmark_annualized_return: float
    annualized_vol: float
    sharpe: float
    benchmark_sharpe: float
    max_drawdown: float
    benchmark_max_drawdown: float
    win_rate: float
    n_trades: int
    exposure: float          # fraction of bars with non-zero position
    turnover: float          # mean |Δposition| per bar

    trades: list[Trade] = field(default_factory=list)

    # diagnostics
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    n_oos_bars: int = 0

    # vol-targeting configuration + diagnostics. When vol targeting is OFF
    # (the default), target_vol_annual is None and the scale series is all
    # 1.0 — positions revert to classic {-1, 0, +1}.
    target_vol_annual: float | None = None
    vol_lookback: int = 20
    max_leverage: float = 1.0
    realized_vol: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    position_scale: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def _signal_from_probability(
    p: pd.Series,
    *,
    long_threshold: float,
    short_threshold: float,
    allow_short: bool,
) -> pd.Series:
    """Map P(up) → {-1, 0, +1} position signal. NaN probs → 0."""
    long_side = (p > long_threshold).astype(int)
    if allow_short:
        short_side = (p < short_threshold).astype(int)
        signal = long_side - short_side
    else:
        signal = long_side
    # NaN probabilities produce NaN in the comparison → astype(int) would fail;
    # we handle by explicit mask.
    signal = signal.where(p.notna(), 0).astype(int)
    return signal


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    dd = equity / running_peak - 1.0
    return float(dd.min())


def _annualized(mean_ret: float, std_ret: float, periods_per_year: int) -> tuple[float, float, float]:
    ann_return = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = std_ret * np.sqrt(periods_per_year)
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else float("nan")
    return float(ann_return), float(ann_vol), float(sharpe)


def _extract_trades(
    positions: pd.Series,
    prices: pd.Series,
    cost_per_turnover: float,
) -> list[Trade]:
    """Enumerate round-trip trades from a positions series.

    A trade is a contiguous interval where the *sign* (direction) of the
    position is constant and non-zero. It opens when sign goes 0 → ±1 or
    flips, and closes when sign returns to 0 or flips. Any final open
    position is closed on the last bar.

    With vol-targeted sizing the position *size* can drift within a trade;
    the reported ``direction`` field is ±1 and the entry-bar size shows up
    in the cost estimate. Aggregate P&L stays correct because the
    bar-level ``strategy_returns`` series is sized correctly throughout.
    """
    trades: list[Trade] = []
    if positions.empty:
        return trades

    pos = positions.to_numpy(dtype=float)
    idx = positions.index
    px = prices.reindex(idx).to_numpy()
    sign = np.sign(pos).astype(int)

    entry_i: int | None = None
    entry_sign: int = 0

    def _close(entry_i: int, exit_i: int, entry_sign: int) -> None:
        entry_px = float(px[entry_i])
        exit_px = float(px[exit_i])
        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px == 0:
            return
        gross = entry_sign * (exit_px / entry_px - 1.0)
        # Cost: one turnover at entry + one at exit. Under variable sizing
        # this is an approximation; the bar-level ``costs`` series uses
        # |Δposition| each bar and is authoritative.
        cost = 2 * cost_per_turnover * abs(float(pos[entry_i]))
        trades.append(
            Trade(
                entry_date=pd.Timestamp(idx[entry_i]),
                exit_date=pd.Timestamp(idx[exit_i]),
                direction=entry_sign,
                entry_price=entry_px,
                exit_price=exit_px,
                holding_days=int(exit_i - entry_i),
                gross_return=float(gross),
                cost=float(cost),
                net_return=float(gross - cost),
            )
        )

    for i in range(len(pos)):
        s = int(sign[i])
        if entry_i is None:
            if s != 0:
                entry_i = i
                entry_sign = s
        else:
            if s != entry_sign:
                _close(entry_i, i, entry_sign)
                entry_i = i if s != 0 else None
                entry_sign = s

    if entry_i is not None:
        _close(entry_i, len(pos) - 1, entry_sign)

    return trades


def backtest(
    prices: pd.DataFrame,
    probabilities: pd.Series,
    *,
    symbol: str = "?",
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    cost_bps: float = 2.0,
    allow_short: bool = False,
    periods_per_year: int = 252,
    target_vol_annual: float | None = None,
    vol_lookback: int = 20,
    max_leverage: float = 1.0,
) -> BacktestReport:
    """Run the backtest.

    Parameters
    ----------
    prices : DataFrame with a ``close`` column and a DatetimeIndex.
    probabilities : P(up) series. NaN is allowed — treated as "no signal".
    long_threshold / short_threshold : entry thresholds on P(up).
    cost_bps : round-trip cost in basis points charged per unit of |Δposition|.
               2.0 bps is a reasonable liquid-equity default. Crypto or small-
               caps are meaningfully higher.
    allow_short : if False, the strategy is long-flat only.
    periods_per_year : 252 for daily equities, 365 for crypto daily.
    target_vol_annual : if set (e.g. 0.10 for 10% annualized), each unit
        signal is scaled so ``size * realized_vol ≈ target_vol_annual``.
        Warmup bars (before ``vol_lookback`` returns exist) sit flat.
        When ``None`` (the default), positions are the classic
        fixed-size ``{-1, 0, +1}`` — unchanged behavior.
    vol_lookback : rolling window for the realized-vol estimate.
    max_leverage : cap on position size. 1.0 never exceeds fully invested;
        higher values allow levering up in low-vol regimes.
    """
    if "close" not in prices.columns:
        raise ValueError("prices must include a 'close' column")
    if not 0.0 < long_threshold < 1.0:
        raise ValueError("long_threshold must be in (0, 1)")
    if not 0.0 < short_threshold < 1.0:
        raise ValueError("short_threshold must be in (0, 1)")
    if short_threshold >= long_threshold:
        raise ValueError("short_threshold must be < long_threshold")

    # Align probabilities to the price index. Anything outside gets NaN.
    px = prices["close"].astype(float).sort_index()
    probs_aligned = probabilities.reindex(px.index)

    # --- 1. Signal → direction ({-1, 0, +1}) for the bar we'll hold NEXT.
    direction = _signal_from_probability(
        probs_aligned,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        allow_short=allow_short,
    )

    # --- 2. Realized vol + position scale.
    asset_return = px.pct_change().fillna(0.0)
    if target_vol_annual is None:
        # Fixed-size classic mode. Scale is a column of 1.0s so the rest of
        # the pipeline stays uniform; position stays integer-valued.
        realized_vol_series = pd.Series(np.nan, index=px.index)
        scale = pd.Series(1.0, index=px.index)
        target_pos = direction.astype(int)
    else:
        realized_vol_series = realized_volatility(
            asset_return, window=vol_lookback, periods_per_year=periods_per_year
        )
        scale = vol_target_scale(
            realized_vol_series,
            target_vol_annual=target_vol_annual,
            max_leverage=max_leverage,
        )
        # Warmup (NaN scale) → size 0 → flat. Direction times sized scale.
        target_pos = direction.astype(float) * scale.fillna(0.0)

    # --- 3. Shift by 1 bar. Position held on day t was decided using info from t-1.
    if target_vol_annual is None:
        held_position = target_pos.shift(1).fillna(0).astype(int)
    else:
        held_position = target_pos.shift(1).fillna(0.0).astype(float)

    # --- 4. Transaction costs.
    cost_per_turnover = cost_bps / 10_000.0
    turnover_bar = held_position.diff().abs().fillna(held_position.abs())
    costs = turnover_bar * cost_per_turnover

    # --- 5. Strategy returns = position * asset_return - costs.
    strat_gross = held_position * asset_return
    strat_net = strat_gross - costs

    # --- 6. Equity curves (start at 1.0 on the first bar).
    equity = (1.0 + strat_net).cumprod()
    bench_equity = (1.0 + asset_return).cumprod()

    # --- 7. Metrics.
    n = len(strat_net)
    oos_mask = probs_aligned.notna()
    n_oos = int(oos_mask.sum())

    mean_ret = float(strat_net.mean())
    std_ret = float(strat_net.std(ddof=0))
    ann_return, ann_vol, sharpe = _annualized(mean_ret, std_ret, periods_per_year)

    b_mean = float(asset_return.mean())
    b_std = float(asset_return.std(ddof=0))
    b_ann_return, _, b_sharpe = _annualized(b_mean, b_std, periods_per_year)

    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    b_total_return = float(bench_equity.iloc[-1] - 1.0) if len(bench_equity) else 0.0

    max_dd = _max_drawdown(equity)
    b_max_dd = _max_drawdown(bench_equity)

    exposure = float((held_position != 0).mean())
    turnover = float(turnover_bar.mean())

    trades = _extract_trades(held_position, px, cost_per_turnover)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.net_return > 0)
    win_rate = wins / n_trades if n_trades else float("nan")

    report = BacktestReport(
        symbol=symbol,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        cost_bps=cost_bps,
        allow_short=allow_short,
        probabilities=probs_aligned,
        positions=held_position,
        strategy_returns=strat_net,
        benchmark_returns=asset_return,
        equity_curve=equity,
        benchmark_equity=bench_equity,
        total_return=total_return,
        benchmark_total_return=b_total_return,
        annualized_return=ann_return,
        benchmark_annualized_return=b_ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        benchmark_sharpe=b_sharpe,
        max_drawdown=max_dd,
        benchmark_max_drawdown=b_max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
        exposure=exposure,
        turnover=turnover,
        trades=trades,
        start_date=pd.Timestamp(px.index[0]) if n else None,
        end_date=pd.Timestamp(px.index[-1]) if n else None,
        n_oos_bars=n_oos,
        target_vol_annual=target_vol_annual,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
        realized_vol=realized_vol_series,
        position_scale=scale,
    )

    log.info(
        "backtest %s: total=%.3f bench=%.3f sharpe=%.2f bench_sharpe=%.2f "
        "dd=%.3f trades=%d exposure=%.2f turnover=%.4f",
        symbol,
        total_return,
        b_total_return,
        sharpe,
        b_sharpe,
        max_dd,
        n_trades,
        exposure,
        turnover,
    )
    return report
