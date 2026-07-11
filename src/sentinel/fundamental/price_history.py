"""Long-horizon price statistics: CAGR, drawdown, recovery, Sharpe.

Pure functions over a stored prices DataFrame (same shape the rest of
Sentinel uses: date index, `close` / `adj_close` columns). No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.fundamental.grades import grade_points, notch

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)

# Absolute CAGR thresholds - long-run price performance is graded against
# universal benchmarks, not sector-relative (see docs/analyze.md).
_CAGR_CUTS: tuple[tuple[float, str], ...] = (
    (0.20, "A+"), (0.15, "A"), (0.12, "A-"),
    (0.10, "B+"), (0.08, "B"), (0.06, "B-"),
    (0.04, "C+"), (0.02, "C"), (0.00, "C-"),
    (-0.03, "D+"), (-0.08, "D"),
)

_DEEP_DRAWDOWN = -0.60


@dataclass
class PriceHistoryScore:
    symbol: str
    years: float
    cagr: dict[int, float] = field(default_factory=dict)
    max_drawdown: float = float("nan")
    drawdown_recovery_days: int | None = None
    sharpe: float = float("nan")
    grade: str | None = None
    score: float | None = None
    summary: str = ""


def _cagr_grade(cagr: float) -> str:
    for cut, letter in _CAGR_CUTS:
        if cagr >= cut:
            return letter
    return "F"


def _max_drawdown(px: pd.Series) -> tuple[float, int | None]:
    peak = px.cummax()
    dd = px / peak - 1.0
    max_dd = float(dd.min())
    if max_dd >= 0:
        return 0.0, None
    trough_date = dd.idxmin()
    peak_level = float(peak.loc[trough_date])
    after = px.loc[trough_date:]
    recovered = after[after >= peak_level]
    if recovered.empty:
        return max_dd, None
    return max_dd, int((recovered.index[0] - trough_date).days)


def long_term_stats(
    prices: pd.DataFrame,
    *,
    symbol: str = "",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    periods_per_year: int = 252,
) -> PriceHistoryScore:
    """Compute the Price History scorecard row from a stored prices frame.

    Uses `adj_close` when present so splits/dividends don't distort CAGR.
    CAGR is only reported for horizons fully covered by the data.
    """
    col = "adj_close" if "adj_close" in prices.columns else "close"
    px = prices[col].dropna().sort_index()
    px.index = pd.to_datetime(px.index)
    if len(px) < 2:
        return PriceHistoryScore(symbol=symbol, years=0.0, summary="insufficient price history")

    end_date = px.index[-1]
    years = (end_date - px.index[0]).days / 365.25

    cagr: dict[int, float] = {}
    for h in horizons:
        if years < h:
            continue
        start_date = end_date - pd.DateOffset(years=h)
        window = px.loc[px.index >= start_date]
        start_px = float(window.iloc[0])
        if start_px <= 0:
            continue
        cagr[h] = (float(px.iloc[-1]) / start_px) ** (1.0 / h) - 1.0

    max_dd, recovery_days = _max_drawdown(px)

    rets = px.pct_change().dropna()
    sharpe = float("nan")
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year))

    if not cagr:
        return PriceHistoryScore(
            symbol=symbol,
            years=years,
            max_drawdown=max_dd,
            drawdown_recovery_days=recovery_days,
            sharpe=sharpe,
            summary="insufficient price history",
        )

    # Grade on the longest available horizon; penalize deep drawdowns a notch.
    longest = max(cagr)
    grade = _cagr_grade(cagr[longest])
    if max_dd <= _DEEP_DRAWDOWN:
        grade = notch(grade, -1)

    parts = [f"{longest}y CAGR {cagr[longest] * 100:.0f}%"]
    if max_dd < 0:
        dd_txt = f"Max DD {abs(max_dd) * 100:.0f}%"
        if recovery_days is not None:
            dd_txt += f" (recovered in {recovery_days / 30.44:.0f}mo)"
        else:
            dd_txt += " (not yet recovered)"
        parts.append(dd_txt)
    if longest < 10:
        parts.append(f"only {years:.1f}y of history")

    return PriceHistoryScore(
        symbol=symbol,
        years=years,
        cagr=cagr,
        max_drawdown=max_dd,
        drawdown_recovery_days=recovery_days,
        sharpe=sharpe,
        grade=grade,
        score=grade_points(grade),
        summary=". ".join(parts) + ".",
    )
