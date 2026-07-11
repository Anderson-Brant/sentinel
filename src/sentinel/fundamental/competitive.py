"""Competitive position: growth and margins vs sector, related tickers.

Until per-peer ingestion exists, the comparison baseline is a static table of
sector medians (large-cap US, rough figures compiled from public aggregates;
see docs/analyze.md for the caveats). Related tickers come from return
correlation against whatever else is in the local store, so the list gets
better as more symbols are ingested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sentinel.fundamental.grades import grade_points, letter_grade
from sentinel.fundamental.valuation import FundamentalsSnapshot

# Sector medians: (revenue growth per year, operating margin).
_SECTOR_BASELINES: dict[str, tuple[float, float]] = {
    "Technology": (0.10, 0.22),
    "Communication Services": (0.07, 0.18),
    "Consumer Cyclical": (0.06, 0.10),
    "Consumer Defensive": (0.04, 0.08),
    "Healthcare": (0.07, 0.12),
    "Financial Services": (0.05, 0.25),
    "Industrials": (0.05, 0.11),
    "Energy": (0.03, 0.12),
    "Utilities": (0.03, 0.14),
    "Real Estate": (0.04, 0.20),
    "Basic Materials": (0.03, 0.12),
}
_DEFAULT_BASELINE = (0.05, 0.12)


@dataclass
class CompetitiveScore:
    symbol: str
    sector: str | None = None
    revenue_growth: float | None = None
    sector_growth: float | None = None
    operating_margin: float | None = None
    sector_margin: float | None = None
    related: list[str] = field(default_factory=list)
    grade: str | None = None
    score: float | None = None
    summary: str = ""


def compute_competitive(snap: FundamentalsSnapshot) -> CompetitiveScore:
    """Compute the Competitive scorecard row. Pure - no network."""
    c = CompetitiveScore(symbol=snap.symbol, sector=snap.sector)

    rev = snap.revenue_history.dropna() if snap.revenue_history is not None else None
    if rev is not None and len(rev) >= 3 and float(rev.iloc[0]) > 0:
        n_years = len(rev) - 1
        c.revenue_growth = (float(rev.iloc[-1]) / float(rev.iloc[0])) ** (1 / n_years) - 1

    oi = snap.operating_income_history
    if oi is not None and rev is not None:
        aligned = pd.concat([oi, rev], axis=1, keys=["oi", "rev"]).dropna()
        aligned = aligned[aligned["rev"] > 0]
        if not aligned.empty:
            c.operating_margin = float((aligned["oi"] / aligned["rev"]).iloc[-1])

    baseline = _SECTOR_BASELINES.get(snap.sector or "", _DEFAULT_BASELINE)
    c.sector_growth, c.sector_margin = baseline

    subscores = []
    if c.revenue_growth is not None:
        delta = c.revenue_growth - c.sector_growth
        subscores.append(_delta_score(delta, scale=0.05))
    if c.operating_margin is not None:
        delta = c.operating_margin - c.sector_margin
        subscores.append(_delta_score(delta, scale=0.06))

    if not subscores:
        c.summary = "insufficient fundamentals data"
        return c

    c.score = sum(subscores) / len(subscores)
    c.grade = letter_grade(c.score)
    c.score = grade_points(c.grade)

    parts = []
    if c.revenue_growth is not None:
        parts.append(
            f"Revenue {c.revenue_growth * 100:+.0f}%/yr vs sector {c.sector_growth * 100:+.0f}%"
        )
    if c.operating_margin is not None:
        parts.append(
            f"Op margin {c.operating_margin * 100:.0f}% vs sector {c.sector_margin * 100:.0f}%"
        )
    c.summary = ". ".join(parts) + "."
    return c


def _delta_score(delta: float, *, scale: float) -> float:
    """Map an above/below-sector delta to a score. `scale` is one grade band."""
    banded = 75.0 + (delta / scale) * 7.5
    return max(40.0, min(98.0, banded))


def related_by_correlation(
    prices_by_symbol: dict[str, pd.DataFrame],
    symbol: str,
    *,
    top_n: int = 5,
    min_overlap: int = 120,
) -> list[str]:
    """Rank other stored symbols by daily-return correlation with `symbol`."""
    symbol = symbol.upper()
    if symbol not in prices_by_symbol:
        return []
    base = prices_by_symbol[symbol]["close"].pct_change().dropna()
    scores: list[tuple[float, str]] = []
    for other, df in prices_by_symbol.items():
        if other.upper() == symbol:
            continue
        rets = df["close"].pct_change().dropna()
        joined = pd.concat([base, rets], axis=1, join="inner").dropna()
        if len(joined) < min_overlap:
            continue
        corr = float(joined.corr().iloc[0, 1])
        scores.append((corr, other.upper()))
    scores.sort(reverse=True)
    return [s for _, s in scores[:top_n]]
