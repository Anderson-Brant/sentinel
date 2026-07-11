"""Business quality: margins, returns on capital, leverage, growth.

Pure functions over a FundamentalsSnapshot. Metrics are scored against
absolute bands (documented in docs/analyze.md); sector-relative calibration
is planned once peer data lands with the competitive row's ingestion work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.fundamental.grades import grade_points, letter_grade
from sentinel.fundamental.valuation import FundamentalsSnapshot

_TAX_RATE = 0.21


@dataclass
class QualityScore:
    symbol: str
    roic: float | None = None
    gross_margin: float | None = None
    gross_margin_stability: float | None = None
    operating_margin: float | None = None
    operating_margin_trend: float | None = None
    net_debt_ebitda: float | None = None
    revenue_growth: float | None = None
    growth_stability: float | None = None
    grade: str | None = None
    score: float | None = None
    summary: str = ""


def _band(value: float, cuts: tuple[tuple[float, float], ...], floor: float) -> float:
    """Score `value` against descending (threshold, score) cuts."""
    for threshold, score in cuts:
        if value >= threshold:
            return score
    return floor


_ROIC_CUTS = ((0.30, 98.0), (0.20, 93.0), (0.15, 87.0), (0.10, 80.0), (0.07, 73.0), (0.04, 65.0), (0.0, 55.0))
_GM_CUTS = ((0.60, 95.0), (0.45, 88.0), (0.35, 80.0), (0.25, 72.0), (0.15, 63.0))
_GROWTH_CUTS = ((0.20, 95.0), (0.12, 88.0), (0.07, 78.0), (0.03, 68.0), (0.0, 58.0))


def _margin_series(numer: pd.Series | None, denom: pd.Series | None) -> pd.Series | None:
    if numer is None or denom is None:
        return None
    aligned = pd.concat([numer, denom], axis=1, keys=["n", "d"]).dropna()
    aligned = aligned[aligned["d"] > 0]
    if aligned.empty:
        return None
    return aligned["n"] / aligned["d"]


def _trend_per_year(series: pd.Series) -> float:
    """Least-squares slope in units per year."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=float)
    slope = np.polyfit(x, series.to_numpy(dtype=float), 1)[0]
    return float(slope)


def compute_quality(snap: FundamentalsSnapshot) -> QualityScore:
    """Compute the Quality scorecard row. Pure - no network."""
    q = QualityScore(symbol=snap.symbol)
    subscores: dict[str, float] = {}

    gm = _margin_series(snap.gross_profit_history, snap.revenue_history)
    if gm is not None and len(gm) >= 2:
        q.gross_margin = float(gm.iloc[-1])
        q.gross_margin_stability = float(gm.std())
        subscores["gm_level"] = _band(q.gross_margin, _GM_CUTS, 50.0)
        stability = q.gross_margin_stability
        subscores["gm_stability"] = _band(
            -stability, ((-0.01, 95.0), (-0.02, 88.0), (-0.04, 78.0), (-0.06, 68.0)), 55.0
        )

    om = _margin_series(snap.operating_income_history, snap.revenue_history)
    if om is not None and len(om) >= 2:
        q.operating_margin = float(om.iloc[-1])
        q.operating_margin_trend = _trend_per_year(om)
        if q.operating_margin_trend >= 0.01:
            subscores["om_trend"] = 90.0
        elif q.operating_margin_trend >= -0.01:
            subscores["om_trend"] = 75.0
        else:
            subscores["om_trend"] = 55.0

    roic = _roic(snap)
    if roic is not None:
        q.roic = roic
        subscores["roic"] = _band(roic, _ROIC_CUTS, 40.0)

    if snap.net_debt is not None and snap.ebitda_ttm is not None and snap.ebitda_ttm > 0:
        q.net_debt_ebitda = snap.net_debt / snap.ebitda_ttm
        subscores["leverage"] = _band(
            -q.net_debt_ebitda,
            ((0.0, 95.0), (-1.0, 88.0), (-2.0, 78.0), (-3.0, 65.0), (-4.0, 55.0)),
            40.0,
        )

    rev = snap.revenue_history.dropna() if snap.revenue_history is not None else None
    if rev is not None and len(rev) >= 3 and float(rev.iloc[0]) > 0:
        n_years = len(rev) - 1
        q.revenue_growth = (float(rev.iloc[-1]) / float(rev.iloc[0])) ** (1 / n_years) - 1
        subscores["growth"] = _band(q.revenue_growth, _GROWTH_CUTS, 45.0)
        yoy = rev.pct_change().dropna()
        if len(yoy) >= 2:
            q.growth_stability = float(yoy.std())
            subscores["growth_stability"] = _band(
                -q.growth_stability, ((-0.03, 92.0), (-0.07, 82.0), (-0.12, 70.0)), 55.0
            )

    if not subscores:
        q.summary = "insufficient fundamentals data"
        return q

    weights = {
        "roic": 0.25,
        "gm_level": 0.15,
        "gm_stability": 0.10,
        "om_trend": 0.10,
        "leverage": 0.15,
        "growth": 0.15,
        "growth_stability": 0.10,
    }
    total_w = sum(weights[k] for k in subscores)
    q.score = sum(subscores[k] * weights[k] for k in subscores) / total_w
    q.grade = letter_grade(q.score)
    q.score = grade_points(q.grade)

    parts = []
    if q.roic is not None:
        parts.append(f"ROIC {q.roic * 100:.0f}%")
    if q.gross_margin is not None:
        gm_txt = f"Gross margin {q.gross_margin * 100:.0f}%"
        if q.gross_margin_stability is not None:
            gm_txt += ", stable" if q.gross_margin_stability <= 0.02 else ", variable"
        parts.append(gm_txt)
    if q.net_debt_ebitda is not None:
        if q.net_debt_ebitda <= 0:
            parts.append("Net cash")
        else:
            parts.append(f"Net debt {q.net_debt_ebitda:.1f}x EBITDA")
    if q.revenue_growth is not None:
        parts.append(f"Revenue {q.revenue_growth * 100:+.0f}%/yr")
    q.summary = ". ".join(parts) + "."
    return q


def _roic(snap: FundamentalsSnapshot) -> float | None:
    """Operating income after tax over invested capital, latest fiscal year."""
    if snap.operating_income_history is None or snap.equity_history is None:
        return None
    oi = snap.operating_income_history.dropna()
    eq = snap.equity_history.dropna()
    if oi.empty or eq.empty:
        return None
    equity = float(eq.iloc[-1])
    debt = float(snap.debt_history.dropna().iloc[-1]) if snap.debt_history is not None and not snap.debt_history.dropna().empty else 0.0
    cash = float(snap.cash_history.dropna().iloc[-1]) if snap.cash_history is not None and not snap.cash_history.dropna().empty else 0.0
    invested = equity + debt - cash
    if invested <= 0:
        return None
    return float(oi.iloc[-1]) * (1 - _TAX_RATE) / invested
