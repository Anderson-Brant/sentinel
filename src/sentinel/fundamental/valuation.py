"""Valuation ratios + percentile vs the stock's own trading history.

Split into a thin yfinance adapter (`fetch_snapshot`, network) and a pure
scoring function (`compute_valuation`) so the scoring logic is testable
offline.

Grading is own-history-relative for now: a stock trading at the 95th
percentile of its own 5-year P/E range grades worse than one at the 20th.
Sector-relative grading lands with the quality factor in v0.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from sentinel.fundamental.grades import grade_points, notch
from sentinel.utils.logging import get_logger

log = get_logger(__name__)

_PEG_CHEAP = 1.0
_PEG_RICH = 3.0

# Own-history percentile -> grade. Calibrated so the middle of a stock's own
# valuation range reads as a middle grade, not a failing one: 50th pct = B-,
# only the most expensive decile of its own history grades D/F.
_PCTILE_CUTS: tuple[tuple[float, str], ...] = (
    (5, "A+"), (15, "A"), (25, "A-"),
    (35, "B+"), (45, "B"), (55, "B-"),
    (65, "C+"), (75, "C"), (85, "C-"),
    (92, "D+"), (97, "D"),
)


def _pctile_grade(pct: float) -> str:
    for cut, letter in _PCTILE_CUTS:
        if pct <= cut:
            return letter
    return "F"


def _ordinal(n: float) -> str:
    i = int(round(n))
    suffix = "th" if 10 <= i % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


@dataclass
class FundamentalsSnapshot:
    """Point-in-time inputs for valuation. Every field may be missing -
    yfinance coverage is uneven and scoring degrades gracefully."""

    symbol: str
    as_of: date
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    trailing_eps: float | None = None
    revenue_ttm: float | None = None
    fcf_ttm: float | None = None
    ebitda_ttm: float | None = None
    net_debt: float | None = None
    dividend_yield: float | None = None
    earnings_growth: float | None = None
    # Annual per-share histories (fiscal year end -> value) for own-history percentiles.
    eps_history: pd.Series | None = None
    revenue_ps_history: pd.Series | None = None
    fcf_ps_history: pd.Series | None = None
    # Monthly closes over the percentile window.
    monthly_close: pd.Series | None = None


@dataclass
class ValuationScore:
    symbol: str
    as_of: date
    pe: float | None = None
    ps: float | None = None
    pfcf: float | None = None
    ev_ebitda: float | None = None
    peg: float | None = None
    dividend_yield: float | None = None
    pe_pctile: float | None = None
    ps_pctile: float | None = None
    pfcf_pctile: float | None = None
    grade: str | None = None
    score: float | None = None
    summary: str = ""

    percentiles: dict[str, float] = field(default_factory=dict)


def _ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom <= 0:
        return None
    return numer / denom


def _history_percentile(
    monthly_close: pd.Series | None,
    per_share_history: pd.Series | None,
    current_ratio: float | None,
) -> float | None:
    """Rank the current price/metric ratio within its own monthly history.

    The annual per-share metric is forward-filled onto the monthly price
    index, i.e. treated as a step function that updates at each fiscal year
    end. Returns a 0-100 percentile; 100 = most expensive it has ever been
    within the window.
    """
    if (
        monthly_close is None
        or per_share_history is None
        or current_ratio is None
        or monthly_close.empty
        or per_share_history.empty
    ):
        return None
    metric = per_share_history[per_share_history > 0].sort_index()
    if metric.empty:
        return None
    px = monthly_close.dropna().sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    metric.index = pd.to_datetime(metric.index).tz_localize(None)
    aligned = metric.reindex(px.index, method="ffill").dropna()
    if len(aligned) < 12:
        return None
    ratios = (px.loc[aligned.index] / aligned).dropna()
    if len(ratios) < 12:
        return None
    return float((ratios < current_ratio).mean() * 100)


def compute_valuation(snap: FundamentalsSnapshot) -> ValuationScore:
    """Compute the Valuation scorecard row. Pure - no network."""
    pe = _ratio(snap.price, snap.trailing_eps)
    ps = _ratio(snap.market_cap, snap.revenue_ttm)
    pfcf = _ratio(snap.market_cap, snap.fcf_ttm)

    ev_ebitda = None
    if snap.market_cap is not None and snap.ebitda_ttm is not None and snap.ebitda_ttm > 0:
        ev = snap.market_cap + (snap.net_debt or 0.0)
        ev_ebitda = ev / snap.ebitda_ttm

    peg = None
    if pe is not None and snap.earnings_growth is not None and snap.earnings_growth > 0:
        peg = pe / (snap.earnings_growth * 100)

    price_ps = _ratio(snap.market_cap, snap.shares_outstanding)
    pe_pct = _history_percentile(snap.monthly_close, snap.eps_history, pe)
    ps_pct = _history_percentile(
        snap.monthly_close,
        snap.revenue_ps_history,
        _ratio(price_ps, _ratio(snap.revenue_ttm, snap.shares_outstanding)),
    )
    pfcf_pct = _history_percentile(
        snap.monthly_close,
        snap.fcf_ps_history,
        _ratio(price_ps, _ratio(snap.fcf_ttm, snap.shares_outstanding)),
    )

    percentiles = {
        k: v
        for k, v in {"pe": pe_pct, "ps": ps_pct, "pfcf": pfcf_pct}.items()
        if v is not None
    }

    grade: str | None = None
    score: float | None = None
    if percentiles:
        mean_pct = sum(percentiles.values()) / len(percentiles)
        grade = _pctile_grade(mean_pct)
        if peg is not None and peg < _PEG_CHEAP:
            grade = notch(grade, +1)
        elif peg is not None and peg > _PEG_RICH:
            grade = notch(grade, -1)
        score = grade_points(grade)

    parts = []
    if pe is not None:
        txt = f"P/E {pe:.0f}"
        if pe_pct is not None:
            txt += f" ({_ordinal(pe_pct)} pct vs own history)"
        parts.append(txt)
    elif ps is not None:
        txt = f"P/S {ps:.1f}"
        if ps_pct is not None:
            txt += f" ({_ordinal(ps_pct)} pct vs own history)"
        parts.append(txt)
    if peg is not None:
        parts.append(f"PEG {peg:.1f}")
    if not parts:
        parts.append("insufficient fundamentals data")

    return ValuationScore(
        symbol=snap.symbol,
        as_of=snap.as_of,
        pe=pe,
        ps=ps,
        pfcf=pfcf,
        ev_ebitda=ev_ebitda,
        peg=peg,
        dividend_yield=snap.dividend_yield,
        pe_pctile=pe_pct,
        ps_pctile=ps_pct,
        pfcf_pctile=pfcf_pct,
        grade=grade,
        score=score,
        summary=". ".join(parts) + ".",
        percentiles=percentiles,
    )


def _get(info: dict, key: str) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # drop NaN


def _per_share_row(stmt: pd.DataFrame | None, label: str, shares: float | None) -> pd.Series | None:
    if stmt is None or stmt.empty or label not in stmt.index:
        return None
    row = stmt.loc[label].dropna()
    if row.empty:
        return None
    if label == "Basic EPS":
        return row.astype(float)
    if not shares:
        return None
    return row.astype(float) / shares


def fetch_snapshot(symbol: str, *, percentile_years: int = 5) -> FundamentalsSnapshot:
    """Pull current fundamentals + annual history for `symbol` from yfinance.

    Every field is best-effort: a missing statement or info key becomes None
    and the scorer works with whatever survived.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    snap = FundamentalsSnapshot(symbol=symbol.upper(), as_of=date.today())

    try:
        info = ticker.info or {}
    except Exception as exc:
        log.warning("yfinance info failed for %s: %s", symbol, exc)
        info = {}

    snap.company_name = info.get("longName") or info.get("shortName")
    snap.sector = info.get("sector")
    snap.industry = info.get("industry")
    snap.price = _get(info, "currentPrice") or _get(info, "regularMarketPrice")
    snap.market_cap = _get(info, "marketCap")
    snap.shares_outstanding = _get(info, "sharesOutstanding")
    snap.trailing_eps = _get(info, "trailingEps")
    snap.revenue_ttm = _get(info, "totalRevenue")
    snap.fcf_ttm = _get(info, "freeCashflow")
    snap.ebitda_ttm = _get(info, "ebitda")
    dy = _get(info, "dividendYield")
    if dy is not None and dy > 1:
        # yfinance switched dividendYield from fraction to percent around 0.2.55.
        dy /= 100
    snap.dividend_yield = dy
    snap.earnings_growth = _get(info, "earningsGrowth")

    total_debt = _get(info, "totalDebt")
    cash = _get(info, "totalCash")
    if total_debt is not None:
        snap.net_debt = total_debt - (cash or 0.0)

    try:
        income = ticker.income_stmt
    except Exception:
        income = None
    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = None

    snap.eps_history = _per_share_row(income, "Basic EPS", snap.shares_outstanding)
    snap.revenue_ps_history = _per_share_row(income, "Total Revenue", snap.shares_outstanding)
    snap.fcf_ps_history = _per_share_row(cashflow, "Free Cash Flow", snap.shares_outstanding)

    try:
        hist = ticker.history(period=f"{percentile_years}y", interval="1mo")
        if hist is not None and not hist.empty:
            snap.monthly_close = hist["Close"].dropna()
    except Exception as exc:
        log.warning("yfinance monthly history failed for %s: %s", symbol, exc)

    return snap
