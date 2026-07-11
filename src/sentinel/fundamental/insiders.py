"""Insider transaction scoring: net buying/selling as % of shares outstanding.

The adapter pulls Yahoo's insider transaction feed via yfinance (sourced from
SEC Form 4 filings). Direct EDGAR ingestion with a durable table is planned;
this gets the row live without it. Scoring is pure and offline-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from sentinel.fundamental.grades import grade_points
from sentinel.utils.logging import get_logger

log = get_logger(__name__)

_BUY_WORDS = ("buy", "purchase", "acquisition")
_SELL_WORDS = ("sale", "sell", "sold")


@dataclass
class InsiderScore:
    symbol: str
    net_pct_6m: float | None = None
    net_pct_12m: float | None = None
    n_buys_6m: int = 0
    n_sells_6m: int = 0
    grade: str | None = None
    score: float | None = None
    summary: str = ""


def _classify(text: str) -> int:
    t = text.lower()
    if any(w in t for w in _BUY_WORDS):
        return 1
    if any(w in t for w in _SELL_WORDS):
        return -1
    return 0


def _net_shares(txns: pd.DataFrame, since: date) -> tuple[float, int, int]:
    window = txns[txns["date"] >= pd.Timestamp(since)]
    net = 0.0
    buys = sells = 0
    for _, row in window.iterrows():
        direction = _classify(str(row["text"]))
        if direction == 0:
            continue
        shares = float(row["shares"]) if pd.notna(row["shares"]) else 0.0
        net += direction * shares
        if direction > 0:
            buys += 1
        else:
            sells += 1
    return net, buys, sells


def compute_insiders(
    txns: pd.DataFrame | None,
    *,
    symbol: str,
    shares_outstanding: float | None,
    as_of: date,
) -> InsiderScore:
    """Compute the Insiders scorecard row from a transactions frame.

    Expects columns: date (datetime), shares (float), text (str, the
    transaction description). Returns an ungraded score when there are no
    classifiable transactions in the window.
    """
    result = InsiderScore(symbol=symbol)
    if (
        txns is None
        or txns.empty
        or shares_outstanding is None
        or shares_outstanding <= 0
        or not {"date", "shares", "text"}.issubset(txns.columns)
    ):
        result.summary = "no insider filings available"
        return result

    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
    if txns["date"].dt.tz is not None:
        txns["date"] = txns["date"].dt.tz_localize(None)
    txns = txns.dropna(subset=["date"])

    net_6m, buys, sells = _net_shares(txns, as_of - timedelta(days=182))
    net_12m, _, _ = _net_shares(txns, as_of - timedelta(days=365))
    result.n_buys_6m = buys
    result.n_sells_6m = sells

    if buys + sells == 0:
        result.summary = "no classifiable insider activity in the last 6mo"
        return result

    result.net_pct_6m = net_6m / shares_outstanding
    result.net_pct_12m = net_12m / shares_outstanding

    pct = result.net_pct_6m * 100
    if pct >= 0.5:
        result.grade, label = "A", "strong accumulation"
    elif pct >= 0.1:
        result.grade, label = "B+", "accumulation"
    elif pct > -0.1:
        result.grade, label = "B-", "neutral"
    elif pct > -0.5:
        result.grade, label = "C", "mild distribution"
    elif pct > -1.5:
        result.grade, label = "D+", "distribution"
    else:
        result.grade, label = "D", "heavy distribution"
    result.score = grade_points(result.grade)

    direction = "buying" if result.net_pct_6m >= 0 else "selling"
    result.summary = (
        f"Net {direction} {abs(pct):.1f}% of shares over 6mo ({label}). "
        f"{buys} buys / {sells} sells."
    )
    return result


def fetch_insider_transactions(symbol: str) -> pd.DataFrame | None:
    """Pull insider transactions for `symbol` from yfinance. Best-effort."""
    import yfinance as yf

    try:
        raw = yf.Ticker(symbol).insider_transactions
    except Exception as exc:
        log.warning("yfinance insider transactions failed for %s: %s", symbol, exc)
        return None
    if raw is None or raw.empty:
        return None

    cols = {c.lower(): c for c in raw.columns}
    date_col = cols.get("start date") or cols.get("date")
    shares_col = cols.get("shares")
    text_col = cols.get("text") or cols.get("transaction")
    if date_col is None or shares_col is None or text_col is None:
        log.warning("unexpected insider transaction columns for %s: %s", symbol, list(raw.columns))
        return None

    return pd.DataFrame(
        {
            "date": raw[date_col],
            "shares": pd.to_numeric(raw[shares_col], errors="coerce"),
            "text": raw[text_col].astype(str),
        }
    )
