"""Market data ingestion via yfinance.

Keeps the adapter thin — the caller is responsible for persistence. This makes
the function easy to unit-test with fixture data.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# Canonical column names used throughout Sentinel.
_EXPECTED = ["open", "high", "low", "close", "adj_close", "volume"]


def ingest_prices(
    symbol: str,
    *,
    start: str | date,
    end: str | date | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for ``symbol`` and return a tidy DataFrame.

    Output columns:
        date (index), symbol, open, high, low, close, adj_close, volume
    """
    if not symbol:
        raise ValueError("symbol is required")

    log.info("Fetching %s from yfinance (start=%s, end=%s, interval=%s)", symbol, start, end, interval)

    raw = yf.download(
        tickers=symbol,
        start=str(start) if start else None,
        end=str(end) if end else None,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {symbol!r}. Check the ticker and date range."
        )

    # yfinance returns a MultiIndex when downloading a single ticker in newer versions.
    # Collapse to single-level columns.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # If adj_close is missing (e.g. crypto via yfinance), fall back to close.
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # Ensure every expected column exists.
    missing = [c for c in _EXPECTED if c not in df.columns]
    if missing:
        raise RuntimeError(f"yfinance response missing columns: {missing}")

    df = df[_EXPECTED].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.dropna(subset=["close"]).sort_index()

    df.insert(0, "symbol", symbol.upper())

    log.info("Retrieved %d rows for %s (%s → %s)", len(df), symbol, df.index.min().date(), df.index.max().date())
    return df
