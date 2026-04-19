"""Crypto OHLCV ingestion via CCXT.

Pulls daily bars from a configurable exchange (Binance by default) and returns
a DataFrame shaped *exactly* like the yfinance equity adapter — columns
``symbol, open, high, low, close, adj_close, volume`` indexed by ``date`` —
so downstream feature / model / backtest code is asset-class agnostic.

External symbol convention mirrors yfinance (``BTC-USD``, ``ETH-USD``). The
adapter maps that to the exchange's unified symbol (``BTC/USDT`` on Binance)
via a configurable quote currency. Storage normalizes USDT/USDC back to USD
so there is one canonical key per asset regardless of quote venue.

The adapter is thin: it fetches, normalizes, and hands the DataFrame back to
the caller (matching ``sentinel.ingestion.market.ingest_prices``). Persistence
happens at the call site, which makes the function trivially unit-testable
with fixture data — no CCXT install required in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Protocol

import pandas as pd

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# Exchange-side quote currencies that map to raw "USD" for storage.
_USD_EQUIVALENT_QUOTES = frozenset({"USDT", "USDC", "BUSD", "DAI", "TUSD"})

# Canonical output columns — must match the yfinance adapter so that a single
# `prices` table can hold both equity and crypto bars.
_PRICE_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]


class ExchangeClient(Protocol):
    """Minimal CCXT surface we depend on; lets tests inject fakes."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None,
        limit: int | None,
    ) -> list[list[Any]]: ...


@dataclass(frozen=True)
class SymbolMap:
    """How external ``BTC-USD`` maps to exchange-native ``BTC/USDT``.

    ``ccxt_symbol`` is what we send to the exchange; ``external_symbol`` is
    what we store, matching yfinance-style tickers.
    """

    base: str
    quote: str

    @property
    def ccxt_symbol(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def external_symbol(self) -> str:
        # USDT/USDC/BUSD → USD for storage. Keeps BTC-USD as the canonical key
        # regardless of which dollar-pegged stablecoin the exchange uses.
        quote = "USD" if self.quote.upper() in _USD_EQUIVALENT_QUOTES else self.quote
        return f"{self.base}-{quote}".upper()


def parse_symbol(symbol: str, *, default_quote: str = "USDT") -> SymbolMap:
    """Parse various input forms into a :class:`SymbolMap`.

    Accepts ``BTC-USD``, ``btc/usdt``, ``ETH:USDT``, or bare ``BTC`` (which
    picks up ``default_quote``). A bare ``-USD`` quote is routed to
    ``default_quote`` since most exchanges quote in USDT/USDC, not raw USD.
    """
    raw = (symbol or "").strip().upper()
    if not raw:
        raise ValueError("symbol is required")

    base: str
    quote: str
    for sep in ("-", "/", ":"):
        if sep in raw:
            base, quote = raw.split(sep, 1)
            break
    else:
        base, quote = raw, default_quote

    if not base or not quote:
        raise ValueError(f"could not parse symbol {symbol!r}")

    # Exchange-side: trade USDT/USDC, not raw USD.
    if quote.upper() == "USD":
        quote = default_quote

    return SymbolMap(base=base, quote=quote.upper())


def _load_ccxt_exchange(exchange_id: str) -> ExchangeClient:
    """Lazy-import ccxt and instantiate the exchange. Tests skip this path."""
    try:
        import ccxt  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "ccxt is required for crypto ingestion. "
            "Install with `pip install -e '.[crypto]'` or `pip install ccxt`."
        ) from e
    cls = getattr(ccxt, exchange_id.lower(), None)
    if cls is None:
        raise RuntimeError(
            f"Unknown ccxt exchange {exchange_id!r}. "
            f"See ccxt.exchanges for the full list."
        )
    return cls({"enableRateLimit": True})


def _to_epoch_ms(value: str | date | datetime) -> int:
    """Convert a str/date/datetime to Unix epoch milliseconds, UTC."""
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ingest_crypto_prices(
    symbol: str,
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    timeframe: str = "1d",
    exchange: str = "binance",
    default_quote: str = "USDT",
    client: ExchangeClient | None = None,
    page_limit: int = 1000,
) -> pd.DataFrame:
    """Fetch crypto OHLCV and return a DataFrame shaped like the equity adapter.

    Output columns:
        ``date`` (index), ``symbol``, ``open``, ``high``, ``low``, ``close``,
        ``adj_close``, ``volume``

    ``adj_close`` mirrors ``close`` — crypto has no dividend adjustments, but
    keeping the column means the feature pipeline stays uniform.

    ``symbol`` can be yfinance-style (``BTC-USD``) or CCXT-style
    (``BTC/USDT``); parsing is forgiving. Output symbol is stored as
    ``BTC-USD`` regardless of quote venue.

    Pagination loops through ``fetch_ohlcv`` calls until the exchange stops
    returning data or ``end`` is reached. Duplicate timestamps across pages
    are collapsed (keep-last).
    """
    mapped = parse_symbol(symbol, default_quote=default_quote)
    if client is None:
        client = _load_ccxt_exchange(exchange)

    since_ms = _to_epoch_ms(start) if start is not None else None
    end_ms = _to_epoch_ms(end) if end is not None else None

    log.info(
        "Fetching %s (%s) from %s since=%s timeframe=%s",
        mapped.ccxt_symbol,
        mapped.external_symbol,
        exchange,
        start,
        timeframe,
    )

    rows: list[list[Any]] = []
    cursor = since_ms
    # Hard safety cap on page count in case an exchange returns a non-monotonic
    # stream. Real-world history is ~10y of daily bars ≈ 4 pages at limit=1000.
    max_pages = 1000
    for _ in range(max_pages):
        batch = client.fetch_ohlcv(
            mapped.ccxt_symbol,
            timeframe=timeframe,
            since=cursor,
            limit=page_limit,
        )
        if not batch:
            break
        if end_ms is not None:
            batch = [b for b in batch if b[0] <= end_ms]
            if not batch:
                break
        rows.extend(batch)
        if len(batch) < page_limit:
            break
        last_ts = batch[-1][0]
        # Defensive: exchange did not advance — stop to avoid infinite loop.
        if cursor is not None and last_ts <= cursor:
            break
        cursor = last_ts + 1
        if end_ms is not None and cursor > end_ms:
            break

    if not rows:
        raise RuntimeError(
            f"No OHLCV rows returned for {mapped.ccxt_symbol} on {exchange}. "
            f"Check the symbol / date range."
        )

    df = pd.DataFrame(
        rows, columns=["ts_ms", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    if timeframe == "1d":
        # Snap to midnight so the date column matches the DATE-keyed `prices`
        # schema. Many exchanges emit bars at 00:00 UTC already, but this is
        # defensive for venues that use 08:00 UTC open times.
        df["date"] = df["date"].dt.floor("D")
    df = df.drop(columns=["ts_ms"]).set_index("date").sort_index()
    # Pagination overlap → keep-last wins (freshest bar for that timestamp).
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])
    df["adj_close"] = df["close"]
    df["symbol"] = mapped.external_symbol
    df = df[_PRICE_COLUMNS]

    log.info(
        "Retrieved %d %s bars for %s (%s → %s)",
        len(df),
        timeframe,
        mapped.external_symbol,
        df.index.min().date(),
        df.index.max().date(),
    )
    return df


__all__ = [
    "ExchangeClient",
    "SymbolMap",
    "parse_symbol",
    "ingest_crypto_prices",
]
