"""Tests for the CCXT crypto adapter.

The adapter is intentionally thin and uses a dependency-injected
``ExchangeClient`` protocol, so we do not need ccxt installed in the test
environment — a ``_FakeExchange`` plays the role of the exchange and we
assert shape / invariants of the returned DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from sentinel.ingestion.crypto import (
    SymbolMap,
    ingest_crypto_prices,
    parse_symbol,
)
from sentinel.scheduling.registry import registered_kinds

# ---------------------------------------------------------------------------
# Fake exchange
# ---------------------------------------------------------------------------


@dataclass
class _FakeExchange:
    """Minimal CCXT-shaped fake.

    ``bars`` is a full history (epoch-ms timestamps); ``fetch_ohlcv`` serves
    them in pages of ``page_limit`` filtered by ``since``.
    """

    bars: list[list[Any]] = field(default_factory=list)
    page_limit: int = 1000
    calls: list[dict[str, Any]] = field(default_factory=list)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None,
        limit: int | None,
    ) -> list[list[Any]]:
        self.calls.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )
        pool = self.bars
        if since is not None:
            pool = [b for b in pool if b[0] >= since]
        cap = limit if limit is not None else self.page_limit
        return list(pool[:cap])


def _day_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso).replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _mk_daily_bars(
    start_iso: str = "2024-01-01", n: int = 5, base_price: float = 100.0
) -> list[list[Any]]:
    start_ms = _day_ms(start_iso)
    day_ms = 24 * 60 * 60 * 1000
    out: list[list[Any]] = []
    for i in range(n):
        p = base_price + i
        out.append(
            [
                start_ms + i * day_ms,
                p,              # open
                p + 1.0,        # high
                p - 1.0,        # low
                p + 0.5,        # close
                1_000_000.0 + i,  # volume
            ]
        )
    return out


# ---------------------------------------------------------------------------
# parse_symbol + SymbolMap
# ---------------------------------------------------------------------------


def test_parse_symbol_yfinance_style():
    m = parse_symbol("BTC-USD")
    assert m.base == "BTC"
    assert m.quote == "USDT"  # USD → default_quote
    assert m.ccxt_symbol == "BTC/USDT"
    assert m.external_symbol == "BTC-USD"


def test_parse_symbol_ccxt_style():
    m = parse_symbol("eth/usdt")
    assert m.base == "ETH"
    assert m.quote == "USDT"
    assert m.ccxt_symbol == "ETH/USDT"
    assert m.external_symbol == "ETH-USD"  # USDT → USD for storage


def test_parse_symbol_bare_ticker_uses_default_quote():
    m = parse_symbol("BTC", default_quote="USDC")
    assert m.ccxt_symbol == "BTC/USDC"
    assert m.external_symbol == "BTC-USD"


def test_parse_symbol_colon_separator():
    m = parse_symbol("SOL:USDT")
    assert m.ccxt_symbol == "SOL/USDT"


def test_parse_symbol_preserves_non_usd_quote():
    m = parse_symbol("BTC-EUR")
    assert m.ccxt_symbol == "BTC/EUR"
    assert m.external_symbol == "BTC-EUR"


def test_parse_symbol_rejects_empty():
    with pytest.raises(ValueError):
        parse_symbol("")
    with pytest.raises(ValueError):
        parse_symbol("   ")


def test_symbol_map_usdc_normalizes_to_usd():
    assert SymbolMap("BTC", "USDC").external_symbol == "BTC-USD"


def test_symbol_map_dai_normalizes_to_usd():
    assert SymbolMap("ETH", "DAI").external_symbol == "ETH-USD"


# ---------------------------------------------------------------------------
# ingest_crypto_prices — happy path
# ---------------------------------------------------------------------------


def test_ingest_returns_price_adapter_shape():
    fake = _FakeExchange(bars=_mk_daily_bars(n=3))
    df = ingest_crypto_prices("BTC-USD", client=fake, start="2024-01-01")

    # Shape must match sentinel.ingestion.market.ingest_prices output so that
    # store.write_prices() accepts it without branching on asset class.
    assert list(df.columns) == [
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert df.index.name == "date"
    assert len(df) == 3
    assert (df["symbol"] == "BTC-USD").all()
    # adj_close mirrors close for crypto (no dividend adjustments).
    assert (df["adj_close"] == df["close"]).all()


def test_ingest_sends_ccxt_symbol_to_exchange():
    fake = _FakeExchange(bars=_mk_daily_bars(n=2))
    ingest_crypto_prices("BTC-USD", client=fake, start="2024-01-01")
    assert fake.calls[0]["symbol"] == "BTC/USDT"
    assert fake.calls[0]["timeframe"] == "1d"


def test_ingest_daily_timestamps_are_floored_to_midnight():
    # Bar emitted at 08:00 UTC should snap to midnight for the DATE-keyed schema.
    eight_am = _day_ms("2024-01-01") + 8 * 60 * 60 * 1000
    fake = _FakeExchange(
        bars=[[eight_am, 100.0, 101.0, 99.0, 100.5, 1_000_000.0]]
    )
    df = ingest_crypto_prices("BTC-USD", client=fake, start="2024-01-01")
    # Index is a DatetimeIndex at exact 00:00:00
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00")


def test_ingest_accepts_date_and_datetime_start():
    fake = _FakeExchange(bars=_mk_daily_bars(n=1))
    # date
    ingest_crypto_prices("BTC-USD", client=fake, start=date(2024, 1, 1))
    # datetime (naive → assumed UTC)
    ingest_crypto_prices("BTC-USD", client=fake, start=datetime(2024, 1, 1))
    # Both calls should have produced non-None since_ms.
    assert all(c["since"] is not None for c in fake.calls)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_ingest_paginates_through_full_history():
    # 2500 bars across 3 pages of 1000.
    bars = _mk_daily_bars(n=2500)
    fake = _FakeExchange(bars=bars, page_limit=1000)
    df = ingest_crypto_prices(
        "BTC-USD", client=fake, start="2018-01-01", page_limit=1000
    )
    assert len(df) == 2500
    assert len(fake.calls) == 3


def test_ingest_deduplicates_overlapping_pages():
    # If the exchange returns overlapping bars across pages (it sometimes
    # includes the ``since`` bar on the next page), we must deduplicate.
    base = _mk_daily_bars(n=3)
    duplicate_ts = _FakeExchange(bars=base + [base[-1]])  # last bar twice
    df = ingest_crypto_prices("BTC-USD", client=duplicate_ts, start="2024-01-01")
    assert len(df) == 3
    assert df.index.is_unique


def test_ingest_stops_when_cursor_does_not_advance():
    # Pathological exchange: every page returns identical bars → we must not
    # loop forever. The adapter caps pages internally; here we verify it
    # tolerates non-advancing cursor without hanging.
    one_bar = _mk_daily_bars(n=1)
    fake = _FakeExchange(bars=one_bar * 2000, page_limit=1)
    df = ingest_crypto_prices("BTC-USD", client=fake, start="2024-01-01")
    assert len(df) == 1  # only one unique timestamp


# ---------------------------------------------------------------------------
# End-time clipping + empty behavior
# ---------------------------------------------------------------------------


def test_ingest_clips_at_end_date():
    bars = _mk_daily_bars(n=10, start_iso="2024-01-01")
    fake = _FakeExchange(bars=bars)
    df = ingest_crypto_prices(
        "BTC-USD", client=fake, start="2024-01-01", end="2024-01-03"
    )
    # Only bars on/before 2024-01-03 should survive.
    assert df.index.max() <= pd.Timestamp("2024-01-03")
    assert len(df) == 3


def test_ingest_raises_when_exchange_returns_nothing():
    fake = _FakeExchange(bars=[])
    with pytest.raises(RuntimeError, match="No OHLCV rows"):
        ingest_crypto_prices("BTC-USD", client=fake, start="2024-01-01")


# ---------------------------------------------------------------------------
# Integration with scheduler registry
# ---------------------------------------------------------------------------


def test_registry_has_ingest_crypto():
    # Kind is registered at import time — smoke-check so the scheduler YAML
    # entry won't KeyError in production.
    assert "ingest-crypto" in registered_kinds()


def test_registered_kinds_includes_full_ingest_family():
    kinds = set(registered_kinds())
    # All four ingestion families should be bookable scheduler jobs.
    assert {"ingest-prices", "ingest-reddit", "ingest-twitter", "ingest-crypto"}.issubset(
        kinds
    )
