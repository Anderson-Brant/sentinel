"""DuckDB storage round-trip tests."""

from __future__ import annotations

from sentinel.storage.duckdb_store import DuckDBStore


def test_prices_roundtrip(tmp_path, synthetic_prices):
    db = tmp_path / "test.duckdb"
    store = DuckDBStore(path=db)
    n = store.write_prices("TEST", synthetic_prices)
    assert n == len(synthetic_prices)

    loaded = store.read_prices("TEST")
    assert len(loaded) == len(synthetic_prices)
    assert list(loaded.columns) == [
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    # Idempotent overwrite.
    n2 = store.write_prices("TEST", synthetic_prices.iloc[:100])
    assert n2 == 100
    assert len(store.read_prices("TEST")) == 100
