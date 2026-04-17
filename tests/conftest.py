"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """Deterministic 500-day OHLCV series. Enough for features + a short walk-forward."""
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    # Drifting random walk with mild autocorrelation so features aren't all noise.
    steps = rng.normal(loc=0.0005, scale=0.012, size=len(dates))
    close = 100 * np.exp(np.cumsum(steps))
    open_ = close * (1 + rng.normal(0, 0.003, size=len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=len(dates))))
    volume = rng.integers(1_000_000, 5_000_000, size=len(dates)).astype(float)

    df = pd.DataFrame(
        {
            "symbol": "TEST",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=pd.Index(dates, name="date"),
    )
    return df
