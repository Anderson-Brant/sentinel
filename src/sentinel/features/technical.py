"""Technical indicator feature generation.

All functions:
    - Take a DataFrame indexed by date with at least ``close`` + ``volume`` columns.
    - Return a DataFrame of new columns aligned to the input index.
    - Use only past information at time t (no lookahead).

Kept in pure pandas so they're easy to test and reason about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def returns(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Simple percent returns over each window."""
    out = {}
    close = prices["close"]
    for w in windows:
        out[f"ret_{w}d"] = close.pct_change(w)
    # Log return 1d — useful for some models.
    out["log_ret_1d"] = np.log(close).diff()
    return pd.DataFrame(out, index=prices.index)


# ---------------------------------------------------------------------------
# Trend & momentum
# ---------------------------------------------------------------------------


def moving_averages(
    prices: pd.DataFrame, sma_windows: list[int], ema_windows: list[int]
) -> pd.DataFrame:
    close = prices["close"]
    out: dict[str, pd.Series] = {}
    for w in sma_windows:
        sma = close.rolling(w).mean()
        out[f"sma_{w}"] = sma
        out[f"close_to_sma_{w}"] = close / sma - 1
    for w in ema_windows:
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
    # Classic crossover flag if 50/200 both present.
    if 50 in sma_windows and 200 in sma_windows:
        out["sma_50_over_200"] = (out["sma_50"] > out["sma_200"]).astype(float)
    return pd.DataFrame(out, index=prices.index)


def momentum(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    close = prices["close"]
    out = {}
    for w in windows:
        # Rate of change: (close_t - close_{t-w}) / close_{t-w}
        out[f"mom_{w}d"] = close.pct_change(w)
        # Rolling high/low distance (useful for breakout-type signals).
        rolling_high = close.rolling(w).max()
        rolling_low = close.rolling(w).min()
        out[f"dist_to_high_{w}d"] = close / rolling_high - 1
        out[f"dist_to_low_{w}d"] = close / rolling_low - 1
    return pd.DataFrame(out, index=prices.index)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def volatility(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    log_ret = np.log(prices["close"]).diff()
    out = {}
    for w in windows:
        out[f"vol_{w}d"] = log_ret.rolling(w).std()
    # True-range-ish: (high - low) / close
    out["hl_range_pct"] = (prices["high"] - prices["low"]) / prices["close"]
    out["hl_range_pct_5d_mean"] = out["hl_range_pct"].rolling(5).mean()
    return pd.DataFrame(out, index=prices.index)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def volume_features(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    vol = prices["volume"].astype(float)
    out = {}
    for w in windows:
        avg = vol.rolling(w).mean()
        out[f"vol_avg_{w}d"] = avg
        out[f"rel_vol_{w}d"] = vol / avg
    # Price-volume interaction: signed return × relative volume (1d return, 20d baseline).
    if 20 in windows:
        out["signed_rel_vol_20d"] = np.sign(prices["close"].pct_change()) * out["rel_vol_20d"]
    return pd.DataFrame(out, index=prices.index)
