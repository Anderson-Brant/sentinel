"""Glue the individual feature generators together into the feature table
consumed by the modeling layer.
"""

from __future__ import annotations

import pandas as pd

from sentinel.config import SentinelConfig
from sentinel.features.targets import directional_target, forward_return
from sentinel.features.technical import (
    momentum,
    moving_averages,
    returns,
    volatility,
    volume_features,
)
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


def build_feature_table(prices: pd.DataFrame, cfg: SentinelConfig) -> pd.DataFrame:
    """Return a wide table: features ⨯ target, indexed by date.

    Rows with NaN features or unknown targets are dropped so the modeling layer
    can assume a clean matrix.
    """
    if prices.empty:
        raise ValueError("prices is empty")

    # Feature blocks.
    blocks: list[pd.DataFrame] = [
        returns(prices, cfg.features.returns["windows"]),
        moving_averages(
            prices,
            sma_windows=cfg.features.moving_averages["sma_windows"],
            ema_windows=cfg.features.moving_averages["ema_windows"],
        ),
        momentum(prices, cfg.features.momentum["windows"]),
        volatility(prices, cfg.features.volatility["windows"]),
        volume_features(prices, cfg.features.volume["windows"]),
    ]

    feats = pd.concat(blocks, axis=1)

    # Drop absolute-level features that don't generalize across price regimes
    # (SMAs, EMAs — we keep the normalized `close_to_sma_*` versions instead).
    level_cols = [c for c in feats.columns if c.startswith(("sma_", "ema_")) and "over" not in c]
    feats = feats.drop(columns=level_cols, errors="ignore")

    # Targets.
    horizon = cfg.targets.horizon_days
    feats["target_direction"] = directional_target(prices, horizon=horizon)
    feats["target_return"] = forward_return(prices, horizon=horizon)

    # Carry symbol if present.
    if "symbol" in prices.columns:
        feats.insert(0, "symbol", prices["symbol"].iloc[0])

    before = len(feats)
    feats = feats.dropna()
    after = len(feats)
    log.info("Feature table: %d rows (dropped %d with NaN), %d cols", after, before - after, feats.shape[1])

    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are actual features (exclude metadata + targets)."""
    excluded = {"symbol", "target_direction", "target_return"}
    return [c for c in df.columns if c not in excluded]
