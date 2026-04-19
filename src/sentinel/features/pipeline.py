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


def build_feature_table(
    prices: pd.DataFrame,
    cfg: SentinelConfig,
    *,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a wide table: features ⨯ target, indexed by date.

    Parameters
    ----------
    prices : OHLCV table indexed by date.
    cfg : loaded :class:`SentinelConfig`.
    sentiment : optional per-date sentiment feature block (see
        :func:`sentinel.features.sentiment.sentiment_features_for_symbol`).
        When provided, its columns are left-joined onto the technical block.
        Missing sentiment days have ``reddit_mention_count`` filled with 0 so
        we keep the price row instead of dropping it — the remaining sentiment
        columns may still be NaN, which the modeling layer can drop or
        impute. Passing ``None`` (the default) keeps the MVP technical-only
        behavior and lets us run the ablation "technical vs hybrid" cleanly.

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

    # Optional sentiment block.
    if sentiment is not None and not sentiment.empty:
        from sentinel.features.sentiment import MENTION_COUNT_COLS

        sent = sentiment.copy()
        sent.index = pd.to_datetime(sent.index)
        # Align to price calendar. Missing days: 0 mentions, 0 sentiment.
        sent = sent.reindex(feats.index)
        # Mention counts (reddit + twitter) default to 0 on no-post days.
        for c in MENTION_COUNT_COLS:
            if c in sent.columns:
                sent[c] = sent[c].fillna(0)
        # Other sentiment columns stay NaN on no-post days — the final dropna
        # would kill every row, so we neutralize them to 0 here (mean/ratios).
        for c in sent.columns:
            if c not in MENTION_COUNT_COLS:
                sent[c] = sent[c].fillna(0.0)
        feats = pd.concat([feats, sent], axis=1)

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
