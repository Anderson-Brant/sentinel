"""Unit tests for the feature engineering layer.

These are deliberately focused on correctness properties, not on "did the model
beat the benchmark" — that's the evaluation layer's job.
"""

from __future__ import annotations

import numpy as np

from sentinel.config import load_config
from sentinel.features.pipeline import build_feature_table, feature_columns
from sentinel.features.targets import directional_target, forward_return
from sentinel.features.technical import (
    momentum,
    moving_averages,
    returns,
    volatility,
    volume_features,
)


def test_returns_basic(synthetic_prices):
    out = returns(synthetic_prices, windows=[1, 5, 20])
    assert {"ret_1d", "ret_5d", "ret_20d", "log_ret_1d"}.issubset(out.columns)
    # First n rows of ret_Nd are NaN.
    assert out["ret_1d"].iloc[0] != out["ret_1d"].iloc[0]  # NaN
    # Returns are finite past the warm-up window.
    assert np.isfinite(out["ret_5d"].iloc[100])


def test_moving_averages_include_crossover(synthetic_prices):
    out = moving_averages(synthetic_prices, sma_windows=[5, 50, 200], ema_windows=[12])
    assert "sma_50_over_200" in out.columns
    # Crossover is binary.
    unique = set(out["sma_50_over_200"].dropna().unique().tolist())
    assert unique.issubset({0.0, 1.0})


def test_momentum_and_volatility_and_volume(synthetic_prices):
    mom = momentum(synthetic_prices, windows=[5, 20])
    vol = volatility(synthetic_prices, windows=[5, 20])
    v = volume_features(synthetic_prices, windows=[5, 20])
    assert "mom_5d" in mom.columns
    assert "vol_5d" in vol.columns
    assert "rel_vol_20d" in v.columns
    assert "signed_rel_vol_20d" in v.columns


def test_forward_return_and_target_alignment(synthetic_prices):
    fr = forward_return(synthetic_prices, horizon=1)
    target = directional_target(synthetic_prices, horizon=1)
    # Last row's forward return is unknown → NaN (target should be NA too).
    assert np.isnan(fr.iloc[-1])
    assert target.iloc[-1] != target.iloc[-1]  # NA comparison → False
    # Direction matches sign of forward return where known.
    known = fr.dropna()
    mapped = (known > 0).astype(int)
    assert (target.loc[known.index].astype(int) == mapped).all()


def test_build_feature_table_has_target_and_no_nan(synthetic_prices):
    cfg = load_config()
    table = build_feature_table(synthetic_prices, cfg=cfg)
    assert "target_direction" in table.columns
    assert "target_return" in table.columns
    # Clean matrix.
    assert not table.isna().any().any()
    # At least some features and a reasonable row count.
    cols = feature_columns(table)
    assert len(cols) >= 15
    assert len(table) > 200


def test_no_lookahead_in_features(synthetic_prices):
    """Sanity: truncating the input shouldn't change historical feature values."""
    cfg = load_config()
    full = build_feature_table(synthetic_prices, cfg=cfg)
    truncated = build_feature_table(synthetic_prices.iloc[:-50], cfg=cfg)

    # Features at dates present in both should match.
    common = truncated.index.intersection(full.index)
    feat_cols = [c for c in feature_columns(full) if c in feature_columns(truncated)]
    left = full.loc[common, feat_cols]
    right = truncated.loc[common, feat_cols]
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), rtol=1e-10, atol=1e-12)
