"""Regime detection + slicing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.regimes import (
    detect_trend_regimes,
    detect_vol_regimes,
    slice_by_regime,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_vol_regime_labels_present_after_warmup():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 300), index=_idx(300))
    reg = detect_vol_regimes(returns, window=20)
    # Warm-up is NaN.
    assert reg.iloc[:19].isna().all()
    # Post-warmup should use all three labels.
    post = reg.iloc[19:].dropna()
    assert set(post.unique()) == {"low", "mid", "high"}


def test_vol_regime_classifies_high_vol_as_high():
    """A low-vol period followed by a high-vol period should be classified as
    'low' and 'high' respectively.
    """
    n = 300
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.002, 150)
    loud = rng.normal(0, 0.05, 150)
    returns = pd.Series(np.concatenate([quiet, loud]), index=_idx(n))
    reg = detect_vol_regimes(returns, window=20)
    # Deep in quiet window → low.
    assert reg.iloc[100] == "low"
    # Deep in loud window → high.
    assert reg.iloc[-20] == "high"


def test_vol_regime_too_few_labels_raises():
    with pytest.raises(ValueError):
        detect_vol_regimes(pd.Series([0.0, 0.0, 0.0]), labels=("only-one",))


def test_trend_regime_bull_and_bear():
    n = 400
    # First half rising → bull after SMA warmup; second half falling → bear.
    prices = pd.Series(
        np.concatenate([100 + np.arange(200), 300 - np.arange(200)]),
        index=_idx(n),
    )
    reg = detect_trend_regimes(prices, fast=20, slow=50)
    # Warm-up is NaN.
    assert reg.iloc[:49].isna().all()
    # Deep in rising half → bull.
    assert reg.iloc[150] == "bull"
    # Deep in falling half (allow a few bars for fast SMA to roll over) → bear.
    assert reg.iloc[-10] == "bear"


def test_trend_regime_fast_must_be_less_than_slow():
    p = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        detect_trend_regimes(p, fast=50, slow=50)
    with pytest.raises(ValueError):
        detect_trend_regimes(p, fast=100, slow=50)


def test_slice_by_regime_fractions_sum_to_one():
    n = 100
    idx = _idx(n)
    s = pd.Series(np.random.default_rng(0).normal(0, 0.01, n), index=idx)
    b = pd.Series(np.random.default_rng(1).normal(0, 0.01, n), index=idx)
    pos = pd.Series(1, index=idx)
    reg = pd.Series(["A"] * 60 + ["B"] * 40, index=idx)

    metrics = slice_by_regime(
        strategy_returns=s, benchmark_returns=b, positions=pos, regime=reg
    )
    labels = {m.label: m for m in metrics}
    assert {"A", "B"} == set(labels)
    assert labels["A"].n_bars == 60
    assert labels["B"].n_bars == 40
    assert abs(sum(m.fraction_of_time for m in metrics) - 1.0) < 1e-12


def test_slice_by_regime_excludes_nan_regime_bars():
    n = 100
    idx = _idx(n)
    s = pd.Series(np.linspace(0.001, 0.002, n), index=idx)
    b = pd.Series(np.linspace(0.001, 0.002, n), index=idx)
    pos = pd.Series(1, index=idx)
    reg = pd.Series(["A"] * 50 + [np.nan] * 50, index=idx)

    metrics = slice_by_regime(
        strategy_returns=s, benchmark_returns=b, positions=pos, regime=reg
    )
    assert len(metrics) == 1
    assert metrics[0].label == "A"
    assert metrics[0].n_bars == 50
    # Fraction denominator is valid bars, not total bars.
    assert metrics[0].fraction_of_time == 1.0


def test_slice_compound_return_matches_hand_calc():
    """On a 3-bar regime with returns 1%, -1%, 2%, total return should be
    (1.01 * 0.99 * 1.02) - 1.
    """
    idx = _idx(3)
    s = pd.Series([0.01, -0.01, 0.02], index=idx)
    b = pd.Series([0.0, 0.0, 0.0], index=idx)
    pos = pd.Series(1, index=idx)
    reg = pd.Series(["x", "x", "x"], index=idx)

    metrics = slice_by_regime(
        strategy_returns=s, benchmark_returns=b, positions=pos, regime=reg
    )
    expected = 1.01 * 0.99 * 1.02 - 1
    assert metrics[0].strategy_total_return == pytest.approx(expected, rel=1e-12)


def test_slice_exposure_reflects_position_activity():
    idx = _idx(4)
    s = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    b = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    pos = pd.Series([0, 1, -1, 0], index=idx)
    reg = pd.Series(["a"] * 4, index=idx)
    metrics = slice_by_regime(
        strategy_returns=s, benchmark_returns=b, positions=pos, regime=reg
    )
    assert metrics[0].exposure == pytest.approx(0.5)


def test_slice_empty_regime_returns_empty():
    idx = _idx(5)
    s = pd.Series([0.0] * 5, index=idx)
    reg = pd.Series([np.nan] * 5, index=idx)
    metrics = slice_by_regime(
        strategy_returns=s, benchmark_returns=s, positions=s, regime=reg
    )
    assert metrics == []
