"""Tests for the position-sizing primitives.

These cover the pure functions in `sentinel.backtest.sizing` independent
of the engine — formula correctness, edge cases (zero vol, NaN warmup),
and parameter validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.backtest.sizing import realized_volatility, vol_target_scale


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


# ---------------------------------------------------------------------------
# realized_volatility
# ---------------------------------------------------------------------------


def test_realized_volatility_warmup_is_nan():
    returns = pd.Series([0.01] * 50, index=_idx(50))
    vol = realized_volatility(returns, window=20)
    # First window-1 = 19 rows must be NaN.
    assert vol.iloc[:19].isna().all()
    # Bar 19 is the first complete window.
    assert not np.isnan(vol.iloc[19])


def test_realized_volatility_constant_returns_yield_zero():
    """A window of identical returns has std=0 → annualized vol=0."""
    returns = pd.Series([0.005] * 30, index=_idx(30))
    vol = realized_volatility(returns, window=10)
    assert vol.iloc[15] == pytest.approx(0.0, abs=1e-12)


def test_realized_volatility_matches_hand_calc():
    """For 0.01, 0.0, 0.02, sample std = sqrt(((0.01 - mean)^2 + ...)/2).
    Annualize by sqrt(252)."""
    returns = pd.Series([0.01, 0.00, 0.02], index=_idx(3))
    vol = realized_volatility(returns, window=3)
    arr = np.array([0.01, 0.00, 0.02])
    expected = float(np.std(arr, ddof=1) * np.sqrt(252))
    assert vol.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_realized_volatility_window_validation():
    with pytest.raises(ValueError, match="window must be >= 2"):
        realized_volatility(pd.Series([0.0, 0.0, 0.0]), window=1)


def test_realized_volatility_periods_per_year_validation():
    with pytest.raises(ValueError, match="periods_per_year must be > 0"):
        realized_volatility(pd.Series([0.0, 0.0]), window=2, periods_per_year=0)


def test_realized_volatility_periods_per_year_scales_linearly_in_sqrt():
    """Doubling periods_per_year should scale annualized vol by sqrt(2)."""
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=_idx(100))
    v_252 = realized_volatility(returns, window=20, periods_per_year=252)
    v_504 = realized_volatility(returns, window=20, periods_per_year=504)
    ratio = v_504.dropna() / v_252.dropna()
    np.testing.assert_allclose(ratio, np.sqrt(2), atol=1e-12)


# ---------------------------------------------------------------------------
# vol_target_scale
# ---------------------------------------------------------------------------


def test_vol_target_scale_inverse_relationship():
    """scale = target / vol. At vol=target, scale=1."""
    vol = pd.Series([0.10, 0.20, 0.05], index=_idx(3))
    scale = vol_target_scale(vol, target_vol_annual=0.10, max_leverage=10.0)
    np.testing.assert_allclose(
        scale.to_numpy(), np.array([1.0, 0.5, 2.0]), atol=1e-12
    )


def test_vol_target_scale_caps_at_max_leverage():
    """Quiet regimes would pump scale above the cap; max_leverage clamps it."""
    vol = pd.Series([0.10, 0.01, 0.001], index=_idx(3))
    scale = vol_target_scale(vol, target_vol_annual=0.10, max_leverage=2.0)
    # At vol=0.10 → scale=1 (under cap). At lower vols → would be 10, 100; capped at 2.
    np.testing.assert_allclose(scale.to_numpy(), np.array([1.0, 2.0, 2.0]))


def test_vol_target_scale_preserves_nan_warmup():
    """NaN inputs must produce NaN outputs (engine treats NaN → flat)."""
    vol = pd.Series([np.nan, np.nan, 0.10], index=_idx(3))
    scale = vol_target_scale(vol, target_vol_annual=0.10)
    assert scale.iloc[0] != scale.iloc[0]  # NaN
    assert scale.iloc[1] != scale.iloc[1]
    assert scale.iloc[2] == pytest.approx(1.0)


def test_vol_target_scale_zero_vol_uses_floor_and_caps():
    """Constant returns produce vol=0; the floor prevents div-by-zero, and the
    cap then clamps the resulting infinity-in-disguise to max_leverage."""
    vol = pd.Series([0.0, 0.0], index=_idx(2))
    scale = vol_target_scale(vol, target_vol_annual=0.10, max_leverage=3.0)
    np.testing.assert_allclose(scale.to_numpy(), np.array([3.0, 3.0]))


def test_vol_target_scale_validates_target():
    vol = pd.Series([0.1])
    with pytest.raises(ValueError, match="target_vol_annual must be > 0"):
        vol_target_scale(vol, target_vol_annual=0.0)
    with pytest.raises(ValueError, match="target_vol_annual must be > 0"):
        vol_target_scale(vol, target_vol_annual=-0.05)


def test_vol_target_scale_validates_max_leverage():
    vol = pd.Series([0.1])
    with pytest.raises(ValueError, match="max_leverage must be > 0"):
        vol_target_scale(vol, target_vol_annual=0.1, max_leverage=0.0)
