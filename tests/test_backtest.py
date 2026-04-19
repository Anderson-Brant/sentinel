"""Backtest engine tests.

These tests pin down the properties that separate an honest backtest from a
broken one. They use small, hand-constructed price paths so every expected
value can be computed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.backtest.engine import backtest


def _trivial_prices(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "adj_close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_always_no_signal_returns_flat_equity():
    """If probabilities are NaN everywhere, the strategy never trades. Equity stays at 1.0."""
    prices = _trivial_prices([100, 101, 102, 103, 104, 105])
    probs = pd.Series([np.nan] * 6, index=prices.index)

    r = backtest(prices, probs, symbol="FLAT", cost_bps=5.0)

    assert r.n_trades == 0
    assert r.exposure == 0.0
    assert r.turnover == 0.0
    np.testing.assert_allclose(r.equity_curve.to_numpy(), 1.0)
    # Benchmark still tracks buy-and-hold.
    assert r.benchmark_total_return > 0


def test_always_long_matches_buy_and_hold_before_costs():
    """If the signal is always long, gross strategy returns == asset returns."""
    prices = _trivial_prices([100, 101, 99, 102, 103])
    # Probabilities above threshold everywhere.
    probs = pd.Series([0.9] * len(prices), index=prices.index)

    r = backtest(prices, probs, symbol="LONG", cost_bps=0.0)

    # After the initial 1-bar shift (position held on day 0 is 0), position is 1 from day 1 on.
    # So strategy return on day t == asset return on day t for t >= 1.
    asset_ret = prices["close"].pct_change().fillna(0.0).to_numpy()
    strat_ret = r.strategy_returns.to_numpy()
    # Day 0: position=0 → 0 strategy return. Asset return is 0 too (pct_change fillna).
    np.testing.assert_allclose(strat_ret[1:], asset_ret[1:], atol=1e-12)


def test_no_lookahead_position_is_shifted():
    """Probability decided on day t must affect the position held on day t+1, not day t."""
    prices = _trivial_prices([100, 100, 100, 200, 100])  # jump on day 3
    # Put the "enter long" signal ON day 2 (decided from close of day 2).
    probs = pd.Series([0.1, 0.1, 0.9, 0.1, 0.1], index=prices.index)

    r = backtest(prices, probs, symbol="LEAK_CHECK", cost_bps=0.0)

    # held_position on day 3 (the jump day) should be 1 because target was set on day 2.
    # held_position on day 2 should still be 0 because target was 0.1 on days 0,1.
    positions = r.positions.to_numpy()
    assert positions[2] == 0, "must not trade on the same bar the signal was decided"
    assert positions[3] == 1, "must trade the bar after the signal"
    # We captured the day-3 jump (100 → 200 = +100%).
    assert r.strategy_returns.iloc[3] == pytest.approx(1.0, rel=1e-9)


def test_transaction_costs_reduce_return():
    """Same signal, higher cost → strictly lower net return."""
    prices = _trivial_prices([100, 101, 102, 101, 103, 104])
    probs = pd.Series([0.1, 0.9, 0.1, 0.9, 0.1, 0.1], index=prices.index)  # flips → costs

    r_no_cost = backtest(prices, probs, symbol="X", cost_bps=0.0)
    r_with_cost = backtest(prices, probs, symbol="X", cost_bps=50.0)  # 50 bps → big

    assert r_with_cost.total_return < r_no_cost.total_return


def test_invalid_thresholds_raise():
    prices = _trivial_prices([100, 101, 102])
    probs = pd.Series([0.5] * 3, index=prices.index)
    with pytest.raises(ValueError):
        backtest(prices, probs, long_threshold=0.4, short_threshold=0.5)
    with pytest.raises(ValueError):
        backtest(prices, probs, long_threshold=1.5, short_threshold=0.5)
    with pytest.raises(ValueError):
        backtest(prices.drop(columns=["close"]), probs)


def test_trade_extraction_counts_round_trips():
    """Position path 0,0,1,1,1,0,0,1,1,0 has two long round-trips."""
    prices = _trivial_prices([100, 101, 102, 103, 104, 103, 102, 103, 104, 103])
    # Build probs so that target_pos → held_position (after shift) matches the path above.
    # held_position = target.shift(1). So target must be 0,1,1,1,0,0,1,1,0 on days 0..8 (9 days).
    # With a 10-bar price series, we set target on all 10 days; only 9 will be observed as held.
    target_probs = [0.6, 0.6, 0.6, 0.6, 0.1, 0.1, 0.6, 0.6, 0.1, 0.1]
    probs = pd.Series(target_probs, index=prices.index)

    r = backtest(prices, probs, symbol="TWO_TRADES", cost_bps=0.0)
    assert r.n_trades == 2
    assert all(t.direction == 1 for t in r.trades)


def test_short_leg_profits_on_decline():
    """allow_short=True + declining price + P(up) < short_threshold → positive P&L."""
    prices = _trivial_prices([100, 99, 98, 97, 96])
    probs = pd.Series([0.1] * len(prices), index=prices.index)

    r_long_only = backtest(prices, probs, cost_bps=0.0, allow_short=False)
    r_short = backtest(prices, probs, cost_bps=0.0, allow_short=True)

    # Long-only on a falling market with sub-threshold probs: position stays 0.
    assert r_long_only.total_return == pytest.approx(0.0, abs=1e-12)
    # With shorts, we profit from the decline.
    assert r_short.total_return > 0


def test_nan_probs_produce_zero_position():
    """NaN probability rows must not produce any position."""
    prices = _trivial_prices([100, 101, 102, 103, 104])
    probs = pd.Series([np.nan, np.nan, 0.9, 0.9, 0.9], index=prices.index)

    r = backtest(prices, probs, cost_bps=0.0)
    # Day 0 and 1: target is 0 (NaN → 0), so held_position on day 1 is 0. Good.
    # Day 2 target becomes 1, so held_position on day 3 is 1.
    assert r.positions.iloc[0] == 0
    assert r.positions.iloc[1] == 0
    assert r.positions.iloc[2] == 0
    assert r.positions.iloc[3] == 1


def test_metrics_have_sane_signs():
    """A profitable strategy has positive annualized return and non-negative exposure ≤ 1."""
    prices = _trivial_prices([100 + i for i in range(30)])  # monotone up
    probs = pd.Series([0.9] * len(prices), index=prices.index)

    r = backtest(prices, probs, cost_bps=0.0)
    assert r.total_return > 0
    assert 0.0 <= r.exposure <= 1.0
    assert r.max_drawdown <= 0.0  # drawdown is non-positive by definition


# ---------------------------------------------------------------------------
# Vol-targeted position sizing
# ---------------------------------------------------------------------------


def test_fixed_size_mode_keeps_int_positions_and_scale_of_one():
    """Default (target_vol_annual=None) must preserve the classic {-1,0,+1} behavior."""
    prices = _trivial_prices([100 + i for i in range(30)])
    probs = pd.Series([0.9] * len(prices), index=prices.index)
    r = backtest(prices, probs, cost_bps=0.0)
    # Positions are all 0 or 1 under always-long signal.
    uniq = set(r.positions.unique().tolist())
    assert uniq.issubset({0, 1})
    # Scale series is all 1.0 in fixed-size mode.
    assert (r.position_scale == 1.0).all()
    # target_vol_annual recorded as None.
    assert r.target_vol_annual is None


def test_vol_target_produces_float_positions_bounded_by_max_leverage():
    """With vol_target on, position values are float, capped by max_leverage."""
    rng = np.random.default_rng(0)
    n = 120
    prices = _trivial_prices((100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))).tolist())
    probs = pd.Series([0.9] * n, index=prices.index)

    r = backtest(
        prices, probs, cost_bps=0.0,
        target_vol_annual=0.10, vol_lookback=20, max_leverage=1.5,
    )
    # Warmup (first vol_lookback bars + 1 for shift) positions are 0.
    assert r.positions.iloc[0] == 0.0
    assert r.positions.iloc[19] == 0.0
    # Post-warmup, positions are non-integer floats bounded by max_leverage.
    post = r.positions.iloc[25:]
    assert post.dtype.kind == "f"
    assert (post.abs() <= 1.5 + 1e-12).all()
    # Report carries through the config + diagnostics.
    assert r.target_vol_annual == 0.10
    assert r.vol_lookback == 20
    assert r.max_leverage == 1.5
    assert r.realized_vol.iloc[19] == pytest.approx(r.realized_vol.iloc[19])  # not NaN
    assert r.realized_vol.iloc[:18].isna().all()


def test_vol_target_no_lookahead():
    """Vol-targeted sizing must still obey the 1-bar shift — a signal on day t can
    only influence P&L on day t+1."""
    # Long flat period then a big jump on day 30. Signal appears on day 29.
    closes = [100.0] * 30 + [200.0] + [200.0] * 10
    prices = _trivial_prices(closes)
    probs = pd.Series([0.1] * len(prices), index=prices.index)
    probs.iloc[29] = 0.9

    r = backtest(
        prices, probs, cost_bps=0.0,
        target_vol_annual=0.10, vol_lookback=10,
    )
    # Position on day 29 (where signal first appears) must be 0.
    assert r.positions.iloc[29] == 0.0
    # Position on day 30 (jump day) must be non-zero.
    assert r.positions.iloc[30] != 0.0


def test_vol_target_lowers_size_in_noisier_regime():
    """Given identical direction signals, a noisier return series should get
    smaller position sizes than a quieter one."""
    n = 80
    rng_q = np.random.default_rng(1)
    rng_l = np.random.default_rng(2)
    # Two price paths with very different vol.
    quiet_rets = rng_q.normal(0, 0.003, n)
    loud_rets = rng_l.normal(0, 0.03, n)
    quiet = 100 * np.exp(np.cumsum(quiet_rets))
    loud = 100 * np.exp(np.cumsum(loud_rets))
    probs = pd.Series([0.9] * n, index=_trivial_prices([100] * n).index)

    r_quiet = backtest(
        _trivial_prices(quiet.tolist()), probs, cost_bps=0.0,
        target_vol_annual=0.10, vol_lookback=20, max_leverage=5.0,
    )
    r_loud = backtest(
        _trivial_prices(loud.tolist()), probs, cost_bps=0.0,
        target_vol_annual=0.10, vol_lookback=20, max_leverage=5.0,
    )
    # Mean post-warmup position size should be strictly larger for quiet.
    qs = r_quiet.positions.iloc[30:].abs().mean()
    ls = r_loud.positions.iloc[30:].abs().mean()
    assert qs > ls


def test_vol_target_cost_grows_with_size_changes():
    """Under vol-targeting, costs are charged on |Δposition| — so swings in
    the scale series produce costs even when the direction never changes."""
    n = 80
    closes = (100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.02, n)))).tolist()
    prices = _trivial_prices(closes)
    probs = pd.Series([0.9] * n, index=prices.index)

    r_no = backtest(prices, probs, cost_bps=0.0, target_vol_annual=0.10, vol_lookback=20)
    r_yes = backtest(prices, probs, cost_bps=20.0, target_vol_annual=0.10, vol_lookback=20)
    # Same direction signal (always long) → identical trades list in count…
    assert r_no.n_trades == r_yes.n_trades
    # …but costs must have eroded returns when bps > 0.
    assert r_yes.total_return < r_no.total_return
