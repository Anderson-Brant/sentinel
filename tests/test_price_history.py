"""Tests for sentinel.fundamental.price_history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.fundamental.price_history import long_term_stats


def _price_frame(dates: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": "TEST", "close": close, "adj_close": close},
        index=pd.Index(dates, name="date"),
    )


def _steady_growth(years: int, annual: float, start: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(end="2026-01-02", periods=years * 252, freq="B")
    daily = (1 + annual) ** (1 / 252) - 1
    close = start * (1 + daily) ** np.arange(len(dates))
    return _price_frame(dates, close)


def test_cagr_recovers_known_growth_rate():
    prices = _steady_growth(years=12, annual=0.15)
    score = long_term_stats(prices)
    for h in (1, 3, 5, 10):
        assert h in score.cagr
        assert score.cagr[h] == pytest.approx(0.15, abs=0.01)


def test_short_history_omits_long_horizons():
    prices = _steady_growth(years=4, annual=0.10)
    score = long_term_stats(prices)
    assert 1 in score.cagr and 3 in score.cagr
    assert 5 not in score.cagr and 10 not in score.cagr
    assert "history" in score.summary


def test_grade_thresholds():
    assert long_term_stats(_steady_growth(12, 0.22)).grade == "A+"
    assert long_term_stats(_steady_growth(12, 0.16)).grade == "A"
    assert long_term_stats(_steady_growth(12, 0.09)).grade == "B"
    assert long_term_stats(_steady_growth(12, -0.10)).grade == "F"


def test_max_drawdown_and_recovery():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    close = np.full(len(dates), 100.0)
    close[100:150] = np.linspace(100, 50, 50)   # crash to -50%
    close[150:250] = np.linspace(50, 110, 100)  # recover past the old peak
    close[250:] = 110.0
    score = long_term_stats(_price_frame(dates, close))
    assert score.max_drawdown == pytest.approx(-0.50, abs=0.01)
    assert score.drawdown_recovery_days is not None
    assert 100 < score.drawdown_recovery_days < 250


def test_unrecovered_drawdown_is_none():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    close = np.concatenate([np.linspace(100, 120, 150), np.linspace(120, 70, 150)])
    score = long_term_stats(_price_frame(dates, close))
    assert score.drawdown_recovery_days is None
    assert "not yet recovered" in score.summary


def test_deep_drawdown_costs_a_notch():
    prices = _steady_growth(12, 0.22)
    crashed = prices.copy()
    n = len(crashed)
    dd = np.ones(n)
    dd[n // 2 : n // 2 + 100] = 0.30  # forced -70% dip
    crashed["close"] = crashed["close"] * dd
    crashed["adj_close"] = crashed["close"]
    score = long_term_stats(crashed)
    clean = long_term_stats(prices)
    assert clean.grade == "A+"
    assert score.max_drawdown < -0.6
    assert score.grade != "A+"


def test_insufficient_data():
    dates = pd.date_range("2026-01-01", periods=1, freq="B")
    score = long_term_stats(_price_frame(dates, np.array([100.0])))
    assert score.grade is None
    assert score.score is None
    assert "insufficient" in score.summary


def test_uses_adj_close_when_present():
    dates = pd.date_range(end="2026-01-02", periods=12 * 252, freq="B")
    daily = (1 + 0.15) ** (1 / 252) - 1
    adj = 100 * (1 + daily) ** np.arange(len(dates))
    df = pd.DataFrame(
        {"symbol": "TEST", "close": np.full(len(dates), 100.0), "adj_close": adj},
        index=pd.Index(dates, name="date"),
    )
    score = long_term_stats(df)
    assert score.cagr[10] == pytest.approx(0.15, abs=0.01)
