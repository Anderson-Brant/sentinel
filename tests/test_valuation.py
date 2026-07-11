"""Tests for sentinel.fundamental.valuation (pure scoring path, no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sentinel.fundamental.valuation import (
    FundamentalsSnapshot,
    _history_percentile,
    compute_valuation,
)


def _snapshot(**overrides) -> FundamentalsSnapshot:
    base = dict(
        symbol="TEST",
        as_of=date(2026, 7, 10),
        price=100.0,
        market_cap=1_000_000_000.0,
        shares_outstanding=10_000_000.0,
        trailing_eps=5.0,
        revenue_ttm=500_000_000.0,
        fcf_ttm=50_000_000.0,
        ebitda_ttm=100_000_000.0,
        net_debt=200_000_000.0,
        dividend_yield=0.01,
        earnings_growth=0.10,
    )
    base.update(overrides)
    return FundamentalsSnapshot(**base)


def test_basic_ratios():
    v = compute_valuation(_snapshot())
    assert v.pe == pytest.approx(20.0)
    assert v.ps == pytest.approx(2.0)
    assert v.pfcf == pytest.approx(20.0)
    assert v.ev_ebitda == pytest.approx(12.0)
    assert v.peg == pytest.approx(2.0)


def test_negative_earnings_gives_no_pe():
    v = compute_valuation(_snapshot(trailing_eps=-3.0))
    assert v.pe is None
    assert v.peg is None
    assert v.ps is not None


def test_no_data_at_all():
    v = compute_valuation(
        FundamentalsSnapshot(symbol="TEST", as_of=date(2026, 7, 10))
    )
    assert v.grade is None
    assert v.score is None
    assert "insufficient" in v.summary


def test_history_percentile_expensive_end():
    # Flat EPS of 5, price ramping 50 -> 100: current P/E 20 is the top of its range.
    months = pd.date_range("2021-01-31", periods=60, freq="ME")
    monthly_close = pd.Series(
        [50 + i for i in range(60)], index=months, dtype=float
    )
    eps = pd.Series([5.0] * 5, index=pd.date_range("2021-12-31", periods=5, freq="YE"))
    pct = _history_percentile(monthly_close, eps, current_ratio=110 / 5)
    assert pct is not None
    assert pct > 95


def test_history_percentile_cheap_end():
    months = pd.date_range("2021-01-31", periods=60, freq="ME")
    monthly_close = pd.Series(
        [100 - i for i in range(60)], index=months, dtype=float
    )
    eps = pd.Series([5.0] * 5, index=pd.date_range("2021-12-31", periods=5, freq="YE"))
    pct = _history_percentile(monthly_close, eps, current_ratio=40 / 5)
    assert pct is not None
    assert pct < 5


def test_history_percentile_needs_enough_months():
    months = pd.date_range("2025-01-31", periods=6, freq="ME")
    monthly_close = pd.Series(range(6), index=months, dtype=float)
    eps = pd.Series([5.0], index=pd.date_range("2024-12-31", periods=1, freq="YE"))
    assert _history_percentile(monthly_close, eps, current_ratio=10.0) is None


def test_grade_uses_percentiles_and_peg():
    months = pd.date_range("2021-01-31", periods=60, freq="ME")
    eps = pd.Series([5.0] * 5, index=pd.date_range("2021-12-31", periods=5, freq="YE"))

    rich = _snapshot(
        price=200.0,
        trailing_eps=5.0,
        earnings_growth=0.05,
        monthly_close=pd.Series([50 + 2 * i for i in range(60)], index=months, dtype=float),
        eps_history=eps,
    )
    v_rich = compute_valuation(rich)
    assert v_rich.grade is not None
    assert v_rich.grade.startswith(("D", "F", "C-"))

    cheap = _snapshot(
        price=40.0,
        trailing_eps=5.0,
        earnings_growth=0.20,
        monthly_close=pd.Series([200 - 2 * i for i in range(60)], index=months, dtype=float),
        eps_history=eps,
    )
    v_cheap = compute_valuation(cheap)
    assert v_cheap.grade is not None
    assert v_cheap.grade.startswith("A")


def test_summary_mentions_pe_and_peg():
    v = compute_valuation(_snapshot())
    assert "P/E 20" in v.summary
    assert "PEG 2.0" in v.summary
