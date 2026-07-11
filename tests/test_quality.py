"""Tests for sentinel.fundamental.quality (pure scoring, no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sentinel.fundamental.quality import compute_quality
from sentinel.fundamental.valuation import FundamentalsSnapshot

_YEARS = pd.date_range("2021-12-31", periods=4, freq="YE")


def _snapshot(**overrides) -> FundamentalsSnapshot:
    revenue = pd.Series([100.0, 115.0, 132.0, 152.0], index=_YEARS)
    base = dict(
        symbol="TEST",
        as_of=date(2026, 7, 11),
        net_debt=-50.0,
        ebitda_ttm=40.0,
        revenue_history=revenue,
        gross_profit_history=revenue * 0.60,
        operating_income_history=revenue * pd.Series([0.20, 0.22, 0.24, 0.25], index=_YEARS),
        net_income_history=revenue * 0.15,
        equity_history=pd.Series([80.0, 90.0, 100.0, 110.0], index=_YEARS),
        debt_history=pd.Series([20.0] * 4, index=_YEARS),
        cash_history=pd.Series([60.0] * 4, index=_YEARS),
    )
    base.update(overrides)
    return FundamentalsSnapshot(**base)


def test_strong_business_grades_well():
    q = compute_quality(_snapshot())
    assert q.grade is not None
    assert q.grade.startswith(("A", "B+"))
    assert q.gross_margin == pytest.approx(0.60)
    assert q.revenue_growth == pytest.approx(0.15, abs=0.005)
    assert q.net_debt_ebitda is not None and q.net_debt_ebitda < 0
    assert "ROIC" in q.summary
    assert "Net cash" in q.summary


def test_roic_computation():
    q = compute_quality(_snapshot())
    # oi 38.0 after 21% tax over invested capital 110 + 20 - 60 = 70.
    assert q.roic == pytest.approx(38.0 * 0.79 / 70.0, rel=1e-6)


def test_weak_business_grades_poorly():
    revenue = pd.Series([100.0, 98.0, 92.0, 88.0], index=_YEARS)
    q = compute_quality(
        _snapshot(
            revenue_history=revenue,
            gross_profit_history=revenue * pd.Series([0.20, 0.16, 0.13, 0.10], index=_YEARS),
            operating_income_history=revenue * pd.Series([0.06, 0.04, 0.01, -0.01], index=_YEARS),
            equity_history=pd.Series([100.0] * 4, index=_YEARS),
            debt_history=pd.Series([300.0] * 4, index=_YEARS),
            cash_history=pd.Series([10.0] * 4, index=_YEARS),
            net_debt=290.0,
            ebitda_ttm=50.0,
        )
    )
    strong = compute_quality(_snapshot())
    assert q.grade is not None and strong.score is not None and q.score is not None
    assert q.score < strong.score
    assert q.grade.startswith(("C", "D", "F"))


def test_no_data_is_ungraded():
    q = compute_quality(FundamentalsSnapshot(symbol="TEST", as_of=date(2026, 7, 11)))
    assert q.grade is None
    assert q.score is None
    assert "insufficient" in q.summary


def test_partial_data_still_scores():
    q = compute_quality(
        FundamentalsSnapshot(
            symbol="TEST",
            as_of=date(2026, 7, 11),
            revenue_history=pd.Series([100.0, 112.0, 125.0, 140.0], index=_YEARS),
        )
    )
    assert q.grade is not None
    assert q.roic is None
    assert q.revenue_growth is not None
