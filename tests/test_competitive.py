"""Tests for sentinel.fundamental.competitive (pure scoring, no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sentinel.fundamental.competitive import compute_competitive, related_by_correlation
from sentinel.fundamental.valuation import FundamentalsSnapshot

_YEARS = pd.date_range("2021-12-31", periods=4, freq="YE")


def _snapshot(growth: float, margin: float, sector: str = "Technology") -> FundamentalsSnapshot:
    revenue = pd.Series([100.0 * (1 + growth) ** i for i in range(4)], index=_YEARS)
    return FundamentalsSnapshot(
        symbol="TEST",
        as_of=date(2026, 7, 11),
        sector=sector,
        revenue_history=revenue,
        operating_income_history=revenue * margin,
    )


def test_outgrowing_sector_grades_well():
    c = compute_competitive(_snapshot(growth=0.30, margin=0.35))
    assert c.grade is not None
    assert c.grade.startswith("A")
    assert c.revenue_growth == pytest.approx(0.30, abs=0.005)
    assert "vs sector" in c.summary


def test_lagging_sector_grades_poorly():
    lag = compute_competitive(_snapshot(growth=-0.05, margin=0.05))
    lead = compute_competitive(_snapshot(growth=0.30, margin=0.35))
    assert lag.score is not None and lead.score is not None
    assert lag.score < lead.score
    assert lag.grade is not None
    assert lag.grade.startswith(("C", "D", "F"))


def test_unknown_sector_uses_default_baseline():
    c = compute_competitive(_snapshot(growth=0.10, margin=0.15, sector="Nonsense"))
    assert c.sector_growth == pytest.approx(0.05)
    assert c.grade is not None


def test_no_data_is_ungraded():
    c = compute_competitive(FundamentalsSnapshot(symbol="TEST", as_of=date(2026, 7, 11)))
    assert c.grade is None
    assert "insufficient" in c.summary


def _price_frame(seed: int, base: np.ndarray | None = None, noise: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    rets = rng.normal(0, 0.01, len(dates)) if base is None else base + rng.normal(0, noise * 0.1, len(dates))
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(rets))}, index=dates)


def test_related_by_correlation_ranks_the_twin_first():
    rng = np.random.default_rng(7)
    common = rng.normal(0.0005, 0.01, 300)
    prices = {
        "AAA": _price_frame(1, base=common),
        "BBB": _price_frame(2, base=common),          # near-twin of AAA
        "CCC": _price_frame(3),                        # unrelated
        "DDD": _price_frame(4),                        # unrelated
    }
    related = related_by_correlation(prices, "AAA", top_n=2)
    assert related[0] == "BBB"


def test_related_requires_overlap():
    prices = {
        "AAA": _price_frame(1),
        "BBB": _price_frame(2).iloc[:30],  # too short
    }
    assert related_by_correlation(prices, "AAA") == []
    assert related_by_correlation(prices, "ZZZ") == []
