"""Tests for scorecard assembly, rendering, and the analyze CLI command."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from typer.testing import CliRunner

from sentinel.analyze.analysis import build_analysis
from sentinel.analyze.render import render_analysis
from sentinel.cli import app
from sentinel.fundamental.valuation import FundamentalsSnapshot

runner = CliRunner()


def _snapshot() -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        symbol="TEST",
        as_of=date(2026, 7, 10),
        company_name="Test Corp",
        sector="Technology",
        industry="Software",
        price=100.0,
        market_cap=1_200_000_000_000.0,
        shares_outstanding=12_000_000_000.0,
        trailing_eps=5.0,
        revenue_ttm=500_000_000_000.0,
        fcf_ttm=50_000_000_000.0,
        ebitda_ttm=100_000_000_000.0,
        net_debt=0.0,
        earnings_growth=0.10,
    )


def test_build_analysis_full(synthetic_prices):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=_snapshot())
    assert analysis.symbol == "TEST"
    assert analysis.company_name == "Test Corp"
    assert analysis.price_history is not None
    assert analysis.valuation is not None
    assert analysis.valuation.pe == pytest.approx(20.0)
    # Valuation has no history series here, so no percentile grade; composite
    # falls back to the rows that did score.
    assert analysis.price_history.grade is not None
    assert analysis.composite_grade is not None


def test_build_analysis_prices_only(synthetic_prices):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=None)
    assert analysis.valuation is None
    assert analysis.price_history is not None
    assert analysis.composite_grade is not None


def test_build_analysis_no_inputs():
    analysis = build_analysis("test", prices=None, snapshot=None)
    assert analysis.price_history is None
    assert analysis.valuation is None
    assert analysis.composite_grade is None


def test_short_history_noted(synthetic_prices):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=None)
    assert any("price history covers" in n for n in analysis.notes)


def test_render_smoke(synthetic_prices, capsys):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=_snapshot())
    render_analysis(analysis)
    out = capsys.readouterr().out
    assert "TEST" in out
    assert "Price hist" in out
    assert "pending" in out
    assert "Composite" in out


def test_render_detail_views(synthetic_prices, capsys):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=_snapshot())
    render_analysis(analysis, detail="price")
    assert "Price history detail" in capsys.readouterr().out
    render_analysis(analysis, detail="valuation")
    out = capsys.readouterr().out
    assert "Valuation detail" in out
    assert "EV/EBITDA" in out


class _FakeStore:
    def __init__(self, prices: pd.DataFrame):
        self._prices = prices
        self.written = None

    def read_prices(self, symbol: str) -> pd.DataFrame:
        return self._prices

    def write_prices(self, symbol: str, df: pd.DataFrame) -> int:
        self.written = df
        return len(df)


def test_cli_analyze(monkeypatch, synthetic_prices):
    monkeypatch.setattr(
        "sentinel.storage.get_store", lambda: _FakeStore(synthetic_prices)
    )
    monkeypatch.setattr(
        "sentinel.fundamental.valuation.fetch_snapshot", lambda symbol: _snapshot()
    )
    result = runner.invoke(app, ["analyze", "TEST"])
    assert result.exit_code == 0
    assert "Composite" in result.stdout


def test_cli_analyze_offline(monkeypatch, synthetic_prices):
    monkeypatch.setattr(
        "sentinel.storage.get_store", lambda: _FakeStore(synthetic_prices)
    )
    result = runner.invoke(app, ["analyze", "TEST", "--offline"])
    assert result.exit_code == 0
    assert "Price hist" in result.stdout


def test_cli_analyze_bad_detail(monkeypatch, synthetic_prices):
    monkeypatch.setattr(
        "sentinel.storage.get_store", lambda: _FakeStore(synthetic_prices)
    )
    result = runner.invoke(app, ["analyze", "TEST", "--offline", "--detail", "nope"])
    assert result.exit_code == 1


def test_cli_analyze_no_prices_offline(monkeypatch):
    monkeypatch.setattr(
        "sentinel.storage.get_store", lambda: _FakeStore(pd.DataFrame())
    )
    result = runner.invoke(app, ["analyze", "TEST", "--offline"])
    assert result.exit_code == 1
