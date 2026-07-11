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

_YEARS = pd.date_range("2021-12-31", periods=4, freq="YE")


def _snapshot() -> FundamentalsSnapshot:
    revenue = pd.Series([380e9, 400e9, 440e9, 500e9], index=_YEARS)
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
        revenue_history=revenue,
        gross_profit_history=revenue * 0.55,
        operating_income_history=revenue * 0.28,
        net_income_history=revenue * 0.22,
        equity_history=pd.Series([200e9, 220e9, 250e9, 280e9], index=_YEARS),
        debt_history=pd.Series([100e9] * 4, index=_YEARS),
        cash_history=pd.Series([150e9] * 4, index=_YEARS),
    )


def _insider_txns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-05-01")],
            "shares": [100_000_000.0, 20_000_000.0],
            "text": ["Purchase", "Sale"],
        }
    )


def test_build_analysis_full(synthetic_prices):
    analysis = build_analysis(
        "test",
        prices=synthetic_prices,
        snapshot=_snapshot(),
        insider_txns=_insider_txns(),
    )
    assert analysis.symbol == "TEST"
    assert analysis.company_name == "Test Corp"
    assert analysis.price_history is not None and analysis.price_history.grade
    assert analysis.valuation is not None
    assert analysis.valuation.pe == pytest.approx(20.0)
    assert analysis.quality is not None and analysis.quality.grade
    assert analysis.competitive is not None and analysis.competitive.grade
    assert analysis.insiders is not None and analysis.insiders.grade
    assert analysis.composite_grade is not None
    assert len(analysis.scored_rows()) >= 4


def test_build_analysis_prices_only(synthetic_prices):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=None)
    assert analysis.valuation is None
    assert analysis.quality is None
    assert analysis.insiders is None
    assert analysis.competitive is None
    assert analysis.price_history is not None
    assert analysis.composite_grade is not None


def test_build_analysis_no_inputs():
    analysis = build_analysis("test", prices=None, snapshot=None)
    assert analysis.price_history is None
    assert analysis.composite_grade is None


def test_no_insider_txns_leaves_row_ungraded(synthetic_prices):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=_snapshot())
    assert analysis.insiders is not None
    assert analysis.insiders.grade is None
    assert "no insider filings" in analysis.insiders.summary


def test_short_history_noted(synthetic_prices):
    analysis = build_analysis("test", prices=synthetic_prices, snapshot=None)
    assert any("price history covers" in n for n in analysis.notes)


def test_render_smoke(synthetic_prices, capsys):
    analysis = build_analysis(
        "test",
        prices=synthetic_prices,
        snapshot=_snapshot(),
        insider_txns=_insider_txns(),
        related_tickers=["AAA", "BBB"],
    )
    render_analysis(analysis)
    out = capsys.readouterr().out
    assert "TEST" in out
    for row in ("Quality", "Valuation", "Price hist", "Insiders", "Competitive"):
        assert row in out
    assert "Composite" in out
    assert "Related: AAA, BBB" in out


def test_render_detail_views(synthetic_prices, capsys):
    analysis = build_analysis(
        "test",
        prices=synthetic_prices,
        snapshot=_snapshot(),
        insider_txns=_insider_txns(),
    )
    for view, title in (
        ("price", "Price history detail"),
        ("valuation", "Valuation detail"),
        ("quality", "Quality detail"),
        ("insiders", "Insiders detail"),
        ("competitive", "Competitive detail"),
    ):
        render_analysis(analysis, detail=view)
        assert title in capsys.readouterr().out


class _FakeStore:
    def __init__(self, prices: pd.DataFrame):
        self._prices = prices
        self.written = None

    def read_prices(self, symbol: str) -> pd.DataFrame:
        return self._prices

    def write_prices(self, symbol: str, df: pd.DataFrame) -> int:
        self.written = df
        return len(df)

    def list_symbols(self) -> list[str]:
        return []


def test_cli_analyze(monkeypatch, synthetic_prices):
    monkeypatch.setattr(
        "sentinel.storage.get_store", lambda: _FakeStore(synthetic_prices)
    )
    monkeypatch.setattr(
        "sentinel.fundamental.valuation.fetch_snapshot", lambda symbol: _snapshot()
    )
    monkeypatch.setattr(
        "sentinel.fundamental.insiders.fetch_insider_transactions",
        lambda symbol: _insider_txns(),
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
