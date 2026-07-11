"""Tests for sentinel.fundamental.insiders (pure scoring, no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sentinel.fundamental.insiders import compute_insiders

AS_OF = date(2026, 7, 11)
SHARES = 1_000_000.0


def _txns(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d, _, _ in rows],
            "shares": [s for _, s, _ in rows],
            "text": [t for _, _, t in rows],
        }
    )


def test_net_buying_grades_high():
    txns = _txns(
        [
            ("2026-06-01", 4000, "Purchase at price 10.00 per share"),
            ("2026-05-01", 3000, "Buy"),
            ("2026-04-01", 1000, "Sale at price 12.00 per share"),
        ]
    )
    score = compute_insiders(txns, symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF)
    assert score.net_pct_6m is not None
    assert score.net_pct_6m > 0.005
    assert score.grade == "A"
    assert "accumulation" in score.summary
    assert score.n_buys_6m == 2
    assert score.n_sells_6m == 1


def test_net_selling_grades_low():
    txns = _txns(
        [
            ("2026-06-15", 8000, "Sale at price 50.00 per share"),
            ("2026-05-15", 4000, "Sale"),
        ]
    )
    score = compute_insiders(txns, symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF)
    assert score.net_pct_6m is not None
    assert score.net_pct_6m < -0.01
    assert score.grade == "D+"
    assert "selling" in score.summary


def test_old_transactions_fall_out_of_window():
    txns = _txns([("2024-01-01", 50_000, "Sale")])
    score = compute_insiders(txns, symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF)
    assert score.grade is None
    assert "no classifiable" in score.summary


def test_twelve_month_window_is_wider():
    txns = _txns(
        [
            ("2026-06-01", 1000, "Purchase"),
            ("2025-09-01", 2000, "Purchase"),
        ]
    )
    score = compute_insiders(txns, symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF)
    assert score.net_pct_6m is not None and score.net_pct_12m is not None
    assert score.net_pct_12m > score.net_pct_6m


def test_no_data_is_ungraded():
    score = compute_insiders(None, symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF)
    assert score.grade is None
    assert "no insider filings" in score.summary

    empty = compute_insiders(
        pd.DataFrame(), symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF
    )
    assert empty.grade is None


def test_unclassifiable_text_is_ignored():
    txns = _txns(
        [
            ("2026-06-01", 5000, "Stock Award (Grant)"),
            ("2026-06-02", 1000, "Purchase"),
        ]
    )
    score = compute_insiders(txns, symbol="TEST", shares_outstanding=SHARES, as_of=AS_OF)
    assert score.n_buys_6m == 1
    assert score.net_pct_6m is not None
    assert score.net_pct_6m == 1000 / SHARES
