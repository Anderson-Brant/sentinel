"""Tests for ticker mention extraction."""

from __future__ import annotations

from sentinel.ingestion.mentions import (
    extract_mentions_for_records,
    extract_tickers,
)


def test_cashtag_is_always_extracted():
    text = "bought some $TSLA and $aapl today, feeling good"
    assert extract_tickers(text) == {"TSLA", "AAPL"}


def test_cashtag_with_suffix():
    text = "$BRK.B and $BTC-USD up big"
    assert extract_tickers(text) == {"BRK.B", "BTC-USD"}


def test_no_whitelist_ignores_bare_uppercase():
    """Without a whitelist, bare uppercase words are *not* matched — too noisy."""
    text = "AAPL earnings tomorrow, the CEO said YOLO"
    assert extract_tickers(text) == set()


def test_whitelist_catches_uncashed_mentions():
    text = "AAPL earnings tomorrow, TSLA also reports"
    got = extract_tickers(text, whitelist={"AAPL", "TSLA", "NVDA"})
    assert got == {"AAPL", "TSLA"}


def test_whitelist_plus_noise_list_filters_common_acronyms():
    """Even if a CEO-like acronym is in the whitelist, the noise list rejects it."""
    text = "The CEO is bullish"
    # CEO is in _NOISE, so even if someone silly put it in the whitelist, it should not match.
    assert extract_tickers(text, whitelist={"CEO", "AAPL"}) == set()


def test_empty_text():
    assert extract_tickers("") == set()
    assert extract_tickers(None) == set()  # type: ignore[arg-type]


def test_extract_mentions_for_records_flattens():
    records = [
        {"id": "p1", "title": "$TSLA to the moon", "body": ""},
        {"id": "p2", "title": "nothing here", "body": "$AAPL and $TSLA"},
        {"id": "p3", "title": "quiet post", "body": "no tickers today"},
    ]
    rows = extract_mentions_for_records(records)
    got = {(r["post_id"], r["symbol"]) for r in rows}
    assert got == {("p1", "TSLA"), ("p2", "AAPL"), ("p2", "TSLA")}


def test_dedups_multiple_mentions_in_one_post():
    text = "$TSLA $TSLA $tsla bullish AF"
    assert extract_tickers(text) == {"TSLA"}
