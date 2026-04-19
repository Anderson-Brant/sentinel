"""Sentiment scoring + aggregation tests.

These tests use a fake scorer so we don't need the real VADER dependency to
pass — the goal is to pin down the shape and math of our aggregation, not to
re-test VADER itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sentinel.features.sentiment import (
    SENTIMENT_FEATURE_COLS,
    score_posts,
    sentiment_features_for_symbol,
)
from sentinel.ingestion.reddit import RedditPost, ingest_posts
from sentinel.storage.duckdb_store import DuckDBStore


class _FakeScorer:
    """Returns compound = +0.5 for bullish keywords, -0.5 for bearish, 0 otherwise."""

    def polarity_scores(self, text: str) -> dict[str, float]:
        t = (text or "").lower()
        if "moon" in t or "bull" in t:
            return {"compound": 0.5, "pos": 0.6, "neg": 0.0, "neu": 0.4}
        if "crash" in t or "dump" in t:
            return {"compound": -0.5, "pos": 0.0, "neg": 0.6, "neu": 0.4}
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def test_score_posts_runs_scorer_on_title_and_body():
    df = pd.DataFrame(
        [
            {"post_id": "p1", "title": "$TSLA to the moon", "body": ""},
            {"post_id": "p2", "title": "CRASH incoming", "body": "market will dump"},
            {"post_id": "p3", "title": "boring update", "body": "no opinion"},
        ]
    )
    out = score_posts(df, scorer=_FakeScorer())
    assert len(out) == 3
    assert set(out.columns) == {
        "post_id",
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
    }
    by_id = {r.post_id: r for r in out.itertuples()}
    assert by_id["p1"].sentiment_compound == 0.5
    assert by_id["p2"].sentiment_compound == -0.5
    assert by_id["p3"].sentiment_compound == 0.0


def test_score_posts_empty_returns_empty_with_right_columns():
    out = score_posts(pd.DataFrame(), scorer=_FakeScorer())
    assert out.empty
    assert "sentiment_compound" in out.columns


def _mk_post(pid: str, title: str, body: str, *, day: int, score=10, comments=5):
    ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc) + timedelta(days=day)
    return RedditPost(
        post_id=pid,
        created_ts=ts,
        subreddit="wallstreetbets",
        author="u/x",
        title=title,
        body=body,
        score=score,
        num_comments=comments,
        url=f"https://r/{pid}",
    )


class _PostFetcher:
    def __init__(self, posts):
        self.posts = posts

    def fetch(self, subreddits, *, limit):
        return list(self.posts)


def test_sentiment_features_per_day_rollup_counts_and_signs(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")

    posts = [
        _mk_post("a1", "$TSLA to the moon", "", day=0),
        _mk_post("a2", "$TSLA bull run", "", day=0),
        _mk_post("a3", "$TSLA crash incoming", "", day=2, score=50, comments=50),
        _mk_post("b1", "$AAPL whatever", "", day=1),
    ]
    from sentinel.config import IngestionRedditConfig

    ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["x"], max_posts_per_run=100),
        symbol_whitelist={"TSLA", "AAPL"},
        fetcher=_PostFetcher(posts),
    )

    # Score with the fake scorer.
    with store._connect() as con:  # noqa: SLF001
        raw = con.execute("SELECT post_id, title, body FROM reddit_posts").fetchdf()
    scored = score_posts(raw, scorer=_FakeScorer())
    store.update_reddit_sentiment(scored)

    feats = sentiment_features_for_symbol(store, "TSLA")

    # Shape: the 3 TSLA posts span day 0 and day 2 → 2 rows.
    assert set(feats.columns) == set(SENTIMENT_FEATURE_COLS)
    assert len(feats) == 2

    day0 = feats.iloc[0]
    day2 = feats.iloc[-1]

    # Day 0: 2 bullish posts → mention_count 2, mean compound 0.5.
    assert day0["reddit_mention_count"] == 2
    assert day0["reddit_sentiment_mean"] == 0.5

    # Day 2: 1 bearish post, mean compound -0.5.
    assert day2["reddit_mention_count"] == 1
    assert day2["reddit_sentiment_mean"] == -0.5


def test_sentiment_features_reindex_fills_gap_days(tmp_path):
    """When an index is provided, no-post days should appear as zero mentions."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")

    posts = [_mk_post("x1", "$TSLA bull", "", day=0)]
    from sentinel.config import IngestionRedditConfig

    ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["x"], max_posts_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_PostFetcher(posts),
    )
    with store._connect() as con:  # noqa: SLF001
        raw = con.execute("SELECT post_id, title, body FROM reddit_posts").fetchdf()
    store.update_reddit_sentiment(score_posts(raw, scorer=_FakeScorer()))

    idx = pd.date_range("2026-04-01", periods=5, freq="D")
    feats = sentiment_features_for_symbol(store, "TSLA", index=idx)

    assert list(feats.index) == list(idx)
    # Day 0 has 1 mention. Others 0.
    assert feats.iloc[0]["reddit_mention_count"] == 1
    assert (feats.iloc[1:]["reddit_mention_count"] == 0).all()


def test_sentiment_features_no_posts_returns_zeros_when_reindexed(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    idx = pd.date_range("2026-04-01", periods=3, freq="D")
    feats = sentiment_features_for_symbol(store, "UNKNOWN", index=idx)
    assert list(feats.index) == list(idx)
    assert (feats["reddit_mention_count"] == 0).all()


def test_engagement_weighted_mean_moves_toward_high_score_post(tmp_path):
    """A heavily upvoted bearish post should drag the weighted mean negative
    even when there are more low-engagement neutral posts.
    """
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    posts = [
        _mk_post("a", "nothing special", "", day=0, score=0, comments=0),
        _mk_post("b", "nothing special", "", day=0, score=0, comments=0),
        # One heavily-engaged bearish post on the same day.
        _mk_post("c", "$TSLA crash", "", day=0, score=1000, comments=500),
    ]
    from sentinel.config import IngestionRedditConfig

    ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["x"], max_posts_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_PostFetcher(posts),
    )
    with store._connect() as con:  # noqa: SLF001
        raw = con.execute("SELECT post_id, title, body FROM reddit_posts").fetchdf()
    store.update_reddit_sentiment(score_posts(raw, scorer=_FakeScorer()))

    feats = sentiment_features_for_symbol(store, "TSLA")
    # Only 1 post mentions TSLA ("c"). But verify wmean == mean when N=1.
    assert np.isclose(feats.iloc[0]["reddit_sentiment_wmean"], -0.5)
