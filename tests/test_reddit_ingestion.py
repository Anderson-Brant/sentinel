"""Reddit ingestion orchestration tests (uses a fake fetcher — no network)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from sentinel.config import IngestionRedditConfig
from sentinel.ingestion.reddit import RedditPost, ingest_posts
from sentinel.storage.duckdb_store import DuckDBStore


class _FakeFetcher:
    """Injectable fetcher that returns a fixed list of posts — no praw involved."""

    def __init__(self, posts):
        self.posts = posts
        self.calls = []

    def fetch(self, subreddits, *, limit):
        self.calls.append((list(subreddits), limit))
        return list(self.posts)


def _post(pid, title, body="", subreddit="wallstreetbets", score=10, comments=5):
    return RedditPost(
        post_id=pid,
        created_ts=datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
        subreddit=subreddit,
        author="someone",
        title=title,
        body=body,
        score=score,
        num_comments=comments,
        url=f"https://reddit.com/{pid}",
    )


def test_ingest_posts_writes_posts_and_mentions(tmp_path):
    store = DuckDBStore(path=tmp_path / "r.duckdb")
    fetcher = _FakeFetcher(
        [
            _post("p1", "$TSLA to the moon"),
            _post("p2", "AAPL earnings tomorrow", body="Holding $AAPL through earnings"),
            _post("p3", "just a vent thread", body="no tickers"),
        ]
    )
    counts = ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["wallstreetbets"], max_posts_per_run=100),
        symbol_whitelist={"AAPL", "TSLA"},
        fetcher=fetcher,
    )
    assert counts["fetched"] == 3
    assert counts["posts_written"] == 3
    # p1 → TSLA. p2 → AAPL (cashtag + whitelist). p3 → nothing.
    assert counts["mentions_written"] >= 2

    # Verify reads.
    tsla_posts = store.read_reddit_posts_for_symbol("TSLA")
    aapl_posts = store.read_reddit_posts_for_symbol("AAPL")
    assert set(tsla_posts["post_id"]) == {"p1"}
    assert set(aapl_posts["post_id"]) == {"p2"}

    # Fetcher was called with the configured subreddits + limit.
    assert fetcher.calls == [(["wallstreetbets"], 100)]


def test_ingest_posts_empty_fetch_is_noop(tmp_path):
    store = DuckDBStore(path=tmp_path / "r.duckdb")
    counts = ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["x"], max_posts_per_run=10),
        fetcher=_FakeFetcher([]),
    )
    assert counts == {"fetched": 0, "posts_written": 0, "mentions_written": 0}


def test_reddit_posts_roundtrip_with_upsert(tmp_path):
    """Writing the same post_id twice should not duplicate rows."""
    store = DuckDBStore(path=tmp_path / "r.duckdb")
    posts_df = pd.DataFrame(
        [
            _post("p1", "v1", score=5).to_dict(),
            _post("p2", "hello $TSLA").to_dict(),
        ]
    )
    assert store.write_reddit_posts(posts_df) == 2

    # Rewrite p1 with a higher score — row should be *updated*, not duplicated.
    updated = pd.DataFrame([_post("p1", "v2", score=999).to_dict()])
    store.write_reddit_posts(updated)

    mentions_df = pd.DataFrame([{"post_id": "p2", "symbol": "TSLA"}])
    store.write_mentions(mentions_df, source="reddit")

    tsla = store.read_reddit_posts_for_symbol("TSLA")
    assert len(tsla) == 1
    assert tsla.iloc[0]["post_id"] == "p2"
