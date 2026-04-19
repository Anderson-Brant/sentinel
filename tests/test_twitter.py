"""Twitter ingestion + tweets storage + sentiment-roll-up tests.

Uses a fake TweetFetcher + FakeScorer so none of these tests need tweepy,
vaderSentiment, or network access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from sentinel.config import IngestionTwitterConfig
from sentinel.features.sentiment import (
    MENTION_COUNT_COLS,
    REDDIT_SENTIMENT_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    TWITTER_SENTIMENT_FEATURE_COLS,
    score_tweets,
    sentiment_features_for_symbol,
)
from sentinel.ingestion.reddit import RedditPost, ingest_posts
from sentinel.ingestion.twitter import (
    Tweet,
    build_cashtag_query,
    ingest_tweets,
)
from sentinel.storage.duckdb_store import DuckDBStore

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeScorer:
    """Returns compound = +0.5 for bullish keywords, -0.5 for bearish, else 0."""

    def polarity_scores(self, text: str) -> dict[str, float]:
        t = (text or "").lower()
        if "moon" in t or "bull" in t:
            return {"compound": 0.5, "pos": 0.6, "neg": 0.0, "neu": 0.4}
        if "crash" in t or "dump" in t:
            return {"compound": -0.5, "pos": 0.0, "neg": 0.6, "neu": 0.4}
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def _mk_tweet(
    tid: str,
    text: str,
    *,
    day: int,
    likes: int = 5,
    retweets: int = 1,
    replies: int = 0,
) -> Tweet:
    ts = datetime(2026, 4, 1, 12, 0, tzinfo=UTC) + timedelta(days=day)
    return Tweet(
        tweet_id=tid,
        created_ts=ts,
        author_id=f"u{tid}",
        author_username=f"user_{tid}",
        text=text,
        lang="en",
        retweet_count=retweets,
        reply_count=replies,
        like_count=likes,
        quote_count=0,
        impression_count=100,
    )


def _mk_post(pid: str, title: str, *, day: int, score=10, comments=5) -> RedditPost:
    ts = datetime(2026, 4, 1, 12, 0, tzinfo=UTC) + timedelta(days=day)
    return RedditPost(
        post_id=pid,
        created_ts=ts,
        subreddit="wallstreetbets",
        author="u/x",
        title=title,
        body="",
        score=score,
        num_comments=comments,
        url=f"https://r/{pid}",
    )


class _FakeTweetFetcher:
    def __init__(self, tweets: list[Tweet]) -> None:
        self.tweets = tweets
        self.last_query: str | None = None
        self.last_limit: int | None = None

    def fetch(self, query: str, *, limit: int) -> list[Tweet]:
        self.last_query = query
        self.last_limit = limit
        return list(self.tweets)


class _FakePostFetcher:
    def __init__(self, posts: list[RedditPost]) -> None:
        self.posts = posts

    def fetch(self, subreddits, *, limit):
        return list(self.posts)


# ---------------------------------------------------------------------------
# build_cashtag_query
# ---------------------------------------------------------------------------


def test_build_cashtag_query_normalizes_and_sorts():
    q = build_cashtag_query(["tsla", "aapl", "SPY", "AAPL"])
    # Sorted, uppercased, deduped, with $ prefix, joined by OR.
    assert q == "($AAPL OR $SPY OR $TSLA) -is:retweet lang:en"


def test_build_cashtag_query_rejects_empty():
    with pytest.raises(ValueError):
        build_cashtag_query([])


# ---------------------------------------------------------------------------
# Orchestration: ingest_tweets
# ---------------------------------------------------------------------------


def test_ingest_tweets_writes_tweets_and_mentions(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    fetcher = _FakeTweetFetcher(
        [
            _mk_tweet("t1", "$TSLA to the moon today", day=0),
            _mk_tweet("t2", "$TSLA looking bullish", day=0),
            _mk_tweet("t3", "Markets crash coming $TSLA $AAPL", day=1),
            _mk_tweet("t4", "unrelated noise about cats", day=1),
        ]
    )
    counts = ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=100),
        symbol_whitelist={"TSLA", "AAPL"},
        fetcher=fetcher,
    )
    assert counts["fetched"] == 4
    assert counts["tweets_written"] == 4
    # Mentions: t1, t2, t3(TSLA), t3(AAPL) → 4 rows. t4 mentions nothing.
    assert counts["mentions_written"] == 4

    # Query was built from whitelist, not passed raw.
    assert fetcher.last_query == "($AAPL OR $TSLA) -is:retweet lang:en"

    # TSLA-tagged tweets include t1, t2, t3 (3 rows).
    tsla = store.read_tweets_for_symbol("TSLA")
    assert set(tsla["tweet_id"]) == {"t1", "t2", "t3"}
    # AAPL gets only t3.
    aapl = store.read_tweets_for_symbol("AAPL")
    assert set(aapl["tweet_id"]) == {"t3"}


def test_ingest_tweets_is_idempotent(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    fetcher = _FakeTweetFetcher(
        [_mk_tweet("t1", "$TSLA bull", day=0), _mk_tweet("t2", "$TSLA moon", day=0)]
    )
    c1 = ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=100),
        symbol_whitelist={"TSLA"},
        fetcher=fetcher,
    )
    c2 = ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=100),
        symbol_whitelist={"TSLA"},
        fetcher=fetcher,
    )
    assert c1["tweets_written"] == c2["tweets_written"] == 2
    # Re-running doesn't duplicate rows.
    all_tweets = store.read_all_tweets()
    assert len(all_tweets) == 2


def test_ingest_tweets_requires_query_or_whitelist(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    with pytest.raises(ValueError):
        ingest_tweets(
            store=store,
            twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=100),
            symbol_whitelist=None,
            fetcher=_FakeTweetFetcher([]),
        )


def test_ingest_tweets_accepts_explicit_query(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    fetcher = _FakeTweetFetcher([_mk_tweet("t1", "$TSLA bull", day=0)])
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=50),
        symbol_whitelist={"TSLA"},
        query="$TSLA lang:en",
        fetcher=fetcher,
    )
    assert fetcher.last_query == "$TSLA lang:en"
    assert fetcher.last_limit == 50


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


def test_tweets_round_trip_preserves_engagement_fields(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    fetcher = _FakeTweetFetcher(
        [_mk_tweet("t99", "$TSLA bull", day=0, likes=42, retweets=7, replies=3)]
    )
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=fetcher,
    )
    row = store.read_tweets_for_symbol("TSLA").iloc[0]
    assert row["like_count"] == 42
    assert row["retweet_count"] == 7
    assert row["reply_count"] == 3
    # Sentiment hasn't been scored yet — it should be NULL.
    assert pd.isna(row["sentiment_compound"])


def test_update_tweet_sentiment_backfills_scores(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    fetcher = _FakeTweetFetcher([_mk_tweet("t1", "$TSLA moon", day=0)])
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=fetcher,
    )
    raw = store.read_all_tweets()
    scored = score_tweets(raw, scorer=_FakeScorer())
    n = store.update_tweet_sentiment(scored)
    assert n == 1

    row = store.read_tweets_for_symbol("TSLA").iloc[0]
    assert row["sentiment_compound"] == 0.5


# ---------------------------------------------------------------------------
# Sentiment rollups — reddit-only, twitter-only, and both
# ---------------------------------------------------------------------------


def test_sentiment_features_twitter_only(tmp_path):
    """With only tweets ingested, twitter_* cols populate, reddit_* stay zero/NaN."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    tweets = [
        _mk_tweet("t1", "$TSLA moon", day=0, likes=10),
        _mk_tweet("t2", "$TSLA bull", day=0, likes=5),
        _mk_tweet("t3", "$TSLA crash", day=2, likes=100),
    ]
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=100),
        symbol_whitelist={"TSLA"},
        fetcher=_FakeTweetFetcher(tweets),
    )
    raw = store.read_all_tweets()
    store.update_tweet_sentiment(score_tweets(raw, scorer=_FakeScorer()))

    feats = sentiment_features_for_symbol(store, "TSLA")
    # Every canonical column exists, reddit or twitter.
    assert set(feats.columns) == set(SENTIMENT_FEATURE_COLS)
    # Two rows: day 0, day 2.
    assert len(feats) == 2

    day0 = feats.iloc[0]
    assert day0["twitter_mention_count"] == 2
    assert day0["twitter_sentiment_mean"] == 0.5
    # Reddit column is absent from actual data → NaN.
    assert pd.isna(day0["reddit_mention_count"])


def test_sentiment_features_reddit_only_still_works(tmp_path):
    """Reddit-only path still fills reddit_* cols; twitter cols are NaN."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    posts = [_mk_post("p1", "$TSLA moon", day=0)]
    from sentinel.config import IngestionRedditConfig

    ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["x"], max_posts_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_FakePostFetcher(posts),
    )
    with store._connect() as con:  # noqa: SLF001
        raw = con.execute("SELECT post_id, title, body FROM reddit_posts").fetchdf()
    from sentinel.features.sentiment import score_posts

    store.update_reddit_sentiment(score_posts(raw, scorer=_FakeScorer()))

    feats = sentiment_features_for_symbol(store, "TSLA")
    assert set(feats.columns) == set(SENTIMENT_FEATURE_COLS)
    assert feats.iloc[0]["reddit_mention_count"] == 1
    assert feats.iloc[0]["reddit_sentiment_mean"] == 0.5
    # No tweets → twitter cols are NaN.
    assert pd.isna(feats.iloc[0]["twitter_mention_count"])


def test_sentiment_features_both_sources_combine_cleanly(tmp_path):
    """With both sources populated, each platform owns its own columns."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    from sentinel.config import IngestionRedditConfig

    # Reddit: 1 bullish post day 0.
    ingest_posts(
        store=store,
        reddit_cfg=IngestionRedditConfig(subreddits=["x"], max_posts_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_FakePostFetcher([_mk_post("p1", "$TSLA moon", day=0)]),
    )
    with store._connect() as con:  # noqa: SLF001
        raw = con.execute("SELECT post_id, title, body FROM reddit_posts").fetchdf()
    from sentinel.features.sentiment import score_posts

    store.update_reddit_sentiment(score_posts(raw, scorer=_FakeScorer()))

    # Twitter: 1 bearish tweet same day.
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_FakeTweetFetcher([_mk_tweet("t1", "$TSLA crash incoming", day=0)]),
    )
    raw_t = store.read_all_tweets()
    store.update_tweet_sentiment(score_tweets(raw_t, scorer=_FakeScorer()))

    feats = sentiment_features_for_symbol(store, "TSLA")
    row = feats.iloc[0]
    assert row["reddit_mention_count"] == 1
    assert row["reddit_sentiment_mean"] == 0.5       # bullish post
    assert row["twitter_mention_count"] == 1
    assert row["twitter_sentiment_mean"] == -0.5     # bearish tweet


def test_sentiment_features_reindex_zero_fills_both_mention_counts(tmp_path):
    """When reindexed, BOTH mention_count columns fill with 0 on no-data days."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    # Only a single tweet on day 0.
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_FakeTweetFetcher([_mk_tweet("t1", "$TSLA bull", day=0)]),
    )
    raw = store.read_all_tweets()
    store.update_tweet_sentiment(score_tweets(raw, scorer=_FakeScorer()))

    idx = pd.date_range("2026-04-01", periods=5, freq="D")
    feats = sentiment_features_for_symbol(store, "TSLA", index=idx)
    assert list(feats.index) == list(idx)
    # Day 0 → 1 twitter mention, 0 reddit mentions.
    assert feats.iloc[0]["twitter_mention_count"] == 1
    assert feats.iloc[0]["reddit_mention_count"] == 0
    # All subsequent days → 0/0 across both.
    for col in MENTION_COUNT_COLS:
        assert (feats.iloc[1:][col] == 0).all()


def test_engagement_weighting_uses_like_retweet_reply(tmp_path):
    """A heavily-liked bearish tweet should drag the weighted mean negative
    even when competing with low-engagement neutral tweets on the same day."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    tweets = [
        _mk_tweet("n1", "$TSLA boring", day=0, likes=0, retweets=0, replies=0),
        _mk_tweet("n2", "$TSLA boring", day=0, likes=0, retweets=0, replies=0),
        # Heavily engaged bearish tweet.
        _mk_tweet(
            "b1", "$TSLA crash incoming", day=0, likes=1000, retweets=500, replies=100
        ),
    ]
    ingest_tweets(
        store=store,
        twitter_cfg=IngestionTwitterConfig(max_tweets_per_run=10),
        symbol_whitelist={"TSLA"},
        fetcher=_FakeTweetFetcher(tweets),
    )
    raw = store.read_all_tweets()
    store.update_tweet_sentiment(score_tweets(raw, scorer=_FakeScorer()))

    feats = sentiment_features_for_symbol(store, "TSLA")
    # Plain mean would be (0 + 0 + (-0.5)) / 3 ≈ -0.167.
    # Weighted mean should be closer to -0.5.
    assert feats.iloc[0]["twitter_sentiment_wmean"] < -0.3


# ---------------------------------------------------------------------------
# Column schemas are intact
# ---------------------------------------------------------------------------


def test_sentiment_col_schemas_are_disjoint_and_union():
    reddit = set(REDDIT_SENTIMENT_FEATURE_COLS)
    twitter = set(TWITTER_SENTIMENT_FEATURE_COLS)
    assert reddit.isdisjoint(twitter)
    assert reddit | twitter == set(SENTIMENT_FEATURE_COLS)
    # Both mention-count cols are included in MENTION_COUNT_COLS.
    assert set(MENTION_COUNT_COLS) == {
        "reddit_mention_count",
        "twitter_mention_count",
    }
