"""Storage protocol.

The modeling, ingestion, and CLI layers all talk to "a store" — they
never needed to know it was DuckDB, they just imported ``DuckDBStore``
directly because it was the only implementation. This module makes the
contract explicit so a second implementation (Postgres / TimescaleDB)
can drop in behind the same surface.

Design notes
------------
* All writes are idempotent at the (symbol, date) or post_id level. A
  re-run of the same ingest should converge to the same state — no
  duplicate rows, no stale trailing rows.
* ``write_features`` is the awkward one: the feature table's columns
  are dynamic (they depend on the configured feature blocks). DuckDB
  resolves this by creating the table on first write from whatever
  DataFrame shows up. Postgres does the same via ``CREATE TABLE`` on
  first write + ``ALTER TABLE ADD COLUMN`` when new feature columns
  arrive. Both hide the mechanics from callers.
* ``read_all_reddit_posts`` replaces what was a private DuckDB
  ``_connect()`` escape hatch in the CLI's ``score-sentiment`` command.
  Every Store op should go through a protocol method; no leaky
  back-end-specific handles.
* ``job_runs`` is the durability layer for :mod:`sentinel.scheduling`.
  Callers append one row per scheduled run; the scheduler reads
  ``last_run_for(name)`` to decide what's due.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from sentinel.scheduling.spec import JobRun


@runtime_checkable
class Store(Protocol):
    """Minimal persistence surface used by the rest of Sentinel."""

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    def write_prices(self, symbol: str, df: pd.DataFrame) -> int: ...
    def read_prices(self, symbol: str) -> pd.DataFrame: ...

    # ------------------------------------------------------------------
    # Features (dynamic schema)
    # ------------------------------------------------------------------
    def write_features(self, symbol: str, df: pd.DataFrame) -> int: ...
    def read_features(self, symbol: str) -> pd.DataFrame: ...

    # ------------------------------------------------------------------
    # Reddit posts + mentions
    # ------------------------------------------------------------------
    def write_reddit_posts(self, posts: pd.DataFrame) -> int: ...
    def write_mentions(self, mentions: pd.DataFrame, *, source: str = "reddit") -> int: ...
    def read_reddit_posts_for_symbol(self, symbol: str) -> pd.DataFrame: ...
    def read_all_reddit_posts(self) -> pd.DataFrame: ...
    def update_reddit_sentiment(self, scored: pd.DataFrame) -> int: ...

    # ------------------------------------------------------------------
    # Tweets
    # ------------------------------------------------------------------
    def write_tweets(self, tweets: pd.DataFrame) -> int: ...
    def read_tweets_for_symbol(self, symbol: str) -> pd.DataFrame: ...
    def read_all_tweets(self) -> pd.DataFrame: ...
    def update_tweet_sentiment(self, scored: pd.DataFrame) -> int: ...

    # ------------------------------------------------------------------
    # Scheduled job runs
    # ------------------------------------------------------------------
    def record_job_run(self, run: JobRun) -> int: ...
    def read_job_runs(
        self, *, job_name: str | None = None, limit: int = 50
    ) -> pd.DataFrame: ...
    def last_run_for(self, job_name: str) -> datetime | None: ...

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_symbols(self) -> list[str]: ...


# The column contract for reddit_posts. Kept here so both backends and the
# sentiment scorer agree on the shape without needing to import each other.
REDDIT_POST_COLUMNS: tuple[str, ...] = (
    "post_id",
    "created_ts",
    "subreddit",
    "author",
    "title",
    "body",
    "score",
    "num_comments",
    "url",
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
)


# Columns for the job_runs table. Shared so both backends emit the same schema
# and the scheduler knows what to expect when reading.
JOB_RUN_COLUMNS: tuple[str, ...] = (
    "job_name",
    "started_at",
    "finished_at",
    "status",
    "rows_written",
    "error",
)


# Column contract for the tweets table. Mirrors REDDIT_POST_COLUMNS — one row
# per tweet, sentiment populated by a second pass. Engagement metrics are
# Twitter-specific (likes / retweets / replies / quotes / impressions) so each
# backend maps them to an integer column.
TWEET_COLUMNS: tuple[str, ...] = (
    "tweet_id",
    "created_ts",
    "author_id",
    "author_username",
    "text",
    "lang",
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "impression_count",
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
)


__all__ = [
    "Store",
    "REDDIT_POST_COLUMNS",
    "TWEET_COLUMNS",
    "JOB_RUN_COLUMNS",
]
