"""DuckDB-backed storage layer.

Schema is intentionally small. Tables:

    prices(symbol, date, open, high, low, close, adj_close, volume,
           PK (symbol, date))

    features(symbol, date, <many floats>, target_direction, target_return,
             PK (symbol, date))

    reddit_posts(post_id PK, created_ts, subreddit, author, title, body,
                 score, num_comments, url, sentiment_compound, sentiment_pos,
                 sentiment_neg, sentiment_neu)

    mentions(post_id, symbol, source, PK (post_id, symbol))

Writes are idempotent:
    - prices/features: DELETE + INSERT per symbol.
    - reddit_posts: upsert by post_id (INSERT OR REPLACE semantics via DELETE+INSERT).
    - mentions: scoped to (source, post_id) — delete all rows for an incoming
      post before re-inserting.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import duckdb
import pandas as pd

from sentinel.config import resolve_db_path
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol     VARCHAR NOT NULL,
    date       DATE    NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    adj_close  DOUBLE,
    volume     DOUBLE,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS reddit_posts (
    post_id             VARCHAR PRIMARY KEY,
    created_ts          TIMESTAMP NOT NULL,
    subreddit           VARCHAR,
    author              VARCHAR,
    title               VARCHAR,
    body                VARCHAR,
    score               INTEGER,
    num_comments        INTEGER,
    url                 VARCHAR,
    sentiment_compound  DOUBLE,
    sentiment_pos       DOUBLE,
    sentiment_neg       DOUBLE,
    sentiment_neu       DOUBLE
);

CREATE TABLE IF NOT EXISTS mentions (
    post_id   VARCHAR NOT NULL,
    symbol    VARCHAR NOT NULL,
    source    VARCHAR NOT NULL,
    PRIMARY KEY (post_id, symbol)
);

CREATE TABLE IF NOT EXISTS tweets (
    tweet_id            VARCHAR PRIMARY KEY,
    created_ts          TIMESTAMP NOT NULL,
    author_id           VARCHAR,
    author_username     VARCHAR,
    text                VARCHAR,
    lang                VARCHAR,
    retweet_count       INTEGER,
    reply_count         INTEGER,
    like_count          INTEGER,
    quote_count         INTEGER,
    impression_count    INTEGER,
    sentiment_compound  DOUBLE,
    sentiment_pos       DOUBLE,
    sentiment_neg       DOUBLE,
    sentiment_neu       DOUBLE
);

CREATE TABLE IF NOT EXISTS job_runs (
    job_name      VARCHAR  NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP NOT NULL,
    status        VARCHAR  NOT NULL,
    rows_written  BIGINT,
    error         VARCHAR,
    PRIMARY KEY (job_name, started_at)
);
"""


class DuckDBStore:
    """Thin wrapper around DuckDB with the handful of ops Sentinel needs."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else resolve_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(_SCHEMA)

    # ------------------------------------------------------------------
    # prices
    # ------------------------------------------------------------------

    def write_prices(self, symbol: str, df: pd.DataFrame) -> int:
        """Idempotently replace all rows for ``symbol``."""
        if df.empty:
            return 0
        symbol = symbol.upper()
        to_write = df.reset_index()
        # Normalize column names to lowercase to match schema.
        to_write.columns = [str(c).lower() for c in to_write.columns]
        if "symbol" not in to_write.columns:
            to_write.insert(0, "symbol", symbol)
        to_write["symbol"] = symbol
        to_write["date"] = pd.to_datetime(to_write["date"]).dt.date

        ordered = to_write[
            ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
        ]

        with self._connect() as con:
            con.execute("DELETE FROM prices WHERE symbol = ?", [symbol])
            con.register("incoming", ordered)
            con.execute("INSERT INTO prices SELECT * FROM incoming")
            con.unregister("incoming")

        return len(ordered)

    def read_prices(self, symbol: str) -> pd.DataFrame:
        """Return prices for ``symbol`` indexed by date ascending."""
        symbol = symbol.upper()
        with self._connect() as con:
            df = con.execute(
                "SELECT date, open, high, low, close, adj_close, volume "
                "FROM prices WHERE symbol = ? ORDER BY date",
                [symbol],
            ).fetchdf()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.insert(0, "symbol", symbol)
        return df

    # ------------------------------------------------------------------
    # features — schema is dynamic, so we use `CREATE OR REPLACE TABLE`
    # scoped per symbol namespace.
    # ------------------------------------------------------------------

    def write_features(self, symbol: str, df: pd.DataFrame) -> int:
        """Write a wide feature table. Creates the table on first write, then
        DELETEs + INSERTs rows for this symbol on subsequent writes.
        """
        if df.empty:
            return 0
        symbol = symbol.upper()
        to_write = df.reset_index()
        to_write.columns = [str(c).lower() for c in to_write.columns]
        if "symbol" not in to_write.columns:
            to_write.insert(0, "symbol", symbol)
        to_write["symbol"] = symbol
        to_write["date"] = pd.to_datetime(to_write["date"]).dt.date

        with self._connect() as con:
            exists = (
                con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'features'"
                ).fetchone()[0]
                > 0
            )
            con.register("incoming", to_write)
            if not exists:
                con.execute("CREATE TABLE features AS SELECT * FROM incoming WHERE 1=0")
                # Best-effort primary key (skip if duckdb version complains).
                with contextlib.suppress(Exception):
                    con.execute("ALTER TABLE features ADD PRIMARY KEY (symbol, date)")

            con.execute("DELETE FROM features WHERE symbol = ?", [symbol])
            con.execute("INSERT INTO features SELECT * FROM incoming")
            con.unregister("incoming")

        return len(to_write)

    def read_features(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        with self._connect() as con:
            exists = (
                con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'features'"
                ).fetchone()[0]
                > 0
            )
            if not exists:
                return pd.DataFrame()
            df = con.execute(
                "SELECT * FROM features WHERE symbol = ? ORDER BY date",
                [symbol],
            ).fetchdf()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    # ------------------------------------------------------------------
    # reddit_posts + mentions
    # ------------------------------------------------------------------

    _REDDIT_COLS = (
        "post_id", "created_ts", "subreddit", "author", "title", "body",
        "score", "num_comments", "url",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
    )

    def write_reddit_posts(self, posts: pd.DataFrame) -> int:
        """Upsert Reddit posts by post_id.

        Missing sentiment columns are tolerated — they get filled with NULL so
        sentiment scoring can be a separate pass run later.
        """
        if posts is None or posts.empty:
            return 0
        df = posts.copy()
        df.columns = [str(c).lower() for c in df.columns]

        # Fill any missing columns.
        for c in self._REDDIT_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[list(self._REDDIT_COLS)]
        df["created_ts"] = pd.to_datetime(df["created_ts"])

        ids = df["post_id"].astype(str).tolist()

        with self._connect() as con:
            # Upsert: delete any existing rows with the same post_ids, then insert.
            con.execute(
                "DELETE FROM reddit_posts WHERE post_id IN "
                "(SELECT * FROM UNNEST(?))",
                [ids],
            )
            con.register("incoming_posts", df)
            con.execute("INSERT INTO reddit_posts SELECT * FROM incoming_posts")
            con.unregister("incoming_posts")

        return len(df)

    def write_mentions(self, mentions: pd.DataFrame, *, source: str = "reddit") -> int:
        """Upsert mentions. ``source`` names the origin system."""
        if mentions is None or mentions.empty:
            return 0
        df = mentions.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if "source" not in df.columns:
            df["source"] = source
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[["post_id", "symbol", "source"]].drop_duplicates()

        post_ids = df["post_id"].astype(str).unique().tolist()
        with self._connect() as con:
            con.execute(
                "DELETE FROM mentions WHERE source = ? AND post_id IN "
                "(SELECT * FROM UNNEST(?))",
                [source, post_ids],
            )
            con.register("incoming_mentions", df)
            con.execute("INSERT INTO mentions SELECT * FROM incoming_mentions")
            con.unregister("incoming_mentions")

        return len(df)

    def read_all_reddit_posts(self) -> pd.DataFrame:
        """Return every row in reddit_posts (``post_id, title, body``).

        Used by ``score-sentiment`` to re-score all posts in a single pass.
        Kept as a protocol method rather than a ``_connect()`` escape hatch
        so the Postgres backend can offer the same operation without CLI
        callers needing to know which store they're talking to.
        """
        with self._connect() as con:
            df = con.execute("SELECT post_id, title, body FROM reddit_posts").fetchdf()
        return df

    def read_reddit_posts_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Return all Reddit posts that mention ``symbol`` (via the mentions table)."""
        symbol = symbol.upper()
        with self._connect() as con:
            df = con.execute(
                """
                SELECT p.*
                FROM reddit_posts p
                JOIN mentions m ON m.post_id = p.post_id
                WHERE m.source = 'reddit' AND m.symbol = ?
                ORDER BY p.created_ts
                """,
                [symbol],
            ).fetchdf()
        if df.empty:
            return df
        df["created_ts"] = pd.to_datetime(df["created_ts"])
        return df

    def update_reddit_sentiment(self, scored: pd.DataFrame) -> int:
        """Write VADER scores back to existing rows in reddit_posts.

        ``scored`` must have columns: post_id, sentiment_compound, sentiment_pos,
        sentiment_neg, sentiment_neu.
        """
        if scored is None or scored.empty:
            return 0
        df = scored[
            ["post_id", "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu"]
        ].copy()
        with self._connect() as con:
            con.register("scored", df)
            con.execute(
                """
                UPDATE reddit_posts
                SET sentiment_compound = scored.sentiment_compound,
                    sentiment_pos      = scored.sentiment_pos,
                    sentiment_neg      = scored.sentiment_neg,
                    sentiment_neu      = scored.sentiment_neu
                FROM scored
                WHERE reddit_posts.post_id = scored.post_id
                """
            )
            con.unregister("scored")
        return len(df)

    # ------------------------------------------------------------------
    # tweets
    # ------------------------------------------------------------------

    _TWEET_COLS = (
        "tweet_id", "created_ts", "author_id", "author_username", "text",
        "lang", "retweet_count", "reply_count", "like_count", "quote_count",
        "impression_count",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
    )

    def write_tweets(self, tweets: pd.DataFrame) -> int:
        """Upsert tweets by tweet_id. Missing sentiment columns get NULL."""
        if tweets is None or tweets.empty:
            return 0
        df = tweets.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for c in self._TWEET_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[list(self._TWEET_COLS)]
        df["created_ts"] = pd.to_datetime(df["created_ts"])

        ids = df["tweet_id"].astype(str).tolist()

        with self._connect() as con:
            con.execute(
                "DELETE FROM tweets WHERE tweet_id IN "
                "(SELECT * FROM UNNEST(?))",
                [ids],
            )
            con.register("incoming_tweets", df)
            con.execute("INSERT INTO tweets SELECT * FROM incoming_tweets")
            con.unregister("incoming_tweets")

        return len(df)

    def read_all_tweets(self) -> pd.DataFrame:
        """Return every tweet (``tweet_id, text``). Used by score-sentiment."""
        with self._connect() as con:
            df = con.execute("SELECT tweet_id, text FROM tweets").fetchdf()
        return df

    def read_tweets_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Return all tweets that mention ``symbol`` (joined via mentions, source='twitter')."""
        symbol = symbol.upper()
        with self._connect() as con:
            df = con.execute(
                """
                SELECT t.*
                FROM tweets t
                JOIN mentions m ON m.post_id = t.tweet_id
                WHERE m.source = 'twitter' AND m.symbol = ?
                ORDER BY t.created_ts
                """,
                [symbol],
            ).fetchdf()
        if df.empty:
            return df
        df["created_ts"] = pd.to_datetime(df["created_ts"])
        return df

    def update_tweet_sentiment(self, scored: pd.DataFrame) -> int:
        """Write sentiment scores back to existing tweets.

        ``scored`` must have columns: tweet_id, sentiment_compound, sentiment_pos,
        sentiment_neg, sentiment_neu.
        """
        if scored is None or scored.empty:
            return 0
        df = scored[
            [
                "tweet_id",
                "sentiment_compound",
                "sentiment_pos",
                "sentiment_neg",
                "sentiment_neu",
            ]
        ].copy()
        with self._connect() as con:
            con.register("scored_t", df)
            con.execute(
                """
                UPDATE tweets
                SET sentiment_compound = scored_t.sentiment_compound,
                    sentiment_pos      = scored_t.sentiment_pos,
                    sentiment_neg      = scored_t.sentiment_neg,
                    sentiment_neu      = scored_t.sentiment_neu
                FROM scored_t
                WHERE tweets.tweet_id = scored_t.tweet_id
                """
            )
            con.unregister("scored_t")
        return len(df)

    # ------------------------------------------------------------------
    # job_runs — durable log for sentinel.scheduling
    # ------------------------------------------------------------------

    def record_job_run(self, run) -> int:
        """Append one ``JobRun`` row. Primary key is (job_name, started_at)."""
        with self._connect() as con:
            con.execute(
                "INSERT INTO job_runs "
                "(job_name, started_at, finished_at, status, rows_written, error) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    run.job_name,
                    run.started_at,
                    run.finished_at,
                    run.status,
                    int(run.rows_written),
                    run.error,
                ],
            )
        return 1

    def read_job_runs(
        self, *, job_name: str | None = None, limit: int = 50
    ) -> pd.DataFrame:
        """Return the most recent ``limit`` job runs, newest first."""
        query = (
            "SELECT job_name, started_at, finished_at, status, rows_written, error "
            "FROM job_runs"
        )
        params: list = []
        if job_name is not None:
            query += " WHERE job_name = ?"
            params.append(job_name)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as con:
            df = con.execute(query, params).fetchdf()
        if df.empty:
            return df
        df["started_at"] = pd.to_datetime(df["started_at"])
        df["finished_at"] = pd.to_datetime(df["finished_at"])
        return df

    def last_run_for(self, job_name: str):
        """Return the ``started_at`` of the most recent successful run, or
        ``None`` if the job has never succeeded. Error/skipped runs are
        deliberately excluded so a failing job keeps getting retried."""
        with self._connect() as con:
            row = con.execute(
                "SELECT MAX(started_at) FROM job_runs "
                "WHERE job_name = ? AND status = 'success'",
                [job_name],
            ).fetchone()
        if row is None or row[0] is None:
            return None
        ts = row[0]
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        return ts

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def list_symbols(self) -> list[str]:
        with self._connect() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
