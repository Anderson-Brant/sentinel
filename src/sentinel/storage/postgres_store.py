"""Postgres / TimescaleDB storage backend.

Mirrors :class:`sentinel.storage.duckdb_store.DuckDBStore` over the same
:class:`~sentinel.storage.base.Store` protocol. Picked by the ``get_store``
factory when the user sets ``SENTINEL_STORAGE_BACKEND=postgres`` and
provides a ``SENTINEL_POSTGRES_DSN``.

Design choices
--------------
* **Lazy import.** ``psycopg`` is an optional dependency — Sentinel must
  still import and run on a machine that doesn't have it. The import
  happens inside ``__init__``, with an actionable install hint on
  failure.
* **TimescaleDB is opt-in and soft-fail.** On first connect, we try
  ``CREATE EXTENSION IF NOT EXISTS timescaledb`` and, if that succeeds,
  convert the time-series tables to hypertables with
  ``create_hypertable(..., if_not_exists => TRUE)``. If Timescale isn't
  installed, we fall back to plain tables — the backend still works,
  just without the partitioning benefits.
* **Features table is dynamic.** Feature columns depend on which
  blocks are turned on in the YAML config. On write, we inspect the
  current ``features`` table and ``ALTER TABLE ADD COLUMN`` for any
  columns the incoming DataFrame has that we haven't seen before. On
  first write the table is created from the DataFrame shape.
* **Every write is idempotent.** Prices and features do
  ``DELETE FROM ... WHERE symbol = ?`` followed by ``INSERT``. Reddit
  posts + mentions delete by the incoming primary keys before inserting.
* **Parameterized SQL.** All user-controlled values go through psycopg's
  parameter binding — never string-formatted into SQL. The only
  identifier we format in is the feature column list, and those are
  sanitized against a conservative allow-list (``[A-Za-z0-9_]+``).
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd

from sentinel.storage.base import (
    JOB_RUN_COLUMNS,
    REDDIT_POST_COLUMNS,
    TWEET_COLUMNS,
    Store,
)
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


_PSYCOPG_INSTALL_HINT = (
    "psycopg is not installed. Install it with `pip install 'psycopg[binary]'` "
    "to use the Postgres storage backend."
)


# Pattern for a safe-to-quote identifier. Feature column names are derived
# programmatically from the technical/sentiment feature modules, so this is
# defense-in-depth rather than strictly necessary — but it keeps any future
# user-controlled feature name from turning into SQL injection.
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"Refusing to quote unsafe identifier: {name!r}")
    return f'"{name}"'


# Map pandas/numpy dtypes to a Postgres type for CREATE TABLE / ALTER TABLE.
def _pg_type_for(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    # object / string / anything else → text
    return "TEXT"


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


_DDL_STATEMENTS = (
    # Prices ------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS prices (
        symbol     TEXT             NOT NULL,
        date       DATE             NOT NULL,
        open       DOUBLE PRECISION,
        high       DOUBLE PRECISION,
        low        DOUBLE PRECISION,
        close      DOUBLE PRECISION,
        adj_close  DOUBLE PRECISION,
        volume     DOUBLE PRECISION,
        PRIMARY KEY (symbol, date)
    )
    """,
    # Reddit posts ------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS reddit_posts (
        post_id             TEXT PRIMARY KEY,
        created_ts          TIMESTAMP NOT NULL,
        subreddit           TEXT,
        author              TEXT,
        title               TEXT,
        body                TEXT,
        score               INTEGER,
        num_comments        INTEGER,
        url                 TEXT,
        sentiment_compound  DOUBLE PRECISION,
        sentiment_pos       DOUBLE PRECISION,
        sentiment_neg       DOUBLE PRECISION,
        sentiment_neu       DOUBLE PRECISION
    )
    """,
    # Mentions ----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS mentions (
        post_id   TEXT NOT NULL,
        symbol    TEXT NOT NULL,
        source    TEXT NOT NULL,
        PRIMARY KEY (post_id, symbol)
    )
    """,
    # Tweets ------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS tweets (
        tweet_id            TEXT PRIMARY KEY,
        created_ts          TIMESTAMP NOT NULL,
        author_id           TEXT,
        author_username     TEXT,
        text                TEXT,
        lang                TEXT,
        retweet_count       INTEGER,
        reply_count         INTEGER,
        like_count          INTEGER,
        quote_count         INTEGER,
        impression_count    INTEGER,
        sentiment_compound  DOUBLE PRECISION,
        sentiment_pos       DOUBLE PRECISION,
        sentiment_neg       DOUBLE PRECISION,
        sentiment_neu       DOUBLE PRECISION
    )
    """,
    # Job runs (scheduler durability) -----------------------------------
    """
    CREATE TABLE IF NOT EXISTS job_runs (
        job_name      TEXT      NOT NULL,
        started_at    TIMESTAMP NOT NULL,
        finished_at   TIMESTAMP NOT NULL,
        status        TEXT      NOT NULL,
        rows_written  BIGINT,
        error         TEXT,
        PRIMARY KEY (job_name, started_at)
    )
    """,
)


class PostgresStore:
    """Postgres / TimescaleDB backend implementing :class:`Store`."""

    def __init__(
        self,
        dsn: str,
        *,
        enable_timescale: bool = True,
        connect_factory: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        dsn : str
            Postgres connection string (``postgresql://user:pass@host/db``).
        enable_timescale : bool, default True
            Attempt to enable the Timescale extension and create hypertables
            on the time-series tables. Soft-fails if Timescale isn't
            available — we'll log a note and keep going with plain tables.
        connect_factory : callable, optional
            Dependency-injection hook used by tests: if supplied, we call
            ``connect_factory(dsn)`` instead of importing psycopg. Keeps
            unit tests from needing psycopg installed.
        """
        if connect_factory is not None:
            self._connect_factory = connect_factory
        else:
            try:
                import psycopg  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover - exercised via stubbed test
                raise ImportError(_PSYCOPG_INSTALL_HINT) from e
            self._connect_factory = lambda d: psycopg.connect(d, autocommit=True)

        self.dsn = dsn
        self.enable_timescale = enable_timescale
        self._timescale_ok: bool | None = None  # resolved on first connect

        self._init_schema()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        con = self._connect_factory(self.dsn)
        try:
            yield con
        finally:
            close = getattr(con, "close", None)
            if callable(close):
                close()

    def _init_schema(self) -> None:
        """Create tables, try Timescale, and convert to hypertables."""
        with self._connect() as con:
            cur = con.cursor()
            for stmt in _DDL_STATEMENTS:
                cur.execute(stmt)

            if self.enable_timescale and self._timescale_ok is None:
                self._timescale_ok = self._try_enable_timescale(cur)

            if self._timescale_ok:
                # Convert time-series tables to hypertables. Idempotent.
                for sql in (
                    "SELECT create_hypertable('prices', 'date', if_not_exists => TRUE, migrate_data => TRUE)",
                    "SELECT create_hypertable('reddit_posts', 'created_ts', if_not_exists => TRUE, migrate_data => TRUE)",
                    "SELECT create_hypertable('tweets', 'created_ts', if_not_exists => TRUE, migrate_data => TRUE)",
                ):
                    try:
                        cur.execute(sql)
                    except Exception as e:  # noqa: BLE001
                        log.warning("hypertable setup skipped: %s", e)

    @staticmethod
    def _try_enable_timescale(cur: Any) -> bool:
        """Return True if the Timescale extension is available and enabled."""
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            log.info("TimescaleDB extension enabled")
            return True
        except Exception as e:  # noqa: BLE001
            log.info("TimescaleDB not available, using plain tables: %s", e)
            return False

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def write_prices(self, symbol: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        symbol = symbol.upper()
        to_write = df.reset_index()
        to_write.columns = [str(c).lower() for c in to_write.columns]
        if "symbol" not in to_write.columns:
            to_write.insert(0, "symbol", symbol)
        to_write["symbol"] = symbol
        to_write["date"] = pd.to_datetime(to_write["date"]).dt.date

        cols = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
        for c in cols:
            if c not in to_write.columns:
                to_write[c] = None
        ordered = to_write[cols]
        rows = list(ordered.itertuples(index=False, name=None))

        with self._connect() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM prices WHERE symbol = %s", (symbol,))
            placeholders = ",".join(["%s"] * len(cols))
            cur.executemany(
                f"INSERT INTO prices ({','.join(cols)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def read_prices(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT {','.join(cols)} FROM prices WHERE symbol = %s ORDER BY date",
                (symbol,),
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.insert(0, "symbol", symbol)
        return df

    # ------------------------------------------------------------------
    # Features (dynamic schema)
    # ------------------------------------------------------------------

    def _existing_feature_columns(self, cur: Any) -> set[str]:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'features'"
        )
        return {r[0] for r in cur.fetchall()}

    def write_features(self, symbol: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        symbol = symbol.upper()
        to_write = df.reset_index()
        to_write.columns = [str(c).lower() for c in to_write.columns]
        if "symbol" not in to_write.columns:
            to_write.insert(0, "symbol", symbol)
        to_write["symbol"] = symbol
        to_write["date"] = pd.to_datetime(to_write["date"]).dt.date

        incoming_cols = list(to_write.columns)
        for c in incoming_cols:
            _quote_ident(c)  # raises on unsafe names; no-op otherwise

        with self._connect() as con:
            cur = con.cursor()
            existing = self._existing_feature_columns(cur)

            if not existing:
                # First write: CREATE the table from the DataFrame shape.
                col_defs: list[str] = []
                for col in incoming_cols:
                    if col == "symbol":
                        col_defs.append('"symbol" TEXT NOT NULL')
                    elif col == "date":
                        col_defs.append('"date" DATE NOT NULL')
                    else:
                        col_defs.append(
                            f'{_quote_ident(col)} {_pg_type_for(to_write[col])}'
                        )
                cur.execute(
                    f"CREATE TABLE features ({', '.join(col_defs)}, "
                    f"PRIMARY KEY (symbol, date))"
                )
            else:
                # Subsequent write: ALTER TABLE to add any new columns.
                for col in incoming_cols:
                    if col in existing:
                        continue
                    cur.execute(
                        f"ALTER TABLE features ADD COLUMN {_quote_ident(col)} "
                        f"{_pg_type_for(to_write[col])}"
                    )

            cur.execute("DELETE FROM features WHERE symbol = %s", (symbol,))
            placeholders = ",".join(["%s"] * len(incoming_cols))
            quoted = ",".join(_quote_ident(c) for c in incoming_cols)
            rows = [
                tuple(None if _is_null(v) else v for v in row)
                for row in to_write.itertuples(index=False, name=None)
            ]
            cur.executemany(
                f"INSERT INTO features ({quoted}) VALUES ({placeholders})",
                rows,
            )
        return len(to_write)

    def read_features(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'features' ORDER BY ordinal_position"
            )
            cols = [r[0] for r in cur.fetchall()]
            if not cols:
                return pd.DataFrame()
            quoted = ",".join(_quote_ident(c) for c in cols)
            cur.execute(
                f"SELECT {quoted} FROM features WHERE symbol = %s ORDER BY date",
                (symbol,),
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    # ------------------------------------------------------------------
    # Reddit posts + mentions
    # ------------------------------------------------------------------

    def write_reddit_posts(self, posts: pd.DataFrame) -> int:
        if posts is None or posts.empty:
            return 0
        df = posts.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for c in REDDIT_POST_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[list(REDDIT_POST_COLUMNS)]
        df["created_ts"] = pd.to_datetime(df["created_ts"])

        ids = df["post_id"].astype(str).tolist()
        rows = [
            tuple(None if _is_null(v) else v for v in row)
            for row in df.itertuples(index=False, name=None)
        ]
        cols = ",".join(REDDIT_POST_COLUMNS)
        placeholders = ",".join(["%s"] * len(REDDIT_POST_COLUMNS))

        with self._connect() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM reddit_posts WHERE post_id = ANY(%s)", (ids,))
            cur.executemany(
                f"INSERT INTO reddit_posts ({cols}) VALUES ({placeholders})",
                rows,
            )
        return len(df)

    def write_mentions(self, mentions: pd.DataFrame, *, source: str = "reddit") -> int:
        if mentions is None or mentions.empty:
            return 0
        df = mentions.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if "source" not in df.columns:
            df["source"] = source
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[["post_id", "symbol", "source"]].drop_duplicates()

        post_ids = df["post_id"].astype(str).unique().tolist()
        rows = list(df.itertuples(index=False, name=None))

        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "DELETE FROM mentions WHERE source = %s AND post_id = ANY(%s)",
                (source, post_ids),
            )
            cur.executemany(
                "INSERT INTO mentions (post_id, symbol, source) VALUES (%s, %s, %s) "
                "ON CONFLICT (post_id, symbol) DO UPDATE SET source = EXCLUDED.source",
                rows,
            )
        return len(df)

    def read_reddit_posts_for_symbol(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        cols = ",".join(f"p.{c}" for c in REDDIT_POST_COLUMNS)
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT {cols} FROM reddit_posts p "
                "JOIN mentions m ON m.post_id = p.post_id "
                "WHERE m.source = 'reddit' AND m.symbol = %s ORDER BY p.created_ts",
                (symbol,),
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=list(REDDIT_POST_COLUMNS))
        df["created_ts"] = pd.to_datetime(df["created_ts"])
        return df

    def read_all_reddit_posts(self) -> pd.DataFrame:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT post_id, title, body FROM reddit_posts")
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["post_id", "title", "body"])
        return pd.DataFrame(rows, columns=["post_id", "title", "body"])

    def update_reddit_sentiment(self, scored: pd.DataFrame) -> int:
        if scored is None or scored.empty:
            return 0
        df = scored[
            [
                "post_id",
                "sentiment_compound",
                "sentiment_pos",
                "sentiment_neg",
                "sentiment_neu",
            ]
        ].copy()
        rows = [
            (
                float(r.sentiment_compound) if r.sentiment_compound is not None else None,
                float(r.sentiment_pos) if r.sentiment_pos is not None else None,
                float(r.sentiment_neg) if r.sentiment_neg is not None else None,
                float(r.sentiment_neu) if r.sentiment_neu is not None else None,
                str(r.post_id),
            )
            for r in df.itertuples(index=False)
        ]
        with self._connect() as con:
            cur = con.cursor()
            cur.executemany(
                "UPDATE reddit_posts SET "
                "sentiment_compound = %s, "
                "sentiment_pos = %s, "
                "sentiment_neg = %s, "
                "sentiment_neu = %s "
                "WHERE post_id = %s",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------
    # Tweets
    # ------------------------------------------------------------------

    def write_tweets(self, tweets: pd.DataFrame) -> int:
        if tweets is None or tweets.empty:
            return 0
        df = tweets.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for c in TWEET_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[list(TWEET_COLUMNS)]
        df["created_ts"] = pd.to_datetime(df["created_ts"])

        ids = df["tweet_id"].astype(str).tolist()
        rows = [
            tuple(None if _is_null(v) else v for v in row)
            for row in df.itertuples(index=False, name=None)
        ]
        cols = ",".join(TWEET_COLUMNS)
        placeholders = ",".join(["%s"] * len(TWEET_COLUMNS))

        with self._connect() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM tweets WHERE tweet_id = ANY(%s)", (ids,))
            cur.executemany(
                f"INSERT INTO tweets ({cols}) VALUES ({placeholders})",
                rows,
            )
        return len(df)

    def read_tweets_for_symbol(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        cols = ",".join(f"t.{c}" for c in TWEET_COLUMNS)
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT {cols} FROM tweets t "
                "JOIN mentions m ON m.post_id = t.tweet_id "
                "WHERE m.source = 'twitter' AND m.symbol = %s "
                "ORDER BY t.created_ts",
                (symbol,),
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=list(TWEET_COLUMNS))
        df["created_ts"] = pd.to_datetime(df["created_ts"])
        return df

    def read_all_tweets(self) -> pd.DataFrame:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT tweet_id, text FROM tweets")
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["tweet_id", "text"])
        return pd.DataFrame(rows, columns=["tweet_id", "text"])

    def update_tweet_sentiment(self, scored: pd.DataFrame) -> int:
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
        rows = [
            (
                float(r.sentiment_compound) if r.sentiment_compound is not None else None,
                float(r.sentiment_pos) if r.sentiment_pos is not None else None,
                float(r.sentiment_neg) if r.sentiment_neg is not None else None,
                float(r.sentiment_neu) if r.sentiment_neu is not None else None,
                str(r.tweet_id),
            )
            for r in df.itertuples(index=False)
        ]
        with self._connect() as con:
            cur = con.cursor()
            cur.executemany(
                "UPDATE tweets SET "
                "sentiment_compound = %s, "
                "sentiment_pos = %s, "
                "sentiment_neg = %s, "
                "sentiment_neu = %s "
                "WHERE tweet_id = %s",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------
    # job_runs — durable log for sentinel.scheduling
    # ------------------------------------------------------------------

    def record_job_run(self, run) -> int:
        """Append one ``JobRun`` row. Primary key is (job_name, started_at)."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO job_runs "
                "(job_name, started_at, finished_at, status, rows_written, error) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    run.job_name,
                    run.started_at,
                    run.finished_at,
                    run.status,
                    int(run.rows_written),
                    run.error,
                ),
            )
        return 1

    def read_job_runs(
        self, *, job_name: str | None = None, limit: int = 50
    ) -> pd.DataFrame:
        """Return the most recent ``limit`` job runs, newest first."""
        cols = list(JOB_RUN_COLUMNS)
        query = f"SELECT {','.join(cols)} FROM job_runs"
        params: list[Any] = []
        if job_name is not None:
            query += " WHERE job_name = %s"
            params.append(job_name)
        query += " ORDER BY started_at DESC LIMIT %s"
        params.append(int(limit))
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows, columns=cols)
        df["started_at"] = pd.to_datetime(df["started_at"])
        df["finished_at"] = pd.to_datetime(df["finished_at"])
        return df

    def last_run_for(self, job_name: str):
        """Return the ``started_at`` of the most recent successful run, or
        ``None`` if the job has never succeeded. Error/skipped runs are
        deliberately excluded so a failing job keeps getting retried."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT MAX(started_at) FROM job_runs "
                "WHERE job_name = %s AND status = 'success'",
                (job_name,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        ts = row[0]
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        return ts

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_symbols(self) -> list[str]:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT DISTINCT symbol FROM prices")
            return [r[0] for r in cur.fetchall()]


def _is_null(v: Any) -> bool:
    """Return True for any form of missing value we might pull from pandas."""
    if v is None:
        return True
    try:
        # NaN propagation: only numeric scalar NaNs count. pd.isna on a
        # Timestamp returns False; on a non-numeric string returns False
        # unless that string is literally NaN which it won't be here.
        if isinstance(v, float) and np.isnan(v):
            return True
    except Exception:  # noqa: BLE001
        pass
    # pd.NA and NaT. pd.NaT is its own NaTType — not a pd.Timestamp — so
    # the isinstance check alone misses it; add an explicit identity test.
    if v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, pd.Timestamp) and pd.isna(v):
        return True
    return False


__all__ = ["PostgresStore"]
