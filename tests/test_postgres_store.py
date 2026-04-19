"""Tests for :mod:`sentinel.storage.postgres_store` and the ``get_store`` factory.

These tests don't require a real Postgres; we inject a fake ``connect_factory``
that records SQL and serves canned responses for ``fetchall`` lookups. Keeps
the fast test path hermetic and still exercises every branch of the backend
(DDL, hypertables soft-fail, idempotent writes, dynamic feature schema,
mentions upsert, factory dispatch).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fake psycopg connection — records SQL + returns canned fetchall responses
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn
        self._last_rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.conn.statements.append((sql, params))

        # Honor a "raise on SQL matching this regex" hook (for timescale soft-fail).
        for pattern, exc in self.conn.raise_on:
            if re.search(pattern, sql, flags=re.IGNORECASE):
                raise exc

        # Resolve queries that expect fetchall to return data.
        self._last_rows = self.conn._answer(sql, params)

    def executemany(self, sql: str, seq: list[Any]) -> None:
        # Record a single entry capturing the SQL + every parameter tuple so
        # tests can assert on the actual rows we would've written.
        self.conn.statements.append((sql, list(seq)))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._last_rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._last_rows[0] if self._last_rows else None


class FakeConnection:
    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.closed = False
        self.raise_on: list[tuple[str, Exception]] = []
        # ``answers`` maps regex patterns → callable(sql, params) -> rows.
        self.answers: dict[str, Any] = answers or {}

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True

    # Helper: return rows for the first regex pattern that matches ``sql``.
    def _answer(self, sql: str, params: Any) -> list[tuple[Any, ...]]:
        for pattern, resolver in self.answers.items():
            if re.search(pattern, sql, flags=re.IGNORECASE | re.DOTALL):
                if callable(resolver):
                    return list(resolver(sql, params))
                return list(resolver)
        return []


class FakeFactory:
    """``connect_factory`` callable that yields a single persistent FakeConnection."""

    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn
        self.calls: list[str] = []

    def __call__(self, dsn: str) -> FakeConnection:
        self.calls.append(dsn)
        return self.conn


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------


def _find(stmts: list[tuple[str, Any]], pattern: str) -> list[tuple[str, Any]]:
    return [(s, p) for (s, p) in stmts if re.search(pattern, s, flags=re.IGNORECASE)]


def test_init_schema_emits_all_tables():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    factory = FakeFactory(conn)
    PostgresStore(dsn="postgresql://fake/db", connect_factory=factory)

    # All three core tables' DDL ran.
    assert _find(conn.statements, r"CREATE TABLE IF NOT EXISTS prices")
    assert _find(conn.statements, r"CREATE TABLE IF NOT EXISTS reddit_posts")
    assert _find(conn.statements, r"CREATE TABLE IF NOT EXISTS mentions")

    # Timescale was attempted.
    assert _find(conn.statements, r"CREATE EXTENSION IF NOT EXISTS timescaledb")

    # Hypertables were created (on the time-series tables).
    ht = _find(conn.statements, r"create_hypertable")
    assert any("prices" in sql for sql, _ in ht)
    assert any("reddit_posts" in sql for sql, _ in ht)


def test_init_schema_soft_fails_when_timescale_missing():
    """If the CREATE EXTENSION statement raises, we skip hypertable DDL and
    keep going with plain tables."""
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    conn.raise_on.append(
        (r"CREATE EXTENSION", RuntimeError("timescale not installed"))
    )
    factory = FakeFactory(conn)
    PostgresStore(dsn="postgresql://fake/db", connect_factory=factory)

    # Core tables still got created.
    assert _find(conn.statements, r"CREATE TABLE IF NOT EXISTS prices")
    # The attempt was made.
    assert _find(conn.statements, r"CREATE EXTENSION IF NOT EXISTS timescaledb")
    # But no hypertable calls were issued because the extension failed.
    assert not _find(conn.statements, r"create_hypertable")


def test_init_schema_respects_enable_timescale_false():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    factory = FakeFactory(conn)
    PostgresStore(
        dsn="postgresql://fake/db",
        enable_timescale=False,
        connect_factory=factory,
    )

    assert not _find(conn.statements, r"CREATE EXTENSION")
    assert not _find(conn.statements, r"create_hypertable")


def test_missing_psycopg_raises_actionable_import_error(monkeypatch):
    """With no connect_factory, we try to import psycopg. If it's missing we
    must surface an install hint rather than a bare ModuleNotFoundError."""
    # Patch __import__ to simulate missing psycopg.
    import builtins

    from sentinel.storage import postgres_store as mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"psycopg is not installed"):
        mod.PostgresStore(dsn="postgresql://fake/db")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def _make_prices(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "open": np.arange(n, dtype=float),
            "high": np.arange(n, dtype=float) + 1,
            "low": np.arange(n, dtype=float) - 1,
            "close": np.arange(n, dtype=float) + 0.5,
            "adj_close": np.arange(n, dtype=float) + 0.5,
            "volume": np.arange(n, dtype=float) * 1000,
        },
        index=pd.Index(dates, name="date"),
    )


def test_write_prices_issues_delete_then_insert():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()  # ignore init DDL

    df = _make_prices(5)
    n = store.write_prices("TEST", df)
    assert n == 5

    # DELETE runs first, then INSERT executemany with 5 rows.
    deletes = _find(conn.statements, r"DELETE FROM prices WHERE symbol")
    inserts = _find(conn.statements, r"INSERT INTO prices")
    assert len(deletes) == 1
    assert deletes[0][1] == ("TEST",)
    assert len(inserts) == 1
    rows = inserts[0][1]
    assert isinstance(rows, list) and len(rows) == 5
    # Each row is an 8-tuple matching the column order.
    assert all(len(r) == 8 for r in rows)
    # Symbol uppercased, date normalized.
    assert all(r[0] == "TEST" for r in rows)


def test_write_prices_empty_returns_zero_no_queries():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()

    assert store.write_prices("TEST", pd.DataFrame()) == 0
    assert conn.statements == []


def test_read_prices_shapes_dataframe():
    from sentinel.storage.postgres_store import PostgresStore

    # Canned response from the SELECT.
    rows = [
        (pd.Timestamp("2024-01-01").date(), 1.0, 1.5, 0.5, 1.25, 1.25, 1000.0),
        (pd.Timestamp("2024-01-02").date(), 2.0, 2.5, 1.5, 2.25, 2.25, 2000.0),
    ]
    conn = FakeConnection(answers={r"SELECT .* FROM prices WHERE symbol": rows})
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )

    loaded = store.read_prices("test")  # lower input
    assert loaded.index.name == "date"
    assert list(loaded.columns) == [
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert len(loaded) == 2
    # Symbol column normalized to upper case.
    assert (loaded["symbol"] == "TEST").all()


def test_read_prices_empty_returns_empty_frame():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()  # no answers → fetchall returns []
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    assert store.read_prices("NOPE").empty


# ---------------------------------------------------------------------------
# Features — dynamic schema
# ---------------------------------------------------------------------------


def test_write_features_creates_table_on_first_write():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection(
        answers={
            # information_schema: table does not exist yet → empty
            r"information_schema\.columns.*features": [],
        }
    )
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()

    df = pd.DataFrame(
        {
            "ret_1": [0.01, 0.02, 0.03],
            "sma_5": [1.1, 1.2, 1.3],
            "target_direction": [1, 0, 1],
        },
        index=pd.Index(pd.date_range("2024-01-01", periods=3), name="date"),
    )
    n = store.write_features("TEST", df)
    assert n == 3

    # CREATE TABLE features (...) ran.
    creates = _find(conn.statements, r"CREATE TABLE features")
    assert len(creates) == 1
    created_sql = creates[0][0]
    # All feature columns are quoted; symbol + date declared with types.
    assert '"symbol" TEXT NOT NULL' in created_sql
    assert '"date" DATE NOT NULL' in created_sql
    assert '"ret_1"' in created_sql
    assert '"target_direction"' in created_sql
    assert "PRIMARY KEY (symbol, date)" in created_sql

    # No ALTER TABLE on first write.
    assert not _find(conn.statements, r"ALTER TABLE features ADD COLUMN")

    # Then DELETE + INSERT.
    assert _find(conn.statements, r"DELETE FROM features WHERE symbol")
    inserts = _find(conn.statements, r"INSERT INTO features")
    assert len(inserts) == 1
    assert len(inserts[0][1]) == 3  # 3 rows


def test_write_features_adds_new_columns_on_subsequent_write():
    from sentinel.storage.postgres_store import PostgresStore

    # Existing columns are symbol, date, ret_1. Incoming adds sma_5, target_direction.
    conn = FakeConnection(
        answers={
            r"information_schema\.columns.*features": [
                ("symbol",),
                ("date",),
                ("ret_1",),
            ],
        }
    )
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()

    df = pd.DataFrame(
        {
            "ret_1": [0.01, 0.02],
            "sma_5": [1.1, 1.2],
            "target_direction": [1, 0],
        },
        index=pd.Index(pd.date_range("2024-01-01", periods=2), name="date"),
    )
    store.write_features("TEST", df)

    alters = _find(conn.statements, r"ALTER TABLE features ADD COLUMN")
    altered_cols = [re.search(r'ADD COLUMN "(\w+)"', s).group(1) for s, _ in alters]
    assert set(altered_cols) == {"sma_5", "target_direction"}
    # ret_1 was pre-existing — no ALTER for it.
    assert "ret_1" not in altered_cols

    # And no CREATE TABLE this time.
    assert not _find(conn.statements, r"CREATE TABLE features")


def test_write_features_rejects_unsafe_column_names():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection(answers={r"information_schema\.columns.*features": []})
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )

    df = pd.DataFrame(
        {"ret_1; DROP TABLE features; --": [0.1, 0.2]},
        index=pd.Index(pd.date_range("2024-01-01", periods=2), name="date"),
    )
    with pytest.raises(ValueError, match="unsafe identifier"):
        store.write_features("TEST", df)


def test_read_features_returns_empty_when_table_absent():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection(answers={r"information_schema\.columns.*features": []})
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    assert store.read_features("TEST").empty


# ---------------------------------------------------------------------------
# Reddit posts + mentions
# ---------------------------------------------------------------------------


def test_write_reddit_posts_upserts_by_post_id():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()

    posts = pd.DataFrame(
        {
            "post_id": ["a", "b"],
            "created_ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "subreddit": ["wsb", "wsb"],
            "author": ["x", "y"],
            "title": ["hi", "lo"],
            "body": ["", ""],
            "score": [1, 2],
            "num_comments": [0, 1],
            "url": ["u1", "u2"],
            # sentiment columns intentionally missing → filled with None
        }
    )
    n = store.write_reddit_posts(posts)
    assert n == 2

    deletes = _find(conn.statements, r"DELETE FROM reddit_posts WHERE post_id = ANY")
    assert len(deletes) == 1
    assert deletes[0][1] == (["a", "b"],)

    inserts = _find(conn.statements, r"INSERT INTO reddit_posts")
    assert len(inserts[0][1]) == 2
    # Each inserted row has 13 columns (REDDIT_POST_COLUMNS length).
    from sentinel.storage.base import REDDIT_POST_COLUMNS

    assert all(len(r) == len(REDDIT_POST_COLUMNS) for r in inserts[0][1])


def test_write_mentions_uses_on_conflict_upsert():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()

    mentions = pd.DataFrame(
        {"post_id": ["a", "b", "a"], "symbol": ["spy", "qqq", "spy"]}
    )  # 'a'/SPY duplicate to verify de-dup
    n = store.write_mentions(mentions, source="reddit")
    assert n == 2  # dedup'd

    inserts = _find(conn.statements, r"INSERT INTO mentions")
    assert "ON CONFLICT" in inserts[0][0]
    rows = inserts[0][1]
    # Symbols upper-cased.
    assert {(r[0], r[1], r[2]) for r in rows} == {
        ("a", "SPY", "reddit"),
        ("b", "QQQ", "reddit"),
    }


def test_read_all_reddit_posts_shape():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection(
        answers={
            r"SELECT post_id, title, body FROM reddit_posts": [
                ("a", "t1", "b1"),
                ("b", "t2", "b2"),
            ]
        }
    )
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    df = store.read_all_reddit_posts()
    assert list(df.columns) == ["post_id", "title", "body"]
    assert len(df) == 2


def test_update_reddit_sentiment_batches_updates():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    conn.statements.clear()

    scored = pd.DataFrame(
        {
            "post_id": ["a", "b"],
            "sentiment_compound": [0.5, -0.3],
            "sentiment_pos": [0.6, 0.1],
            "sentiment_neg": [0.1, 0.4],
            "sentiment_neu": [0.3, 0.5],
        }
    )
    n = store.update_reddit_sentiment(scored)
    assert n == 2

    updates = _find(conn.statements, r"UPDATE reddit_posts SET")
    assert len(updates) == 1
    rows = updates[0][1]
    assert len(rows) == 2
    # Each row is (compound, pos, neg, neu, post_id)
    assert rows[0][-1] == "a"
    assert rows[1][-1] == "b"


def test_list_symbols():
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection(
        answers={r"SELECT DISTINCT symbol FROM prices": [("SPY",), ("QQQ",)]}
    )
    store = PostgresStore(
        dsn="postgresql://fake/db",
        connect_factory=FakeFactory(conn),
        enable_timescale=False,
    )
    assert sorted(store.list_symbols()) == ["QQQ", "SPY"]


# ---------------------------------------------------------------------------
# _is_null / _pg_type_for helpers
# ---------------------------------------------------------------------------


def test_is_null_handles_various_missing_values():
    from sentinel.storage.postgres_store import _is_null

    assert _is_null(None)
    assert _is_null(float("nan"))
    assert _is_null(pd.NA)
    assert _is_null(pd.NaT)
    assert not _is_null(0)
    assert not _is_null(0.0)
    assert not _is_null("")
    assert not _is_null(pd.Timestamp("2024-01-01"))


def test_pg_type_mapping():
    from sentinel.storage.postgres_store import _pg_type_for

    assert _pg_type_for(pd.Series([1, 2, 3], dtype="int64")) == "BIGINT"
    assert _pg_type_for(pd.Series([1.0, 2.0], dtype="float64")) == "DOUBLE PRECISION"
    assert _pg_type_for(pd.Series([True, False])) == "BOOLEAN"
    assert _pg_type_for(pd.to_datetime(pd.Series(["2024-01-01"]))) == "TIMESTAMP"
    assert _pg_type_for(pd.Series(["a", "b"])) == "TEXT"


# ---------------------------------------------------------------------------
# get_store() factory
# ---------------------------------------------------------------------------


def test_factory_defaults_to_duckdb(tmp_path, monkeypatch):
    from sentinel.config import load_secrets
    from sentinel.storage import get_store
    from sentinel.storage.duckdb_store import DuckDBStore

    # Clear the lru_cache so test env vars take effect.
    load_secrets.cache_clear()
    monkeypatch.delenv("SENTINEL_STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("SENTINEL_DB_PATH", str(tmp_path / "factory.duckdb"))

    store = get_store()
    assert isinstance(store, DuckDBStore)


def test_factory_dispatches_to_postgres_with_override():
    """Use overrides (not env) so we don't need real env manipulation."""
    from sentinel.storage import get_store
    from sentinel.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    factory = FakeFactory(conn)

    # Monkey-patch PostgresStore to accept our fake factory via overrides is
    # awkward; instead, build it directly and just check the factory raises
    # ValueError when no DSN is supplied.
    with pytest.raises(ValueError, match="SENTINEL_POSTGRES_DSN is not set"):
        get_store(backend="postgres")

    # With a DSN + a pre-built PostgresStore, we verify the postgres path
    # constructs one when we inject connect_factory via monkeypatch:
    import sentinel.storage as storage_mod

    orig_loader = storage_mod._load_postgres_store

    def fake_loader():
        def ctor(dsn, enable_timescale=True):
            return PostgresStore(
                dsn=dsn, enable_timescale=enable_timescale, connect_factory=factory
            )

        return ctor

    storage_mod._load_postgres_store = fake_loader  # type: ignore[assignment]
    try:
        store = get_store(backend="postgres", dsn="postgresql://fake/db")
    finally:
        storage_mod._load_postgres_store = orig_loader  # type: ignore[assignment]
    assert isinstance(store, PostgresStore)
    assert factory.calls  # DSN passed through


def test_factory_rejects_unknown_backend():
    from sentinel.storage import get_store

    with pytest.raises(ValueError, match="Unknown SENTINEL_STORAGE_BACKEND"):
        get_store(backend="mongodb")


def test_factory_respects_env_var(monkeypatch, tmp_path):
    """Round-trip via env vars — ensures the config surface is wired end-to-end."""
    from sentinel.config import load_secrets
    from sentinel.storage import get_store

    load_secrets.cache_clear()
    monkeypatch.setenv("SENTINEL_STORAGE_BACKEND", "postgres")
    monkeypatch.delenv("SENTINEL_POSTGRES_DSN", raising=False)

    with pytest.raises(ValueError, match="SENTINEL_POSTGRES_DSN is not set"):
        get_store()

    # Reset the cache for other tests.
    load_secrets.cache_clear()
