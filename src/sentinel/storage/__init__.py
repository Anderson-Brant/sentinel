"""Storage adapters.

Two backends ship in-tree:

* :class:`~sentinel.storage.duckdb_store.DuckDBStore` — default. Zero
  setup, backed by a single file. Appropriate for local development and
  single-user research work.
* :class:`~sentinel.storage.postgres_store.PostgresStore` — opt-in
  Postgres / TimescaleDB backend. Appropriate for multi-user, scheduled,
  or deployed setups. Enabled by setting
  ``SENTINEL_STORAGE_BACKEND=postgres`` and ``SENTINEL_POSTGRES_DSN``.

Callers should go through :func:`get_store`; the factory reads secrets
and instantiates the right backend. The ``DuckDBStore`` / ``PostgresStore``
classes remain importable directly for tests and advanced usage.
"""

from __future__ import annotations

from typing import Any

from sentinel.storage.base import Store
from sentinel.storage.duckdb_store import DuckDBStore


def _load_postgres_store() -> type:
    # Lazy-imported so the Postgres module's `psycopg` import doesn't fire
    # on every `from sentinel.storage import ...` when the user is just
    # running the DuckDB default path.
    from sentinel.storage.postgres_store import PostgresStore

    return PostgresStore


def get_store(**overrides: Any) -> Store:
    """Return a backend instance selected by the ``SENTINEL_STORAGE_BACKEND``
    setting (default ``duckdb``).

    ``overrides`` lets tests inject config without setting env vars:

        * ``backend``: ``"duckdb"`` or ``"postgres"``.
        * ``dsn``: Postgres DSN (required when backend is postgres).
        * ``enable_timescale``: bool, forwarded to ``PostgresStore``.
        * ``path``: filesystem path for ``DuckDBStore``.
    """
    from sentinel.config import load_secrets

    secrets = load_secrets()
    backend = (overrides.get("backend") or secrets.storage_backend or "duckdb").lower()

    if backend == "duckdb":
        return DuckDBStore(path=overrides.get("path"))

    if backend == "postgres":
        dsn = overrides.get("dsn") or secrets.postgres_dsn
        if not dsn:
            raise ValueError(
                "SENTINEL_POSTGRES_DSN is not set but "
                "SENTINEL_STORAGE_BACKEND=postgres. "
                "Set the DSN (postgresql://user:pass@host/db) in your environment."
            )
        enable_ts = overrides.get(
            "enable_timescale",
            getattr(secrets, "postgres_enable_timescale", True),
        )
        PostgresStore = _load_postgres_store()
        return PostgresStore(dsn=dsn, enable_timescale=bool(enable_ts))

    raise ValueError(
        f"Unknown SENTINEL_STORAGE_BACKEND={backend!r}. Use 'duckdb' or 'postgres'."
    )


__all__ = ["Store", "DuckDBStore", "get_store"]
