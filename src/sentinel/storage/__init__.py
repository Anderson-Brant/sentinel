"""Storage adapters. DuckDB is the MVP backend; Postgres is roadmap."""

from sentinel.storage.duckdb_store import DuckDBStore

__all__ = ["DuckDBStore"]
