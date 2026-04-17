"""DuckDB-backed storage layer.

Schema is intentionally small for the MVP. Tables:

    prices(symbol VARCHAR, date DATE, open, high, low, close, adj_close, volume,
           PRIMARY KEY (symbol, date))

    features(symbol VARCHAR, date DATE, <many float columns>, target_direction,
             target_return, PRIMARY KEY (symbol, date))

Writes are idempotent via DELETE + INSERT per (symbol).
"""

from __future__ import annotations

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
                try:
                    con.execute("ALTER TABLE features ADD PRIMARY KEY (symbol, date)")
                except Exception:  # noqa: BLE001
                    pass
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
    # introspection
    # ------------------------------------------------------------------

    def list_symbols(self) -> list[str]:
        with self._connect() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
