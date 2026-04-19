"""Configuration loading.

Precedence (highest wins):
  1. Explicit argument passed to ``load_config(path=...)``.
  2. ``SENTINEL_CONFIG`` environment variable.
  3. ``config/default.yaml`` at the repo root.

Secrets (API keys, etc.) come from environment variables / .env, never YAML.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env at import time so all downstream modules see it.
load_dotenv()


def repo_root() -> Path:
    """Return the repository root (where pyproject.toml lives)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fall back to cwd if we're installed site-packages style.
    return Path.cwd()


# ---------------------------------------------------------------------------
# Secrets / runtime settings — from env / .env
# ---------------------------------------------------------------------------


class Secrets(BaseSettings):
    """Env-backed settings. Missing values are fine for the MVP loop."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_path: str = Field(default="data/sentinel.duckdb", alias="SENTINEL_DB_PATH")
    log_level: str = Field(default="INFO", alias="SENTINEL_LOG_LEVEL")

    # Storage backend. "duckdb" keeps the zero-setup local behavior;
    # "postgres" switches to the Postgres / TimescaleDB backend using
    # `postgres_dsn`.
    storage_backend: str = Field(default="duckdb", alias="SENTINEL_STORAGE_BACKEND")
    postgres_dsn: str | None = Field(default=None, alias="SENTINEL_POSTGRES_DSN")
    postgres_enable_timescale: bool = Field(
        default=True, alias="SENTINEL_POSTGRES_TIMESCALE"
    )

    reddit_client_id: str | None = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str | None = Field(default=None, alias="REDDIT_USER_AGENT")

    twitter_bearer_token: str | None = Field(default=None, alias="TWITTER_BEARER_TOKEN")


# ---------------------------------------------------------------------------
# YAML config — declarative knobs for features/models
# ---------------------------------------------------------------------------


class IngestionMarketConfig(BaseModel):
    default_start: str = "2015-01-01"
    default_interval: str = "1d"


class IngestionRedditConfig(BaseModel):
    subreddits: list[str] = Field(default_factory=lambda: ["wallstreetbets"])
    max_posts_per_run: int = 500


class IngestionTwitterConfig(BaseModel):
    max_tweets_per_run: int = 500


class IngestionCryptoConfig(BaseModel):
    """Crypto OHLCV ingestion via CCXT.

    ``default_quote`` is the exchange-side quote currency (most venues trade
    USDT/USDC, not raw USD). Symbols are stored in yfinance-style ``BTC-USD``
    so downstream features are asset-class agnostic.
    """

    default_start: str = "2020-01-01"
    default_interval: str = "1d"
    default_exchange: str = "binance"
    default_quote: str = "USDT"


class IngestionConfig(BaseModel):
    market: IngestionMarketConfig = Field(default_factory=IngestionMarketConfig)
    reddit: IngestionRedditConfig = Field(default_factory=IngestionRedditConfig)
    twitter: IngestionTwitterConfig = Field(default_factory=IngestionTwitterConfig)
    crypto: IngestionCryptoConfig = Field(default_factory=IngestionCryptoConfig)


class FeaturesConfig(BaseModel):
    returns: dict[str, list[int]] = Field(default_factory=lambda: {"windows": [1, 5, 10, 20]})
    moving_averages: dict[str, list[int]] = Field(
        default_factory=lambda: {"sma_windows": [5, 10, 20, 50, 200], "ema_windows": [12, 26]}
    )
    volatility: dict[str, list[int]] = Field(default_factory=lambda: {"windows": [5, 10, 20]})
    momentum: dict[str, list[int]] = Field(default_factory=lambda: {"windows": [5, 10, 20]})
    volume: dict[str, list[int]] = Field(default_factory=lambda: {"windows": [5, 20]})


class TargetsConfig(BaseModel):
    horizon_days: int = 1


class WalkForwardConfig(BaseModel):
    n_splits: int = 5
    min_train_size: int = 252


class ModelingConfig(BaseModel):
    default_model: str = "logistic"
    random_state: int = 42
    test_fraction: float = 0.2
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)


class ScheduledJobConfig(BaseModel):
    """One entry in ``scheduler.jobs``. The ``kind`` string maps to a job
    function in :mod:`sentinel.scheduling.registry`; ``params`` is forwarded
    verbatim as kwargs."""

    name: str
    kind: str
    interval: str | int | float  # "15m", "1h", bare seconds — parsed at load
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class SchedulerConfig(BaseModel):
    """``scheduler:`` section. Empty by default — the section is optional."""

    jobs: list[ScheduledJobConfig] = Field(default_factory=list)
    tick_seconds: int = 30


class SentinelConfig(BaseModel):
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("SENTINEL_CONFIG")
    if env_path:
        return Path(env_path)
    return repo_root() / "config" / "default.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> SentinelConfig:
    """Load + validate the YAML config. Cached per process."""
    cfg_path = _resolve_config_path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        raw = {}
    return SentinelConfig.model_validate(raw)


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    return Secrets()


def resolve_db_path() -> Path:
    """Return an absolute path to the DuckDB file, creating parent dirs as needed."""
    raw = load_secrets().db_path
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
