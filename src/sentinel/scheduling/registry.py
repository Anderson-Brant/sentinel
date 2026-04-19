"""Job registry — mapping ``kind`` string → callable.

Each registered job is a ``JobFn`` with the signature::

    job_fn(store: Store, **params) -> JobResult

where ``JobResult`` is a plain dict with:

* ``rows_written`` — integer row count written to the store.
* ``detail`` — short human-readable one-liner for logs / CLI.

Keeping the registry data-driven means adding a new scheduled operation is a
one-line change here plus the function itself; no branching in the scheduler
core or CLI.

Imports for the actual ingestion / feature pipeline are *inside* each job
function so importing this module is cheap and doesn't pull in yfinance / praw
for callers that only need to register their own kinds.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from sentinel.storage.base import Store
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


class JobResult(TypedDict, total=False):
    rows_written: int
    detail: str


JobFn = Callable[..., JobResult]


_REGISTRY: dict[str, JobFn] = {}


def register(kind: str) -> Callable[[JobFn], JobFn]:
    """Decorator to register a job function for a ``kind`` string."""

    def _wrap(fn: JobFn) -> JobFn:
        if kind in _REGISTRY:
            raise ValueError(f"Job kind {kind!r} already registered")
        _REGISTRY[kind] = fn
        return fn

    return _wrap


def get_job(kind: str) -> JobFn:
    try:
        return _REGISTRY[kind]
    except KeyError as e:
        raise KeyError(
            f"Unknown job kind {kind!r}. Registered: {sorted(_REGISTRY)}"
        ) from e


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in jobs
# ---------------------------------------------------------------------------


@register("ingest-prices")
def ingest_prices_job(
    store: Store,
    *,
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
) -> JobResult:
    """Pull OHLCV for each symbol and persist it.

    Each symbol is a separate yfinance call; failures are aggregated into the
    detail string but do not abort subsequent symbols — we'd rather refresh
    SPY than let a bad ticker block the whole run.
    """
    from sentinel.config import load_config
    from sentinel.ingestion.market import ingest_prices as _ingest

    cfg = load_config()
    start = start or cfg.ingestion.market.default_start
    interval = interval or cfg.ingestion.market.default_interval

    total = 0
    failed: list[str] = []
    for symbol in symbols:
        try:
            df = _ingest(symbol, start=start, end=end, interval=interval)
            n = store.write_prices(symbol, df)
            total += n
        except Exception as e:  # noqa: BLE001 — per-symbol isolation
            failed.append(f"{symbol}: {e}")
            log.warning("ingest-prices failed for %s: %s", symbol, e)

    ok = len(symbols) - len(failed)
    detail = f"{ok}/{len(symbols)} symbols ok, {total} rows"
    if failed:
        detail += f" (failures: {'; '.join(failed)})"
    return {"rows_written": total, "detail": detail}


@register("ingest-reddit")
def ingest_reddit_job(
    store: Store,
    *,
    whitelist: list[str] | None = None,
    subreddits: list[str] | None = None,
    limit: int | None = None,
) -> JobResult:
    """Fetch Reddit posts, extract ticker mentions, persist both."""
    from sentinel.config import IngestionRedditConfig, load_config
    from sentinel.ingestion.reddit import ingest_posts

    cfg = load_config()
    reddit_cfg = cfg.ingestion.reddit
    overrides: dict[str, Any] = {}
    if subreddits is not None:
        overrides["subreddits"] = list(subreddits)
    if limit is not None:
        overrides["max_posts_per_run"] = int(limit)
    if overrides:
        reddit_cfg = IngestionRedditConfig(
            **{**reddit_cfg.model_dump(), **overrides}
        )

    wl = {str(t).upper() for t in whitelist} if whitelist else None
    counts = ingest_posts(store=store, reddit_cfg=reddit_cfg, symbol_whitelist=wl)

    total = counts.get("posts_written", 0) + counts.get("mentions_written", 0)
    detail = (
        f"fetched={counts.get('fetched', 0)}, "
        f"posts={counts.get('posts_written', 0)}, "
        f"mentions={counts.get('mentions_written', 0)}"
    )
    return {"rows_written": total, "detail": detail}


@register("ingest-twitter")
def ingest_twitter_job(
    store: Store,
    *,
    whitelist: list[str] | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> JobResult:
    """Fetch tweets matching a cashtag query, extract mentions, persist both."""
    from sentinel.config import IngestionTwitterConfig, load_config
    from sentinel.ingestion.twitter import ingest_tweets

    cfg = load_config()
    twitter_cfg = cfg.ingestion.twitter
    if limit is not None:
        twitter_cfg = IngestionTwitterConfig(
            **{**twitter_cfg.model_dump(), "max_tweets_per_run": int(limit)}
        )

    wl = {str(t).upper() for t in whitelist} if whitelist else None
    counts = ingest_tweets(
        store=store, twitter_cfg=twitter_cfg, symbol_whitelist=wl, query=query
    )

    total = counts.get("tweets_written", 0) + counts.get("mentions_written", 0)
    detail = (
        f"fetched={counts.get('fetched', 0)}, "
        f"tweets={counts.get('tweets_written', 0)}, "
        f"mentions={counts.get('mentions_written', 0)}"
    )
    return {"rows_written": total, "detail": detail}


@register("ingest-crypto")
def ingest_crypto_job(
    store: Store,
    *,
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
    exchange: str | None = None,
    quote: str | None = None,
) -> JobResult:
    """Pull crypto OHLCV for each symbol via CCXT and persist it.

    Crypto bars share the ``prices`` table with equities — symbols are stored
    in yfinance-style (``BTC-USD``) regardless of which stablecoin the
    exchange quotes in.
    """
    from sentinel.config import load_config
    from sentinel.ingestion.crypto import ingest_crypto_prices

    cfg = load_config()
    cc = cfg.ingestion.crypto
    start = start or cc.default_start
    interval = interval or cc.default_interval
    exchange = exchange or cc.default_exchange
    quote = quote or cc.default_quote

    total = 0
    failed: list[str] = []
    for symbol in symbols:
        try:
            df = ingest_crypto_prices(
                symbol,
                start=start,
                end=end,
                timeframe=interval,
                exchange=exchange,
                default_quote=quote,
            )
            stored_symbol = (
                str(df["symbol"].iloc[0]) if not df.empty else symbol.upper()
            )
            n = store.write_prices(stored_symbol, df)
            total += n
        except Exception as e:  # noqa: BLE001 — per-symbol isolation
            failed.append(f"{symbol}: {e}")
            log.warning("ingest-crypto failed for %s: %s", symbol, e)

    ok = len(symbols) - len(failed)
    detail = f"{ok}/{len(symbols)} symbols ok, {total} rows"
    if failed:
        detail += f" (failures: {'; '.join(failed)})"
    return {"rows_written": total, "detail": detail}


@register("score-sentiment")
def score_sentiment_job(
    store: Store,
    *,
    scorer: str = "vader",
    model_name: str | None = None,
    batch_size: int = 16,
) -> JobResult:
    """Re-score all stored Reddit posts AND tweets in a single pass."""
    from sentinel.features.sentiment import score_posts, score_tweets

    posts = store.read_all_reddit_posts()
    tweets = store.read_all_tweets()
    if posts.empty and tweets.empty:
        return {"rows_written": 0, "detail": "no Reddit posts or tweets in storage"}

    fb = None
    if scorer.lower() != "vader":
        from sentinel.features.finbert import DEFAULT_MODEL, FinBertScorer

        fb = FinBertScorer(
            model_name=model_name or DEFAULT_MODEL, batch_size=batch_size
        )

    n_posts = 0
    if not posts.empty:
        scored = score_posts(posts) if fb is None else score_posts(posts, scorer=fb)
        n_posts = store.update_reddit_sentiment(scored)

    n_tweets = 0
    if not tweets.empty:
        scored_t = (
            score_tweets(tweets) if fb is None else score_tweets(tweets, scorer=fb)
        )
        n_tweets = store.update_tweet_sentiment(scored_t)

    total = n_posts + n_tweets
    return {
        "rows_written": total,
        "detail": f"scored {n_posts} posts + {n_tweets} tweets with {scorer}",
    }


@register("build-features")
def build_features_job(
    store: Store,
    *,
    symbols: list[str],
    with_sentiment: bool = False,
) -> JobResult:
    """Recompute the feature table for each symbol and persist it."""
    import pandas as pd

    from sentinel.config import load_config
    from sentinel.features.pipeline import build_feature_table
    from sentinel.features.sentiment import sentiment_features_for_symbol

    cfg = load_config()
    total = 0
    failed: list[str] = []
    for symbol in symbols:
        try:
            prices = store.read_prices(symbol)
            if prices.empty:
                failed.append(f"{symbol}: no prices")
                continue
            sentiment_df = None
            if with_sentiment:
                sentiment_df = sentiment_features_for_symbol(
                    store, symbol, index=pd.to_datetime(prices.index)
                )
            features = build_feature_table(prices, cfg=cfg, sentiment=sentiment_df)
            n = store.write_features(symbol, features)
            total += n
        except Exception as e:  # noqa: BLE001 — per-symbol isolation
            failed.append(f"{symbol}: {e}")
            log.warning("build-features failed for %s: %s", symbol, e)

    ok = len(symbols) - len(failed)
    detail = f"{ok}/{len(symbols)} symbols ok, {total} feature rows"
    if failed:
        detail += f" (failures: {'; '.join(failed)})"
    return {"rows_written": total, "detail": detail}


__all__ = [
    "JobResult",
    "JobFn",
    "register",
    "get_job",
    "registered_kinds",
]
