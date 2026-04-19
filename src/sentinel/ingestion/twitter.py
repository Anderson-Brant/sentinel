"""Twitter / X ingestion adapter.

Pulls recent tweets matching cashtags (or an explicit query) via `tweepy` v2,
extracts ticker mentions (via :mod:`sentinel.ingestion.mentions`), and
persists both raw tweets and mention rows to the configured store.

Design notes
------------
- ``tweepy`` is imported lazily inside :class:`TwitterClient` so this module
  can be imported (and the rest of the test suite run) without the optional
  ``social`` extra installed.
- The fetch step is factored into a :class:`TweetFetcher` protocol so tests
  can inject a fake that returns canned records without touching the network.
  This mirrors :class:`sentinel.ingestion.reddit.PostFetcher`.
- Sentiment scoring is intentionally *not* done here — raw tweets land first,
  and :mod:`sentinel.features.sentiment` fills in the scores in a second pass.
  That separation keeps the network-bound step idempotent and lets sentiment
  re-runs happen without re-hitting the Twitter API.
- Query construction: when no explicit ``query`` is passed, we build one from
  the whitelist by OR-ing cashtags (``$AAPL OR $TSLA``). The v2 ``recent
  search`` endpoint allows up to 512 characters per query, which is plenty
  for a resume-grade whitelist of tickers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from sentinel.config import IngestionTwitterConfig, Secrets
from sentinel.ingestion.mentions import extract_mentions_for_records
from sentinel.storage.base import Store
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass — matches the tweets table (sentiment columns filled later)
# ---------------------------------------------------------------------------


@dataclass
class Tweet:
    """One tweet, ready for storage.

    Field names match :data:`sentinel.storage.base.TWEET_COLUMNS` (minus the
    four ``sentiment_*`` columns which are filled by a later scoring pass).
    """

    tweet_id: str
    created_ts: datetime
    author_id: str | None
    author_username: str | None
    text: str
    lang: str | None
    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int
    impression_count: int

    def to_dict(self) -> dict:
        d = asdict(self)
        # Storage uses naive UTC timestamps — tweepy returns tz-aware datetimes.
        if isinstance(d["created_ts"], datetime) and d["created_ts"].tzinfo is not None:
            d["created_ts"] = d["created_ts"].astimezone(UTC).replace(tzinfo=None)
        return d


# ---------------------------------------------------------------------------
# Fetch protocol + tweepy-backed client
# ---------------------------------------------------------------------------


class TweetFetcher(Protocol):
    """Anything that can yield Tweet objects given a query + limit."""

    def fetch(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[Tweet]:  # pragma: no cover - protocol
        ...


class TwitterClient:
    """Thin wrapper around tweepy v2. Lazy-imports tweepy on first use.

    Requires ``TWITTER_BEARER_TOKEN`` in the environment (or .env). The
    client is read-only — we never post anything back.
    """

    def __init__(self, secrets: Secrets | None = None) -> None:
        self._secrets = secrets or Secrets()
        self._client = None  # lazily constructed

    # -- internals ------------------------------------------------------

    def _twitter(self):
        if self._client is not None:
            return self._client
        if not self._secrets.twitter_bearer_token:
            raise RuntimeError(
                "Twitter credentials missing: TWITTER_BEARER_TOKEN. "
                "Set it in your environment or .env file."
            )
        try:
            import tweepy  # type: ignore
        except ImportError as e:  # pragma: no cover - import path
            raise RuntimeError(
                "The `tweepy` package is required for Twitter ingestion. "
                "Install with `pip install -e '.[social]'`."
            ) from e

        self._client = tweepy.Client(
            bearer_token=self._secrets.twitter_bearer_token,
            wait_on_rate_limit=True,
        )
        return self._client

    # -- public ---------------------------------------------------------

    def fetch(self, query: str, *, limit: int) -> list[Tweet]:
        """Run a recent-search against the v2 API, paginating up to ``limit``."""
        import tweepy  # type: ignore  # pragma: no cover - handled by _twitter()

        client = self._twitter()
        out: list[Tweet] = []
        log.info("Twitter recent_search: %r (limit=%d)", query, limit)

        # Per v2 docs, max_results is 10..100 per page. Let Paginator handle pages.
        try:
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=[
                    "id",
                    "text",
                    "created_at",
                    "author_id",
                    "lang",
                    "public_metrics",
                ],
                expansions=["author_id"],
                user_fields=["username"],
                max_results=min(max(10, limit), 100),
                limit=max(1, -(-limit // 100)),  # ceil(limit/100) pages
            )
        except Exception as e:  # pragma: no cover - network
            log.warning("Twitter paginator construction failed: %s", e)
            return out

        seen = 0
        for resp in paginator:
            if resp is None or resp.data is None:
                continue
            users = {
                u.id: u for u in (getattr(resp, "includes", {}) or {}).get("users", [])
            }
            for t in resp.data:
                if seen >= limit:
                    break
                out.append(_tweet_to_record(t, users))
                seen += 1
            if seen >= limit:
                break
        return out


def _tweet_to_record(t, users: dict) -> Tweet:
    """Convert a tweepy v2 Tweet into our dataclass. Tolerant to missing fields."""
    created = getattr(t, "created_at", None)
    created_ts = (
        created
        if isinstance(created, datetime)
        else datetime.now(tz=UTC)
    )
    metrics = getattr(t, "public_metrics", {}) or {}
    author_id = getattr(t, "author_id", None)
    user = users.get(author_id) if author_id is not None else None
    username = str(getattr(user, "username", "")) if user is not None else None

    return Tweet(
        tweet_id=str(getattr(t, "id", "")),
        created_ts=created_ts,
        author_id=str(author_id) if author_id is not None else None,
        author_username=username or None,
        text=str(getattr(t, "text", "") or ""),
        lang=str(getattr(t, "lang", "") or "") or None,
        retweet_count=int(metrics.get("retweet_count", 0) or 0),
        reply_count=int(metrics.get("reply_count", 0) or 0),
        like_count=int(metrics.get("like_count", 0) or 0),
        quote_count=int(metrics.get("quote_count", 0) or 0),
        impression_count=int(metrics.get("impression_count", 0) or 0),
    )


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def build_cashtag_query(whitelist: Iterable[str]) -> str:
    """Turn a ticker whitelist into a v2 recent-search query.

    The v2 operator for cashtags is ``$TICKER``. We OR them together and
    drop retweets (``-is:retweet``) since RTs duplicate the underlying text
    and would inflate mention counts without adding signal.
    """
    tickers = sorted({str(t).upper() for t in whitelist if t})
    if not tickers:
        raise ValueError(
            "Cannot build a Twitter query with no tickers. Pass --whitelist "
            "or a non-empty symbol whitelist."
        )
    cashtags = " OR ".join(f"${t}" for t in tickers)
    return f"({cashtags}) -is:retweet lang:en"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def ingest_tweets(
    *,
    store: Store | None = None,
    twitter_cfg: IngestionTwitterConfig | None = None,
    symbol_whitelist: Iterable[str] | None = None,
    query: str | None = None,
    fetcher: TweetFetcher | None = None,
) -> dict[str, int]:
    """Fetch tweets, extract ticker mentions, persist both.

    Parameters
    ----------
    store : Target store. Created via ``get_store()`` if ``None``.
    twitter_cfg : Twitter ingest settings (currently just ``max_tweets_per_run``).
    symbol_whitelist : Tickers to track. Used both to build the search query
        (cashtag OR) and to filter mention extraction to known tickers — so
        we never record spurious tickers like "$LOL".
    query : Optional raw query string. If supplied it replaces the one built
        from the whitelist; the whitelist is still used for mention filtering.
    fetcher : Optional injected fetcher (tests pass a fake here).

    Returns
    -------
    Dict with counts: ``{"fetched", "tweets_written", "mentions_written"}``.
    """
    if store is None:
        from sentinel.storage import get_store  # local import — avoids cycle
        store = get_store()

    twitter_cfg = twitter_cfg or IngestionTwitterConfig()
    fetcher = fetcher or TwitterClient()

    wl_set = {str(t).upper() for t in symbol_whitelist} if symbol_whitelist else None
    if query is None:
        if not wl_set:
            raise ValueError(
                "ingest_tweets() needs either a `query` or a non-empty "
                "`symbol_whitelist` to build one from."
            )
        query = build_cashtag_query(wl_set)

    tweets = fetcher.fetch(query, limit=twitter_cfg.max_tweets_per_run)
    log.info("Fetched %d total tweets", len(tweets))

    if not tweets:
        return {"fetched": 0, "tweets_written": 0, "mentions_written": 0}

    rows = [t.to_dict() for t in tweets]
    tweets_df = pd.DataFrame(rows)

    # Twitter has no "title" — everything is in ``text``. The mentions
    # extractor already handles a single text field via ``text_fields``.
    mention_records = extract_mentions_for_records(
        [{"id": r["tweet_id"], "text": r["text"]} for r in rows],
        text_fields=("text",),
        whitelist=wl_set,
    )
    mentions_df = (
        pd.DataFrame(mention_records)
        if mention_records
        else pd.DataFrame(columns=["post_id", "symbol"])
    )

    tweets_written = store.write_tweets(tweets_df)
    mentions_written = store.write_mentions(mentions_df, source="twitter")

    log.info(
        "Wrote %d tweets (+ %d mention rows across %d unique tickers)",
        tweets_written,
        mentions_written,
        mentions_df["symbol"].nunique() if not mentions_df.empty else 0,
    )
    return {
        "fetched": len(tweets),
        "tweets_written": tweets_written,
        "mentions_written": mentions_written,
    }


__all__ = [
    "Tweet",
    "TwitterClient",
    "TweetFetcher",
    "build_cashtag_query",
    "ingest_tweets",
]
