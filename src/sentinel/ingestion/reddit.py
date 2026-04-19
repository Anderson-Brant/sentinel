"""Reddit ingestion adapter.

Pulls recent posts from configured subreddits via `praw`, extracts ticker
mentions (via :mod:`sentinel.ingestion.mentions`), and persists both raw posts
and mention rows to the DuckDB store.

Design notes
------------
- ``praw`` is imported lazily inside :class:`RedditClient` so this module can
  be imported (and the rest of the test suite run) without the optional
  ``social`` extra installed.
- The fetch step is factored into a ``PostFetcher`` protocol so tests can
  inject a fake that returns canned records without touching the network.
- Sentiment scoring is intentionally *not* done here — raw posts land first,
  and :mod:`sentinel.features.sentiment` fills in the scores in a second pass.
  That separation keeps the network-bound step idempotent and lets sentiment
  re-runs happen without refetching Reddit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Protocol

import pandas as pd

from sentinel.config import IngestionRedditConfig, Secrets
from sentinel.ingestion.mentions import extract_mentions_for_records
from sentinel.storage.duckdb_store import DuckDBStore
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass — matches the reddit_posts table (sentiment columns filled later)
# ---------------------------------------------------------------------------


@dataclass
class RedditPost:
    post_id: str
    created_ts: datetime
    subreddit: str
    author: str | None
    title: str
    body: str
    score: int
    num_comments: int
    url: str

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure the timestamp is timezone-naive UTC — DuckDB TIMESTAMP is
        # naive by default, and mixing tz-aware and naive values errors out.
        if isinstance(d["created_ts"], datetime) and d["created_ts"].tzinfo is not None:
            d["created_ts"] = d["created_ts"].astimezone(timezone.utc).replace(tzinfo=None)
        return d


# ---------------------------------------------------------------------------
# Fetch protocol + praw-backed client
# ---------------------------------------------------------------------------


class PostFetcher(Protocol):
    """Anything that can yield RedditPost objects given subreddit + limit."""

    def fetch(
        self,
        subreddits: Iterable[str],
        *,
        limit: int,
    ) -> list[RedditPost]:  # pragma: no cover - protocol
        ...


class RedditClient:
    """Thin wrapper around praw. Lazy-imports praw on first use.

    Credentials come from :class:`Secrets` (env vars / .env). All three must
    be set — praw will refuse to authenticate otherwise.
    """

    def __init__(self, secrets: Secrets | None = None) -> None:
        self._secrets = secrets or Secrets()
        self._reddit = None  # lazily constructed

    # -- internals ------------------------------------------------------

    def _client(self):
        if self._reddit is not None:
            return self._reddit
        missing = [
            name
            for name, val in [
                ("REDDIT_CLIENT_ID", self._secrets.reddit_client_id),
                ("REDDIT_CLIENT_SECRET", self._secrets.reddit_client_secret),
                ("REDDIT_USER_AGENT", self._secrets.reddit_user_agent),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(
                "Reddit credentials missing: "
                + ", ".join(missing)
                + ". Set them in your environment or .env file."
            )
        try:
            import praw  # type: ignore
        except ImportError as e:  # pragma: no cover - import path
            raise RuntimeError(
                "The `praw` package is required for Reddit ingestion. "
                "Install with `pip install -e '.[social]'`."
            ) from e

        self._reddit = praw.Reddit(
            client_id=self._secrets.reddit_client_id,
            client_secret=self._secrets.reddit_client_secret,
            user_agent=self._secrets.reddit_user_agent,
            check_for_async=False,
        )
        # Read-only — we never post anything back.
        self._reddit.read_only = True
        return self._reddit

    # -- public ---------------------------------------------------------

    def fetch(
        self,
        subreddits: Iterable[str],
        *,
        limit: int,
    ) -> list[RedditPost]:
        """Fetch the newest ``limit`` submissions from each subreddit."""
        reddit = self._client()
        out: list[RedditPost] = []
        for sub in subreddits:
            log.info("Fetching up to %d posts from r/%s", limit, sub)
            try:
                for submission in reddit.subreddit(sub).new(limit=limit):
                    out.append(_submission_to_post(submission))
            except Exception as e:  # pragma: no cover - network errors
                log.warning("Error fetching r/%s: %s", sub, e)
        return out


def _submission_to_post(submission) -> RedditPost:
    """Convert a praw Submission into our dataclass. Tolerant to missing fields."""
    created = getattr(submission, "created_utc", None)
    created_ts = (
        datetime.fromtimestamp(created, tz=timezone.utc) if created else datetime.utcnow()
    )
    author = getattr(submission, "author", None)
    author_name = str(author) if author is not None else None
    return RedditPost(
        post_id=str(getattr(submission, "id", "")),
        created_ts=created_ts,
        subreddit=str(getattr(submission, "subreddit", "")),
        author=author_name,
        title=str(getattr(submission, "title", "") or ""),
        body=str(getattr(submission, "selftext", "") or ""),
        score=int(getattr(submission, "score", 0) or 0),
        num_comments=int(getattr(submission, "num_comments", 0) or 0),
        url=str(getattr(submission, "url", "") or ""),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def ingest_posts(
    *,
    store: DuckDBStore | None = None,
    reddit_cfg: IngestionRedditConfig | None = None,
    symbol_whitelist: set[str] | None = None,
    fetcher: PostFetcher | None = None,
) -> dict[str, int]:
    """Fetch Reddit posts, extract ticker mentions, persist both.

    Parameters
    ----------
    store : Target DuckDB store. Created with defaults if ``None``.
    reddit_cfg : Subreddit + limit settings. Defaults if ``None``.
    symbol_whitelist : Optional set of tickers to catch without a ``$`` prefix.
                       Highly recommended in production to cut noise.
    fetcher : Optional injected fetcher (tests pass a fake here).

    Returns
    -------
    Dict with counts: ``{"fetched", "posts_written", "mentions_written"}``.
    """
    store = store or DuckDBStore()
    reddit_cfg = reddit_cfg or IngestionRedditConfig()
    fetcher = fetcher or RedditClient()

    posts = fetcher.fetch(reddit_cfg.subreddits, limit=reddit_cfg.max_posts_per_run)
    log.info("Fetched %d total posts from Reddit", len(posts))

    if not posts:
        return {"fetched": 0, "posts_written": 0, "mentions_written": 0}

    # Build DataFrames for the storage layer.
    rows = [p.to_dict() for p in posts]
    posts_df = pd.DataFrame(rows)

    mention_records = extract_mentions_for_records(
        # `extract_mentions_for_records` expects an ``id`` field.
        [{"id": r["post_id"], "title": r["title"], "body": r["body"]} for r in rows],
        whitelist=symbol_whitelist,
    )
    mentions_df = (
        pd.DataFrame(mention_records) if mention_records else pd.DataFrame(columns=["post_id", "symbol"])
    )

    posts_written = store.write_reddit_posts(posts_df)
    mentions_written = store.write_mentions(mentions_df, source="reddit")

    log.info(
        "Wrote %d posts (+ %d mention rows across %d unique tickers)",
        posts_written,
        mentions_written,
        mentions_df["symbol"].nunique() if not mentions_df.empty else 0,
    )
    return {
        "fetched": len(posts),
        "posts_written": posts_written,
        "mentions_written": mentions_written,
    }


__all__ = [
    "RedditPost",
    "RedditClient",
    "PostFetcher",
    "ingest_posts",
]
