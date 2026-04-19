"""Sentiment feature generation.

Public entry points:

    1. :func:`score_posts` — VADER-scores each raw Reddit post. Intended to be
       called after :func:`sentinel.ingestion.reddit.ingest_posts` has landed
       rows in ``reddit_posts``.
    2. :func:`score_tweets` — VADER-scores each raw tweet. Mirror of
       :func:`score_posts` for the ``tweets`` table.
    3. :func:`sentiment_features_for_symbol` — rolls up the scored posts and
       tweets into a per-date feature block that joins cleanly onto the
       technical table.

The scoring and aggregation steps are deliberately separate: scoring is a
per-row transform that can be re-run independently (e.g. after upgrading the
scorer), while aggregation is a per-(symbol, date) reduction that belongs in
the features layer.

VADER (`vaderSentiment`) is imported lazily so the module itself imports
without the ``social`` extra installed. A simple fallback scorer can be
injected for tests that don't want to pull the dependency in.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from sentinel.storage.base import Store
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class Scorer(Protocol):
    """Anything that maps a text string to a VADER-style dict."""

    def polarity_scores(self, text: str) -> dict[str, float]:  # pragma: no cover
        ...


def _vader_scorer() -> Scorer:
    """Lazy-construct a VADER SentimentIntensityAnalyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except ImportError as e:  # pragma: no cover - import path
        raise RuntimeError(
            "`vaderSentiment` is required for sentiment scoring. "
            "Install with `pip install -e '.[social]'`."
        ) from e
    return SentimentIntensityAnalyzer()


def _score_rows(
    df: pd.DataFrame,
    *,
    id_col: str,
    text_fields: tuple[str, ...],
    scorer: Scorer | None,
) -> pd.DataFrame:
    """Internal: score arbitrary rows, emitting ``id_col`` + sentiment cols."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                id_col,
                "sentiment_compound",
                "sentiment_pos",
                "sentiment_neg",
                "sentiment_neu",
            ]
        )

    scorer = scorer or _vader_scorer()

    out = []
    for _, row in df.iterrows():
        text = " ".join(str(row.get(f) or "") for f in text_fields).strip()
        scores = scorer.polarity_scores(text) if text else {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 0.0,
        }
        out.append(
            {
                id_col: row[id_col],
                "sentiment_compound": float(scores.get("compound", 0.0)),
                "sentiment_pos": float(scores.get("pos", 0.0)),
                "sentiment_neg": float(scores.get("neg", 0.0)),
                "sentiment_neu": float(scores.get("neu", 0.0)),
            }
        )
    return pd.DataFrame(out)


def score_posts(
    posts: pd.DataFrame,
    *,
    scorer: Scorer | None = None,
    text_fields: tuple[str, ...] = ("title", "body"),
) -> pd.DataFrame:
    """Attach VADER scores to each row in ``posts``.

    Returns a DataFrame with columns ``post_id``, ``sentiment_compound``,
    ``sentiment_pos``, ``sentiment_neg``, ``sentiment_neu`` — ready to feed
    into :meth:`Store.update_reddit_sentiment`.

    Title and body are concatenated before scoring because post titles on
    r/wallstreetbets often carry the whole sentiment signal ("GME TO THE
    MOON") with an empty body.
    """
    return _score_rows(posts, id_col="post_id", text_fields=text_fields, scorer=scorer)


def score_tweets(
    tweets: pd.DataFrame,
    *,
    scorer: Scorer | None = None,
    text_fields: tuple[str, ...] = ("text",),
) -> pd.DataFrame:
    """Attach VADER scores to each row in ``tweets``.

    Returns a DataFrame with columns ``tweet_id``, ``sentiment_compound``,
    ``sentiment_pos``, ``sentiment_neg``, ``sentiment_neu`` — ready to feed
    into :meth:`Store.update_tweet_sentiment`.
    """
    return _score_rows(tweets, id_col="tweet_id", text_fields=text_fields, scorer=scorer)


# ---------------------------------------------------------------------------
# Aggregation — per-symbol, per-date rollups
# ---------------------------------------------------------------------------


# Reddit-sourced columns. Stable — the ablation harness refers to these by name.
REDDIT_SENTIMENT_FEATURE_COLS = (
    "reddit_mention_count",
    "reddit_sentiment_mean",
    "reddit_sentiment_wmean",        # engagement-weighted (score + num_comments)
    "reddit_sentiment_pos_ratio",
    "reddit_sentiment_neg_ratio",
    "reddit_mention_zscore_20",      # mention count vs trailing 20-day mean/std
    "reddit_sentiment_mean_5",       # 5-day rolling sentiment mean
    "reddit_sentiment_mean_20",      # 20-day rolling sentiment mean
)


# Twitter-sourced columns. Mirrors the Reddit set — same shape, same derivation.
TWITTER_SENTIMENT_FEATURE_COLS = (
    "twitter_mention_count",
    "twitter_sentiment_mean",
    "twitter_sentiment_wmean",        # engagement-weighted (like + retweet + reply)
    "twitter_sentiment_pos_ratio",
    "twitter_sentiment_neg_ratio",
    "twitter_mention_zscore_20",
    "twitter_sentiment_mean_5",
    "twitter_sentiment_mean_20",
)


# Combined — everything the sentiment block emits. Consumed by the ablation
# harness to distinguish "sentiment" columns from "technical" ones.
SENTIMENT_FEATURE_COLS = REDDIT_SENTIMENT_FEATURE_COLS + TWITTER_SENTIMENT_FEATURE_COLS


# Columns that represent raw mention counts and should be zero-filled on days
# with no data (rather than NaN-filled, which would drop the price row).
MENTION_COUNT_COLS = ("reddit_mention_count", "twitter_mention_count")


def _daily_rollup(
    posts: pd.DataFrame,
    *,
    prefix: str,
    weight_expr,
) -> pd.DataFrame:
    """Collapse (post × date) → one row per date, emitting ``<prefix>_*`` cols.

    Parameters
    ----------
    posts : DataFrame with columns ``created_ts``, ``sentiment_compound``,
        ``sentiment_pos``, ``sentiment_neg`` plus whatever fields ``weight_expr``
        needs to compute per-row engagement weight.
    prefix : column prefix — ``reddit`` or ``twitter``.
    weight_expr : callable(df) → pd.Series of non-negative engagement weights.
    """
    df = posts.copy()
    df["created_ts"] = pd.to_datetime(df["created_ts"])
    df["date"] = df["created_ts"].dt.floor("D")

    # Engagement weight. +1 so an all-zero day still produces a finite mean.
    df["_w"] = weight_expr(df).astype(float) + 1.0
    df["_compound"] = df["sentiment_compound"].fillna(0.0)
    df["_wc"] = df["_w"] * df["_compound"]

    grouped = df.groupby("date")
    wmean = grouped["_wc"].sum() / grouped["_w"].sum()

    return pd.DataFrame(
        {
            f"{prefix}_mention_count": grouped.size(),
            f"{prefix}_sentiment_mean": grouped["sentiment_compound"].mean(),
            f"{prefix}_sentiment_wmean": wmean,
            f"{prefix}_sentiment_pos_ratio": grouped["sentiment_pos"].mean(),
            f"{prefix}_sentiment_neg_ratio": grouped["sentiment_neg"].mean(),
        }
    )


def _add_rolling(daily: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Attach 5d/20d rolling means + 20d mention z-score to a daily frame."""
    mention_col = f"{prefix}_mention_count"
    mean_col = f"{prefix}_sentiment_mean"

    daily[f"{prefix}_sentiment_mean_5"] = (
        daily[mean_col].rolling(5, min_periods=1).mean()
    )
    daily[f"{prefix}_sentiment_mean_20"] = (
        daily[mean_col].rolling(20, min_periods=1).mean()
    )

    rolling_mean = daily[mention_col].rolling(20, min_periods=5).mean()
    rolling_std = daily[mention_col].rolling(20, min_periods=5).std()
    daily[f"{prefix}_mention_zscore_20"] = (
        (daily[mention_col] - rolling_mean) / rolling_std.replace(0, np.nan)
    )
    return daily


def _reddit_weights(df: pd.DataFrame) -> pd.Series:
    """Engagement weight for a Reddit post. Clip downvoted scores to 0 so they
    don't *amplify* sentiment in the opposite direction."""
    return (
        np.clip(df["score"].fillna(0), 0, None)
        + df["num_comments"].fillna(0)
    )


def _twitter_weights(df: pd.DataFrame) -> pd.Series:
    """Engagement weight for a tweet. Sum of like, retweet and reply counts."""
    parts = []
    for col in ("like_count", "retweet_count", "reply_count"):
        if col in df.columns:
            parts.append(df[col].fillna(0))
        else:
            parts.append(pd.Series(0.0, index=df.index))
    return sum(parts)


def _reddit_block(posts: pd.DataFrame) -> pd.DataFrame:
    """Reddit subset of the combined sentiment block."""
    if posts.empty:
        return pd.DataFrame(columns=list(REDDIT_SENTIMENT_FEATURE_COLS))
    daily = _daily_rollup(posts, prefix="reddit", weight_expr=_reddit_weights)
    daily = _add_rolling(daily, prefix="reddit")
    return daily.reindex(columns=list(REDDIT_SENTIMENT_FEATURE_COLS))


def _twitter_block(tweets: pd.DataFrame) -> pd.DataFrame:
    """Twitter subset of the combined sentiment block."""
    if tweets.empty:
        return pd.DataFrame(columns=list(TWITTER_SENTIMENT_FEATURE_COLS))
    daily = _daily_rollup(tweets, prefix="twitter", weight_expr=_twitter_weights)
    daily = _add_rolling(daily, prefix="twitter")
    return daily.reindex(columns=list(TWITTER_SENTIMENT_FEATURE_COLS))


def sentiment_features_for_symbol(
    store: Store,
    symbol: str,
    *,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a per-date sentiment feature block for ``symbol``.

    Combines Reddit and Twitter into a single wide frame with all of
    :data:`SENTIMENT_FEATURE_COLS`. If ``index`` is provided, the frame is
    reindexed to it (missing days → zero mentions, NaN sentiment) so it
    joins cleanly onto the technical feature table.

    Either source can be empty — the other still produces its block, with
    the missing source emitting all-zero / NaN columns. This keeps the
    ablation harness ("sentiment only") honest when only one platform is
    connected.
    """
    posts = store.read_reddit_posts_for_symbol(symbol)
    tweets = _read_tweets_for_symbol_compat(store, symbol)

    reddit = _reddit_block(posts)
    twitter = _twitter_block(tweets)

    # Outer join on date so a day that only has reddit posts gets twitter cols
    # as NaN (and vice versa).
    feats = reddit.join(twitter, how="outer") if not reddit.empty else twitter
    if feats is None or feats.empty:
        log.info(
            "No Reddit or Twitter data found for %s — returning empty frame",
            symbol,
        )
        if index is None:
            return pd.DataFrame(columns=list(SENTIMENT_FEATURE_COLS))
        empty = pd.DataFrame(0.0, index=index, columns=list(SENTIMENT_FEATURE_COLS))
        for col in MENTION_COUNT_COLS:
            empty[col] = 0
        empty.index.name = "date"
        return empty

    # Canonical column order.
    feats = feats.reindex(columns=list(SENTIMENT_FEATURE_COLS))

    if index is not None:
        feats = feats.reindex(index)
        for col in MENTION_COUNT_COLS:
            feats[col] = feats[col].fillna(0)

    feats.index.name = "date"
    return feats


def _read_tweets_for_symbol_compat(store: Store, symbol: str) -> pd.DataFrame:
    """Tolerant read: an old store that predates the tweets table returns
    empty rather than raising. Keeps the sentiment pipeline importable
    against older checkouts in the test matrix."""
    reader = getattr(store, "read_tweets_for_symbol", None)
    if reader is None:
        return pd.DataFrame()
    try:
        return reader(symbol)
    except Exception as e:  # noqa: BLE001 — defensive, logged below
        log.debug("read_tweets_for_symbol fell through for %s: %s", symbol, e)
        return pd.DataFrame()


__all__ = [
    "REDDIT_SENTIMENT_FEATURE_COLS",
    "TWITTER_SENTIMENT_FEATURE_COLS",
    "SENTIMENT_FEATURE_COLS",
    "MENTION_COUNT_COLS",
    "score_posts",
    "score_tweets",
    "sentiment_features_for_symbol",
]
