"""Twitter / X ingestion — STUB.

Planned implementation:
    - Use `tweepy` (v2 API) with TWITTER_BEARER_TOKEN.
    - Query recent search for $CASHTAG or ticker keywords.
    - Persist tweets + engagement metrics to the `tweets` table.
    - Roll up mention counts + sentiment in the features layer.
"""

from __future__ import annotations

from typing import Any

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


def ingest_tweets(symbol: str, **_: Any) -> None:  # pragma: no cover - stub
    raise NotImplementedError(
        "Twitter/X ingestion is on the roadmap. "
        "Install the social extras (`pip install -e '.[social]'`) and implement this adapter."
    )
