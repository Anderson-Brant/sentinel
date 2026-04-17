"""Reddit ingestion — STUB.

Planned implementation:
    - Use `praw` with credentials from env vars (REDDIT_CLIENT_ID/SECRET/USER_AGENT).
    - For each configured subreddit, pull recent posts + top comments.
    - Extract ticker mentions (regex on $TICKER or whitelist-matched uppercase tokens).
    - Persist raw records to the `reddit_posts` table; derive mentions into `mentions`.
    - Compute rolling mention counts + engagement-weighted aggregates in the features layer.
"""

from __future__ import annotations

from typing import Any

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


def ingest_posts(symbol: str, **_: Any) -> None:  # pragma: no cover - stub
    raise NotImplementedError(
        "Reddit ingestion is on the roadmap. "
        "Install the social extras (`pip install -e '.[social]'`) and implement this adapter."
    )
