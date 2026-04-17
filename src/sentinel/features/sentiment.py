"""Sentiment feature generation — STUB.

Planned implementation:
    - Read raw posts/comments/tweets from storage.
    - Score each record with VADER (or a finance-tuned model later).
    - Aggregate per (symbol, date) into:
        * mean / median sentiment
        * engagement-weighted sentiment
        * positive / negative / neutral ratios
        * mention count + z-score vs trailing baseline
        * rolling sentiment momentum + volatility
    - Join onto the technical feature table on (symbol, date).

Kept as a stub in the MVP — the models train on technical features only for now.
"""

from __future__ import annotations

import pandas as pd


def sentiment_features(*_, **__) -> pd.DataFrame:  # pragma: no cover - stub
    """Return an empty DataFrame. Replace with real logic once ingestion lands."""
    return pd.DataFrame()
