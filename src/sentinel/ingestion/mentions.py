"""Ticker mention extraction from free-text posts/comments/tweets.

Two strategies are combined:

1. **Cashtag** — ``$TSLA``, ``$BTC-USD``. Unambiguous; always extracted.
2. **Whitelist word** — if a whitelist of tickers is provided, uppercase tokens
   that match it are also extracted. This catches things like "AAPL earnings
   tomorrow" without the dollar-sign prefix. Without a whitelist this is
   off by default — matching arbitrary 3–5-letter uppercase words produces
   way too much noise ("CEO", "USA", "MONDAY", etc.).

Return type is a **set** of ticker strings (uppercased, no leading ``$``) so
the same ticker mentioned 5 times in one post counts once — engagement
weighting happens later in the features layer.
"""

from __future__ import annotations

import re

# $AAPL, $BRK.B, $BTC-USD, $GOOG.L — letters, optional dot/dash suffix.
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,6}(?:[.\-][A-Za-z]{1,4})?)")

# Uppercase words 2–6 chars long, not followed by letters/digits.
_UPPER_WORD_RE = re.compile(r"\b([A-Z]{2,6})\b")

# Common false positives when matching against a whitelist from free text.
# Not exhaustive — this is a second line of defense on top of the whitelist.
_NOISE = frozenset(
    {
        "CEO", "CFO", "CTO", "COO", "IPO", "ETF", "EPS", "FED", "USA", "YOLO",
        "FOMO", "USD", "EUR", "GBP", "JPY", "GDP", "CPI", "PCE", "Q1", "Q2",
        "Q3", "Q4", "EV", "AI", "ML", "NLP", "LLM", "API", "SQL", "SEC",
        "IRS", "DOJ", "FTC", "DOW", "SPY", "QQQ",  # keep SPY/QQQ? They ARE
        # tickers — don't put them in the noise list if you want to track them.
        # Leaving them here means users MUST pass them via the whitelist.
    }
)


def extract_tickers(text: str, whitelist: set[str] | None = None) -> set[str]:
    """Return the set of tickers mentioned in ``text`` (uppercased, no $).

    Parameters
    ----------
    text : the post/comment/tweet body to scan.
    whitelist : optional set of tickers to recognize *without* a leading ``$``.
                Without it, only cashtags are returned. Pass this to catch
                uncashed mentions for a known universe.
    """
    if not text:
        return set()

    found: set[str] = set()

    for m in _CASHTAG_RE.finditer(text):
        found.add(m.group(1).upper().lstrip("$"))

    if whitelist:
        wl = {w.upper() for w in whitelist}
        for m in _UPPER_WORD_RE.finditer(text):
            tok = m.group(1).upper()
            if tok in wl and tok not in _NOISE:
                found.add(tok)

    return found


def extract_mentions_for_records(
    records: list[dict],
    *,
    text_fields: tuple[str, ...] = ("title", "body"),
    whitelist: set[str] | None = None,
) -> list[dict]:
    """Flatten (record × ticker) into mention rows.

    Each input record must have an ``id`` field; the output rows carry
    ``post_id`` + ``symbol``. Caller decides the record's source (reddit,
    twitter, etc.) and adds that column if needed.
    """
    out: list[dict] = []
    for r in records:
        text = " ".join(str(r.get(f) or "") for f in text_fields)
        for ticker in extract_tickers(text, whitelist=whitelist):
            out.append({"post_id": r["id"], "symbol": ticker})
    return out
