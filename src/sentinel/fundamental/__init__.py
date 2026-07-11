"""Long-term factor data: price history, valuation, quality, insiders, competitive.

Feeds the `sentinel analyze` scorecard. Each submodule computes one row.
"""

from sentinel.fundamental.grades import GRADES, letter_grade, notch
from sentinel.fundamental.price_history import PriceHistoryScore, long_term_stats
from sentinel.fundamental.valuation import (
    FundamentalsSnapshot,
    ValuationScore,
    compute_valuation,
    fetch_snapshot,
)

__all__ = [
    "GRADES",
    "letter_grade",
    "notch",
    "PriceHistoryScore",
    "long_term_stats",
    "FundamentalsSnapshot",
    "ValuationScore",
    "compute_valuation",
    "fetch_snapshot",
]
