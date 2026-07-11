"""Long-term factor data: price history, valuation, quality, insiders, competitive.

Feeds the `sentinel analyze` scorecard. Each submodule computes one row.
"""

from sentinel.fundamental.competitive import (
    CompetitiveScore,
    compute_competitive,
    related_by_correlation,
)
from sentinel.fundamental.grades import GRADES, letter_grade, notch
from sentinel.fundamental.insiders import (
    InsiderScore,
    compute_insiders,
    fetch_insider_transactions,
)
from sentinel.fundamental.price_history import PriceHistoryScore, long_term_stats
from sentinel.fundamental.quality import QualityScore, compute_quality
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
    "QualityScore",
    "compute_quality",
    "InsiderScore",
    "compute_insiders",
    "fetch_insider_transactions",
    "CompetitiveScore",
    "compute_competitive",
    "related_by_correlation",
]
