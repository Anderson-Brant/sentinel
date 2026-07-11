"""The Analysis dataclass and its assembly.

All five scorecard rows are computed here from already-fetched inputs; the
renderer prints whatever could be scored and marks the rest unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from sentinel.fundamental.competitive import CompetitiveScore, compute_competitive
from sentinel.fundamental.grades import letter_grade
from sentinel.fundamental.insiders import InsiderScore, compute_insiders
from sentinel.fundamental.price_history import PriceHistoryScore, long_term_stats
from sentinel.fundamental.quality import QualityScore, compute_quality
from sentinel.fundamental.valuation import (
    FundamentalsSnapshot,
    ValuationScore,
    compute_valuation,
)


@dataclass
class Analysis:
    symbol: str
    as_of: date
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None

    quality: QualityScore | None = None
    valuation: ValuationScore | None = None
    price_history: PriceHistoryScore | None = None
    insiders: InsiderScore | None = None
    competitive: CompetitiveScore | None = None

    related_tickers: list[str] = field(default_factory=list)
    composite_grade: str | None = None
    notes: list[str] = field(default_factory=list)

    def scored_rows(self) -> list[tuple[str, object]]:
        rows = [
            ("quality", self.quality),
            ("valuation", self.valuation),
            ("price history", self.price_history),
            ("insiders", self.insiders),
            ("competitive", self.competitive),
        ]
        return [(name, s) for name, s in rows if s is not None and getattr(s, "grade", None)]


def build_analysis(
    symbol: str,
    *,
    prices: pd.DataFrame | None,
    snapshot: FundamentalsSnapshot | None,
    insider_txns: pd.DataFrame | None = None,
    related_tickers: list[str] | None = None,
) -> Analysis:
    """Assemble the scorecard from already-fetched inputs. Pure - no network."""
    symbol = symbol.upper()
    analysis = Analysis(symbol=symbol, as_of=date.today())

    if snapshot is not None:
        analysis.company_name = snapshot.company_name
        analysis.sector = snapshot.sector
        analysis.industry = snapshot.industry
        analysis.market_cap = snapshot.market_cap

    if prices is not None and not prices.empty:
        analysis.price_history = long_term_stats(prices, symbol=symbol)

    if snapshot is not None:
        analysis.valuation = compute_valuation(snapshot)
        analysis.quality = compute_quality(snapshot)
        analysis.competitive = compute_competitive(snapshot)
        analysis.insiders = compute_insiders(
            insider_txns,
            symbol=symbol,
            shares_outstanding=snapshot.shares_outstanding,
            as_of=analysis.as_of,
        )

    analysis.related_tickers = related_tickers or []

    scores = [
        s.score  # type: ignore[attr-defined]
        for _, s in analysis.scored_rows()
        if getattr(s, "score", None) is not None
    ]
    if scores:
        analysis.composite_grade = letter_grade(sum(scores) / len(scores))

    if analysis.price_history is not None and analysis.price_history.years < 10:
        analysis.notes.append(f"price history covers {analysis.price_history.years:.1f}y")

    return analysis
