"""The Analysis dataclass and its assembly.

Rows land one milestone at a time: price history + valuation (v0.2),
quality (v0.3), insiders (v0.5), competitive (v0.6). The renderer prints
pending rows for whatever isn't here yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from sentinel.fundamental.grades import letter_grade
from sentinel.fundamental.price_history import PriceHistoryScore, long_term_stats
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

    price_history: PriceHistoryScore | None = None
    valuation: ValuationScore | None = None

    composite_grade: str | None = None
    notes: list[str] = field(default_factory=list)


def build_analysis(
    symbol: str,
    *,
    prices: pd.DataFrame | None,
    snapshot: FundamentalsSnapshot | None,
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

    scores = [
        s.score
        for s in (analysis.price_history, analysis.valuation)
        if s is not None and s.score is not None and s.grade not in (None, "")
    ]
    if scores:
        analysis.composite_grade = letter_grade(sum(scores) / len(scores))

    if analysis.price_history is not None and analysis.price_history.years < 10:
        analysis.notes.append(f"price history covers {analysis.price_history.years:.1f}y")

    return analysis
