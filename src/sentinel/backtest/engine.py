"""Backtest engine — STUB.

Planned shape (sketch):

    def backtest(
        prices: pd.DataFrame,
        probabilities: pd.Series,
        *,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        cost_bps: float = 2.0,
        allow_short: bool = False,
    ) -> BacktestReport:
        '''
        Convert predicted probabilities into positions, apply transaction costs,
        compute an equity curve + Sharpe / max drawdown / win rate / turnover.
        Must use probability_t to size position for t+1 (no look-ahead).
        '''

Intentionally left unimplemented until the MVP loop is validated end-to-end.
"""

from __future__ import annotations


def backtest(*_, **__):  # pragma: no cover - stub
    raise NotImplementedError(
        "Backtest engine is on the near-term roadmap. "
        "Implement signal→position→equity with costs + proper lag."
    )
