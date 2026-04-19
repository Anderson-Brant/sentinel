"""Backtest engine: convert probabilities into positions → equity → metrics."""

from sentinel.backtest.engine import BacktestReport, Trade, backtest

__all__ = ["BacktestReport", "Trade", "backtest"]
