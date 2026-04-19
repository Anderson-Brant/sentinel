"""Ablation harness: technical-only vs sentiment-only vs hybrid.

Motivation
----------
The README's top design principle is honest evaluation. Adding a sentiment
block to a feature table almost *always* makes in-sample accuracy look
better — more parameters, more ways to fit. The real question is whether
sentiment carries predictive signal *out-of-sample*, so we compare three
carefully matched variants on the same walk-forward splits:

    technical : price-derived features only (returns, MAs, vol, momentum)
    sentiment : sentiment-derived features only (mention counts, rolling
                sentiment, engagement-weighted)
    hybrid    : technical + sentiment

The three variants share the exact same index, target, walk-forward splits,
and random state — the only thing that changes is which feature columns are
visible to the model. That keeps "does sentiment add value" a clean test.

Inputs
------
``run_ablation`` takes a *hybrid* feature table (the output of
:func:`sentinel.features.pipeline.build_feature_table` called with
``sentiment=...``) plus the list of sentiment-column names. It slices the
same table three ways — no refitting pipelines, no risk of the three
variants diverging on different data.

Optionally a ``prices`` frame can be passed, in which case each variant also
gets its own walk-forward backtest so the comparison includes strategy-level
metrics (Sharpe, total return, max drawdown), not just classification scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import pandas as pd

from sentinel.config import SentinelConfig
from sentinel.evaluation.walk_forward import (
    WalkForwardReport,
    walk_forward_evaluate,
    walk_forward_predictions,
)
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


Variant = Literal["technical", "sentiment", "hybrid"]
VARIANTS: tuple[Variant, ...] = ("technical", "sentiment", "hybrid")


@dataclass
class AblationVariantResult:
    """One row of the ablation table."""

    variant: Variant
    n_features: int
    wf_report: WalkForwardReport
    #: filled in when prices were passed; otherwise None
    backtest_report: object | None = None

    # --- convenience accessors so the renderer stays boring -------------

    @property
    def mean_accuracy(self) -> float:
        return self.wf_report.mean_accuracy

    @property
    def mean_f1(self) -> float:
        return self.wf_report.mean_f1

    @property
    def mean_roc_auc(self) -> float:
        return self.wf_report.mean_roc_auc

    @property
    def mean_naive_accuracy(self) -> float:
        return self.wf_report.mean_naive_accuracy


@dataclass
class AblationReport:
    """Top-level ablation result — ordered list of per-variant rows."""

    symbol: str
    model_name: str
    results: list[AblationVariantResult] = field(default_factory=list)

    def by_variant(self) -> dict[Variant, AblationVariantResult]:
        return {r.variant: r for r in self.results}

    # --- verdict helpers ------------------------------------------------

    def sentiment_uplift_accuracy(self) -> float | None:
        """Δ(accuracy) between hybrid and technical-only. None if either is missing."""
        rs = self.by_variant()
        if "hybrid" in rs and "technical" in rs:
            return rs["hybrid"].mean_accuracy - rs["technical"].mean_accuracy
        return None

    def sentiment_uplift_sharpe(self) -> float | None:
        """Δ(Sharpe) between hybrid and technical-only backtests. None if no backtest was run."""
        rs = self.by_variant()
        if "hybrid" not in rs or "technical" not in rs:
            return None
        h = rs["hybrid"].backtest_report
        t = rs["technical"].backtest_report
        if h is None or t is None:
            return None
        return float(h.sharpe - t.sharpe)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


_META_COLS = frozenset({"symbol", "target_direction", "target_return"})


def _subset_features(
    features: pd.DataFrame,
    *,
    keep: Iterable[str],
) -> pd.DataFrame:
    """Return a copy of ``features`` containing only ``keep`` columns + meta/target.

    The ``feature_columns`` helper used by walk-forward derives features from
    whatever-is-not-meta, so we physically drop the other columns — it's
    simpler than teaching the downstream code to honor a whitelist.
    """
    keep_set = set(keep)
    cols = [c for c in features.columns if c in _META_COLS or c in keep_set]
    return features.loc[:, cols].copy()


def run_ablation(
    features: pd.DataFrame,
    *,
    symbol: str,
    model_name: str,
    cfg: SentinelConfig,
    sentiment_columns: Iterable[str],
    prices: pd.DataFrame | None = None,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    cost_bps: float = 2.0,
    allow_short: bool = False,
    periods_per_year: int = 252,
) -> AblationReport:
    """Run the tech/sentiment/hybrid ablation on a single hybrid feature table.

    Parameters
    ----------
    features : Hybrid feature table — must contain ``target_direction`` and
               the technical + sentiment blocks. Rows with NaN already dropped.
    symbol, model_name, cfg : forwarded to ``walk_forward_evaluate``.
    sentiment_columns : Column names that belong to the sentiment block. Used
                        to split the table. Columns in this list that are
                        missing from ``features`` are silently ignored.
    prices : Optional OHLCV frame. If provided, each variant also runs a
             full walk-forward backtest for strategy-level comparison.
    long_threshold, short_threshold, cost_bps, allow_short, periods_per_year :
        forwarded to the backtest engine when ``prices`` is provided.
    """
    if "target_direction" not in features.columns:
        raise ValueError("features must include 'target_direction'")

    sentiment_cols = [c for c in sentiment_columns if c in features.columns]
    if not sentiment_cols:
        raise ValueError(
            "No sentiment columns found in features. Did you call "
            "`build_feature_table(prices, cfg, sentiment=...)`?"
        )

    technical_cols = [
        c for c in features.columns if c not in _META_COLS and c not in sentiment_cols
    ]
    if not technical_cols:
        raise ValueError("No technical feature columns found.")

    variant_columns: dict[Variant, list[str]] = {
        "technical": technical_cols,
        "sentiment": sentiment_cols,
        "hybrid": technical_cols + sentiment_cols,
    }

    # Lazy import of the backtest to avoid circular import at module load.
    backtest_fn = None
    if prices is not None:
        from sentinel.backtest.engine import backtest as backtest_fn  # type: ignore

    results: list[AblationVariantResult] = []
    for v in VARIANTS:
        subset = _subset_features(features, keep=variant_columns[v])
        log.info(
            "ablation[%s] %s: %d rows × %d feature cols",
            symbol,
            v,
            len(subset),
            len(variant_columns[v]),
        )
        wf = walk_forward_evaluate(subset, model_name=model_name, cfg=cfg)

        bt = None
        if backtest_fn is not None and prices is not None:
            probs = walk_forward_predictions(subset, model_name=model_name, cfg=cfg)
            bt = backtest_fn(
                prices=prices,
                probabilities=probs,
                symbol=f"{symbol}[{v}]",
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                cost_bps=cost_bps,
                allow_short=allow_short,
                periods_per_year=periods_per_year,
            )

        results.append(
            AblationVariantResult(
                variant=v,
                n_features=len(variant_columns[v]),
                wf_report=wf,
                backtest_report=bt,
            )
        )

    return AblationReport(symbol=symbol, model_name=model_name, results=results)


__all__ = [
    "Variant",
    "VARIANTS",
    "AblationVariantResult",
    "AblationReport",
    "run_ablation",
]
