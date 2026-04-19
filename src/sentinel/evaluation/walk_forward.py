"""Rolling-origin (walk-forward) validation.

For each of ``n_splits`` splits:
    - Train on all rows up to fold start (minimum ``min_train_size``).
    - Test on the next contiguous chunk.
    - Never use the future.

Reports per-fold metrics + aggregated means. Also compares against a naive
"predict yesterday's direction" baseline on the same folds so the reader
can see whether the model beats a free heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sentinel.config import SentinelConfig
from sentinel.features.pipeline import feature_columns
from sentinel.models.baseline import build_classifier
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FoldMetrics:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    accuracy: float
    f1: float
    roc_auc: float
    naive_accuracy: float  # "predict yesterday's sign"


@dataclass
class WalkForwardReport:
    model_name: str
    folds: list[FoldMetrics]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_f1(self) -> float:
        return float(np.mean([f.f1 for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_roc_auc(self) -> float:
        vals = [f.roc_auc for f in self.folds if not np.isnan(f.roc_auc)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_naive_accuracy(self) -> float:
        return (
            float(np.mean([f.naive_accuracy for f in self.folds])) if self.folds else float("nan")
        )


def walk_forward_evaluate(
    features: pd.DataFrame, *, model_name: str, cfg: SentinelConfig
) -> WalkForwardReport:
    """Evaluate ``model_name`` on ``features`` using rolling-origin splits."""
    if "target_direction" not in features.columns:
        raise ValueError("features table must include 'target_direction'")

    feats = features.sort_index()
    feat_cols = feature_columns(feats)
    X = feats[feat_cols].astype(float).to_numpy()
    y = feats["target_direction"].astype(int).to_numpy()
    yesterday_sign = (feats["target_direction"].shift(1).fillna(0).astype(int)).to_numpy()
    dates = feats.index

    n = len(feats)
    min_train = cfg.modeling.walk_forward.min_train_size
    n_splits = cfg.modeling.walk_forward.n_splits

    if n < min_train + n_splits:
        raise ValueError(
            f"Not enough rows for walk-forward ({n} < min_train {min_train} + n_splits {n_splits})."
        )

    # Equal-size test folds over the tail.
    tail = n - min_train
    fold_size = max(1, tail // n_splits)

    folds: list[FoldMetrics] = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        pipeline = build_classifier(model_name, random_state=cfg.modeling.random_state)
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        try:
            y_proba = pipeline.predict_proba(X_te)[:, 1]
            roc = (
                float(roc_auc_score(y_te, y_proba))
                if len(np.unique(y_te)) > 1
                else float("nan")
            )
        except Exception:  # noqa: BLE001
            roc = float("nan")

        naive = float(accuracy_score(y_te, yesterday_sign[test_start:test_end]))

        folds.append(
            FoldMetrics(
                fold=i + 1,
                train_start=pd.Timestamp(dates[0]),
                train_end=pd.Timestamp(dates[train_end - 1]),
                test_start=pd.Timestamp(dates[test_start]),
                test_end=pd.Timestamp(dates[test_end - 1]),
                n_train=int(train_end),
                n_test=int(test_end - test_start),
                accuracy=float(accuracy_score(y_te, y_pred)),
                f1=float(f1_score(y_te, y_pred, zero_division=0)),
                roc_auc=roc,
                naive_accuracy=naive,
            )
        )

    report = WalkForwardReport(model_name=model_name, folds=folds)
    log.info(
        "walk-forward %s: mean_acc=%.3f mean_f1=%.3f mean_roc=%.3f (naive=%.3f) over %d folds",
        model_name,
        report.mean_accuracy,
        report.mean_f1,
        report.mean_roc_auc,
        report.mean_naive_accuracy,
        len(folds),
    )
    return report


def walk_forward_predictions(
    features: pd.DataFrame, *, model_name: str, cfg: SentinelConfig
) -> pd.Series:
    """Produce out-of-sample P(up) probabilities using rolling-origin splits.

    Returns a Series aligned to ``features.index``. Rows inside the initial
    warm-up (before ``min_train_size``) are NaN — the backtest should interpret
    those as "no signal, no position".

    This is the series the backtest engine should consume. Feeding it
    probabilities from a single model fit on the whole history would leak
    future information into past positions.
    """
    if "target_direction" not in features.columns:
        raise ValueError("features table must include 'target_direction'")

    feats = features.sort_index()
    feat_cols = feature_columns(feats)
    X = feats[feat_cols].astype(float).to_numpy()
    y = feats["target_direction"].astype(int).to_numpy()

    n = len(feats)
    min_train = cfg.modeling.walk_forward.min_train_size
    n_splits = cfg.modeling.walk_forward.n_splits
    if n < min_train + n_splits:
        raise ValueError(
            f"Not enough rows for walk-forward ({n} < min_train {min_train} + n_splits {n_splits})."
        )

    tail = n - min_train
    fold_size = max(1, tail // n_splits)
    probs = pd.Series(np.nan, index=feats.index, dtype=float, name="p_up")

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        pipeline = build_classifier(model_name, random_state=cfg.modeling.random_state)
        pipeline.fit(X[:train_end], y[:train_end])
        probs.iloc[test_start:test_end] = pipeline.predict_proba(X[test_start:test_end])[:, 1]

    n_oos = int(probs.notna().sum())
    log.info("walk-forward predictions: %d OOS rows over %d folds", n_oos, n_splits)
    return probs
