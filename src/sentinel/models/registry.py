"""Train / save / load models and run prediction on the latest feature row.

Intentionally minimal:
    - One pickle per (symbol, model_name).
    - Saves a ``TrainResult`` dataclass alongside the fitted pipeline so we
      can render comparison tables without re-running training.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sentinel.config import SentinelConfig, repo_root
from sentinel.features.pipeline import feature_columns
from sentinel.models.baseline import build_classifier
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


ARTIFACT_ROOT = repo_root() / "artifacts" / "models"


@dataclass
class TrainResult:
    model_name: str
    symbol: str
    feature_names: list[str]
    pipeline: Any  # sklearn Pipeline — typed as Any to avoid a hard sklearn import on load
    holdout_accuracy: float
    holdout_f1: float
    holdout_roc_auc: float
    baseline_accuracy: float  # "always predict majority class"
    n_train: int
    n_test: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def _time_split(
    features: pd.DataFrame, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features.sort_index()
    n = len(features)
    split_at = int(n * (1 - test_fraction))
    return features.iloc[:split_at], features.iloc[split_at:]


def train_model(features: pd.DataFrame, *, model_name: str, cfg: SentinelConfig) -> TrainResult:
    """Time-ordered train/test split → fit → report holdout metrics."""
    if "target_direction" not in features.columns:
        raise ValueError("features table must include 'target_direction'")
    if len(features) < 100:
        raise ValueError(f"Not enough rows to train ({len(features)}). Ingest more history.")

    feat_cols = feature_columns(features)
    X = features[feat_cols].astype(float).to_numpy()
    y = features["target_direction"].astype(int).to_numpy()

    train_df, test_df = _time_split(features, cfg.modeling.test_fraction)
    X_train = train_df[feat_cols].astype(float).to_numpy()
    y_train = train_df["target_direction"].astype(int).to_numpy()
    X_test = test_df[feat_cols].astype(float).to_numpy()
    y_test = test_df["target_direction"].astype(int).to_numpy()

    pipeline = build_classifier(model_name, random_state=cfg.modeling.random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        roc = float("nan")

    majority = int(np.round(np.mean(y_train)))
    baseline_pred = np.full_like(y_test, fill_value=majority)

    result = TrainResult(
        model_name=model_name,
        symbol=str(features.get("symbol", pd.Series(["?"])).iloc[0]),
        feature_names=feat_cols,
        pipeline=pipeline,
        holdout_accuracy=float(accuracy_score(y_test, y_pred)),
        holdout_f1=float(f1_score(y_test, y_pred, zero_division=0)),
        holdout_roc_auc=roc,
        baseline_accuracy=float(accuracy_score(y_test, baseline_pred)),
        n_train=len(y_train),
        n_test=len(y_test),
        metadata={"total_rows": int(len(X)), "class_balance": float(np.mean(y))},
    )
    log.info(
        "[%s/%s] holdout acc=%.3f f1=%.3f roc=%.3f (baseline acc=%.3f)",
        result.symbol,
        result.model_name,
        result.holdout_accuracy,
        result.holdout_f1,
        result.holdout_roc_auc,
        result.baseline_accuracy,
    )
    return result


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------


def _artifact_path(symbol: str, model_name: str) -> Path:
    return ARTIFACT_ROOT / f"{symbol.upper()}__{model_name}.pkl"


def save_model(symbol: str, model_name: str, result: TrainResult) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(symbol, model_name)
    with path.open("wb") as f:
        pickle.dump(result, f)
    return path


def load_model(symbol: str, model_name: str) -> TrainResult | None:
    path = _artifact_path(symbol, model_name)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Predict latest
# ---------------------------------------------------------------------------


@dataclass
class LatestPrediction:
    as_of: pd.Timestamp
    direction: int          # 0 or 1
    probability_up: float   # in [0, 1]
    label: str              # "bullish" | "bearish" | "neutral"


def predict_latest(result: TrainResult, features: pd.DataFrame) -> LatestPrediction:
    row = features.iloc[[-1]]
    X = row[result.feature_names].astype(float).to_numpy()
    proba_up = float("nan")
    try:
        proba_up = float(result.pipeline.predict_proba(X)[0, 1])
    except Exception:  # noqa: BLE001
        pass
    direction = int(result.pipeline.predict(X)[0])

    if np.isnan(proba_up):
        label = "bullish" if direction == 1 else "bearish"
    elif proba_up >= 0.6:
        label = "bullish"
    elif proba_up <= 0.4:
        label = "bearish"
    else:
        label = "neutral"

    return LatestPrediction(
        as_of=pd.Timestamp(row.index[-1]),
        direction=direction,
        probability_up=proba_up,
        label=label,
    )
