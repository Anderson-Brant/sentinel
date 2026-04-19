"""Classifier factory.

Returns an unfit sklearn Pipeline for any supported model name so that
downstream code (walk-forward, ablation, regime slicing) can call
``fit / predict / predict_proba`` uniformly without caring what's inside.

Supported families:

    logistic          — LogisticRegression, class-weight balanced
    random_forest     — RandomForestClassifier, class-weight balanced
    xgboost           — XGBClassifier (optional dep, lazy-imported)
    lightgbm          — LGBMClassifier (optional dep, lazy-imported)

Linear / distance-based models are wrapped in a StandardScaler step.
Tree-based models are *not* — scaling is a no-op for trees and the
extra pipeline step only adds noise to the pickle and to evaluation time.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.models.gradient_boosted import (
    GBM_MODELS,
    make_lightgbm_classifier,
    make_xgboost_classifier,
)


# Models whose inputs benefit from zero-mean / unit-variance scaling.
_SCALED_MODELS = {"logistic"}

# Tree-based models — scaling is unnecessary, we wrap in a single-step
# Pipeline purely for interface uniformity.
_TREE_MODELS = {"random_forest"} | GBM_MODELS

SUPPORTED: frozenset[str] = frozenset(_SCALED_MODELS | _TREE_MODELS)


def build_classifier(name: str, *, random_state: int = 42) -> Pipeline:
    """Return an unfit sklearn pipeline for the requested model."""
    name = name.lower()
    if name == "logistic":
        clf = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
        )
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    elif name == "xgboost":
        clf = make_xgboost_classifier(random_state=random_state)
    elif name == "lightgbm":
        clf = make_lightgbm_classifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model {name!r}. Supported: {sorted(SUPPORTED)}")

    if name in _SCALED_MODELS:
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", clf),
            ]
        )
    # Tree models: single-step pipeline, interface-uniform but no scaler.
    return Pipeline([("clf", clf)])
