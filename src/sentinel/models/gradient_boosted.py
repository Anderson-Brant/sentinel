"""Gradient-boosted tree adapters: XGBoost and LightGBM.

Both libraries are *optional* dependencies. We lazy-import them inside the
factory so:

    * `sentinel` keeps importing cleanly when neither lib is installed
    * tests can run without xgboost / lightgbm on the test machine
    * the CLI surfaces a clear, actionable install message if a user
      requests `--model xgboost` without the library present

Each factory returns an sklearn-compatible estimator that exposes the
standard ``fit / predict / predict_proba`` API. They drop straight into
``sentinel.models.baseline.build_classifier`` so the rest of the codebase
(walk-forward, ablation, regime slicing) doesn't need to know what the
underlying model is.

Defaults are intentionally conservative — shallow trees, modest learning
rate, row+column subsampling — to keep variance reasonable on financial
time-series targets where signal-to-noise is low and overfit risk is the
dominant failure mode.
"""

from __future__ import annotations

from typing import Any

_XGBOOST_INSTALL_HINT = (
    "xgboost is not installed. Install it with `pip install xgboost` "
    "(or `pip install -e \".[gbm]\"` once the optional extra is wired up)."
)
_LIGHTGBM_INSTALL_HINT = (
    "lightgbm is not installed. Install it with `pip install lightgbm` "
    "(or `pip install -e \".[gbm]\"` once the optional extra is wired up)."
)


def make_xgboost_classifier(*, random_state: int = 42, **overrides: Any) -> Any:
    """Return an unfit XGBoost classifier with sensible defaults.

    Lazy-imports xgboost. Raises a clear ImportError with an install hint
    if the library is missing.

    Parameters
    ----------
    random_state:
        Seed forwarded to ``XGBClassifier``.
    **overrides:
        Override any default hyperparameter (e.g. ``max_depth=6``). Useful
        for sweeps without editing this file.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised via stubbed test
        raise ImportError(_XGBOOST_INSTALL_HINT) from e

    params: dict[str, Any] = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def make_lightgbm_classifier(*, random_state: int = 42, **overrides: Any) -> Any:
    """Return an unfit LightGBM classifier with sensible defaults.

    Lazy-imports lightgbm. Raises a clear ImportError with an install hint
    if the library is missing.

    Parameters mirror :func:`make_xgboost_classifier`. Defaults differ
    because LightGBM is leaf-wise (uses ``num_leaves``), not depth-wise.
    """
    try:
        from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised via stubbed test
        raise ImportError(_LIGHTGBM_INSTALL_HINT) from e

    params: dict[str, Any] = dict(
        n_estimators=500,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight="balanced",
        objective="binary",
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


# Public model names supported by this module. baseline.py reads this set
# to decide whether to dispatch into the GBM factories.
GBM_MODELS: frozenset[str] = frozenset({"xgboost", "lightgbm"})
