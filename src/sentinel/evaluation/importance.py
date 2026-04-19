"""Feature importance reporting — permutation + SHAP.

Why this exists
---------------
The rest of Sentinel answers *does the model work*. This module answers
*why does it work* (and equally importantly: *what is it actually leaning
on*). Without this, a walk-forward Sharpe of 1.2 is a claim with no
provenance — you can't tell whether the edge is coming from a robust
momentum feature or an over-fit sentiment artifact, and you can't tell
whether a silent data-pipeline change has quietly changed what the model
cares about.

Two methods are supported:

* ``permutation_importance`` — model-agnostic, zero extra deps. For each
  feature, shuffle its column ``n_repeats`` times and measure how much
  the score drops vs. the un-shuffled baseline. Handled here without
  sklearn's ``permutation_importance`` so the pipeline works even when
  sklearn's API drifts (it has twice across 0.24/1.0) and so the tests
  can run in the zero-sklearn sandbox.

* ``shap_importance`` — faithful local explanations via ``shap``. We use
  the unified ``shap.Explainer`` entry point so the same call works for
  tree models (TreeExplainer under the hood), linear models (Linear),
  and falls back to KernelExplainer otherwise. shap is lazy-imported so
  Sentinel still runs without it.

Both return an :class:`ImportanceResult` sorted descending by mean
importance with a matching ``std_importance`` column so callers can show
error bars or just skim the top-N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


_SHAP_INSTALL_HINT = (
    "shap is not installed. Install it with `pip install shap` "
    "to enable SHAP-based feature importance."
)


@dataclass
class ImportanceResult:
    """Sorted (descending) importance report.

    ``df`` has three columns: ``feature``, ``mean_importance``,
    ``std_importance``. Units depend on the method — for permutation
    importance it's a score *drop*; for SHAP it's mean ``|shap|``.
    """

    method: str  # "permutation" | "shap"
    df: pd.DataFrame
    scoring: str | None = None  # only set for permutation

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.df.head(n)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def _score(pipeline: Any, X: np.ndarray, y: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        preds = pipeline.predict(X)
        return float(np.mean(np.asarray(preds) == np.asarray(y)))
    if scoring == "roc_auc":
        # Lazy import so sandbox tests don't require sklearn.
        from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

        proba = pipeline.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, proba))
    raise ValueError(f"Unknown scoring {scoring!r}. Use 'accuracy' or 'roc_auc'.")


def permutation_importance(
    pipeline: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: Sequence[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "accuracy",
) -> ImportanceResult:
    """Permutation importance of every column of ``X``.

    For each feature, the column is shuffled ``n_repeats`` times and the
    pipeline is re-scored. ``mean_importance`` is the mean ``baseline −
    shuffled_score`` drop; ``std_importance`` is the std across repeats.

    Positive drops = the feature matters to the model.
    Near-zero or slightly negative drops = the feature is essentially a
    decorative passenger.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_cols = X_arr.shape[1]

    if n_cols != len(feature_names):
        raise ValueError(
            f"X has {n_cols} columns but feature_names has {len(feature_names)} names."
        )
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    rng = np.random.default_rng(random_state)
    baseline = _score(pipeline, X_arr, y_arr, scoring)

    means = np.zeros(n_cols, dtype=float)
    stds = np.zeros(n_cols, dtype=float)

    for j in range(n_cols):
        drops = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            X_perm = X_arr.copy()
            # Shuffle along axis 0 for column j only — preserves the joint
            # distribution of every *other* feature vs. the target.
            rng.shuffle(X_perm[:, j])
            drops[r] = baseline - _score(pipeline, X_perm, y_arr, scoring)
        means[j] = float(drops.mean())
        stds[j] = float(drops.std(ddof=0))

    df = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_importance": means,
                "std_importance": stds,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    log.info(
        "permutation importance (%s): baseline=%.4f, top=%s",
        scoring,
        baseline,
        df.iloc[0]["feature"] if not df.empty else "—",
    )
    return ImportanceResult(method="permutation", df=df, scoring=scoring)


# ---------------------------------------------------------------------------
# SHAP importance
# ---------------------------------------------------------------------------


def _unwrap_pipeline(pipeline: Any) -> tuple[Any, list[Any]]:
    """Return (final_estimator, pre_steps). Pre-steps are applied to X so
    shap sees the same input the classifier sees."""
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return pipeline, []
    final = steps[-1][1]
    pre = [s for _, s in steps[:-1]]
    return final, pre


def shap_importance(
    pipeline: Any,
    X: np.ndarray | pd.DataFrame,
    feature_names: Sequence[str],
    *,
    max_samples: int = 500,
    random_state: int = 42,
) -> ImportanceResult:
    """SHAP-based feature importance via ``shap.Explainer``.

    Scaler / preprocessing steps in the pipeline are applied to ``X``
    before shap sees it, so the importances are in the same feature
    space as the classifier's inputs. Subsamples to ``max_samples`` for
    tractability on larger datasets (SHAP is O(n * feature_count) for
    tree models and worse for kernel-based explainers).

    Returns mean |shap value| per feature, with std across samples.
    """
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised via stubbed test
        raise ImportError(_SHAP_INSTALL_HINT) from e

    X_arr = np.asarray(X, dtype=float)
    if X_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X_arr.shape[1]} columns but feature_names has {len(feature_names)} names."
        )

    final, pre = _unwrap_pipeline(pipeline)
    X_transformed = X_arr
    for step in pre:
        X_transformed = step.transform(X_transformed)
    X_transformed = np.asarray(X_transformed, dtype=float)

    n = X_transformed.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sample = X_transformed[idx]
    else:
        X_sample = X_transformed

    explainer = shap.Explainer(final, X_sample)
    result = explainer(X_sample)
    values = np.asarray(getattr(result, "values", result))

    # For binary classifiers that return per-class SHAP values, shape is
    # (n_samples, n_features, n_classes). Take the positive class.
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape {values.shape}; expected 2D or 3D."
        )

    abs_vals = np.abs(values)
    mean_imp = abs_vals.mean(axis=0)
    std_imp = abs_vals.std(axis=0, ddof=0)

    df = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_importance": mean_imp.astype(float),
                "std_importance": std_imp.astype(float),
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    log.info(
        "shap importance: %d samples, top=%s",
        X_sample.shape[0],
        df.iloc[0]["feature"] if not df.empty else "—",
    )
    return ImportanceResult(method="shap", df=df)
