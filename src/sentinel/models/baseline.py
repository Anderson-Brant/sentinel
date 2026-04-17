"""Baseline classifier factory.

Only two models in the MVP so we can show the pipeline works end-to-end
against a sane baseline. Both are wrapped in a `StandardScaler` pipeline
so downstream code can call `.fit` / `.predict` / `.predict_proba`
uniformly without caring which model is inside.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED = {"logistic", "random_forest"}


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
    else:
        raise ValueError(f"Unsupported model {name!r}. Supported: {sorted(SUPPORTED)}")

    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )
