"""Tests for the feature importance module.

Two things we want to prove:

    1. ``permutation_importance`` correctly ranks a known-signal column
       first, even for a deliberately dumb oracle classifier — the
       algorithm, not the model, is what's on trial.
    2. ``shap_importance`` passes the right things to ``shap.Explainer``
       and correctly reduces the output into a sorted DataFrame. We stub
       shap via ``sys.modules`` so this test doesn't need shap installed.

We also pin a couple of error cases (shape mismatch, bad scoring name,
missing shap library) because those are the calls that fail silently in
interactive use.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.importance import (
    ImportanceResult,
    permutation_importance,
    shap_importance,
)


# ---------------------------------------------------------------------------
# Permutation importance — synthetic signal
# ---------------------------------------------------------------------------


class _OracleByColumn:
    """Dumb classifier: predicts whichever target value the given column's
    sign matches. Used so we can construct a dataset where we *know* which
    column carries the signal and verify the algorithm finds it."""

    def __init__(self, signal_col: int):
        self.signal_col = signal_col

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.signal_col] > 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = (X[:, self.signal_col] > 0).astype(float)
        return np.column_stack([1 - p, p])


@pytest.fixture
def synthetic_binary():
    """3 features; feature 1 IS the target, features 0 and 2 are pure noise."""
    rng = np.random.default_rng(0)
    n = 200
    noise0 = rng.standard_normal(n)
    signal = rng.standard_normal(n)
    noise2 = rng.standard_normal(n)
    X = np.column_stack([noise0, signal, noise2])
    y = (signal > 0).astype(int)
    return X, y, ["noise_a", "signal", "noise_b"]


def test_permutation_importance_ranks_signal_first(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    result = permutation_importance(clf, X, y, names, n_repeats=5, random_state=0)

    assert isinstance(result, ImportanceResult)
    assert result.method == "permutation"
    # Highest mean drop should be the signal column.
    assert result.df.iloc[0]["feature"] == "signal"
    # Noise features should have substantially smaller drops.
    top_drop = float(result.df.iloc[0]["mean_importance"])
    other_drops = result.df.iloc[1:]["mean_importance"].astype(float).to_numpy()
    assert top_drop > 0.25  # meaningful drop — ~1.0 minus noise accuracy
    assert all(abs(d) < top_drop / 2 for d in other_drops)


def test_permutation_importance_returns_sorted_descending(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    result = permutation_importance(clf, X, y, names, n_repeats=3, random_state=0)
    means = result.df["mean_importance"].to_numpy()
    assert (np.diff(means) <= 0).all()  # non-increasing


def test_permutation_importance_has_std_column(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    result = permutation_importance(clf, X, y, names, n_repeats=5, random_state=0)
    assert set(result.df.columns) == {"feature", "mean_importance", "std_importance"}
    assert (result.df["std_importance"] >= 0).all()


def test_permutation_importance_n_repeats_1_works(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    result = permutation_importance(clf, X, y, names, n_repeats=1, random_state=0)
    # Single repeat -> std is trivially zero
    assert (result.df["std_importance"] == 0.0).all()


def test_permutation_importance_rejects_zero_repeats(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(clf, X, y, names, n_repeats=0, random_state=0)


def test_permutation_importance_rejects_shape_mismatch(synthetic_binary):
    X, y, _ = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    with pytest.raises(ValueError, match="feature_names"):
        permutation_importance(clf, X, y, ["only", "two"], n_repeats=1, random_state=0)


def test_permutation_importance_rejects_unknown_scoring(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    with pytest.raises(ValueError, match="Unknown scoring"):
        permutation_importance(clf, X, y, names, n_repeats=1, scoring="mse")


def test_permutation_importance_seed_reproducible(synthetic_binary):
    X, y, names = synthetic_binary
    clf = _OracleByColumn(signal_col=1)
    a = permutation_importance(clf, X, y, names, n_repeats=3, random_state=7)
    b = permutation_importance(clf, X, y, names, n_repeats=3, random_state=7)
    pd.testing.assert_frame_equal(a.df, b.df)


def test_permutation_importance_top_n():
    # Single-feature case so we don't depend on fixture state.
    X = np.array([[0.1], [0.2], [0.3], [0.4]])
    y = np.array([0, 0, 1, 1])

    class _AlwaysOne:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    result = permutation_importance(
        _AlwaysOne(), X, y, ["f"], n_repeats=2, random_state=1
    )
    assert len(result.top(5)) == 1
    assert len(result.top(0)) == 0


# ---------------------------------------------------------------------------
# SHAP importance — with stubbed shap
# ---------------------------------------------------------------------------


def _make_fake_shap(values: np.ndarray):
    """Return a fake ``shap`` module whose ``Explainer(...)(X)`` yields an
    object with ``.values == values``. Records every call."""

    log: dict[str, list] = {"Explainer_init": [], "Explainer_call": []}

    class _Result:
        def __init__(self, v):
            self.values = v

    class _Explainer:
        def __init__(self, model, background):
            log["Explainer_init"].append((model, np.asarray(background).shape))
            self._model = model

        def __call__(self, X):
            log["Explainer_call"].append(np.asarray(X).shape)
            return _Result(values)

    mod = types.ModuleType("shap")
    mod.Explainer = _Explainer
    return mod, log


@pytest.fixture
def shap_2d_values():
    # 4 samples x 3 features — feature 1 dominates
    return np.array(
        [
            [0.1, 0.9, -0.05],
            [-0.2, 1.0, 0.0],
            [0.05, -0.8, 0.1],
            [0.0, 0.85, -0.02],
        ]
    )


def test_shap_importance_passes_final_estimator_to_explainer(monkeypatch, shap_2d_values):
    mod, log = _make_fake_shap(shap_2d_values)
    monkeypatch.setitem(sys.modules, "shap", mod)

    # Pipeline-like object: scaler step that records transform, then a "clf".
    class _Scaler:
        def __init__(self):
            self.calls = 0

        def transform(self, X):
            self.calls += 1
            return np.asarray(X) * 2.0  # a visible scaling

    class _Clf:
        name = "final-estimator"

    scaler = _Scaler()
    clf = _Clf()
    pipeline = types.SimpleNamespace(steps=[("scaler", scaler), ("clf", clf)])

    X = np.random.default_rng(0).standard_normal((4, 3))
    result = shap_importance(pipeline, X, ["a", "b", "c"], max_samples=500)

    # Scaler's transform was applied before shap saw the data.
    assert scaler.calls == 1
    # shap.Explainer got the final estimator, not the pipeline.
    model_sent, shape_sent = log["Explainer_init"][-1]
    assert model_sent is clf
    assert shape_sent == (4, 3)


def test_shap_importance_mean_abs_and_sort(monkeypatch, shap_2d_values):
    mod, _ = _make_fake_shap(shap_2d_values)
    monkeypatch.setitem(sys.modules, "shap", mod)

    class _Clf:
        pass

    pipeline = types.SimpleNamespace(steps=[("clf", _Clf())])
    X = np.zeros((4, 3))
    result = shap_importance(pipeline, X, ["a", "b", "c"], max_samples=500)

    # Feature "b" has the biggest |shap| values, so it should be #1.
    assert result.df.iloc[0]["feature"] == "b"
    # Mean |b| = mean(|0.9, 1.0, 0.8, 0.85|) = 0.8875
    assert math.isclose(float(result.df.iloc[0]["mean_importance"]), 0.8875, rel_tol=1e-9)
    # Rows are sorted descending by mean_importance.
    means = result.df["mean_importance"].to_numpy()
    assert (np.diff(means) <= 0).all()


def test_shap_importance_handles_3d_values(monkeypatch):
    # Simulate a binary classifier shap output: (n, f, 2). We want class 1.
    class0 = np.zeros((4, 3))
    class1 = np.array(
        [
            [0.1, 0.5, 0.0],
            [-0.1, 0.6, 0.0],
            [0.0, 0.4, 0.0],
            [0.05, 0.55, 0.0],
        ]
    )
    values = np.stack([class0, class1], axis=-1)  # (4, 3, 2)
    mod, _ = _make_fake_shap(values)
    monkeypatch.setitem(sys.modules, "shap", mod)

    class _Clf:
        pass

    pipeline = types.SimpleNamespace(steps=[("clf", _Clf())])
    X = np.zeros((4, 3))
    result = shap_importance(pipeline, X, ["a", "b", "c"], max_samples=500)
    # Class-1 importances should drive the ranking, so "b" wins.
    assert result.df.iloc[0]["feature"] == "b"
    # Class-0 is all zeros, so if we'd accidentally aggregated everything
    # the top importance would be small — confirm we actually took class 1.
    assert float(result.df.iloc[0]["mean_importance"]) > 0.45


def test_shap_importance_subsamples_when_too_large(monkeypatch):
    values = np.random.default_rng(0).standard_normal((10, 2))  # 10 rows (matches the 10 we subsample to)
    mod, log = _make_fake_shap(values)
    monkeypatch.setitem(sys.modules, "shap", mod)

    class _Clf:
        pass

    pipeline = types.SimpleNamespace(steps=[("clf", _Clf())])
    X = np.random.default_rng(1).standard_normal((100, 2))
    shap_importance(pipeline, X, ["a", "b"], max_samples=10, random_state=42)

    _, bg_shape = log["Explainer_init"][-1]
    call_shape = log["Explainer_call"][-1]
    assert bg_shape == (10, 2)  # subsampled background
    assert call_shape == (10, 2)


def test_shap_importance_rejects_shape_mismatch(monkeypatch, shap_2d_values):
    mod, _ = _make_fake_shap(shap_2d_values)
    monkeypatch.setitem(sys.modules, "shap", mod)

    class _Clf:
        pass

    pipeline = types.SimpleNamespace(steps=[("clf", _Clf())])
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="feature_names"):
        shap_importance(pipeline, X, ["only-one-name"])


def test_shap_importance_missing_lib_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)  # block real import

    class _Clf:
        pass

    pipeline = types.SimpleNamespace(steps=[("clf", _Clf())])
    with pytest.raises(ImportError, match="shap is not installed"):
        shap_importance(pipeline, np.zeros((3, 2)), ["a", "b"])
