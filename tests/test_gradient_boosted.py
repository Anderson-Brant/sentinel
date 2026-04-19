"""Tests for the gradient-boosted tree adapters.

These tests stub out ``xgboost`` and ``lightgbm`` via ``sys.modules`` so
the suite passes regardless of whether the optional libs are installed.
We assert on:

    * the install-hint ImportError fires when the lib is missing
    * the factory passes the right keyword args
    * overrides win over defaults
    * the resulting Pipeline has no scaler step (trees don't need one)
    * the registered model name flows through ``build_classifier``
"""

from __future__ import annotations

import sys
import types

import pytest

from sentinel.models.baseline import SUPPORTED, build_classifier
from sentinel.models import gradient_boosted as gb


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _make_fake_classifier_module(class_name: str):
    """Create a fake xgboost/lightgbm-like module exposing one classifier
    that records the kwargs it was constructed with."""
    captured: dict[str, dict] = {"kwargs": {}}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)
            self.kwargs = dict(kwargs)
            self._fitted = False

        # sklearn-compatible surface so a Pipeline will accept it
        def fit(self, X, y):
            self._fitted = True
            return self

        def predict(self, X):
            return [0] * len(X)

        def predict_proba(self, X):
            return [[0.5, 0.5] for _ in X]

        def get_params(self, deep=True):
            return dict(self.kwargs)

        def set_params(self, **params):
            self.kwargs.update(params)
            return self

    mod = types.ModuleType(f"fake_{class_name.lower()}")
    setattr(mod, class_name, FakeClassifier)
    return mod, captured


@pytest.fixture
def fake_xgboost(monkeypatch):
    mod, captured = _make_fake_classifier_module("XGBClassifier")
    monkeypatch.setitem(sys.modules, "xgboost", mod)
    return captured


@pytest.fixture
def fake_lightgbm(monkeypatch):
    mod, captured = _make_fake_classifier_module("LGBMClassifier")
    monkeypatch.setitem(sys.modules, "lightgbm", mod)
    return captured


# ---------------------------------------------------------------------------
# Missing-library error path
# ---------------------------------------------------------------------------


def test_xgboost_missing_lib_raises_install_hint(monkeypatch):
    # Force ImportError by removing xgboost from sys.modules and blocking import.
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match="xgboost is not installed"):
        gb.make_xgboost_classifier()


def test_lightgbm_missing_lib_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(ImportError, match="lightgbm is not installed"):
        gb.make_lightgbm_classifier()


# ---------------------------------------------------------------------------
# Factory defaults / overrides
# ---------------------------------------------------------------------------


def test_xgboost_defaults_are_conservative(fake_xgboost):
    gb.make_xgboost_classifier(random_state=7)
    kw = fake_xgboost["kwargs"]
    assert kw["random_state"] == 7
    assert kw["n_estimators"] == 400
    assert kw["max_depth"] == 4
    assert kw["learning_rate"] == 0.05
    assert kw["objective"] == "binary:logistic"
    assert kw["eval_metric"] == "logloss"
    assert kw["tree_method"] == "hist"
    # Subsampling should be on by default (anti-overfit on noisy targets).
    assert 0.0 < kw["subsample"] <= 1.0
    assert 0.0 < kw["colsample_bytree"] <= 1.0


def test_xgboost_overrides_win_over_defaults(fake_xgboost):
    gb.make_xgboost_classifier(max_depth=8, learning_rate=0.2, custom_thing="hi")
    kw = fake_xgboost["kwargs"]
    assert kw["max_depth"] == 8
    assert kw["learning_rate"] == 0.2
    assert kw["custom_thing"] == "hi"


def test_lightgbm_defaults_are_conservative(fake_lightgbm):
    gb.make_lightgbm_classifier(random_state=11)
    kw = fake_lightgbm["kwargs"]
    assert kw["random_state"] == 11
    assert kw["n_estimators"] == 500
    assert kw["num_leaves"] == 31
    assert kw["learning_rate"] == 0.05
    assert kw["objective"] == "binary"
    assert kw["class_weight"] == "balanced"
    # LightGBM is leaf-wise — depth defaults to unlimited.
    assert kw["max_depth"] == -1
    assert kw["verbose"] == -1


def test_lightgbm_overrides_win_over_defaults(fake_lightgbm):
    gb.make_lightgbm_classifier(num_leaves=127, class_weight=None)
    kw = fake_lightgbm["kwargs"]
    assert kw["num_leaves"] == 127
    assert kw["class_weight"] is None


# ---------------------------------------------------------------------------
# Registry / Pipeline integration
# ---------------------------------------------------------------------------


def test_supported_set_includes_gbms():
    assert "xgboost" in SUPPORTED
    assert "lightgbm" in SUPPORTED


def test_gbm_models_registered_constant():
    assert gb.GBM_MODELS == frozenset({"xgboost", "lightgbm"})


def test_build_classifier_xgboost_returns_unscaled_pipeline(fake_xgboost):
    pipe = build_classifier("xgboost", random_state=3)
    # Tree models: Pipeline is single-step, no scaler.
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["clf"]
    assert fake_xgboost["kwargs"]["random_state"] == 3


def test_build_classifier_lightgbm_returns_unscaled_pipeline(fake_lightgbm):
    pipe = build_classifier("lightgbm", random_state=5)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["clf"]
    assert fake_lightgbm["kwargs"]["random_state"] == 5


def test_build_classifier_logistic_keeps_scaler():
    # Logistic is the only currently-scaled model; tree models must not be.
    pipe = build_classifier("logistic", random_state=1)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["scaler", "clf"]


def test_build_classifier_unknown_name_lists_supported():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_classifier("transformer")


def test_pipeline_with_fake_xgboost_can_fit_and_predict_proba(fake_xgboost):
    pipe = build_classifier("xgboost")
    # Smoke: fit + predict_proba flow through Pipeline against the fake.
    pipe.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    out = pipe.predict_proba([[0.5, 0.5]])
    assert len(out) == 1
    assert len(out[0]) == 2
