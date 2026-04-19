"""Tests for the experiment tracking abstraction.

We stub `mlflow` via sys.modules so the tests pass whether or not mlflow is
installed on the runner. What we're pinning down:

    * NullTracker truly no-ops (important — it's the default when --track is off)
    * MLflowTracker forwards calls to mlflow with the right arguments
    * Missing mlflow raises an actionable ImportError with the install hint
    * log_metrics silently drops NaN / inf / None (mlflow rejects them)
    * log_params coerces None → "None" string (mlflow rejects None values)
    * get_tracker() picks the right implementation based on the flag
"""

from __future__ import annotations

import sys
import types

import pytest

from sentinel.tracking import NullTracker, get_tracker
from sentinel.tracking import mlflow_tracker as tracking_mod

# ---------------------------------------------------------------------------
# Fake mlflow helper
# ---------------------------------------------------------------------------


def _make_fake_mlflow():
    """Build a types.ModuleType that records every call made to it."""
    log: dict[str, list] = {
        "set_tracking_uri": [],
        "set_experiment": [],
        "start_run_calls": [],
        "log_params": [],
        "log_metric": [],
        "log_artifact": [],
        "log_dict": [],
        "set_tag": [],
        "run_context_entered": [],
        "run_context_exited": [],
    }

    class _RunCtx:
        def __init__(self, run_name=None):
            self.run_name = run_name
            self.info = types.SimpleNamespace(run_id="fake-id", run_name=run_name)

        def __enter__(self):
            log["run_context_entered"].append(self.run_name)
            return self

        def __exit__(self, exc_type, exc, tb):
            log["run_context_exited"].append(self.run_name)
            return False

    mod = types.ModuleType("mlflow")

    def set_tracking_uri(uri):
        log["set_tracking_uri"].append(uri)

    def set_experiment(name):
        log["set_experiment"].append(name)

    def start_run(run_name=None):
        log["start_run_calls"].append(run_name)
        return _RunCtx(run_name=run_name)

    def log_params(params):
        log["log_params"].append(dict(params))

    def log_metric(k, v, step=None):
        log["log_metric"].append((k, v, step))

    def log_artifact(path):
        log["log_artifact"].append(path)

    def log_dict(d, name):
        log["log_dict"].append((dict(d), name))

    def set_tag(k, v):
        log["set_tag"].append((k, v))

    mod.set_tracking_uri = set_tracking_uri
    mod.set_experiment = set_experiment
    mod.start_run = start_run
    mod.log_params = log_params
    mod.log_metric = log_metric
    mod.log_artifact = log_artifact
    mod.log_dict = log_dict
    mod.set_tag = set_tag

    return mod, log


@pytest.fixture
def fake_mlflow(monkeypatch):
    mod, log = _make_fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", mod)
    return log


# ---------------------------------------------------------------------------
# NullTracker: truly inert
# ---------------------------------------------------------------------------


def test_null_tracker_all_methods_are_noops():
    t = NullTracker()
    # All methods callable, return None, never raise.
    with t.start_run(run_name="whatever") as run:
        assert run is None
    assert t.log_params({"a": 1}) is None
    assert t.log_metrics({"b": 2.0}) is None
    assert t.log_metrics({"c": float("nan")}) is None  # robust to NaN
    assert t.log_artifact("path") is None
    assert t.log_dict({"k": "v"}, "name") is None
    assert t.set_tag("key", "val") is None


def test_get_tracker_returns_null_when_track_false():
    t = get_tracker(track=False)
    assert isinstance(t, NullTracker)


# ---------------------------------------------------------------------------
# MLflowTracker: forwards calls correctly
# ---------------------------------------------------------------------------


def test_mlflow_tracker_sets_uri_and_experiment_on_init(fake_mlflow):
    tracking_mod.MLflowTracker(experiment="sentinel-test", tracking_uri="file:///tmp/mlruns")
    assert fake_mlflow["set_tracking_uri"] == ["file:///tmp/mlruns"]
    assert fake_mlflow["set_experiment"] == ["sentinel-test"]


def test_mlflow_tracker_no_uri_skips_set_tracking_uri(fake_mlflow):
    tracking_mod.MLflowTracker(experiment="sentinel")
    assert fake_mlflow["set_tracking_uri"] == []  # not called without a URI
    assert fake_mlflow["set_experiment"] == ["sentinel"]


def test_mlflow_tracker_start_run_is_a_context(fake_mlflow):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    with t.start_run(run_name="r1") as run:
        assert run is not None
        assert run.info.run_name == "r1"
    assert fake_mlflow["run_context_entered"] == ["r1"]
    assert fake_mlflow["run_context_exited"] == ["r1"]


def test_mlflow_tracker_log_params_forwards(fake_mlflow):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    t.log_params({"model": "logistic", "cost_bps": 2.0, "allow_short": False})
    assert fake_mlflow["log_params"][-1] == {
        "model": "logistic",
        "cost_bps": 2.0,
        "allow_short": False,
    }


def test_mlflow_tracker_log_params_coerces_none(fake_mlflow):
    """mlflow.log_params rejects None values — MLflowTracker coerces to 'None'."""
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    t.log_params({"target_vol_annual": None, "symbol": "SPY"})
    sent = fake_mlflow["log_params"][-1]
    assert sent["target_vol_annual"] == "None"
    assert sent["symbol"] == "SPY"


def test_mlflow_tracker_log_metrics_forwards_only_finite(fake_mlflow):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    t.log_metrics(
        {
            "sharpe": 1.23,
            "win_rate": float("nan"),
            "div_zero": float("inf"),
            "missing": None,
            "not_numeric": "oops",
            "zero": 0.0,
        }
    )
    sent = {k: v for k, v, _ in fake_mlflow["log_metric"]}
    # Only finite numeric metrics make it through.
    assert sent == {"sharpe": 1.23, "zero": 0.0}


def test_mlflow_tracker_log_metrics_forwards_step(fake_mlflow):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    t.log_metrics({"loss": 0.5}, step=7)
    assert fake_mlflow["log_metric"][-1] == ("loss", 0.5, 7)


def test_mlflow_tracker_artifact_path_is_coerced_to_str(fake_mlflow, tmp_path):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    p = tmp_path / "model.pkl"
    p.write_bytes(b"dummy")
    t.log_artifact(p)
    # Path -> str is important because mlflow's underlying API expects str.
    assert fake_mlflow["log_artifact"][-1] == str(p)


def test_mlflow_tracker_set_tag_forwards(fake_mlflow):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    t.set_tag("symbol", "SPY")
    assert fake_mlflow["set_tag"][-1] == ("symbol", "SPY")


def test_mlflow_tracker_log_dict_forwards(fake_mlflow):
    t = tracking_mod.MLflowTracker(experiment="sentinel")
    t.log_dict({"a": 1}, "params.json")
    assert fake_mlflow["log_dict"][-1] == ({"a": 1}, "params.json")


# ---------------------------------------------------------------------------
# Missing lib → actionable error
# ---------------------------------------------------------------------------


def test_mlflow_missing_lib_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # block real import
    with pytest.raises(ImportError, match="mlflow is not installed"):
        tracking_mod.MLflowTracker(experiment="sentinel")


def test_get_tracker_returns_mlflow_when_track_true(fake_mlflow):
    t = get_tracker(track=True, experiment="exp-a", tracking_uri="file:///tmp/x")
    assert isinstance(t, tracking_mod.MLflowTracker)
    assert fake_mlflow["set_experiment"] == ["exp-a"]
    assert fake_mlflow["set_tracking_uri"] == ["file:///tmp/x"]
