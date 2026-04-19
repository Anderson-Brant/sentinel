"""Experiment tracking abstraction.

Design: a tiny ``Tracker`` protocol plus two implementations — ``NullTracker``
(no-ops) and ``MLflowTracker`` (lazy-imports mlflow). Callers hit
:func:`get_tracker` once, receive a tracker that conforms to the protocol,
and call ``log_params / log_metrics / log_artifact`` without ever branching
on whether tracking is enabled.

Why lazy import: ``mlflow`` is a heavy optional dependency. Sentinel has to
import and run cleanly without it; only users who pass ``--track`` pay the
cost.

Why no-op via NullTracker instead of ``if tracker: ...``: branches at every
call site are noisy, easy to forget, and easy to diverge. A proper null
object is the textbook fix.

Usage::

    tracker = get_tracker(track=track_flag, experiment="sentinel",
                         tracking_uri=mlflow_uri)
    with tracker.start_run(run_name="SPY__logistic"):
        tracker.log_params({"model": "logistic", "cost_bps": 2.0})
        tracker.log_metrics({"sharpe": 1.23, "total_return": 0.18})

"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


_MLFLOW_INSTALL_HINT = (
    "mlflow is not installed. Install it with `pip install mlflow` "
    "to enable --track."
)


@runtime_checkable
class Tracker(Protocol):
    """Minimal tracker surface.

    ``start_run`` is a context manager so callers can scope each run with
    ``with tracker.start_run(...) as run:``. Concrete classes decide what
    "a run" means; the NullTracker just yields None.
    """

    def start_run(
        self, run_name: str | None = None
    ) -> "contextlib.AbstractContextManager[Any]": ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path) -> None: ...
    def log_dict(self, d: dict[str, Any], name: str) -> None: ...
    def set_tag(self, key: str, value: str) -> None: ...


class NullTracker:
    """Zero-dependency no-op. Safe to use when ``--track`` is off."""

    @contextlib.contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[None]:
        yield None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str | Path) -> None:
        return None

    def log_dict(self, d: dict[str, Any], name: str) -> None:
        return None

    def set_tag(self, key: str, value: str) -> None:
        return None


class MLflowTracker:
    """Forwards logging calls to mlflow. Lazy-imports on construction.

    On init, optionally sets the tracking URI (default: mlflow's own default,
    typically ``./mlruns``) and creates or switches to the named experiment.

    :meth:`log_metrics` silently drops NaN/inf/None values — MLflow rejects
    them, and swallowing one bad metric shouldn't take down a whole run.
    :meth:`log_params` coerces ``None`` to the string ``"None"`` for the same
    reason.
    """

    def __init__(
        self,
        *,
        experiment: str = "sentinel",
        tracking_uri: str | None = None,
    ) -> None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - exercised via stubbed test
            raise ImportError(_MLFLOW_INSTALL_HINT) from e
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.experiment = experiment
        self.tracking_uri = tracking_uri

    @contextlib.contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[Any]:
        with self._mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        cleaned = {k: ("None" if v is None else v) for k, v in params.items()}
        self._mlflow.log_params(cleaned)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(f) or math.isinf(f):
                continue
            self._mlflow.log_metric(k, f, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def log_dict(self, d: dict[str, Any], name: str) -> None:
        self._mlflow.log_dict(d, name)

    def set_tag(self, key: str, value: str) -> None:
        self._mlflow.set_tag(key, value)


def get_tracker(
    *,
    track: bool,
    experiment: str = "sentinel",
    tracking_uri: str | None = None,
) -> Tracker:
    """Return a ``Tracker``: ``NullTracker`` if ``track`` is False, else
    ``MLflowTracker`` (lazy-imports mlflow)."""
    if not track:
        return NullTracker()
    return MLflowTracker(experiment=experiment, tracking_uri=tracking_uri)
