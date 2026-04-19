"""Optional experiment tracking. Use :func:`get_tracker` to obtain a tracker
that either no-ops (``track=False``) or forwards to MLflow (``track=True``)."""

from sentinel.tracking.mlflow_tracker import (
    MLflowTracker,
    NullTracker,
    Tracker,
    get_tracker,
)

__all__ = ["Tracker", "NullTracker", "MLflowTracker", "get_tracker"]
