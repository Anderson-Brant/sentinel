"""Scheduler data types + interval parsing.

Keep these dataclasses pure — no I/O, no dependencies on the store. That
keeps them cheap to test and easy to serialize if we ever move scheduling
state out of the local Store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Interval parsing
# ---------------------------------------------------------------------------

# Accept "30s", "15m", "2h", "1d" (case-insensitive), or a bare integer/float
# meaning seconds. This is intentionally narrow — a full cron grammar would
# be overkill for the jobs we run (everything is "every N units").
_INTERVAL_RE = re.compile(r"^\s*(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>[smhd])?\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_interval(spec: int | float | str) -> int:
    """Convert a human-readable interval to whole seconds.

    Examples
    --------
    >>> parse_interval("15m")
    900
    >>> parse_interval("1h")
    3600
    >>> parse_interval(30)
    30
    """
    if isinstance(spec, (int, float)):
        if spec <= 0:
            raise ValueError(f"Interval must be positive, got {spec!r}")
        return int(spec)
    if not isinstance(spec, str):
        raise TypeError(f"Interval must be str or number, got {type(spec).__name__}")
    m = _INTERVAL_RE.match(spec)
    if not m:
        raise ValueError(
            f"Unrecognized interval {spec!r}. Use e.g. '30s', '15m', '2h', '1d'."
        )
    n = float(m.group("n"))
    unit = (m.group("unit") or "s").lower()
    seconds = n * _UNIT_SECONDS[unit]
    if seconds <= 0:
        raise ValueError(f"Interval must be positive, got {spec!r}")
    return int(seconds)


# ---------------------------------------------------------------------------
# JobSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobSpec:
    """Declarative description of a scheduled job.

    Fields
    ------
    name : Unique identifier. Used in logs, job_runs, and CLI filters.
    kind : Registry key naming which operation to run (see
           :mod:`sentinel.scheduling.registry`).
    interval_seconds : Minimum gap between successful runs.
    params : kind-specific keyword arguments forwarded to the job function.
    enabled : Lets the YAML config disable a job without deleting it.
    """

    name: str
    kind: str
    interval_seconds: int
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        kind: str,
        interval: int | float | str,
        params: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> JobSpec:
        return cls(
            name=name,
            kind=kind,
            interval_seconds=parse_interval(interval),
            params=dict(params or {}),
            enabled=enabled,
        )


# ---------------------------------------------------------------------------
# JobRun — persisted result of a single invocation
# ---------------------------------------------------------------------------


STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"
_STATUSES = {STATUS_SUCCESS, STATUS_ERROR, STATUS_SKIPPED}


@dataclass
class JobRun:
    job_name: str
    started_at: datetime
    finished_at: datetime
    status: str
    rows_written: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                f"JobRun.status must be one of {sorted(_STATUSES)}, got {self.status!r}"
            )
        if self.finished_at < self.started_at:
            raise ValueError("JobRun.finished_at precedes started_at")

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


__all__ = [
    "JobSpec",
    "JobRun",
    "parse_interval",
    "STATUS_SUCCESS",
    "STATUS_ERROR",
    "STATUS_SKIPPED",
]
