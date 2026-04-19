"""Scheduling package — declarative recurring ingestion / feature jobs.

Public surface::

    from sentinel.scheduling import JobSpec, Scheduler, load_jobs_from_config

Typical usage::

    from sentinel.config import load_config
    from sentinel.storage import get_store

    store = get_store()
    jobs = load_jobs_from_config(load_config())
    Scheduler(store, jobs).run_forever(tick_seconds=30)

The jobs themselves live in :mod:`sentinel.scheduling.registry`. New job
kinds are added by decorating a function with ``@register("my-kind")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentinel.scheduling.registry import (
    get_job,
    register,
    registered_kinds,
)
from sentinel.scheduling.scheduler import Scheduler
from sentinel.scheduling.spec import (
    STATUS_ERROR,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    JobRun,
    JobSpec,
    parse_interval,
)

if TYPE_CHECKING:
    from sentinel.config import SentinelConfig


def load_jobs_from_config(cfg: SentinelConfig) -> list[JobSpec]:
    """Convert the YAML ``scheduler.jobs`` section into JobSpec instances."""
    return [
        JobSpec.from_config(
            name=j.name,
            kind=j.kind,
            interval=j.interval,
            params=j.params,
            enabled=j.enabled,
        )
        for j in cfg.scheduler.jobs
    ]


__all__ = [
    "JobSpec",
    "JobRun",
    "Scheduler",
    "parse_interval",
    "register",
    "get_job",
    "registered_kinds",
    "load_jobs_from_config",
    "STATUS_SUCCESS",
    "STATUS_ERROR",
    "STATUS_SKIPPED",
]
