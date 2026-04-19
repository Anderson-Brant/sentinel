"""Scheduler loop.

Design
------
The scheduler is deliberately boring. Given a set of ``JobSpec``s and a
``Store``:

1. On each ``run_once``, look up every job's last successful run in the
   ``job_runs`` table.
2. Compute the set of jobs whose last run is older than their interval (or
   that have never run).
3. Execute each due job inside its own try/except. Record a ``JobRun`` row
   — success, error, or skipped — regardless of outcome.
4. In daemon mode (``run_forever``), sleep for ``tick_seconds`` and repeat.

The design choices worth calling out:

* **No in-memory state.** Every "what's due?" decision consults the store.
  That makes the scheduler idempotent across crashes — if the process dies
  mid-loop, the next startup picks up where it left off based on what's
  actually in ``job_runs``.
* **Clock + sleeper are injected.** The defaults are
  ``datetime.utcnow()`` and ``time.sleep``; tests pass fakes that advance
  time deterministically.
* **Per-job error isolation.** One failing job never aborts the loop; the
  failure goes into ``job_runs.status = 'error'`` and the scheduler moves
  on to the next due job.
* **Graceful stop.** ``run_forever`` catches ``KeyboardInterrupt`` and
  returns cleanly so operators can Ctrl-C without stack traces.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable
from datetime import datetime

from sentinel.scheduling.registry import get_job
from sentinel.scheduling.spec import (
    STATUS_ERROR,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    JobRun,
    JobSpec,
)
from sentinel.storage.base import Store
from sentinel.utils.logging import get_logger

log = get_logger(__name__)


# A "clock" returns a naive UTC datetime (matching our storage convention).
Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


def _default_clock() -> datetime:
    # naive UTC to match job_runs schema
    return datetime.utcnow()


class Scheduler:
    """Owns a set of jobs and decides when each one should run.

    Parameters
    ----------
    store : a Sentinel Store with the ``job_runs`` methods.
    jobs : iterable of JobSpec. Duplicates by ``name`` are rejected up-front.
    clock : callable returning naive UTC datetime. Injectable for tests.
    sleeper : callable(float) for ``run_forever``'s inter-tick pauses.
    """

    def __init__(
        self,
        store: Store,
        jobs: Iterable[JobSpec],
        *,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self.store = store
        self.jobs: list[JobSpec] = list(jobs)
        seen: set[str] = set()
        for job in self.jobs:
            if job.name in seen:
                raise ValueError(f"Duplicate job name: {job.name!r}")
            seen.add(job.name)
        self.clock: Clock = clock or _default_clock
        self.sleeper: Sleeper = sleeper or time.sleep

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def due_jobs(self, *, now: datetime | None = None) -> list[tuple[JobSpec, datetime | None]]:
        """Return ``(spec, last_run_at)`` pairs for every job currently due.

        A job is due if it is enabled AND (has never run OR its last run is
        older than ``interval_seconds`` before ``now``).
        """
        now = now or self.clock()
        due: list[tuple[JobSpec, datetime | None]] = []
        for spec in self.jobs:
            if not spec.enabled:
                continue
            last = self.store.last_run_for(spec.name)
            if last is None or (now - last).total_seconds() >= spec.interval_seconds:
                due.append((spec, last))
        return due

    def next_due_at(self, spec: JobSpec) -> datetime | None:
        """When this job is next due to run, based on its last run.

        ``None`` means "run at next tick" (never run before, or disabled).
        """
        if not spec.enabled:
            return None
        last = self.store.last_run_for(spec.name)
        if last is None:
            return None
        from datetime import timedelta

        return last + timedelta(seconds=spec.interval_seconds)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_once(self, *, now: datetime | None = None) -> list[JobRun]:
        """Run every currently-due job exactly once; return all persisted runs."""
        now = now or self.clock()
        results: list[JobRun] = []
        for spec, _last in self.due_jobs(now=now):
            results.append(self._run_job(spec))
        return results

    def _run_job(self, spec: JobSpec) -> JobRun:
        started = self.clock()
        log.info("scheduler: starting job %s (%s)", spec.name, spec.kind)
        try:
            fn = get_job(spec.kind)
            result = fn(self.store, **spec.params) or {}
            finished = self.clock()
            run = JobRun(
                job_name=spec.name,
                started_at=started,
                finished_at=finished,
                status=STATUS_SUCCESS,
                rows_written=int(result.get("rows_written", 0)),
                error=None,
            )
            log.info(
                "scheduler: %s ok in %.2fs — %s",
                spec.name,
                run.duration_seconds,
                result.get("detail", ""),
            )
        except Exception as e:  # noqa: BLE001 — isolating per-job failures
            finished = self.clock()
            run = JobRun(
                job_name=spec.name,
                started_at=started,
                finished_at=finished,
                status=STATUS_ERROR,
                rows_written=0,
                error=f"{type(e).__name__}: {e}",
            )
            log.error(
                "scheduler: %s failed after %.2fs:\n%s",
                spec.name,
                run.duration_seconds,
                "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            )
        self.store.record_job_run(run)
        return run

    def run_forever(
        self,
        *,
        tick_seconds: int = 30,
        max_ticks: int | None = None,
    ) -> int:
        """Daemon loop. Returns the number of ticks executed.

        ``max_ticks`` is a test hook — set it and the loop exits after N ticks
        without needing a signal. Production callers leave it as ``None``.
        """
        tick = 0
        try:
            while True:
                if max_ticks is not None and tick >= max_ticks:
                    break
                self.run_once()
                tick += 1
                if max_ticks is not None and tick >= max_ticks:
                    break
                self.sleeper(tick_seconds)
        except KeyboardInterrupt:
            log.info("scheduler: received interrupt, stopping cleanly")
        return tick

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def record_skipped(self, spec: JobSpec, *, reason: str) -> JobRun:
        """Persist a 'skipped' row. Useful when a job is disabled at the CLI."""
        now = self.clock()
        run = JobRun(
            job_name=spec.name,
            started_at=now,
            finished_at=now,
            status=STATUS_SKIPPED,
            rows_written=0,
            error=reason,
        )
        self.store.record_job_run(run)
        return run


__all__ = ["Scheduler", "Clock", "Sleeper"]
