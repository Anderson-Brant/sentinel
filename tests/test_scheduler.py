"""Tests for the scheduler core, job registry, and job_runs storage.

No network, no real ingestion — we replace the job functions and let the
scheduler drive the store through an injectable clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from sentinel.scheduling import (
    STATUS_ERROR,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    JobRun,
    JobSpec,
    Scheduler,
    load_jobs_from_config,
    parse_interval,
)
from sentinel.scheduling.registry import _REGISTRY, get_job, register, registered_kinds
from sentinel.storage.duckdb_store import DuckDBStore


# ---------------------------------------------------------------------------
# Interval parsing
# ---------------------------------------------------------------------------


def test_parse_interval_unit_strings():
    assert parse_interval("30s") == 30
    assert parse_interval("15m") == 900
    assert parse_interval("2h") == 7200
    assert parse_interval("1d") == 86400


def test_parse_interval_case_insensitive_and_float():
    assert parse_interval("15M") == 900
    assert parse_interval("0.5h") == 1800


def test_parse_interval_numeric_seconds():
    assert parse_interval(45) == 45
    assert parse_interval(12.9) == 12


def test_parse_interval_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_interval("tomorrow")
    with pytest.raises(ValueError):
        parse_interval(0)
    with pytest.raises(ValueError):
        parse_interval(-5)
    with pytest.raises(TypeError):
        parse_interval([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JobRun validation
# ---------------------------------------------------------------------------


def test_job_run_rejects_unknown_status():
    with pytest.raises(ValueError):
        JobRun(
            job_name="x",
            started_at=datetime(2026, 1, 1),
            finished_at=datetime(2026, 1, 1),
            status="bogus",
        )


def test_job_run_rejects_inverted_times():
    with pytest.raises(ValueError):
        JobRun(
            job_name="x",
            started_at=datetime(2026, 1, 2),
            finished_at=datetime(2026, 1, 1),
            status=STATUS_SUCCESS,
        )


# ---------------------------------------------------------------------------
# DuckDB job_runs round-trip
# ---------------------------------------------------------------------------


def _run(store: DuckDBStore, name: str, status: str, started: datetime, *, rows=0, err=None):
    store.record_job_run(
        JobRun(
            job_name=name,
            started_at=started,
            finished_at=started + timedelta(seconds=1),
            status=status,
            rows_written=rows,
            error=err,
        )
    )


def test_duckdb_job_runs_roundtrip(tmp_path):
    store = DuckDBStore(path=tmp_path / "j.duckdb")
    _run(store, "A", STATUS_SUCCESS, datetime(2026, 4, 18, 10, 0), rows=5)
    _run(store, "A", STATUS_ERROR, datetime(2026, 4, 18, 11, 0), err="boom")
    _run(store, "B", STATUS_SUCCESS, datetime(2026, 4, 18, 12, 0), rows=2)

    hist = store.read_job_runs(limit=10)
    assert len(hist) == 3
    # ordered newest first
    assert list(hist["job_name"]) == ["B", "A", "A"]

    only_a = store.read_job_runs(job_name="A", limit=10)
    assert len(only_a) == 2
    assert set(only_a["job_name"]) == {"A"}


def test_duckdb_last_run_ignores_errors(tmp_path):
    store = DuckDBStore(path=tmp_path / "j.duckdb")
    _run(store, "A", STATUS_SUCCESS, datetime(2026, 4, 18, 10, 0))
    _run(store, "A", STATUS_ERROR, datetime(2026, 4, 18, 11, 0), err="boom")
    # last successful was the earlier one
    last = store.last_run_for("A")
    assert last == datetime(2026, 4, 18, 10, 0)


def test_duckdb_last_run_none_when_never_run(tmp_path):
    store = DuckDBStore(path=tmp_path / "j.duckdb")
    assert store.last_run_for("nope") is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_builtins():
    kinds = registered_kinds()
    # The four built-ins must be present.
    for k in ("ingest-prices", "ingest-reddit", "score-sentiment", "build-features"):
        assert k in kinds


def test_register_duplicate_rejected():
    try:
        @register("ingest-prices")  # type: ignore[misc]
        def _dup(store):  # pragma: no cover - should raise
            return {}
    except ValueError as e:
        assert "already registered" in str(e)
    else:  # pragma: no cover
        raise AssertionError("duplicate register should raise")


def test_get_job_unknown_kind_raises():
    with pytest.raises(KeyError):
        get_job("nope")


# ---------------------------------------------------------------------------
# Scheduler: due calculation + dispatch
# ---------------------------------------------------------------------------


class FakeClock:
    """Advanceable clock so tests can skip forward without sleep."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _reg(kind: str):
    """Register a job ``kind`` and return a call-counter + cleanup."""
    calls: list[dict] = []

    def fn(store, **params):
        calls.append(params)
        return {"rows_written": len(calls), "detail": "ok"}

    _REGISTRY[kind] = fn
    return calls, lambda: _REGISTRY.pop(kind, None)


def test_scheduler_runs_due_jobs_and_persists(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    calls_a, cleanup_a = _reg("test-kind-a")
    calls_b, cleanup_b = _reg("test-kind-b")
    try:
        clock = FakeClock(datetime(2026, 4, 18, 9, 0))
        specs = [
            JobSpec(name="a", kind="test-kind-a", interval_seconds=60, params={"x": 1}),
            JobSpec(name="b", kind="test-kind-b", interval_seconds=300),
        ]
        sched = Scheduler(store, specs, clock=clock, sleeper=lambda _: None)

        # Tick 0: both never-run, both due.
        runs = sched.run_once()
        assert {r.job_name for r in runs} == {"a", "b"}
        assert all(r.status == STATUS_SUCCESS for r in runs)
        assert len(calls_a) == 1
        assert calls_a[0] == {"x": 1}

        # Tick 1, +30s: neither due yet (a is 60s, b is 300s).
        clock.advance(30)
        assert sched.run_once() == []

        # Tick 2, +60s from start: only A due.
        clock.advance(30)
        runs = sched.run_once()
        assert [r.job_name for r in runs] == ["a"]
        assert len(calls_a) == 2
        assert len(calls_b) == 1

        # History is durable.
        hist = store.read_job_runs(limit=10)
        assert len(hist) == 3
    finally:
        cleanup_a()
        cleanup_b()


def test_scheduler_error_isolates_and_retries(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")

    def bomb(store, **params):
        raise RuntimeError("nope")

    _REGISTRY["test-bomb"] = bomb
    calls_ok, cleanup_ok = _reg("test-ok")
    try:
        clock = FakeClock(datetime(2026, 4, 18, 9, 0))
        specs = [
            JobSpec(name="boom", kind="test-bomb", interval_seconds=60),
            JobSpec(name="fine", kind="test-ok", interval_seconds=60),
        ]
        sched = Scheduler(store, specs, clock=clock, sleeper=lambda _: None)

        runs = sched.run_once()
        by_name = {r.job_name: r for r in runs}
        assert by_name["boom"].status == STATUS_ERROR
        assert "RuntimeError" in (by_name["boom"].error or "")
        assert by_name["fine"].status == STATUS_SUCCESS  # unaffected
        assert len(calls_ok) == 1

        # Failing job never registered a successful run → still due next tick.
        clock.advance(1)  # any forward motion at all
        due_names = {spec.name for spec, _ in sched.due_jobs()}
        assert "boom" in due_names
        assert "fine" not in due_names  # just ran successfully
    finally:
        _REGISTRY.pop("test-bomb", None)
        cleanup_ok()


def test_scheduler_respects_enabled_flag(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    calls, cleanup = _reg("test-off")
    try:
        specs = [
            JobSpec(name="off", kind="test-off", interval_seconds=60, enabled=False),
        ]
        sched = Scheduler(
            store, specs, clock=FakeClock(datetime(2026, 4, 18)), sleeper=lambda _: None
        )
        assert sched.run_once() == []
        assert calls == []
    finally:
        cleanup()


def test_scheduler_duplicate_name_rejected(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    with pytest.raises(ValueError):
        Scheduler(
            store,
            [
                JobSpec(name="a", kind="test-kind-a", interval_seconds=60),
                JobSpec(name="a", kind="test-kind-b", interval_seconds=60),
            ],
        )


def test_run_forever_max_ticks(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    calls, cleanup = _reg("test-tick")
    sleeps: list[float] = []
    try:
        clock = FakeClock(datetime(2026, 4, 18))

        def sleeper(n: float) -> None:
            sleeps.append(n)
            clock.advance(n)

        specs = [JobSpec(name="t", kind="test-tick", interval_seconds=1)]
        sched = Scheduler(store, specs, clock=clock, sleeper=sleeper)
        ticks = sched.run_forever(tick_seconds=2, max_ticks=3)
        assert ticks == 3
        # 3 ticks → 3 calls (interval=1, sleeper advances 2s between).
        assert len(calls) == 3
        # Sleeper invoked between ticks (2 times for 3 ticks).
        assert len(sleeps) == 2
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# load_jobs_from_config
# ---------------------------------------------------------------------------


def test_load_jobs_from_config_parses_intervals():
    from sentinel.config import ScheduledJobConfig, SchedulerConfig, SentinelConfig

    cfg = SentinelConfig(
        scheduler=SchedulerConfig(
            jobs=[
                ScheduledJobConfig(
                    name="daily-prices",
                    kind="ingest-prices",
                    interval="1d",
                    params={"symbols": ["SPY"]},
                ),
                ScheduledJobConfig(
                    name="wsb",
                    kind="ingest-reddit",
                    interval="15m",
                    enabled=False,
                ),
            ]
        )
    )
    jobs = load_jobs_from_config(cfg)
    assert [j.name for j in jobs] == ["daily-prices", "wsb"]
    assert jobs[0].interval_seconds == 86400
    assert jobs[0].params == {"symbols": ["SPY"]}
    assert jobs[1].interval_seconds == 900
    assert jobs[1].enabled is False


# ---------------------------------------------------------------------------
# record_skipped
# ---------------------------------------------------------------------------


def test_record_skipped_persists(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    spec = JobSpec(name="x", kind="ingest-prices", interval_seconds=60)
    sched = Scheduler(
        store, [spec], clock=FakeClock(datetime(2026, 4, 18)), sleeper=lambda _: None
    )
    run = sched.record_skipped(spec, reason="maintenance")
    assert run.status == STATUS_SKIPPED
    assert run.error == "maintenance"
    hist = store.read_job_runs(job_name="x", limit=5)
    assert len(hist) == 1
    assert hist.iloc[0]["status"] == STATUS_SKIPPED
    # Skipped runs do NOT count as "last successful" — last_run_for stays None.
    assert store.last_run_for("x") is None
