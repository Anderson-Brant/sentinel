"""Scheduler commands: run, status, history."""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel.config import load_config

schedule_app = typer.Typer(
    help="Run + inspect scheduled ingestion / feature jobs.", no_args_is_help=True
)

console = Console()


@schedule_app.command("run")
def schedule_run(
    once: bool = typer.Option(
        False,
        "--once/--forever",
        help="Run one pass over due jobs and exit (default), or loop forever.",
    ),
    tick_seconds: int | None = typer.Option(
        None,
        "--tick-seconds",
        help="Seconds between ticks in --forever mode. Defaults to scheduler.tick_seconds.",
    ),
) -> None:
    """Run scheduled jobs defined under ``scheduler.jobs`` in config."""
    from sentinel.scheduling import Scheduler, load_jobs_from_config
    from sentinel.storage import get_store

    cfg = load_config()
    jobs = load_jobs_from_config(cfg)
    if not jobs:
        console.print(
            "[yellow]No scheduler.jobs configured.[/yellow] "
            "Add jobs to your YAML config - see README."
        )
        raise typer.Exit(code=0)

    store = get_store()
    sched = Scheduler(store, jobs)

    if once:
        runs = sched.run_once()
        if not runs:
            console.print("[dim]No jobs due.[/dim]")
            return
        for r in runs:
            colour = {"success": "green", "error": "red", "skipped": "yellow"}.get(
                r.status, "white"
            )
            console.print(
                f"[{colour}]{r.status}[/{colour}] [cyan]{r.job_name}[/cyan] "
                f"in {r.duration_seconds:.2f}s, rows={r.rows_written}"
                + (f" - {r.error}" if r.error else "")
            )
        return

    tick = tick_seconds if tick_seconds is not None else cfg.scheduler.tick_seconds
    console.print(
        f"[green]✓[/green] Scheduler running. Jobs: "
        f"{', '.join(j.name for j in jobs)}. Tick: {tick}s. Ctrl-C to stop."
    )
    sched.run_forever(tick_seconds=tick)
    console.print("[dim]Scheduler stopped.[/dim]")


@schedule_app.command("status")
def schedule_status() -> None:
    """Show every configured job and when it last ran / is next due."""
    from rich.table import Table

    from sentinel.scheduling import Scheduler, load_jobs_from_config
    from sentinel.storage import get_store

    cfg = load_config()
    jobs = load_jobs_from_config(cfg)
    if not jobs:
        console.print("[yellow]No scheduler.jobs configured.[/yellow]")
        raise typer.Exit(code=0)

    store = get_store()
    sched = Scheduler(store, jobs)
    now = sched.clock()

    table = Table(title="Scheduled jobs")
    table.add_column("name", style="cyan", no_wrap=True)
    table.add_column("kind")
    table.add_column("interval", justify="right")
    table.add_column("enabled", justify="center")
    table.add_column("last run", style="dim")
    table.add_column("next due", style="dim")

    for spec in jobs:
        last = store.last_run_for(spec.name)
        next_due = sched.next_due_at(spec)
        last_str = last.isoformat(timespec="seconds") if last else "-"
        if not spec.enabled:
            next_str = "disabled"
        elif next_due is None:
            next_str = "next tick"
        elif next_due <= now:
            next_str = "due now"
        else:
            delta = next_due - now
            next_str = f"in {int(delta.total_seconds())}s"
        table.add_row(
            spec.name,
            spec.kind,
            f"{spec.interval_seconds}s",
            "✓" if spec.enabled else "✗",
            last_str,
            next_str,
        )
    console.print(table)


@schedule_app.command("history")
def schedule_history(
    job: str | None = typer.Option(
        None, "--job", help="Filter to a single job name."
    ),
    limit: int = typer.Option(
        20, "--limit", help="Max rows to show, newest first.", min=1
    ),
) -> None:
    """Show the most recent rows of the ``job_runs`` log."""
    from rich.table import Table

    from sentinel.storage import get_store

    store = get_store()
    df = store.read_job_runs(job_name=job, limit=limit)
    if df.empty:
        console.print("[dim]No job runs recorded yet.[/dim]")
        return

    table = Table(title=f"Job history ({len(df)} row{'s' if len(df) != 1 else ''})")
    table.add_column("job", style="cyan")
    table.add_column("started", style="dim")
    table.add_column("dur", justify="right")
    table.add_column("status")
    table.add_column("rows", justify="right")
    table.add_column("error", style="red")

    for row in df.itertuples(index=False):
        status = str(row.status)
        colour = {"success": "green", "error": "red", "skipped": "yellow"}.get(
            status, "white"
        )
        duration = (row.finished_at - row.started_at).total_seconds()
        table.add_row(
            str(row.job_name),
            row.started_at.isoformat(timespec="seconds"),
            f"{duration:.2f}s",
            f"[{colour}]{status}[/{colour}]",
            str(int(row.rows_written) if row.rows_written is not None else 0),
            (row.error or "")[:80],
        )
    console.print(table)
