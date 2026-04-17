"""Rich rendering for evaluation reports & prediction output."""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.evaluation.walk_forward import WalkForwardReport
from sentinel.models.registry import LatestPrediction

console = Console()


def _fmt(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def render_evaluation(symbol: str, model: str, report: WalkForwardReport) -> None:
    console.rule(f"[bold]{symbol} — {model} walk-forward evaluation[/bold]")

    table = Table(show_lines=False)
    table.add_column("Fold", justify="right", style="cyan")
    table.add_column("Train end")
    table.add_column("Test range")
    table.add_column("n_test", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Naive acc", justify="right", style="dim")

    for f in report.folds:
        table.add_row(
            str(f.fold),
            f.train_end.date().isoformat(),
            f"{f.test_start.date()} → {f.test_end.date()}",
            str(f.n_test),
            _fmt(f.accuracy),
            _fmt(f.f1),
            _fmt(f.roc_auc),
            _fmt(f.naive_accuracy),
        )

    # Summary row.
    table.add_section()
    table.add_row(
        "[bold]mean[/bold]",
        "",
        "",
        "",
        f"[bold]{_fmt(report.mean_accuracy)}[/bold]",
        f"[bold]{_fmt(report.mean_f1)}[/bold]",
        f"[bold]{_fmt(report.mean_roc_auc)}[/bold]",
        f"[bold]{_fmt(report.mean_naive_accuracy)}[/bold]",
    )

    console.print(table)

    # Interpretation line.
    delta = report.mean_accuracy - report.mean_naive_accuracy
    verdict = (
        f"[green]Beats naive baseline by {delta:+.3f}.[/green]"
        if delta > 0
        else f"[red]Underperforms naive baseline by {delta:+.3f}.[/red]"
    )
    console.print(Panel(verdict, title="Verdict", border_style="dim"))


def render_prediction(symbol: str, model: str, pred: LatestPrediction) -> None:
    label_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(
        pred.label, "white"
    )
    panel = Panel.fit(
        f"[bold]{symbol}[/bold]  ·  as of [cyan]{pred.as_of.date()}[/cyan]\n"
        f"Model: [bold]{model}[/bold]\n"
        f"Direction: {'UP' if pred.direction == 1 else 'DOWN'}\n"
        f"P(up) = [bold]{_fmt(pred.probability_up)}[/bold]\n"
        f"Signal: [bold {label_color}]{pred.label.upper()}[/bold {label_color}]",
        title="Latest prediction",
        border_style=label_color,
    )
    console.print(panel)
