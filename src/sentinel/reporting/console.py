"""Rich rendering for evaluation reports & prediction output."""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.backtest.engine import BacktestReport
from sentinel.evaluation.ablation import AblationReport
from sentinel.evaluation.importance import ImportanceResult
from sentinel.evaluation.regimes import RegimeReport
from sentinel.evaluation.walk_forward import WalkForwardReport
from sentinel.models.registry import LatestPrediction

console = Console()


def _fmt(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.2f}%"


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


def render_backtest(report: BacktestReport, *, show_trades: int = 5) -> None:
    """Render a BacktestReport: summary comparison table + optional trade list."""
    header = f"[bold]{report.symbol} — backtest[/bold]  "
    header += f"(long>{report.long_threshold:.2f}"
    if report.allow_short:
        header += f", short<{report.short_threshold:.2f}"
    header += f", costs={report.cost_bps:.1f}bps)"
    console.rule(header)

    if report.start_date is not None and report.end_date is not None:
        span = (
            f"[dim]{report.start_date.date()} → {report.end_date.date()} "
            f"· {report.n_oos_bars} OOS bars[/dim]"
        )
        console.print(span)

    # Side-by-side strategy vs buy-and-hold.
    table = Table(show_lines=False, title="Strategy vs. buy-and-hold", title_style="bold")
    table.add_column("Metric")
    table.add_column("Strategy", justify="right", style="cyan")
    table.add_column("Benchmark", justify="right", style="dim")
    table.add_column("Δ", justify="right")

    def _row(label: str, strat: float, bench: float, pct: bool = False) -> None:
        if pct:
            s, b = _fmt_pct(strat), _fmt_pct(bench)
            delta_num = strat - bench
            d = _fmt_pct(delta_num)
        else:
            s, b = _fmt(strat), _fmt(bench)
            delta_num = strat - bench
            d = _fmt(delta_num)
        color = "green" if delta_num > 0 else ("red" if delta_num < 0 else "white")
        table.add_row(label, s, b, f"[{color}]{d}[/{color}]")

    _row("Total return", report.total_return, report.benchmark_total_return, pct=True)
    _row(
        "Annualized return",
        report.annualized_return,
        report.benchmark_annualized_return,
        pct=True,
    )
    _row("Sharpe", report.sharpe, report.benchmark_sharpe)
    _row("Max drawdown", report.max_drawdown, report.benchmark_max_drawdown, pct=True)

    # Strategy-only metrics.
    table.add_section()
    table.add_row("Annualized vol", _fmt_pct(report.annualized_vol), "—", "—")
    table.add_row("Exposure", f"{report.exposure * 100:.1f}%", "100.0%", "—")
    table.add_row("Turnover / bar", f"{report.turnover:.4f}", "—", "—")
    table.add_row("# trades", str(report.n_trades), "—", "—")
    win_rate_str = (
        f"{report.win_rate * 100:.1f}%"
        if report.n_trades and not math.isnan(report.win_rate)
        else "—"
    )
    table.add_row("Win rate", win_rate_str, "—", "—")

    console.print(table)

    # Verdict.
    beats_sharpe = report.sharpe - report.benchmark_sharpe
    beats_return = report.total_return - report.benchmark_total_return
    if beats_sharpe > 0 and beats_return > 0:
        verdict_str = (
            f"[green]Beats buy-and-hold on both Sharpe ({beats_sharpe:+.2f}) "
            f"and total return ({beats_return * 100:+.2f}%) after costs.[/green]"
        )
        color = "green"
    elif beats_sharpe > 0 or beats_return > 0:
        verdict_str = (
            f"[yellow]Mixed vs. buy-and-hold: Sharpe Δ={beats_sharpe:+.2f}, "
            f"return Δ={beats_return * 100:+.2f}%.[/yellow]"
        )
        color = "yellow"
    else:
        verdict_str = (
            f"[red]Underperforms buy-and-hold: Sharpe Δ={beats_sharpe:+.2f}, "
            f"return Δ={beats_return * 100:+.2f}%.[/red]"
        )
        color = "red"
    console.print(Panel(verdict_str, title="Verdict", border_style=color))

    # Recent trades.
    if show_trades and report.trades:
        trade_table = Table(title=f"Last {min(show_trades, len(report.trades))} trades", show_lines=False)
        trade_table.add_column("Entry", style="cyan")
        trade_table.add_column("Exit")
        trade_table.add_column("Dir", justify="center")
        trade_table.add_column("Held", justify="right")
        trade_table.add_column("Gross", justify="right")
        trade_table.add_column("Cost", justify="right", style="dim")
        trade_table.add_column("Net", justify="right")
        for t in report.trades[-show_trades:]:
            net_color = "green" if t.net_return > 0 else "red"
            trade_table.add_row(
                t.entry_date.date().isoformat(),
                t.exit_date.date().isoformat(),
                "LONG" if t.direction == 1 else "SHORT",
                str(t.holding_days),
                _fmt_pct(t.gross_return),
                _fmt_pct(t.cost),
                f"[{net_color}]{_fmt_pct(t.net_return)}[/{net_color}]",
            )
        console.print(trade_table)


def render_ablation(report: AblationReport) -> None:
    """Render an ablation comparison: side-by-side variants + verdict on sentiment uplift."""
    console.rule(
        f"[bold]{report.symbol} — {report.model_name} ablation[/bold]  "
        "[dim](technical / sentiment / hybrid)[/dim]"
    )

    table = Table(show_lines=False)
    table.add_column("Variant", style="cyan")
    table.add_column("# feats", justify="right")
    table.add_column("Mean acc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Naive acc", justify="right", style="dim")

    has_backtest = any(r.backtest_report is not None for r in report.results)
    if has_backtest:
        table.add_column("Return", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD", justify="right")

    for r in report.results:
        row = [
            r.variant,
            str(r.n_features),
            _fmt(r.mean_accuracy),
            _fmt(r.mean_f1),
            _fmt(r.mean_roc_auc),
            _fmt(r.mean_naive_accuracy),
        ]
        if has_backtest:
            bt = r.backtest_report
            if bt is None:
                row += ["—", "—", "—"]
            else:
                row += [_fmt_pct(bt.total_return), _fmt(bt.sharpe), _fmt_pct(bt.max_drawdown)]
        table.add_row(*row)

    console.print(table)

    # --- Verdict -------------------------------------------------------
    acc_uplift = report.sentiment_uplift_accuracy()
    sharpe_uplift = report.sentiment_uplift_sharpe()
    lines: list[str] = []
    if acc_uplift is not None:
        color = "green" if acc_uplift > 0 else ("yellow" if acc_uplift == 0 else "red")
        lines.append(
            f"Accuracy uplift (hybrid − technical): "
            f"[{color}]{acc_uplift:+.3f}[/{color}]"
        )
    if sharpe_uplift is not None:
        color = "green" if sharpe_uplift > 0 else ("yellow" if sharpe_uplift == 0 else "red")
        lines.append(
            f"Sharpe uplift (hybrid − technical): "
            f"[{color}]{sharpe_uplift:+.2f}[/{color}]"
        )

    if acc_uplift is not None and acc_uplift > 0.005:
        lines.append("[green]→ Sentiment appears to add out-of-sample value.[/green]")
    elif acc_uplift is not None and acc_uplift < -0.005:
        lines.append(
            "[red]→ Sentiment hurts out-of-sample performance. "
            "Likely noise or coverage gap.[/red]"
        )
    elif acc_uplift is not None:
        lines.append(
            "[yellow]→ Sentiment has no meaningful impact at this data scale.[/yellow]"
        )

    if lines:
        console.print(Panel("\n".join(lines), title="Verdict", border_style="dim"))


def render_regime_analysis(reports: list[RegimeReport]) -> None:
    """Render one table per regime axis (e.g. volatility, trend).

    Each row compares strategy vs benchmark on the within-regime subset of
    returns — this is the honest answer to "when does your strategy work?"
    """
    if not reports:
        console.print("[yellow]No regime reports to render.[/yellow]")
        return

    for r in reports:
        axis_title = r.axis.capitalize()
        console.rule(
            f"[bold]{r.symbol} — regime analysis: {axis_title}[/bold]  "
            f"[dim]{r.description}[/dim]"
        )

        if not r.metrics:
            console.print(
                f"[yellow]No valid bars for {axis_title} — "
                "is the window longer than the backtest?[/yellow]"
            )
            continue

        table = Table(show_lines=False)
        table.add_column("Regime", style="cyan")
        table.add_column("% time", justify="right")
        table.add_column("n bars", justify="right", style="dim")
        table.add_column("Strat ret", justify="right")
        table.add_column("Bench ret", justify="right", style="dim")
        table.add_column("Δ return", justify="right")
        table.add_column("Strat Sharpe", justify="right")
        table.add_column("Bench Sharpe", justify="right", style="dim")
        table.add_column("Strat DD", justify="right")
        table.add_column("Exposure", justify="right")

        for m in r.metrics:
            delta = m.strategy_total_return - m.benchmark_total_return
            delta_color = "green" if delta > 0 else ("red" if delta < 0 else "white")
            table.add_row(
                m.label,
                f"{m.fraction_of_time * 100:.1f}%",
                str(m.n_bars),
                _fmt_pct(m.strategy_total_return),
                _fmt_pct(m.benchmark_total_return),
                f"[{delta_color}]{_fmt_pct(delta)}[/{delta_color}]",
                _fmt(m.strategy_sharpe),
                _fmt(m.benchmark_sharpe),
                _fmt_pct(m.strategy_max_drawdown),
                f"{m.exposure * 100:.1f}%",
            )

        console.print(table)

        # Honest-assessment line: where is the strategy's edge concentrated?
        by = r.by_label()
        edges = {
            label: m.strategy_total_return - m.benchmark_total_return
            for label, m in by.items()
        }
        if edges:
            best_label = max(edges, key=edges.get)
            worst_label = min(edges, key=edges.get)
            if edges[best_label] > 0:
                note = (
                    f"Largest edge in [green]{best_label}[/green] regime "
                    f"({_fmt_pct(edges[best_label])}); "
                )
            else:
                note = (
                    f"[red]No regime shows positive edge vs buy-and-hold[/red]; "
                )
            if edges[worst_label] < 0:
                note += (
                    f"underperforms most in [red]{worst_label}[/red] regime "
                    f"({_fmt_pct(edges[worst_label])})."
                )
            else:
                note += (
                    f"smallest edge in [dim]{worst_label}[/dim] regime "
                    f"({_fmt_pct(edges[worst_label])})."
                )
            console.print(Panel(note, title="Where the edge lives", border_style="dim"))


def render_importance(
    symbol: str,
    model: str,
    result: ImportanceResult,
    *,
    top: int = 20,
) -> None:
    """Render a top-N feature importance table with a visual magnitude bar.

    The bar is scaled so the top feature fills it — the rest show their
    *relative* importance, which is what actually matters for reading
    the table at a glance.
    """
    method_label = {"permutation": "permutation importance", "shap": "SHAP importance"}.get(
        result.method, result.method
    )
    header = f"[bold]{symbol} — {model} {method_label}[/bold]"
    if result.method == "permutation" and result.scoring:
        header += f"  [dim](scoring={result.scoring})[/dim]"
    console.rule(header)

    rows = result.df.head(top)
    if rows.empty:
        console.print("[yellow]No features to report.[/yellow]")
        return

    # Scale the bar relative to the largest *positive* importance in the
    # top-N window. Negative drops (a feature the model actively hurts
    # its own score by trusting) are rendered with an empty bar so the
    # sign is obvious without the bar misleading the reader.
    max_mag = float(rows["mean_importance"].max()) if not rows.empty else 0.0
    bar_width = 20

    table = Table(show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Feature", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("± Std", justify="right", style="dim")
    table.add_column("Relative", justify="left")

    for i, row in enumerate(rows.itertuples(index=False), start=1):
        mean = float(row.mean_importance)
        std = float(row.std_importance)
        if max_mag > 0 and mean > 0:
            filled = max(1, int(round((mean / max_mag) * bar_width)))
            filled = min(filled, bar_width)
            bar = "█" * filled + "·" * (bar_width - filled)
            bar_color = "green"
        else:
            bar = "·" * bar_width
            bar_color = "red" if mean < 0 else "dim"
        table.add_row(
            str(i),
            str(row.feature),
            _fmt(mean),
            _fmt(std),
            f"[{bar_color}]{bar}[/{bar_color}]",
        )

    console.print(table)

    # Summary note: if the top two features dominate, flag it — that's
    # often a sign of a brittle model.
    total = float(rows["mean_importance"].clip(lower=0).sum())
    if total > 0 and len(rows) >= 2:
        top2 = float(rows["mean_importance"].clip(lower=0).iloc[:2].sum())
        concentration = top2 / total
        if concentration > 0.75:
            console.print(
                Panel(
                    f"Top-2 features explain [bold]{concentration * 100:.0f}%[/bold] of "
                    f"the importance in this window — the model is concentrated. "
                    f"A feature pipeline change here would likely break it.",
                    title="Concentration note",
                    border_style="yellow",
                )
            )


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
