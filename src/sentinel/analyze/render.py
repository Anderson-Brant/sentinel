"""Terminal rendering for the analyze scorecard.

One line per dimension: letter grade + 1-line evidence. Detail views expand
a single row into its underlying numbers.
"""

from __future__ import annotations

import math

from rich.console import Console
from rich.table import Table
from rich.text import Text

from sentinel.analyze.analysis import Analysis

console = Console()

_PENDING = {
    "Quality": "v0.3",
    "Insiders": "v0.5",
    "Competitive": "v0.6",
}


def _grade_style(grade: str) -> str:
    if grade.startswith("A"):
        return "bold green"
    if grade.startswith("B"):
        return "bold cyan"
    if grade.startswith("C"):
        return "bold yellow"
    return "bold red"


def _fmt_mcap(mcap: float | None) -> str:
    if mcap is None:
        return ""
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if mcap >= cut:
            return f"${mcap / cut:.1f}{suffix} mcap"
    return f"${mcap:,.0f} mcap"


def _fmt(x: float | None, spec: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:{spec}}"


def _header(analysis: Analysis) -> str:
    bits = [f"[bold]{analysis.symbol}[/bold]"]
    if analysis.company_name:
        bits.append(analysis.company_name)
    if analysis.industry or analysis.sector:
        bits.append(analysis.industry or analysis.sector or "")
    mcap = _fmt_mcap(analysis.market_cap)
    if mcap:
        bits.append(mcap)
    bits.append(f"as of {analysis.as_of.isoformat()}")
    return " · ".join(bits)


def render_analysis(analysis: Analysis, *, detail: str | None = None) -> None:
    console.print(_header(analysis))
    console.print()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Row", min_width=13)
    table.add_column("Grade", min_width=4)
    table.add_column("Evidence")

    def _row(label: str, grade: str | None, summary: str) -> None:
        if grade:
            table.add_row(label, Text(grade, style=_grade_style(grade)), summary)
        else:
            table.add_row(label, "—", f"[dim]{summary}[/dim]")

    _row("Quality", None, f"(pending, {_PENDING['Quality']})")
    if analysis.valuation is not None:
        _row("Valuation", analysis.valuation.grade, analysis.valuation.summary)
    else:
        _row("Valuation", None, "no fundamentals data")
    if analysis.price_history is not None:
        _row("Price hist", analysis.price_history.grade, analysis.price_history.summary)
    else:
        _row("Price hist", None, "no price data")
    _row("Insiders", None, f"(pending, {_PENDING['Insiders']})")
    _row("Competitive", None, f"(pending, {_PENDING['Competitive']})")

    console.print(table)
    console.print()

    if analysis.composite_grade:
        scored = []
        if analysis.price_history is not None and analysis.price_history.grade:
            scored.append("price history")
        if analysis.valuation is not None and analysis.valuation.grade:
            scored.append("valuation")
        composite = Text()
        composite.append("Composite: ")
        composite.append(
            analysis.composite_grade, style=_grade_style(analysis.composite_grade)
        )
        composite.append(f"   ({' + '.join(scored)} only)", style="dim")
        console.print(composite)
    else:
        console.print("[dim]Composite: — (no rows scored)[/dim]")

    for note in analysis.notes:
        console.print(f"[dim]note: {note}[/dim]")

    if detail == "price":
        _render_price_detail(analysis)
    elif detail == "valuation":
        _render_valuation_detail(analysis)


def _render_price_detail(analysis: Analysis) -> None:
    ph = analysis.price_history
    if ph is None:
        console.print("[red]No price history to detail.[/red]")
        return
    console.print()
    table = Table(title="Price history detail", title_style="bold", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for h in sorted(ph.cagr):
        table.add_row(f"CAGR {h}y", f"{ph.cagr[h] * 100:+.1f}%")
    table.add_row("Max drawdown", f"{ph.max_drawdown * 100:.1f}%")
    table.add_row(
        "Recovery",
        "not yet" if ph.drawdown_recovery_days is None else f"{ph.drawdown_recovery_days}d",
    )
    table.add_row("Sharpe (full window)", _fmt(ph.sharpe))
    table.add_row("History", f"{ph.years:.1f}y")
    table.add_row("Numeric score", _fmt(ph.score, ".1f"))
    console.print(table)


def _render_valuation_detail(analysis: Analysis) -> None:
    v = analysis.valuation
    if v is None:
        console.print("[red]No valuation data to detail.[/red]")
        return
    console.print()
    table = Table(title="Valuation detail", title_style="bold", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Pctile vs own history", justify="right")
    table.add_row("P/E (trailing)", _fmt(v.pe, ".1f"), _fmt(v.pe_pctile, ".0f"))
    table.add_row("P/S", _fmt(v.ps, ".1f"), _fmt(v.ps_pctile, ".0f"))
    table.add_row("P/FCF", _fmt(v.pfcf, ".1f"), _fmt(v.pfcf_pctile, ".0f"))
    table.add_row("EV/EBITDA", _fmt(v.ev_ebitda, ".1f"), "")
    table.add_row("PEG", _fmt(v.peg, ".1f"), "")
    table.add_row(
        "Dividend yield",
        "-" if v.dividend_yield is None else f"{v.dividend_yield * 100:.2f}%",
        "",
    )
    table.add_row("Numeric score", _fmt(v.score, ".1f"), "")
    console.print(table)
