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

DETAIL_VIEWS = ("quality", "valuation", "price", "insiders", "competitive")


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


def _fmt_pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


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

    def _row(label: str, score: object | None, fallback: str) -> None:
        grade = getattr(score, "grade", None) if score is not None else None
        summary = getattr(score, "summary", "") if score is not None else ""
        if grade:
            table.add_row(label, Text(grade, style=_grade_style(grade)), summary)
        else:
            table.add_row(label, "—", f"[dim]{summary or fallback}[/dim]")

    _row("Quality", analysis.quality, "no fundamentals data")
    _row("Valuation", analysis.valuation, "no fundamentals data")
    _row("Price hist", analysis.price_history, "no price data")
    _row("Insiders", analysis.insiders, "no insider filings available")
    _row("Competitive", analysis.competitive, "no fundamentals data")

    console.print(table)
    console.print()

    if analysis.composite_grade:
        scored = [name for name, _ in analysis.scored_rows()]
        composite = Text()
        composite.append("Composite: ")
        composite.append(
            analysis.composite_grade, style=_grade_style(analysis.composite_grade)
        )
        if len(scored) < 5:
            composite.append(f"   ({' + '.join(scored)} only)", style="dim")
        console.print(composite)
    else:
        console.print("[dim]Composite: — (no rows scored)[/dim]")

    if analysis.related_tickers:
        console.print(f"[dim]Related: {', '.join(analysis.related_tickers)}[/dim]")

    for note in analysis.notes:
        console.print(f"[dim]note: {note}[/dim]")

    if detail == "price":
        _render_price_detail(analysis)
    elif detail == "valuation":
        _render_valuation_detail(analysis)
    elif detail == "quality":
        _render_quality_detail(analysis)
    elif detail == "insiders":
        _render_insiders_detail(analysis)
    elif detail == "competitive":
        _render_competitive_detail(analysis)


def _detail_table(title: str) -> Table:
    table = Table(title=title, title_style="bold", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    return table


def _render_price_detail(analysis: Analysis) -> None:
    ph = analysis.price_history
    if ph is None:
        console.print("[red]No price history to detail.[/red]")
        return
    console.print()
    table = _detail_table("Price history detail")
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


def _render_quality_detail(analysis: Analysis) -> None:
    q = analysis.quality
    if q is None:
        console.print("[red]No quality data to detail.[/red]")
        return
    console.print()
    table = _detail_table("Quality detail")
    table.add_row("ROIC", _fmt_pct(q.roic))
    table.add_row("Gross margin", _fmt_pct(q.gross_margin))
    table.add_row("Gross margin stability (std)", _fmt_pct(q.gross_margin_stability))
    table.add_row("Operating margin", _fmt_pct(q.operating_margin))
    table.add_row(
        "Op margin trend",
        "-" if q.operating_margin_trend is None else f"{q.operating_margin_trend * 100:+.1f}pp/yr",
    )
    table.add_row(
        "Net debt / EBITDA",
        "net cash" if q.net_debt_ebitda is not None and q.net_debt_ebitda <= 0 else _fmt(q.net_debt_ebitda, ".1f"),
    )
    table.add_row("Revenue growth", _fmt_pct(q.revenue_growth))
    table.add_row("Growth stability (std of YoY)", _fmt_pct(q.growth_stability))
    table.add_row("Numeric score", _fmt(q.score, ".1f"))
    console.print(table)


def _render_insiders_detail(analysis: Analysis) -> None:
    i = analysis.insiders
    if i is None:
        console.print("[red]No insider data to detail.[/red]")
        return
    console.print()
    table = _detail_table("Insiders detail")
    table.add_row("Net 6mo (% of shares)", _fmt_pct(i.net_pct_6m))
    table.add_row("Net 12mo (% of shares)", _fmt_pct(i.net_pct_12m))
    table.add_row("Buys (6mo)", str(i.n_buys_6m))
    table.add_row("Sells (6mo)", str(i.n_sells_6m))
    table.add_row("Numeric score", _fmt(i.score, ".1f"))
    console.print(table)


def _render_competitive_detail(analysis: Analysis) -> None:
    c = analysis.competitive
    if c is None:
        console.print("[red]No competitive data to detail.[/red]")
        return
    console.print()
    table = _detail_table("Competitive detail")
    table.add_row("Sector", c.sector or "-")
    table.add_row("Revenue growth", _fmt_pct(c.revenue_growth))
    table.add_row("Sector median growth", _fmt_pct(c.sector_growth))
    table.add_row("Operating margin", _fmt_pct(c.operating_margin))
    table.add_row("Sector median margin", _fmt_pct(c.sector_margin))
    table.add_row("Numeric score", _fmt(c.score, ".1f"))
    console.print(table)
