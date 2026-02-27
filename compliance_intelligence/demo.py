#!/usr/bin/env python3
"""CLI demo — run all scenarios through the compliance pipeline with rich output.

Usage:
    python demo.py                  # Run all scenarios
    python demo.py --scenario high_risk  # Run a specific scenario
    python demo.py --list           # List available scenarios
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich import box

console = Console()


def print_report(scenario_key: str, scenario_info: dict, state) -> None:
    """Pretty-print a compliance report."""
    report = state.report
    if not report:
        console.print("[red]No report generated![/red]")
        return

    decision = report.decision
    risk = report.risk_score

    # Verdict colors
    verdict_colors = {
        "approve": "green",
        "flag_for_review": "yellow",
        "escalate": "bright_red",
        "block": "red bold",
        "request_more_info": "blue",
    }
    vc = verdict_colors.get(decision.verdict.value, "white") if decision else "white"

    # Header
    console.print()
    console.rule(f"[bold]{scenario_info['name']}[/bold]", style="bright_blue")

    # Executive Summary
    console.print(Panel(
        report.executive_summary,
        title="[bold]Executive Summary[/bold]",
        border_style="bright_blue",
        padding=(1, 2),
    ))

    # Risk Score Table
    if risk:
        table = Table(title="Risk Breakdown", box=box.ROUNDED, border_style="dim")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Bar", min_width=20)

        for label, score in [
            ("Entity", risk.entity_risk_score),
            ("Transaction", risk.transaction_risk_score),
            ("Pattern", risk.pattern_risk_score),
            ("Geopolitical", risk.geopolitical_risk_score),
        ]:
            bar_len = int(score / 5)
            if score >= 75:
                color = "red"
            elif score >= 50:
                color = "bright_red"
            elif score >= 25:
                color = "yellow"
            else:
                color = "green"
            bar = f"[{color}]{'█' * bar_len}{'░' * (20 - bar_len)}[/{color}]"
            table.add_row(label, f"[{color}]{score:.0f}/100[/{color}]", bar)

        table.add_section()
        overall_color = vc
        table.add_row(
            "[bold]OVERALL[/bold]",
            f"[{overall_color}]{risk.overall_score:.1f}/100[/{overall_color}]",
            f"[{overall_color}]{risk.overall_level.value.upper()}[/{overall_color}]",
        )
        console.print(table)

    # Verdict
    if decision:
        verdict_text = decision.verdict.value.replace("_", " ").upper()
        console.print(Panel(
            f"[{vc}]{verdict_text}[/{vc}]  (confidence: {decision.confidence:.0%})\n\n"
            f"{decision.rationale}",
            title="[bold]Compliance Decision[/bold]",
            border_style=vc.split()[0] if " " in vc else vc,
        ))

    # Alerts
    if report.alerts:
        alert_table = Table(title=f"Alerts ({len(report.alerts)})", box=box.SIMPLE, border_style="dim")
        alert_table.add_column("Sev", width=8)
        alert_table.add_column("Title", style="white")
        alert_table.add_column("Description", style="dim")
        for alert in report.alerts:
            sev_color = {"critical": "red", "warning": "yellow", "info": "blue"}.get(alert.severity.value, "white")
            alert_table.add_row(
                f"[{sev_color}]{alert.severity.value.upper()}[/{sev_color}]",
                alert.title,
                alert.description[:120] + ("..." if len(alert.description) > 120 else ""),
            )
        console.print(alert_table)

    # Recommended Actions
    if decision and decision.recommended_actions:
        tree = Tree("[bold]Recommended Actions[/bold]")
        for i, action in enumerate(decision.recommended_actions, 1):
            tree.add(f"[cyan]{i}.[/cyan] {action}")
        console.print(tree)

    # Processing trace (condensed)
    console.print(f"\n[dim]Processing time: {report.processing_time_ms:.0f}ms | "
                  f"Re-analysis cycles: {report.reanalysis_count} | "
                  f"Steps: {len(state.processing_steps)}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Compliance Intelligence CLI Demo")
    parser.add_argument("--scenario", "-s", help="Run a specific scenario")
    parser.add_argument("--list", "-l", action="store_true", help="List available scenarios")
    parser.add_argument("--trace", "-t", action="store_true", help="Show full agent trace")
    args = parser.parse_args()

    # Import here so startup is fast for --help
    from app.scenarios import SCENARIOS
    from app.graph.workflow import run_compliance_pipeline

    if args.list:
        table = Table(title="Available Scenarios", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Expected", style="yellow")
        table.add_column("Description", style="dim")
        for key, info in SCENARIOS.items():
            table.add_row(key, info["name"], info["expected_verdict"], info["description"][:80])
        console.print(table)
        return

    # Banner
    console.print(Panel.fit(
        "[bold bright_blue]Compliance Intelligence[/bold bright_blue]\n"
        "[dim]Multi-Agent AML/KYC Transaction Monitoring[/dim]\n"
        "[dim]LangGraph + Deep Agents + Pydantic v2[/dim]",
        border_style="bright_blue",
    ))

    scenarios_to_run = {}
    if args.scenario:
        if args.scenario not in SCENARIOS:
            console.print(f"[red]Unknown scenario: {args.scenario}[/red]")
            console.print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        scenarios_to_run[args.scenario] = SCENARIOS[args.scenario]
    else:
        scenarios_to_run = SCENARIOS

    for key, info in scenarios_to_run.items():
        transaction = info["factory"]()
        console.print(f"\n[bright_blue]▶ Running:[/bright_blue] {info['name']}")
        console.print(f"  [dim]TXN {transaction.transaction_id} | "
                      f"{transaction.transaction_type.value} | "
                      f"{transaction.currency.value} {transaction.amount:,.2f}[/dim]")

        start = time.time()
        state = run_compliance_pipeline(transaction)
        elapsed = (time.time() - start) * 1000

        if state.report:
            state.report.processing_time_ms = round(elapsed, 1)

        print_report(key, info, state)

        if args.trace and state.processing_steps:
            console.print("\n[bold dim]Agent Trace:[/bold dim]")
            for step in state.processing_steps:
                console.print(f"  [dim]{step}[/dim]")

    console.print("\n")
    console.rule("[bold green]All scenarios complete[/bold green]")


if __name__ == "__main__":
    main()
