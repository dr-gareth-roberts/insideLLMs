"""Optimize-prompt command: thin CLI wrapper over prompt optimization utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insideLLMs.optimization import OptimizationStrategy, optimize_prompt

from .._output import (
    Colors,
    colorize,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
    print_success,
)


def _parse_strategies(raw: str | None) -> list[OptimizationStrategy] | None:
    if raw is None:
        return None
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    parsed: list[OptimizationStrategy] = []
    for name in names:
        try:
            parsed.append(OptimizationStrategy(name))
        except ValueError as exc:
            allowed = ", ".join(strategy.value for strategy in OptimizationStrategy)
            raise ValueError(f"Unknown strategy '{name}'. Allowed: {allowed}") from exc
    return parsed or None


def cmd_optimize_prompt(args: argparse.Namespace) -> int:
    """Execute the optimize-prompt command."""
    prompt = args.prompt
    if args.input_file:
        try:
            prompt = Path(args.input_file).read_text(encoding="utf-8")
        except Exception as exc:
            print_error(f"Could not read --input-file: {exc}")
            return 1

    if not prompt:
        print_error("Provide a prompt as positional arg or via --input-file")
        return 1

    try:
        strategies = _parse_strategies(args.strategies)
    except ValueError as exc:
        print_error(str(exc))
        return 1

    report = optimize_prompt(prompt, strategies)
    payload = {
        **report.to_dict(),
        "original_prompt": report.original_prompt,
        "optimized_prompt": report.optimized_prompt,
    }

    if args.format == "json":
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            print_success(f"Wrote optimization report: {args.output}")
        else:
            print(rendered)
        return 0

    print_header("Optimize Prompt")
    print_key_value("Token reduction", report.token_reduction)
    print_key_value("Estimated quality change", f"{report.estimated_quality_change:.3f}")
    print_key_value("Strategies applied", ", ".join(s.value for s in report.strategies_applied) or "-")

    if args.show_diff:
        print_subheader("Original")
        print(colorize(report.original_prompt, Colors.DIM))
        print_subheader("Optimized")
        print(report.optimized_prompt)

    if report.suggestions:
        print_subheader("Suggestions")
        for suggestion in report.suggestions[:10]:
            print(f"  - {suggestion}")

    if args.output:
        Path(args.output).write_text(report.optimized_prompt, encoding="utf-8")
        print_success(f"Wrote optimized prompt: {args.output}")

    print()
    print_info(
        "Next: run your harness with this prompt and gate with "
        f"{colorize('insidellms diff --fail-on-changes', Colors.GREEN)}."
    )
    return 0

