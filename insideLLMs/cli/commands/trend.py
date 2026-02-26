"""Trend command: show metric trajectories across runs."""

import argparse
import json
from pathlib import Path
from typing import Any

from insideLLMs.tracking import (
    check_thresholds,
    compute_trends,
    format_sparkline,
    index_run,
    load_index,
)

from .._output import (
    Colors,
    colorize,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
    print_warning,
)


def cmd_trend(args: argparse.Namespace) -> int:
    """Show metric trends across indexed runs."""
    index_path = Path(args.index)

    # If --add is given, index a new run first
    if getattr(args, "add", None):
        run_dir = Path(args.add)
        if not run_dir.is_dir():
            print_error(f"Run directory not found: {run_dir}")
            return 1
        entry = index_run(index_path, run_dir, config_label=getattr(args, "label", ""))
        print_info(f"Indexed run {entry['run_id']} ({entry['record_count']} records)")

    entries = load_index(index_path)
    if not entries:
        print_error(f"No runs indexed in {index_path}")
        print_info("Use --add <run-dir> to index a completed run.")
        return 1

    last_n = getattr(args, "last", 0)
    if last_n and last_n > 0:
        entries = entries[-last_n:]

    output_format = getattr(args, "format", "text")
    metric = getattr(args, "metric", "accuracy")
    trend_data = compute_trends(entries, metric=metric)

    # Check thresholds
    violations: list[dict[str, Any]] = []
    threshold_arg = getattr(args, "threshold", None)
    if threshold_arg is not None:
        violations = check_thresholds(entries, {metric: threshold_arg})

    if output_format == "json":
        payload = {
            "metric": metric,
            "runs": len(entries),
            "trend": trend_data,
            "violations": violations,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if violations and getattr(args, "fail_on_threshold", False):
            return 2
        return 0

    # Text output
    print_header(f"Trend: {metric}")
    print_key_value("Runs", len(entries))

    if trend_data:
        values = [t["value"] for t in trend_data]
        spark = format_sparkline(values)
        print_key_value("Sparkline", spark)
        print()

        print_subheader("Run History")
        for t in trend_data:
            delta_str = ""
            if t["delta"] is not None:
                sign = "+" if t["delta"] >= 0 else ""
                if t["delta"] < 0:
                    delta_str = colorize(f" ({sign}{t['delta']:.4f})", Colors.RED)
                elif t["delta"] > 0:
                    delta_str = colorize(f" ({sign}{t['delta']:.4f})", Colors.GREEN)
                else:
                    delta_str = f" ({sign}{t['delta']:.4f})"
            ts = t["timestamp"][:19] if t["timestamp"] else "?"
            print(f"  {ts}  {t['run_id'][:16]:<16}  {t['value']:.4f}{delta_str}")

    if violations:
        print()
        print_subheader("Threshold Violations")
        for v in violations:
            print_warning(
                f"{v['metric']} dropped to {v['value']:.4f} "
                f"(threshold {v['threshold']:.4f}) at run {v['run_id']}"
            )
        if getattr(args, "fail_on_threshold", False):
            return 2

    return 0
