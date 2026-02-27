"""Interactive snapshot helpers for diff workflows."""

from __future__ import annotations

import shutil
import textwrap
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SNAPSHOT_CANDIDATE_FILES: tuple[str, ...] = (
    "records.jsonl",
    "results.jsonl",
    "manifest.json",
    "summary.json",
    "report.html",
    "config.resolved.yaml",
)


def _summary_display(summary: Mapping[str, Any] | None) -> str:
    """Produce compact text for interactive baseline/candidate review."""
    if not isinstance(summary, Mapping):
        return "-"

    output_value = summary.get("output")
    if output_value is not None:
        return str(output_value)

    status = summary.get("status")
    metric_name = summary.get("primary_metric")
    metric_value = summary.get("primary_score")
    if metric_name and metric_value is not None:
        return f"status={status}; {metric_name}={metric_value}"
    if status:
        return f"status={status}"
    return "-"


def _side_by_side_lines(
    baseline_text: str,
    candidate_text: str,
    *,
    width: int = 44,
) -> list[str]:
    left = textwrap.wrap(baseline_text, width=width) or [""]
    right = textwrap.wrap(candidate_text, width=width) or [""]
    lines = [
        "    baseline".ljust(width + 6) + "| candidate",
        "    " + "-" * width + "+" + "-" * (width + 1),
    ]
    for line_l, line_r in zip_longest(left, right, fillvalue=""):
        lines.append(f"    {line_l:<{width}} | {line_r}")
    return lines


def build_interactive_review_lines(
    diff_report: Mapping[str, Any],
    *,
    limit: int,
    width: int = 44,
    dim_text: Callable[[str], str] | None = None,
) -> list[str]:
    """Build a concise interactive review view for snapshot acceptance."""
    regressions = diff_report.get("regressions", [])
    changes = diff_report.get("changes", [])
    only_baseline = diff_report.get("only_baseline", [])
    only_candidate = diff_report.get("only_candidate", [])
    review_items = [*regressions, *changes]
    dim = dim_text or (lambda s: s)

    lines: list[str] = []
    if not review_items and not only_baseline and not only_candidate:
        lines.append("  No differences to review.")
        return lines

    shown = 0
    for item in review_items[:limit]:
        if not isinstance(item, Mapping):
            continue
        label = item.get("label") if isinstance(item.get("label"), Mapping) else {}
        model_label = label.get("model", item.get("model_id", "unknown"))
        probe_label = label.get("probe", item.get("probe_id", "unknown"))
        example_label = label.get("example", item.get("example_id", "unknown"))
        detail = item.get("detail") or item.get("kind", "change")
        lines.append(f"  {model_label} | {probe_label} | example {example_label}: {detail}")
        baseline_text = _summary_display(item.get("baseline"))
        candidate_text = _summary_display(item.get("candidate"))
        lines.extend(_side_by_side_lines(baseline_text, candidate_text, width=width))
        shown += 1

    remaining = len(review_items) - shown
    if remaining > 0:
        lines.append(dim(f"  ... and {remaining} more reviewed changes"))

    if isinstance(only_baseline, list) and only_baseline:
        lines.append("  Missing in candidate:")
        for item in only_baseline[:limit]:
            if not isinstance(item, Mapping):
                continue
            label = item.get("label") if isinstance(item.get("label"), Mapping) else {}
            lines.append(
                f"    {label.get('model', item.get('model_id', 'unknown'))} | "
                f"{label.get('probe', item.get('probe_id', 'unknown'))} | "
                f"example {label.get('example', item.get('example_id', 'unknown'))}"
            )
        if len(only_baseline) > limit:
            lines.append(dim(f"    ... and {len(only_baseline) - limit} more"))

    if isinstance(only_candidate, list) and only_candidate:
        lines.append("  New in candidate:")
        for item in only_candidate[:limit]:
            if not isinstance(item, Mapping):
                continue
            label = item.get("label") if isinstance(item.get("label"), Mapping) else {}
            lines.append(
                f"    {label.get('model', item.get('model_id', 'unknown'))} | "
                f"{label.get('probe', item.get('probe_id', 'unknown'))} | "
                f"example {label.get('example', item.get('example_id', 'unknown'))}"
            )
        if len(only_candidate) > limit:
            lines.append(dim(f"    ... and {len(only_candidate) - limit} more"))

    return lines


def print_interactive_review(
    diff_report: Mapping[str, Any],
    *,
    limit: int,
    emit: Callable[[str], None] = print,
    emit_subheader: Callable[[str], None] | None = None,
    width: int = 44,
    dim_text: Callable[[str], str] | None = None,
) -> None:
    """Print interactive review lines using caller-provided emitters."""
    if emit_subheader is not None:
        emit_subheader("Interactive Snapshot Review")
    else:
        emit("Interactive Snapshot Review")
    for line in build_interactive_review_lines(
        diff_report,
        limit=limit,
        width=width,
        dim_text=dim_text,
    ):
        emit(line)


def prompt_accept_snapshot(
    *,
    input_func: Callable[[str], str] | None = None,
    prompt: str = "Accept new behavior as baseline? [y/N]: ",
) -> bool:
    """Ask whether to accept candidate outputs as the new baseline."""
    input_func = input_func or input
    try:
        response = input_func(prompt).strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes"}


def copy_candidate_artifacts_to_baseline(
    run_dir_baseline: Path,
    run_dir_candidate: Path,
    *,
    artifact_names: Sequence[str] = SNAPSHOT_CANDIDATE_FILES,
) -> list[str]:
    """Copy canonical candidate artifacts into the baseline run directory."""
    copied: list[str] = []
    run_dir_baseline.mkdir(parents=True, exist_ok=True)
    for artifact_name in artifact_names:
        source = run_dir_candidate / artifact_name
        if source.exists() and source.is_file():
            shutil.copy2(source, run_dir_baseline / artifact_name)
            copied.append(artifact_name)
    return copied


__all__ = [
    "SNAPSHOT_CANDIDATE_FILES",
    "build_interactive_review_lines",
    "copy_candidate_artifacts_to_baseline",
    "print_interactive_review",
    "prompt_accept_snapshot",
]
