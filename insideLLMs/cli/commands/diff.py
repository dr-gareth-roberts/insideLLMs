"""Diff command: compare two run directories for behavioural regressions."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from insideLLMs.runtime.diffing import (
    DiffGatePolicy,
    build_diff_computation,
    compute_diff_exit_code,
    judge_diff_report,
)
from insideLLMs.runtime.diffing_interactive import (
    copy_candidate_artifacts_to_baseline,
    print_interactive_review,
    prompt_accept_snapshot,
)
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

from .._output import (
    Colors,
    colorize,
    print_error,
    print_header,
    print_key_value,
    print_subheader,
    print_warning,
)
from .._record_utils import _json_default, _read_jsonl_records


def _print_judge_review(judge_report: dict[str, Any], *, limit: int) -> None:
    """Render judge verdict details for terminal output."""
    summary = judge_report.get("summary") if isinstance(judge_report.get("summary"), dict) else {}
    verdicts = (
        judge_report.get("verdicts") if isinstance(judge_report.get("verdicts"), list) else []
    )

    print_subheader("Judge Verdict")
    print_key_value("Policy", judge_report.get("policy", "strict"))
    print_key_value("Breaking", "yes" if judge_report.get("breaking") else "no")
    print_key_value("Breaking items", summary.get("breaking", 0))
    print_key_value("Review items", summary.get("review", 0))
    print_key_value("Acceptable items", summary.get("acceptable", 0))

    if not verdicts:
        print("  No judged items.")
        return

    for item in verdicts[:limit]:
        if not isinstance(item, dict):
            continue
        label = item.get("label") if isinstance(item.get("label"), dict) else {}
        model_label = label.get("model", item.get("model_id", "unknown"))
        probe_label = label.get("probe", item.get("probe_id", "unknown"))
        example_label = label.get("example", item.get("example_id", "unknown"))
        decision = item.get("decision", "review")
        reason = item.get("reason", "")
        detail = item.get("detail") or item.get("kind") or "change"
        print(
            f"  [{decision}] {model_label} | {probe_label} | example {example_label}: "
            f"{detail} ({reason})"
        )
    if len(verdicts) > limit:
        print(colorize(f"  ... and {len(verdicts) - limit} more", Colors.DIM))


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two run directories and report behavioural regressions."""
    run_dir_a = Path(args.run_dir_a)
    run_dir_b = Path(args.run_dir_b)

    if not run_dir_a.exists() or not run_dir_a.is_dir():
        print_error(f"Run directory not found: {run_dir_a}")
        return 1
    if not run_dir_b.exists() or not run_dir_b.is_dir():
        print_error(f"Run directory not found: {run_dir_b}")
        return 1

    records_path_a = run_dir_a / "records.jsonl"
    records_path_b = run_dir_b / "records.jsonl"

    if not records_path_a.exists():
        print_error(f"records.jsonl not found in: {run_dir_a}")
        return 1
    if not records_path_b.exists():
        print_error(f"records.jsonl not found in: {run_dir_b}")
        return 1

    try:
        records_a = _read_jsonl_records(records_path_a)
        records_b = _read_jsonl_records(records_path_b)
    except Exception as e:
        print_error(f"Could not read records.jsonl: {e}")
        return 1

    if not records_a or not records_b:
        print_error("Both run directories must contain records to compare")
        return 1

    output_format = getattr(args, "format", "text")
    output_path = getattr(args, "output", None)
    interactive = bool(getattr(args, "interactive", False))
    if output_path and output_format != "json":
        print_warning("--output is only used with --format json")
    if interactive and output_format == "json":
        print_error("--interactive requires text output; remove --format json")
        return 1
    if interactive and not sys.stdin.isatty():
        print_warning(
            "--interactive is running in a non-interactive shell; snapshot prompt may default"
        )
    gate_policy = DiffGatePolicy(
        fail_on_regressions=bool(args.fail_on_regressions),
        fail_on_changes=bool(args.fail_on_changes),
        fail_on_trace_violations=bool(args.fail_on_trace_violations),
        fail_on_trace_drift=bool(args.fail_on_trace_drift),
        fail_on_trajectory_drift=bool(getattr(args, "fail_on_trajectory_drift", False)),
    )
    try:
        computation = build_diff_computation(
            records_baseline=records_a,
            records_candidate=records_b,
            baseline_label=str(run_dir_a),
            candidate_label=str(run_dir_b),
            output_fingerprint_ignore=args.output_fingerprint_ignore,
            validate_output=bool(getattr(args, "validate_output", False)),
            schema_version=str(getattr(args, "schema_version", DEFAULT_SCHEMA_VERSION)),
            validation_mode=str(getattr(args, "validation_mode", "strict")),
        )
    except Exception as e:
        print_error(f"Could not compute diff report: {e}")
        return 1
    diff_report = computation.diff_report
    regressions = computation.regressions
    improvements = computation.improvements
    changes = computation.changes
    only_a = computation.only_baseline
    only_b = computation.only_candidate
    trace_drifts = computation.trace_drifts
    trace_violation_increases = computation.trace_violation_increases
    trajectory_drifts = computation.trajectory_drifts
    judge_report: dict[str, Any] | None = None

    if bool(getattr(args, "judge", False)):
        judge_computation = judge_diff_report(
            diff_report,
            policy=str(getattr(args, "judge_policy", "strict")),
            limit=int(getattr(args, "judge_limit", args.limit)),
        )
        judge_report = judge_computation.judge_report

    if computation.baseline_duplicates:
        print_warning(
            f"Baseline has {computation.baseline_duplicates} duplicate key(s); first occurrence used"
        )
    if computation.candidate_duplicates:
        print_warning(
            f"Comparison has {computation.candidate_duplicates} duplicate key(s); first occurrence used"
        )

    if output_format == "json":
        payload_obj = dict(diff_report)
        if judge_report is not None:
            payload_obj["judge"] = judge_report
        payload = json.dumps(payload_obj, indent=2, default=_json_default, sort_keys=True)
        if output_path:
            Path(output_path).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return compute_diff_exit_code(computation, gate_policy)

    has_differences = computation.has_differences

    print_header("Behavioural Diff")
    print_key_value("Baseline", run_dir_a)
    print_key_value("Comparison", run_dir_b)
    print_key_value("Common keys", diff_report["counts"]["common"])
    print_key_value("Only in baseline", diff_report["counts"]["only_baseline"])
    print_key_value("Only in comparison", diff_report["counts"]["only_candidate"])
    print_key_value("Regressions", diff_report["counts"]["regressions"])
    print_key_value("Improvements", diff_report["counts"]["improvements"])
    print_key_value("Other changes", diff_report["counts"]["other_changes"])
    if trace_drifts:
        print_key_value("Trace drifts", diff_report["counts"]["trace_drifts"])
    if trace_violation_increases:
        print_key_value(
            "Trace violation increases", diff_report["counts"]["trace_violation_increases"]
        )
    if trajectory_drifts:
        print_key_value("Trajectory drifts", diff_report["counts"]["trajectory_drifts"])

    def print_section(title: str, items: list[tuple[str, str, str, str]]) -> None:
        if not items:
            return
        print_subheader(title)
        for model_label, probe_label, example_id, detail in items[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}: {detail}")
        if len(items) > args.limit:
            print(colorize(f"  ... and {len(items) - args.limit} more", Colors.DIM))

    print_section("Regressions", regressions)
    print_section("Improvements", improvements)
    print_section("Other Changes", changes)
    print_section("Trace Drifts", trace_drifts)
    print_section("Trace Violation Increases", trace_violation_increases)
    print_section("Trajectory Drifts", trajectory_drifts)

    if only_a:
        print_subheader("Missing in Comparison")
        for model_label, probe_label, example_id in only_a[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}")
        if len(only_a) > args.limit:
            print(colorize(f"  ... and {len(only_a) - args.limit} more", Colors.DIM))

    if only_b:
        print_subheader("New in Comparison")
        for model_label, probe_label, example_id in only_b[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}")
        if len(only_b) > args.limit:
            print(colorize(f"  ... and {len(only_b) - args.limit} more", Colors.DIM))

    if judge_report is not None:
        _print_judge_review(judge_report, limit=args.limit)

    if interactive:
        print_interactive_review(
            diff_report,
            limit=args.limit,
            emit_subheader=print_subheader,
            dim_text=lambda text: colorize(text, Colors.DIM),
        )
        if has_differences:
            if prompt_accept_snapshot():
                copied = copy_candidate_artifacts_to_baseline(run_dir_a, run_dir_b)
                if copied:
                    print_subheader("Baseline Updated")
                    print(f"  Copied {len(copied)} artifact(s) from candidate to baseline:")
                    for artifact_name in copied:
                        print(f"  - {artifact_name}")
                else:
                    print_warning("No candidate artifacts were copied to baseline")
                return 0
            print_warning("Baseline unchanged (interactive review declined).")
        else:
            print("  Baseline already matches candidate.")

    return compute_diff_exit_code(computation, gate_policy)
