"""Diff command: compare two run directories for behavioural regressions."""

import argparse
import json
from pathlib import Path
from typing import Any

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
from .._record_utils import (
    _json_default,
    _metric_mismatch_context,
    _metric_mismatch_details,
    _metric_mismatch_reason,
    _output_fingerprint,
    _output_summary,
    _output_text,
    _primary_score,
    _read_jsonl_records,
    _record_key,
    _record_label,
    _status_string,
    _trace_fingerprint,
    _trace_violation_count,
    _trace_violations,
)


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
    if output_path and output_format != "json":
        print_warning("--output is only used with --format json")

    ignore_keys: set[str] = set()
    for entry in args.output_fingerprint_ignore or []:
        for item in str(entry).split(","):
            item = item.strip()
            if item:
                ignore_keys.add(item.lower())

    ignore_keys_set = ignore_keys if ignore_keys else None

    def build_index(
        records: list[dict[str, Any]],
    ) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        duplicates = 0
        for record in records:
            key = _record_key(record)
            if key in index:
                duplicates += 1
                continue
            index[key] = record
        return index, duplicates

    def record_identity(record: dict[str, Any]) -> dict[str, Any]:
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        label = _record_label(record)
        key = _record_key(record)
        return {
            "record_key": {"model_id": key[0], "probe_id": key[1], "item_id": key[2]},
            "model_id": key[0],
            "probe_id": key[1],
            "example_id": label[2],
            "replicate_key": custom.get("replicate_key"),
            "label": {"model": label[0], "probe": label[1], "example": label[2]},
        }

    def record_summary(record: dict[str, Any]) -> dict[str, Any]:
        scores = record.get("scores") if isinstance(record.get("scores"), dict) else None
        _metric_name, metric_value = _primary_score(record)
        return {
            "status": _status_string(record),
            "primary_metric": record.get("primary_metric"),
            "primary_score": metric_value,
            "scores_keys": sorted(scores.keys()) if isinstance(scores, dict) else None,
            "output": _output_summary(record, ignore_keys_set),
        }

    index_a, dup_a = build_index(records_a)
    index_b, dup_b = build_index(records_b)

    if dup_a:
        print_warning(f"Baseline has {dup_a} duplicate key(s); first occurrence used")
    if dup_b:
        print_warning(f"Comparison has {dup_b} duplicate key(s); first occurrence used")

    all_keys = set(index_a) | set(index_b)
    regressions: list[tuple[str, str, str, str]] = []
    improvements: list[tuple[str, str, str, str]] = []
    changes: list[tuple[str, str, str, str]] = []
    only_a: list[tuple[str, str, str]] = []
    only_b: list[tuple[str, str, str]] = []

    regressions_json: list[dict[str, Any]] = []
    improvements_json: list[dict[str, Any]] = []
    changes_json: list[dict[str, Any]] = []
    only_a_json: list[dict[str, Any]] = []
    only_b_json: list[dict[str, Any]] = []
    # Trace tracking
    trace_drifts: list[tuple[str, str, str, str]] = []
    trace_drifts_json: list[dict[str, Any]] = []
    trace_violation_increases: list[tuple[str, str, str, str]] = []
    trace_violation_increases_json: list[dict[str, Any]] = []

    for key in sorted(all_keys):
        record_a = index_a.get(key)
        record_b = index_b.get(key)

        if record_a is None:
            only_b.append(_record_label(record_b))  # type: ignore[arg-type]  # record_b is not None here (checked via all_keys)
            only_b_json.append(record_identity(record_b))  # type: ignore[arg-type]  # record_b is not None here
            continue
        if record_b is None:
            only_a.append(_record_label(record_a))
            only_a_json.append(record_identity(record_a))
            continue

        label = _record_label(record_a)
        status_a = _status_string(record_a)
        status_b = _status_string(record_b)
        score_name_a, score_a = _primary_score(record_a)
        score_name_b, score_b = _primary_score(record_b)
        identity = record_identity(record_a)
        summary_a = record_summary(record_a)
        summary_b = record_summary(record_b)

        if status_a == "success" and status_b != "success":
            regressions.append((*label, f"status {status_a} -> {status_b}"))
            regressions_json.append(
                {
                    **identity,
                    "kind": "status_regression",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )
            continue
        if status_a != "success" and status_b == "success":
            improvements.append((*label, f"status {status_a} -> {status_b}"))
            improvements_json.append(
                {
                    **identity,
                    "kind": "status_improvement",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )
            continue

        metrics_compared = False
        if score_a is not None and score_b is not None and score_name_a == score_name_b:
            delta = score_b - score_a
            metrics_compared = True
            if delta < 0:
                regressions.append(
                    (*label, f"{score_name_a} {score_a:.4f} -> {score_b:.4f} (delta {delta:.4f})")
                )
                regressions_json.append(
                    {
                        **identity,
                        "kind": "metric_regression",
                        "metric": score_name_a,
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "delta": delta,
                    }
                )
                continue
            if delta > 0:
                improvements.append(
                    (*label, f"{score_name_a} {score_a:.4f} -> {score_b:.4f} (delta +{delta:.4f})")
                )
                improvements_json.append(
                    {
                        **identity,
                        "kind": "metric_improvement",
                        "metric": score_name_a,
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "delta": delta,
                    }
                )

        if not metrics_compared and (score_a is not None or score_b is not None):
            reason = _metric_mismatch_reason(record_a, record_b) or "type_mismatch"
            detail = _metric_mismatch_details(record_a, record_b)
            changes.append((*label, f"metrics not comparable:{reason}; {detail}"))
            changes_json.append(
                {
                    **identity,
                    "kind": "metrics_not_comparable",
                    "reason": reason,
                    "baseline": summary_a,
                    "candidate": summary_b,
                    "details": _metric_mismatch_context(record_a, record_b),
                }
            )

        if metrics_compared and status_a == "success" and status_b == "success":
            scores_a = record_a.get("scores") if isinstance(record_a.get("scores"), dict) else None
            scores_b = record_b.get("scores") if isinstance(record_b.get("scores"), dict) else None
            if isinstance(scores_a, dict) and isinstance(scores_b, dict):
                keys_a = sorted(scores_a.keys())
                keys_b = sorted(scores_b.keys())
                missing_in_b = sorted(set(keys_a) - set(keys_b))
                missing_in_a = sorted(set(keys_b) - set(keys_a))
                if missing_in_a or missing_in_b:
                    changes.append(
                        (
                            *label,
                            "metric_key_missing: "
                            f"baseline_missing={missing_in_a}, candidate_missing={missing_in_b}",
                        )
                    )
                    changes_json.append(
                        {
                            **identity,
                            "kind": "metric_key_missing",
                            "baseline_missing": missing_in_a,
                            "candidate_missing": missing_in_b,
                            "baseline": summary_a,
                            "candidate": summary_b,
                        }
                    )

        output_a = _output_text(record_a)
        output_b = _output_text(record_b)
        if output_a is not None or output_b is not None:
            if output_a != output_b:
                changes.append((*label, "output changed"))
                changes_json.append(
                    {
                        **identity,
                        "kind": "output_changed",
                        "baseline": summary_a,
                        "candidate": summary_b,
                    }
                )
        else:
            fingerprint_a = _output_fingerprint(record_a, ignore_keys=ignore_keys_set)
            fingerprint_b = _output_fingerprint(record_b, ignore_keys=ignore_keys_set)
            if fingerprint_a != fingerprint_b:
                if fingerprint_a and fingerprint_b:
                    changes.append(
                        (*label, f"output fingerprint {fingerprint_a} -> {fingerprint_b}")
                    )
                else:
                    changes.append((*label, "output changed (structured)"))
                changes_json.append(
                    {
                        **identity,
                        "kind": "output_changed",
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "baseline_fingerprint": fingerprint_a,
                        "candidate_fingerprint": fingerprint_b,
                    }
                )
        if status_a != status_b:
            changes.append((*label, f"status {status_a} -> {status_b}"))
            changes_json.append(
                {
                    **identity,
                    "kind": "status_changed",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )

        # Trace drift detection
        trace_fp_a = _trace_fingerprint(record_a)
        trace_fp_b = _trace_fingerprint(record_b)
        if trace_fp_a and trace_fp_b and trace_fp_a != trace_fp_b:
            trace_drifts.append((*label, f"trace {trace_fp_a[:12]} -> {trace_fp_b[:12]}"))
            trace_drifts_json.append(
                {
                    **identity,
                    "kind": "trace_drift",
                    "baseline_trace_fingerprint": trace_fp_a,
                    "candidate_trace_fingerprint": trace_fp_b,
                }
            )

        # Trace violation comparison
        violations_a = _trace_violation_count(record_a)
        violations_b = _trace_violation_count(record_b)
        if violations_b > violations_a:
            trace_violation_increases.append(
                (*label, f"violations {violations_a} -> {violations_b}")
            )
            trace_violation_increases_json.append(
                {
                    **identity,
                    "kind": "trace_violations_increased",
                    "baseline_violations": violations_a,
                    "candidate_violations": violations_b,
                    "candidate_violation_details": _trace_violations(record_b),
                }
            )

    diff_report = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "baseline": str(run_dir_a),
        "candidate": str(run_dir_b),
        "run_ids": {
            "baseline": sorted(
                {record.get("run_id") for record in records_a if record.get("run_id")}
            ),
            "candidate": sorted(
                {record.get("run_id") for record in records_b if record.get("run_id")}
            ),
        },
        "counts": {
            "common": len(all_keys) - len(only_a) - len(only_b),
            "only_baseline": len(only_a),
            "only_candidate": len(only_b),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "other_changes": len(changes),
            "trace_drifts": len(trace_drifts),
            "trace_violation_increases": len(trace_violation_increases),
        },
        "duplicates": {"baseline": dup_a, "candidate": dup_b},
        "regressions": regressions_json,
        "improvements": improvements_json,
        "changes": changes_json,
        "only_baseline": only_a_json,
        "only_candidate": only_b_json,
        "trace_drifts": trace_drifts_json,
        "trace_violation_increases": trace_violation_increases_json,
    }

    if output_format == "json":
        payload = json.dumps(diff_report, indent=2, default=_json_default, sort_keys=True)
        if output_path:
            Path(output_path).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        if args.fail_on_regressions and regressions:
            return 2
        if args.fail_on_changes and (regressions or changes or only_a or only_b):
            return 2
        if args.fail_on_trace_violations and trace_violation_increases:
            return 3
        if args.fail_on_trace_drift and trace_drifts:
            return 4
        return 0

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

    if args.fail_on_regressions and regressions:
        return 2
    if args.fail_on_changes and (regressions or changes or only_a or only_b):
        return 2
    if args.fail_on_trace_violations and trace_violation_increases:
        return 3
    if args.fail_on_trace_drift and trace_drifts:
        return 4
    return 0
