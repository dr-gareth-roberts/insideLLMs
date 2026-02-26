"""Report command: rebuild summary and HTML reports from records."""

import argparse
import json
from pathlib import Path

from insideLLMs.analysis.statistics import generate_summary_report

from .._output import print_error, print_success, print_warning
from .._record_utils import _json_default, _parse_datetime, _read_jsonl_records
from .._report_builder import _build_basic_harness_report, _build_experiments_from_records


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuild summary.json and report.html from records.jsonl."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1

    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        print_error(f"records.jsonl not found in: {run_dir}")
        return 1

    try:
        records = _read_jsonl_records(records_path)
    except Exception as e:
        print_error(f"Could not read records.jsonl: {e}")
        return 1

    if not records:
        print_error("No records found in records.jsonl")
        return 1

    experiments, derived_config, schema_version = _build_experiments_from_records(records)
    if not experiments:
        print_error("No experiments could be reconstructed from records")
        return 1

    run_ids = {record.get("run_id") for record in records if record.get("run_id")}
    run_id = sorted(run_ids)[0] if run_ids else None
    if run_id and len(run_ids) > 1:
        print_warning(f"Multiple run_ids found; using {run_id}")

    generated_at = None
    if run_id:
        try:
            from insideLLMs.runtime.runner import _deterministic_base_time, _deterministic_run_times

            base_time = _deterministic_base_time(str(run_id))
            _, generated_at = _deterministic_run_times(base_time, len(records))
        except (ValueError, KeyError):
            generated_at = None

    if generated_at is None:
        generated_at = max(
            (dt for dt in (_parse_datetime(r.get("completed_at")) for r in records) if dt),
            default=None,
        )

    summary = generate_summary_report(experiments, include_ci=True)
    summary_payload = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "summary": summary,
        "config": derived_config,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, default=_json_default, sort_keys=True)
    print_success(f"Summary written to: {summary_path}")

    report_title = args.report_title or "Behavioural Probe Report"
    report_path = run_dir / "report.html"
    try:
        from insideLLMs.visualization import create_interactive_html_report

        create_interactive_html_report(
            experiments,
            title=report_title,
            save_path=str(report_path),
            generated_at=generated_at,
        )
    except ImportError:
        report_html = _build_basic_harness_report(
            experiments,
            summary,
            report_title,
            generated_at=generated_at,
        )
        with open(report_path, "w") as f:
            f.write(report_html)

    print_success(f"Report written to: {report_path}")
    return 0
