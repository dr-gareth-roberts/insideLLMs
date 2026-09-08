"""Report command: rebuild summary and HTML reports from records."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from insideLLMs.analysis.statistics import generate_summary_report
from insideLLMs.runtime._artifact_utils import _require_unsealed_run_directory
from insideLLMs.runtime.runner import _deterministic_base_time, _deterministic_run_times
from insideLLMs.schemas import OutputValidator, SchemaRegistry

from .._output import _cli_version_string, print_error, print_success, print_warning
from .._record_utils import _json_default, _parse_datetime, _read_jsonl_records
from .._report_builder import _build_basic_harness_report, _build_experiments_from_records


def _read_optional_object(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_diagnostic_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_health_metadata(value: object, label: str) -> dict[str, Any]:
    health = _validate_diagnostic_mapping(value, label)
    if "healthy" in health and not isinstance(health["healthy"], bool):
        raise ValueError(f"{label}.healthy must be a boolean")
    for field in ("record_count", "expected_count"):
        count = health.get(field)
        if count is not None and (
            not isinstance(count, int) or isinstance(count, bool) or count < 0
        ):
            raise ValueError(f"{label}.{field} must be a non-negative integer or null")
    reasons = health.get("reasons")
    if reasons is not None and (
        not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons)
    ):
        raise ValueError(f"{label}.reasons must be an array of strings")
    status_counts = health.get("status_counts")
    if status_counts is not None and (
        not isinstance(status_counts, dict)
        or not all(
            isinstance(status, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for status, count in status_counts.items()
        )
    ):
        raise ValueError(f"{label}.status_counts must map statuses to non-negative integers")
    return health


def _validate_diagnostic_list(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be an array of objects")
    return value


def _merge_authoritative_fields(
    existing: dict[str, Any], authoritative: dict[str, Any], label: str, warnings: list[str]
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in authoritative.items():
        if label == "abort" and key == "secondary_diagnostics":
            # Manifest order first, then summary-only evidence; exact duplicates
            # collapse so repeated rebuilds retain details without accumulating.
            combined = []
            for diagnostic in [*value, *existing.get(key, [])]:
                if diagnostic not in combined:
                    combined.append(diagnostic)
            merged[key] = combined
            continue
        if key in merged and merged[key] != value:
            warnings.append(f"Metadata conflict: manifest {label}.{key} overrides existing summary")
        merged[key] = value
    return merged


def _validated_authoritative_metadata(
    manifest: dict[str, Any] | None, old_payload: dict[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    old_summary: dict[str, Any] = {}
    if old_payload is not None:
        old_summary = _validate_diagnostic_mapping(
            old_payload.get("summary"), "summary.json summary"
        )

    diagnostics: dict[str, Any] = {}
    for key in ("run_completed", "abort", "health", "secondary_diagnostics", "metadata_warnings"):
        if key in old_summary:
            diagnostics[key] = old_summary[key]

    if (
        "run_completed" in diagnostics
        and diagnostics["run_completed"] is not None
        and not isinstance(diagnostics["run_completed"], bool)
    ):
        raise ValueError("summary run_completed must be true, false, or null")
    for key in ("abort", "health"):
        if key in diagnostics:
            if key == "health":
                _validate_health_metadata(diagnostics[key], "summary health")
            else:
                abort = _validate_diagnostic_mapping(diagnostics[key], "summary abort")
                if "secondary_diagnostics" in abort:
                    _validate_diagnostic_list(
                        abort["secondary_diagnostics"], "summary abort.secondary_diagnostics"
                    )
    if "secondary_diagnostics" in diagnostics:
        _validate_diagnostic_list(
            diagnostics["secondary_diagnostics"], "summary secondary_diagnostics"
        )

    old_warnings = diagnostics.get("metadata_warnings", [])
    if not isinstance(old_warnings, list) or not all(
        isinstance(warning, str) for warning in old_warnings
    ):
        raise ValueError("summary metadata_warnings must be an array of strings")
    warnings = list(old_warnings)

    if manifest is None:
        if "run_completed" not in diagnostics:
            diagnostics["run_completed"] = False if "abort" in diagnostics else None
        return diagnostics, warnings

    if "run_completed" in manifest and not isinstance(manifest["run_completed"], bool):
        raise ValueError("manifest run_completed must be a boolean")
    custom = manifest.get("custom", {})
    custom = _validate_diagnostic_mapping(custom, "manifest custom")
    manifest_supplies_completion = "run_completed" in manifest or "abort" in custom
    manifest_completion = manifest.get("run_completed")
    if "run_completed" not in manifest and "abort" in custom:
        manifest_completion = False

    old_completion = diagnostics.get("run_completed")
    if (
        manifest_supplies_completion
        and old_completion is not None
        and manifest_completion != old_completion
    ):
        warnings.append(
            "Metadata conflict: manifest completion overrides existing summary completion"
        )
    if manifest_supplies_completion:
        diagnostics["run_completed"] = manifest_completion
    elif "run_completed" not in diagnostics:
        diagnostics["run_completed"] = False if "abort" in diagnostics else None
    for key in ("abort", "health"):
        if key in custom:
            authoritative = (
                _validate_health_metadata(custom[key], "manifest custom.health")
                if key == "health"
                else _validate_diagnostic_mapping(custom[key], "manifest custom.abort")
            )
            if key == "abort" and "secondary_diagnostics" in authoritative:
                _validate_diagnostic_list(
                    authoritative["secondary_diagnostics"],
                    "manifest custom.abort.secondary_diagnostics",
                )
            existing = diagnostics.get(key, {})
            diagnostics[key] = _merge_authoritative_fields(existing, authoritative, key, warnings)
    if diagnostics["run_completed"] is False:
        health = diagnostics.get("health")
        if isinstance(health, dict) and health.get("healthy") is True:
            warnings.append(
                "Metadata conflict: incomplete manifest completion overrides summary health.healthy"
            )
            health["healthy"] = False
    if warnings:
        diagnostics["metadata_warnings"] = warnings
    return diagnostics, warnings


def _owned_stage_path(run_dir: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".insidellms-report-", suffix=suffix, dir=run_dir)
    os.close(descriptor)
    return Path(name)


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuild summary.json and report.html from records.jsonl."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1

    try:
        _require_unsealed_run_directory(run_dir)
    except ValueError as error:
        print_error(
            f"Cannot rebuild report in place: {error}. Use a fresh derivative/export directory."
        )
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

    run_ids = {record["run_id"] for record in records if record.get("run_id")}
    run_id = sorted(run_ids)[0] if run_ids else None
    if run_id and len(run_ids) > 1:
        print_warning(f"Multiple run_ids found; using {run_id}")

    generated_at = None
    if run_id:
        try:
            base_time = _deterministic_base_time(str(run_id))
            _, generated_at = _deterministic_run_times(base_time, len(records))
        except (ValueError, KeyError):
            generated_at = None

    if generated_at is None:
        generated_at = max(
            (dt for dt in (_parse_datetime(r.get("completed_at")) for r in records) if dt),
            default=None,
        )

    try:
        manifest = _read_optional_object(run_dir / "manifest.json", "manifest.json")
        old_payload = _read_optional_object(run_dir / "summary.json", "summary.json")
        diagnostic_metadata, metadata_warnings = _validated_authoritative_metadata(
            manifest, old_payload
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print_error(f"Could not validate report metadata: {error}")
        return 1

    summary = generate_summary_report(experiments, include_ci=True)
    summary.update(diagnostic_metadata)
    report_metadata: dict[str, Any] = {
        "from_records": True,
        "tool": "insidellms report",
        "tool_version": _cli_version_string(),
        "records_file": "records.jsonl",
    }
    summary_payload = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "summary": summary,
        "config": derived_config,
        "report_metadata": report_metadata,
    }

    report_title = args.report_title or "Behavioural Probe Report"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.html"
    summary_stage: Path | None = None
    report_stage: Path | None = None
    try:
        summary_stage = _owned_stage_path(run_dir, ".summary.json")
        report_stage = _owned_stage_path(run_dir, ".report.html")
        summary_text = json.dumps(summary_payload, indent=2, default=_json_default, sort_keys=True)
        summary_stage.write_text(summary_text, encoding="utf-8")
        schema_payload = {
            key: summary_payload[key]
            for key in ("schema_version", "generated_at", "summary", "config")
        }
        OutputValidator().validate(
            SchemaRegistry.HARNESS_SUMMARY,
            schema_payload,
            schema_version=schema_version,
            mode="strict",
        )
        try:
            from insideLLMs.analysis.visualization import create_interactive_html_report

            create_interactive_html_report(
                experiments,
                title=report_title,
                save_path=str(report_stage),
                generated_at=generated_at,
                run_health=diagnostic_metadata,
            )
            if not report_stage.read_text(encoding="utf-8"):
                raise ValueError("interactive report builder did not write HTML")
        except ImportError:
            report_html = _build_basic_harness_report(
                experiments,
                summary,
                report_title,
                generated_at=generated_at,
                run_health=diagnostic_metadata,
            )
            report_stage.write_text(report_html, encoding="utf-8")
        staged_html = report_stage.read_text(encoding="utf-8")
        if "<html" not in staged_html.lower() or "</html>" not in staged_html.lower():
            raise ValueError("generated report is not a complete HTML document")
        os.replace(summary_stage, summary_path)
        os.replace(report_stage, report_path)
    except Exception as error:
        print_error(f"Could not rebuild report: {error}")
        return 1
    finally:
        if summary_stage is not None:
            summary_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)

    print_success(f"Summary written to: {summary_path}")
    print_success(f"Report written to: {report_path}")
    if summary.get("run_completed") is not True:
        state = "incomplete" if summary.get("run_completed") is False else "completion unknown"
        print_warning(f"Report generated successfully, but run status is {state}")
    for warning in metadata_warnings:
        print_warning(warning)
    return 0
