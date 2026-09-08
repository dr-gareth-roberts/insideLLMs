"""Run health checks independent of behavioural comparison and scoring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def describe_run_abort(error: Exception) -> dict[str, Any]:
    """Describe an early stop without including its prompt or partial payload."""
    original = getattr(error, "original_error", None) or error
    reason = getattr(original, "abort_reason", None)
    if isinstance(reason, dict):
        return dict(reason)
    return {
        "code": "execution_error",
        "error_type": type(original).__name__,
        "message": str(original),
    }


def assess_run_health(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    run_completed: bool = True,
) -> dict[str, Any]:
    """Require a complete, nonempty collection of successful executions.

    Execution success is distinct from a probe's behavioural score. A completed
    run can be unhealthy, and an incorrect model answer can execute successfully.
    """
    counts = Counter(str(record.get("status", "unknown")) for record in records)
    reasons: list[str] = []
    if not run_completed:
        reasons.append("Run did not complete")
    if not records:
        reasons.append("Run contains no records")
    if counts.get("success", 0) == 0:
        reasons.append("Run contains no successful executions")
    for status, count in sorted(counts.items()):
        if status != "success":
            reasons.append(f"{count} record(s) have status {status}")
    if expected_count is not None and len(records) != expected_count:
        reasons.append(f"Expected {expected_count} records, found {len(records)}")
    return {
        "healthy": not reasons,
        "reasons": reasons,
        "record_count": len(records),
        "expected_count": expected_count,
        "status_counts": dict(sorted(counts.items())),
    }


def check_run_directory_health(run_dir: str | Path) -> dict[str, Any]:
    """Check persisted evidence, including manifest counts, before CI diffing.

    Malformed or missing artefacts raise an error; a well-formed unhealthy run
    returns the same health payload as ``assess_run_health``. Older manifests
    without expected-count metadata are checked against their declared count.
    A legacy finalization timestamp implies completion only when no explicit
    abort or unhealthy declaration contradicts it.
    """
    directory = Path(run_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be an object")
    records = []
    for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Every result record must be an object")
            records.append(record)
    custom = manifest.get("custom") or {}
    declared_health = custom.get("health") or {}
    health = assess_run_health(
        records,
        expected_count=declared_health.get("expected_count", manifest.get("record_count")),
        run_completed=(
            "abort" not in custom
            and (
                manifest.get("run_completed") is True
                or (
                    manifest.get("schema_version") == "1.0.0"
                    and "run_completed" not in manifest
                    and declared_health.get("healthy") is not False
                    and bool(manifest.get("completed_at"))
                )
            )
        ),
    )
    if declared_health.get("healthy") is False:
        health["reasons"].append("Manifest declares unhealthy run")
    for field, actual in (
        ("record_count", len(records)),
        ("success_count", health["status_counts"].get("success", 0)),
        ("error_count", health["status_counts"].get("error", 0)),
    ):
        if manifest.get(field) != actual:
            health["reasons"].append(f"Manifest {field} does not match records")
    health["healthy"] = not health["reasons"]
    return health
