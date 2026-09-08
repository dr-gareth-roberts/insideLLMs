"""Validate async histories read-only, then recover verified unattempted tails."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def replace_resume_records(path: Path, retained: bytes) -> None:
    """Atomically retain exact bytes without platform newline translation."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".records-resume-", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(retained)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_resume_history(path: Path) -> tuple[list[dict[str, Any]], list[bytes], bytes]:
    """Keep exact record bytes and tolerate only an invalid unterminated last line."""
    original = path.read_bytes() if path.exists() else b""
    records = []
    lines = []
    pending = b""
    raw_lines = original.splitlines(keepends=True)
    for index, line in enumerate(raw_lines):
        pending += line
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index == len(raw_lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Invalid JSONL record on line {index + 1} in {path}") from exc
        if not isinstance(record, dict):
            raise ValueError("Every resume record must be an object")
        records.append(record)
        lines.append(pending if pending.endswith(b"\n") else pending + b"\n")
        pending = b""
    else:
        if pending and lines:
            lines[-1] += pending
    return records, lines, original


def attempted_prefix_length(records: list[dict[str, Any]]) -> int:
    """Only the runner's explicit, evidence-free fail-fast suffix is recoverable."""
    first_skipped = next(
        (index for index, record in enumerate(records) if record.get("status") == "skipped"),
        len(records),
    )
    if first_skipped == len(records):
        return first_skipped
    if not any(
        record.get("status") in {"error", "timeout"}
        and (record.get("error") is not None or record.get("error_type") is not None)
        for record in records[:first_skipped]
    ):
        raise ValueError("Skipped resume suffix has no preceding attempted error")
    for record in records[first_skipped:]:
        custom = record.get("custom")
        if (
            record.get("status") != "skipped"
            or not isinstance(custom, dict)
            or custom.get("execution") != {"attempted": False, "reason": "stop_on_error"}
            or custom["execution"]["attempted"] is not False
            or custom.get("output_fingerprint") is not None
            or set(custom) - {"record_index", "replicate_key", "output_fingerprint", "execution"}
            or any(
                record.get(field) is not None
                for field in (
                    "output",
                    "output_text",
                    "error",
                    "error_type",
                    "latency_ms",
                    "primary_metric",
                )
            )
            or any(record.get(field) not in (None, {}) for field in ("usage", "scores", "metadata"))
            or any(
                field
                not in {
                    "schema_version",
                    "run_id",
                    "started_at",
                    "completed_at",
                    "model",
                    "probe",
                    "dataset",
                    "example_id",
                    "input",
                    "messages",
                    "messages_hash",
                    "messages_storage",
                    "output",
                    "output_text",
                    "scores",
                    "primary_metric",
                    "usage",
                    "latency_ms",
                    "status",
                    "error",
                    "error_type",
                    "custom",
                    "metadata",
                }
                for field in record
            )
        ):
            raise ValueError("Ambiguous skipped history; cannot resume safely")
    return first_skipped
