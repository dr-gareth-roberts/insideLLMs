"""Record processing utilities for the insideLLMs CLI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    serialize_value as _serialize_value,
)
from insideLLMs._serialization import (
    stable_json_dumps as _stable_json_dumps,
)
from insideLLMs.runtime import diffing as _runtime_diffing
from insideLLMs.types import (
    ProbeCategory,
    ResultStatus,
)

_record_key = _runtime_diffing._record_key
_record_label = _runtime_diffing._record_label
_status_string = _runtime_diffing._status_string
_output_text = _runtime_diffing._output_text
_strip_volatile_keys = _runtime_diffing._strip_volatile_keys
_output_summary = _runtime_diffing._output_summary
_output_fingerprint = _runtime_diffing._output_fingerprint
_trace_fingerprint = _runtime_diffing._trace_fingerprint
_trace_violations = _runtime_diffing._trace_violations
_trace_violation_count = _runtime_diffing._trace_violation_count
_primary_score = _runtime_diffing._primary_score
_metric_mismatch_reason = _runtime_diffing._metric_mismatch_reason
_metric_mismatch_context = _runtime_diffing._metric_mismatch_context
_metric_mismatch_details = _runtime_diffing._metric_mismatch_details


def _json_default(obj: Any) -> Any:
    """JSON default handler for CLI writes."""
    return _serialize_value(obj)


def _write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    strict_serialization: bool = True,
) -> None:
    """Write a list of dictionaries to a JSON Lines file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            try:
                line = _stable_json_dumps(record, strict=strict_serialization)
            except StrictSerializationError as exc:
                raise ValueError(
                    "strict_serialization requires JSON-stable values in records.jsonl emission."
                ) from exc
            f.write(line + "\n")


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a value into a datetime object."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _status_from_record(value: Any) -> ResultStatus:
    """Convert a value to a ResultStatus enum."""
    if isinstance(value, ResultStatus):
        return value
    try:
        return ResultStatus(str(value))
    except (ValueError, TypeError):
        return ResultStatus.ERROR


def _probe_category_from_value(value: Any) -> ProbeCategory:
    """Convert a value to a ProbeCategory enum."""
    if isinstance(value, ProbeCategory):
        return value
    try:
        return ProbeCategory(str(value))
    except Exception as _:
        return ProbeCategory.CUSTOM


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON Lines file."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if isinstance(obj, dict):
                records.append(obj)
    return records
