"""Record processing utilities for the insideLLMs CLI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    fingerprint_value as _fingerprint_value,
)
from insideLLMs._serialization import (
    serialize_value as _serialize_value,
)
from insideLLMs._serialization import (
    stable_json_dumps as _stable_json_dumps,
)
from insideLLMs.types import (
    ProbeCategory,
    ResultStatus,
)


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
    except Exception:
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


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Extract a unique key tuple from a result record."""
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    harness = custom.get("harness") if isinstance(custom.get("harness"), dict) else {}
    model_spec = record.get("model") if isinstance(record.get("model"), dict) else {}
    probe_spec = record.get("probe") if isinstance(record.get("probe"), dict) else {}

    model_id = (
        harness.get("model_id")
        or model_spec.get("model_id")
        or harness.get("model_name")
        or "model"
    )
    probe_id = (
        harness.get("probe_type")
        or probe_spec.get("probe_id")
        or harness.get("probe_name")
        or "probe"
    )
    replicate_key = custom.get("replicate_key")
    example_id = record.get("example_id") or harness.get("example_index")
    stable_id = record.get("messages_hash") or _fingerprint_value(record.get("input"))
    chosen_id = replicate_key or stable_id or example_id or "0"
    return (str(model_id), str(probe_id), str(chosen_id))


def _record_label(record: dict[str, Any]) -> tuple[str, str, str]:
    """Extract human-readable labels from a result record."""
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    harness = custom.get("harness") if isinstance(custom.get("harness"), dict) else {}
    model_spec = record.get("model") if isinstance(record.get("model"), dict) else {}
    probe_spec = record.get("probe") if isinstance(record.get("probe"), dict) else {}

    model_name = harness.get("model_name") or model_spec.get("model_id") or "model"
    model_id = harness.get("model_id") or model_spec.get("model_id")
    if model_id and model_id != model_name:
        model_label = f"{model_name} ({model_id})"
    else:
        model_label = str(model_name)

    probe_name = harness.get("probe_name") or probe_spec.get("probe_id") or "probe"
    example_id = record.get("example_id") or harness.get("example_index") or "0"
    return (str(model_label), str(probe_name), str(example_id))


def _status_string(record: dict[str, Any]) -> str:
    status = record.get("status")
    return str(status) if status is not None else "unknown"


def _output_text(record: dict[str, Any]) -> Optional[str]:
    output_text = record.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = record.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        value = output.get("output_text") or output.get("text")
        if isinstance(value, str):
            return value
    return None


def _strip_volatile_keys(value: Any, ignore_keys: set[str]) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key).lower()
            if key_str in ignore_keys:
                continue
            cleaned[key] = _strip_volatile_keys(item, ignore_keys)
        return cleaned
    if isinstance(value, list):
        return [_strip_volatile_keys(item, ignore_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_volatile_keys(item, ignore_keys) for item in value)
    return value


def _output_summary(
    record: dict[str, Any], ignore_keys: Optional[set[str]]
) -> Optional[dict[str, Any]]:
    from ._output import _trim_text

    output_text = _output_text(record)
    if isinstance(output_text, str):
        return {"type": "text", "preview": _trim_text(output_text), "length": len(output_text)}
    output = record.get("output")
    if output is None:
        return None
    return {
        "type": "structured",
        "fingerprint": _output_fingerprint(record, ignore_keys=ignore_keys),
    }


def _output_fingerprint(
    record: dict[str, Any], ignore_keys: Optional[set[str]] = None
) -> Optional[str]:
    output = record.get("output")
    if ignore_keys:
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        override = custom.get("output_fingerprint")
        if output is None and isinstance(override, str):
            return override
        if output is None:
            return None
        sanitized = _strip_volatile_keys(output, ignore_keys)
        return _fingerprint_value(sanitized)

    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    override = custom.get("output_fingerprint")
    if isinstance(override, str):
        return override
    if output is None:
        return None
    return _fingerprint_value(output)


def _trace_fingerprint(record: dict[str, Any]) -> Optional[str]:
    """Extract trace fingerprint from ResultRecord.custom."""
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    fp = custom.get("trace_fingerprint")
    if isinstance(fp, str):
        return fp

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    fingerprint = trace.get("fingerprint") if isinstance(trace.get("fingerprint"), dict) else {}
    value = fingerprint.get("value")
    if not isinstance(value, str) or not value:
        return None
    # If the bundle stores raw 64-hex without prefix, normalize to the legacy "sha256:<hex>" form.
    if ":" not in value and len(value) == 64:
        return f"sha256:{value}"
    return value


def _trace_violations(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract trace violations from ResultRecord.custom."""
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    violations = custom.get("trace_violations")
    if isinstance(violations, list):
        return violations

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    violations = trace.get("violations")
    return violations if isinstance(violations, list) else []


def _trace_violation_count(record: dict[str, Any]) -> int:
    """Count trace violations in a record."""
    return len(_trace_violations(record))


def _primary_score(record: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    metric = record.get("primary_metric")
    if metric and metric in scores and isinstance(scores[metric], (int, float)):
        return str(metric), float(scores[metric])
    if "score" in scores and isinstance(scores["score"], (int, float)):
        return "score", float(scores["score"])
    return None, None


def _metric_mismatch_reason(record_a: dict[str, Any], record_b: dict[str, Any]) -> Optional[str]:
    scores_a = record_a.get("scores")
    scores_b = record_b.get("scores")
    if not isinstance(scores_a, dict) or not isinstance(scores_b, dict):
        return "type_mismatch"

    keys_a = sorted(scores_a.keys())
    keys_b = sorted(scores_b.keys())
    if not keys_a or not keys_b:
        return "missing_scores"

    primary_a = record_a.get("primary_metric")
    primary_b = record_b.get("primary_metric")
    if primary_a and primary_b and primary_a != primary_b:
        return "primary_metric_mismatch"
    if not primary_a or not primary_b:
        return "missing_primary_metric"

    if primary_a not in scores_a or primary_b not in scores_b:
        return "metric_key_missing"

    value_a = scores_a.get(primary_a)
    value_b = scores_b.get(primary_b)
    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return "non_numeric_metric"

    return "type_mismatch"


def _metric_mismatch_context(record_a: dict[str, Any], record_b: dict[str, Any]) -> dict[str, Any]:
    scores_a = record_a.get("scores")
    scores_b = record_b.get("scores")
    primary_a = record_a.get("primary_metric")
    primary_b = record_b.get("primary_metric")

    keys_a = sorted(scores_a.keys()) if isinstance(scores_a, dict) else None
    keys_b = sorted(scores_b.keys()) if isinstance(scores_b, dict) else None

    context = {
        "baseline": {
            "primary_metric": primary_a,
            "scores_keys": keys_a,
            "metric_value": scores_a.get(primary_a) if isinstance(scores_a, dict) else None,
        },
        "candidate": {
            "primary_metric": primary_b,
            "scores_keys": keys_b,
            "metric_value": scores_b.get(primary_b) if isinstance(scores_b, dict) else None,
        },
    }

    custom = record_a.get("custom") if isinstance(record_a.get("custom"), dict) else {}
    replicate_key = custom.get("replicate_key")
    if replicate_key:
        context["replicate_key"] = replicate_key

    return context


def _metric_mismatch_details(record_a: dict[str, Any], record_b: dict[str, Any]) -> str:
    context = _metric_mismatch_context(record_a, record_b)

    baseline = context.get("baseline", {})
    candidate = context.get("candidate", {})
    details = [
        f"baseline.primary_metric={baseline.get('primary_metric')!r}",
        f"candidate.primary_metric={candidate.get('primary_metric')!r}",
    ]

    if baseline.get("scores_keys") is not None:
        details.append(f"baseline.scores keys={baseline.get('scores_keys')!r}")
    else:
        details.append("baseline.scores type=None")

    if candidate.get("scores_keys") is not None:
        details.append(f"candidate.scores keys={candidate.get('scores_keys')!r}")
    else:
        details.append("candidate.scores type=None")

    if baseline.get("primary_metric") is not None:
        details.append(
            f"baseline.{baseline.get('primary_metric')}={baseline.get('metric_value')!r}"
        )
    if candidate.get("primary_metric") is not None:
        details.append(
            f"candidate.{candidate.get('primary_metric')}={candidate.get('metric_value')!r}"
        )

    if "replicate_key" in context:
        details.append(f"replicate_key={context['replicate_key']}")

    return "; ".join(details)
