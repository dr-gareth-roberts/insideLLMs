"""Deterministic diff computation utilities.

This module provides a framework-level API for building canonical diff reports
between two run outputs. It is intentionally runtime-scoped so CLI, automation,
and library users can compose on top of the same core behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from insideLLMs._serialization import (
    fingerprint_value as _fingerprint_value,
)
from insideLLMs.runtime._base import _normalize_validation_mode
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

JudgePolicy = Literal["strict", "balanced"]


@dataclass
class DiffComputation:
    """Structured outputs from diff computation."""

    diff_report: dict[str, Any]
    regressions: list[tuple[str, str, str, str]]
    improvements: list[tuple[str, str, str, str]]
    changes: list[tuple[str, str, str, str]]
    only_baseline: list[tuple[str, str, str]]
    only_candidate: list[tuple[str, str, str]]
    trace_drifts: list[tuple[str, str, str, str]]
    trace_violation_increases: list[tuple[str, str, str, str]]
    trajectory_drifts: list[tuple[str, str, str, str]]
    baseline_duplicates: int
    candidate_duplicates: int

    @property
    def has_differences(self) -> bool:
        """Whether any behavioral/trace differences are present."""
        return bool(
            self.regressions
            or self.changes
            or self.only_baseline
            or self.only_candidate
            or self.trace_drifts
            or self.trace_violation_increases
            or self.trajectory_drifts
        )


@dataclass
class DiffJudgeComputation:
    """Deterministic judge summary layered on top of DiffReport."""

    judge_report: dict[str, Any]
    breaking: bool
    breaking_count: int
    review_count: int
    acceptable_count: int


@dataclass(frozen=True)
class DiffGatePolicy:
    """Typed gate configuration for diff exit-code policy."""

    fail_on_regressions: bool = False
    fail_on_changes: bool = False
    fail_on_trace_violations: bool = False
    fail_on_trace_drift: bool = False
    fail_on_trajectory_drift: bool = False


def _trim_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
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


def _output_text(record: dict[str, Any]) -> str | None:
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


def _output_fingerprint(record: dict[str, Any], ignore_keys: set[str] | None = None) -> str | None:
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


def _output_summary(record: dict[str, Any], ignore_keys: set[str] | None) -> dict[str, Any] | None:
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


def _trace_fingerprint(record: dict[str, Any]) -> str | None:
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    fp = custom.get("trace_fingerprint")
    if isinstance(fp, str):
        return fp

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    fingerprint = trace.get("fingerprint") if isinstance(trace.get("fingerprint"), dict) else {}
    value = fingerprint.get("value")
    if not isinstance(value, str) or not value:
        return None
    if ":" not in value and len(value) == 64:
        return f"sha256:{value}"
    return value


def _trace_violations(record: dict[str, Any]) -> list[dict[str, Any]]:
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    violations = custom.get("trace_violations")
    if isinstance(violations, list):
        return violations

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    violations = trace.get("violations")
    return violations if isinstance(violations, list) else []


def _trace_violation_count(record: dict[str, Any]) -> int:
    return len(_trace_violations(record))


def _trace_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    if isinstance(custom.get("trace_events"), list):
        return [event for event in custom["trace_events"] if isinstance(event, dict)]

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    if isinstance(trace.get("events"), list):
        return [event for event in trace["events"] if isinstance(event, dict)]

    output = record.get("output")
    if isinstance(output, dict) and isinstance(output.get("trace_events"), list):
        return [event for event in output["trace_events"] if isinstance(event, dict)]
    return []


def _tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    output = record.get("output")
    if isinstance(output, dict) and isinstance(output.get("tool_calls"), list):
        return [call for call in output["tool_calls"] if isinstance(call, dict)]
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    if isinstance(custom.get("tool_calls"), list):
        return [call for call in custom["tool_calls"] if isinstance(call, dict)]

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    derived = trace.get("derived") if isinstance(trace.get("derived"), dict) else {}
    tool_calls = derived.get("tool_calls") if isinstance(derived.get("tool_calls"), dict) else {}
    sequence = tool_calls.get("sequence")
    if isinstance(sequence, list):
        return [
            {"tool_name": str(tool_name), "arguments": None, "step": index}
            for index, tool_name in enumerate(sequence)
            if tool_name is not None
        ]
    return []


def _trajectory_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = _trace_events(record)
    if events:
        steps: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            kind = str(event.get("kind") or "event")
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            seq = event.get("seq")
            step_raw = payload.get("step")
            step = step_raw if isinstance(step_raw, int) else seq if isinstance(seq, int) else index
            tool_name = payload.get("tool_name")
            if kind == "tool_call_start":
                steps.append(
                    {
                        "kind": kind,
                        "step": step,
                        "tool_name": str(tool_name) if tool_name is not None else None,
                        "tool_call_id": payload.get("tool_call_id"),
                        "arguments_fingerprint": _fingerprint_value(payload.get("arguments")),
                    }
                )
                continue
            if kind == "tool_result":
                steps.append(
                    {
                        "kind": kind,
                        "step": step,
                        "tool_name": str(tool_name) if tool_name is not None else None,
                        "tool_call_id": payload.get("tool_call_id"),
                        "result_fingerprint": _fingerprint_value(payload.get("result")),
                    }
                )
                continue
            if kind in {"generate_start", "generate_end", "error", "custom"}:
                steps.append({"kind": kind, "step": step})
        if steps:
            return steps

    tool_calls = _tool_calls(record)
    if tool_calls:
        return [
            {
                "kind": "tool_call_start",
                "step": index,
                "tool_name": str(call.get("tool_name") or call.get("name") or "tool"),
                "tool_call_id": call.get("tool_call_id"),
                "arguments_fingerprint": _fingerprint_value(
                    call.get("arguments") if call.get("arguments") is not None else call.get("args")
                ),
            }
            for index, call in enumerate(tool_calls)
        ]
    return []


def _trajectory_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    steps = _trajectory_steps(record)
    if not steps:
        return None
    tool_sequence = [
        step["tool_name"]
        for step in steps
        if step.get("kind") == "tool_call_start" and isinstance(step.get("tool_name"), str)
    ]
    summary: dict[str, Any] = {
        "step_count": len(steps),
        "tool_call_count": len(tool_sequence),
        "tool_sequence": tool_sequence,
        "fingerprint": _fingerprint_value(steps),
        "steps": steps[:25],
    }
    if len(steps) > 25:
        summary["truncated_steps"] = len(steps) - 25
    return summary


def _primary_score(record: dict[str, Any]) -> tuple[str | None, float | None]:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    metric = record.get("primary_metric")
    if metric and metric in scores and isinstance(scores[metric], (int, float)):
        return str(metric), float(scores[metric])
    if "score" in scores and isinstance(scores["score"], (int, float)):
        return "score", float(scores["score"])
    return None, None


def _metric_mismatch_reason(record_a: dict[str, Any], record_b: dict[str, Any]) -> str | None:
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


def _normalize_ignore_keys(raw_ignore_keys: list[str] | None) -> set[str] | None:
    ignore_keys: set[str] = set()
    for entry in raw_ignore_keys or []:
        for item in str(entry).split(","):
            item = item.strip()
            if item:
                ignore_keys.add(item.lower())
    return ignore_keys if ignore_keys else None


def _build_index(
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


def _record_identity(record: dict[str, Any]) -> dict[str, Any]:
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


def _record_summary(record: dict[str, Any], ignore_keys: set[str] | None) -> dict[str, Any]:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else None
    _metric_name, metric_value = _primary_score(record)
    return {
        "status": _status_string(record),
        "primary_metric": record.get("primary_metric"),
        "primary_score": metric_value,
        "scores_keys": sorted(scores.keys()) if isinstance(scores, dict) else None,
        "output": _output_summary(record, ignore_keys),
    }


def _validate_diff_report(
    diff_report: dict[str, Any],
    *,
    schema_version: str,
    validation_mode: str,
) -> None:
    from insideLLMs.schemas import OutputValidator, SchemaRegistry

    validator = OutputValidator(SchemaRegistry())
    validator.validate(
        SchemaRegistry.DIFF_REPORT,
        diff_report,
        schema_version=schema_version,
        mode=_normalize_validation_mode(validation_mode),
    )


def build_diff_computation(
    *,
    records_baseline: list[dict[str, Any]],
    records_candidate: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    output_fingerprint_ignore: list[str] | None = None,
    validate_output: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
) -> DiffComputation:
    """Build deterministic diff artifacts and categorized change lists."""
    ignore_keys = _normalize_ignore_keys(output_fingerprint_ignore)

    index_a, dup_a = _build_index(records_baseline)
    index_b, dup_b = _build_index(records_candidate)
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
    trace_drifts: list[tuple[str, str, str, str]] = []
    trace_drifts_json: list[dict[str, Any]] = []
    trace_violation_increases: list[tuple[str, str, str, str]] = []
    trace_violation_increases_json: list[dict[str, Any]] = []
    trajectory_drifts: list[tuple[str, str, str, str]] = []
    trajectory_drifts_json: list[dict[str, Any]] = []

    for key in sorted(all_keys):
        record_a = index_a.get(key)
        record_b = index_b.get(key)

        if record_a is None:
            only_b.append(_record_label(record_b))  # type: ignore[arg-type]
            only_b_json.append(_record_identity(record_b))  # type: ignore[arg-type]
            continue
        if record_b is None:
            only_a.append(_record_label(record_a))
            only_a_json.append(_record_identity(record_a))
            continue

        label = _record_label(record_a)
        status_a = _status_string(record_a)
        status_b = _status_string(record_b)
        score_name_a, score_a = _primary_score(record_a)
        score_name_b, score_b = _primary_score(record_b)
        identity = _record_identity(record_a)
        summary_a = _record_summary(record_a, ignore_keys)
        summary_b = _record_summary(record_b, ignore_keys)

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
            fingerprint_a = _output_fingerprint(record_a, ignore_keys=ignore_keys)
            fingerprint_b = _output_fingerprint(record_b, ignore_keys=ignore_keys)
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

        trajectory_a = _trajectory_summary(record_a)
        trajectory_b = _trajectory_summary(record_b)
        trajectory_fp_a = (
            trajectory_a.get("fingerprint")
            if isinstance(trajectory_a, dict) and isinstance(trajectory_a.get("fingerprint"), str)
            else None
        )
        trajectory_fp_b = (
            trajectory_b.get("fingerprint")
            if isinstance(trajectory_b, dict) and isinstance(trajectory_b.get("fingerprint"), str)
            else None
        )
        if (trajectory_a or trajectory_b) and trajectory_fp_a != trajectory_fp_b:
            step_count_a = int(trajectory_a.get("step_count", 0)) if trajectory_a else 0
            step_count_b = int(trajectory_b.get("step_count", 0)) if trajectory_b else 0
            tool_count_a = int(trajectory_a.get("tool_call_count", 0)) if trajectory_a else 0
            tool_count_b = int(trajectory_b.get("tool_call_count", 0)) if trajectory_b else 0
            detail = (
                f"trajectory steps {step_count_a} -> {step_count_b}; "
                f"tool calls {tool_count_a} -> {tool_count_b}"
            )
            trajectory_drifts.append((*label, detail))
            trajectory_drifts_json.append(
                {
                    **identity,
                    "kind": "trajectory_drift",
                    "detail": detail,
                    "baseline_trajectory_fingerprint": trajectory_fp_a,
                    "candidate_trajectory_fingerprint": trajectory_fp_b,
                    "baseline_trajectory": trajectory_a,
                    "candidate_trajectory": trajectory_b,
                }
            )

    diff_report = {
        "schema_version": schema_version,
        "baseline": baseline_label,
        "candidate": candidate_label,
        "run_ids": {
            "baseline": sorted(
                {record.get("run_id") for record in records_baseline if record.get("run_id")}
            ),
            "candidate": sorted(
                {record.get("run_id") for record in records_candidate if record.get("run_id")}
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
            "trajectory_drifts": len(trajectory_drifts),
        },
        "duplicates": {"baseline": dup_a, "candidate": dup_b},
        "regressions": regressions_json,
        "improvements": improvements_json,
        "changes": changes_json,
        "only_baseline": only_a_json,
        "only_candidate": only_b_json,
        "trace_drifts": trace_drifts_json,
        "trace_violation_increases": trace_violation_increases_json,
        "trajectory_drifts": trajectory_drifts_json,
    }

    if validate_output:
        _validate_diff_report(
            diff_report,
            schema_version=schema_version,
            validation_mode=validation_mode,
        )

    return DiffComputation(
        diff_report=diff_report,
        regressions=regressions,
        improvements=improvements,
        changes=changes,
        only_baseline=only_a,
        only_candidate=only_b,
        trace_drifts=trace_drifts,
        trace_violation_increases=trace_violation_increases,
        trajectory_drifts=trajectory_drifts,
        baseline_duplicates=dup_a,
        candidate_duplicates=dup_b,
    )


def _judge_section_rule(section: str) -> tuple[str, str]:
    if section in {"regressions", "trace_violation_increases", "only_baseline", "only_candidate"}:
        return ("breaking", "high-confidence breaking change")
    if section == "improvements":
        return ("acceptable", "positive behavior delta")
    if section in {"changes", "trace_drifts", "trajectory_drifts"}:
        return ("review", "requires review")
    return ("review", "unclassified change")


def _as_label(item: Mapping[str, Any]) -> dict[str, Any]:
    label = item.get("label")
    return label if isinstance(label, dict) else {}


def judge_diff_report(
    diff_report: Mapping[str, Any],
    *,
    policy: JudgePolicy = "strict",
    limit: int | None = None,
) -> DiffJudgeComputation:
    """Apply a deterministic judge policy to a diff report."""
    sections = (
        "regressions",
        "changes",
        "only_baseline",
        "only_candidate",
        "trace_violation_increases",
        "trace_drifts",
        "trajectory_drifts",
        "improvements",
    )

    verdicts: list[dict[str, Any]] = []
    for section in sections:
        items = diff_report.get(section)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            decision, reason = _judge_section_rule(section)
            label = _as_label(item)
            detail = item.get("detail") or item.get("kind")
            verdicts.append(
                {
                    "section": section,
                    "decision": decision,
                    "reason": reason,
                    "kind": item.get("kind"),
                    "detail": detail,
                    "model_id": item.get("model_id"),
                    "probe_id": item.get("probe_id"),
                    "example_id": item.get("example_id"),
                    "label": label or None,
                }
            )

    limit_value = limit if isinstance(limit, int) and limit > 0 else None
    shown_verdicts = verdicts[:limit_value] if limit_value else verdicts

    breaking_count = sum(1 for verdict in verdicts if verdict.get("decision") == "breaking")
    review_count = sum(1 for verdict in verdicts if verdict.get("decision") == "review")
    acceptable_count = sum(1 for verdict in verdicts if verdict.get("decision") == "acceptable")

    if policy == "strict":
        breaking = breaking_count > 0 or review_count > 0
    else:
        breaking = breaking_count > 0

    judge_report: dict[str, Any] = {
        "policy": policy,
        "breaking": breaking,
        "summary": {
            "total_items": len(verdicts),
            "shown_items": len(shown_verdicts),
            "breaking": breaking_count,
            "review": review_count,
            "acceptable": acceptable_count,
        },
        "verdicts": shown_verdicts,
    }
    if len(shown_verdicts) < len(verdicts):
        judge_report["summary"]["truncated"] = len(verdicts) - len(shown_verdicts)

    return DiffJudgeComputation(
        judge_report=judge_report,
        breaking=breaking,
        breaking_count=breaking_count,
        review_count=review_count,
        acceptable_count=acceptable_count,
    )


def compute_diff_exit_code(
    computation: DiffComputation,
    policy: DiffGatePolicy | None = None,
) -> int:
    """Compute canonical diff gating exit code from a diff computation."""
    policy = policy or DiffGatePolicy()
    if policy.fail_on_regressions and computation.regressions:
        return 2
    if policy.fail_on_changes and (
        computation.regressions
        or computation.changes
        or computation.only_baseline
        or computation.only_candidate
    ):
        return 2
    if policy.fail_on_trace_violations and computation.trace_violation_increases:
        return 3
    if policy.fail_on_trace_drift and computation.trace_drifts:
        return 4
    if policy.fail_on_trajectory_drift and computation.trajectory_drifts:
        return 5
    return 0


__all__ = [
    "compute_diff_exit_code",
    "DiffComputation",
    "DiffGatePolicy",
    "DiffJudgeComputation",
    "JudgePolicy",
    "build_diff_computation",
    "judge_diff_report",
]
