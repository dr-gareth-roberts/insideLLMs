"""The shared per-item scoring contract used by runners and probe batches.

Dataset mappings opt into evaluation with ``reference_answer`` (or the
``reference`` alias). An explicit null calls reference-free evaluators; an
omitted field leaves the item unscored. Reference fields never reach ``run``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from insideLLMs.types import ProbeResult, ResultStatus

if TYPE_CHECKING:
    from insideLLMs.probes.base import Probe

REFERENCE_FIELDS = ("reference_answer", "reference")


def required_output_field(
    output: Any,
    field: str,
    expected_types: type | tuple[type, ...],
    *,
    allow_none: bool = False,
) -> Any:
    """Read and validate a scorer field from a live object or saved mapping."""
    if isinstance(output, Mapping):
        if field not in output:
            raise ValueError(f"Persisted probe output is missing required field '{field}'.")
        value = output[field]
    else:
        try:
            value = getattr(output, field)
        except AttributeError as exc:
            raise TypeError(f"Probe output has no required field '{field}'.") from exc
    if value is None and allow_none:
        return None
    accepts_bool = expected_types is bool or (
        isinstance(expected_types, tuple) and bool in expected_types
    )
    if isinstance(value, bool) and not accepts_bool:
        raise TypeError(f"Probe output field '{field}' has an invalid type.")
    if not isinstance(value, expected_types):
        raise TypeError(f"Probe output field '{field}' has an invalid type.")
    return value


def probe_input(probe: Probe, item: Any) -> Any:
    """Keep held-out answers out of scored probes' model prompts."""
    from insideLLMs.probes.base import ScoredProbe

    if isinstance(probe, ScoredProbe) and isinstance(item, dict):
        return {key: value for key, value in item.items() if key not in REFERENCE_FIELDS}
    return item


def _normalized_scores(evaluation: dict[str, Any]) -> tuple[dict[str, float], str]:
    if not isinstance(evaluation.get("is_correct"), bool):
        raise ValueError("ScoredProbe evaluation must include a boolean 'is_correct'.")
    supplied = evaluation.get("scores", {})
    if not isinstance(supplied, dict):
        raise ValueError("ScoredProbe evaluation 'scores' must be a mapping.")
    scores = dict(supplied)
    if "score" in evaluation:
        scores["score"] = evaluation["score"]
    scores["accuracy"] = float(evaluation["is_correct"])
    for name, value in scores.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("ScoredProbe scores must be named finite numbers.")
    primary = evaluation.get("primary_metric", "score" if "score" in scores else "accuracy")
    if not isinstance(primary, str) or primary not in scores:
        raise ValueError("ScoredProbe primary_metric must name an evaluated score.")
    return {name: float(value) for name, value in scores.items()}, primary


def evaluate_probe_result(probe: Probe, result: ProbeResult) -> ProbeResult:
    """Evaluate exactly once while retaining the original input and raw output."""
    from insideLLMs.probes.base import ScoredProbe

    if not isinstance(probe, ScoredProbe) or result.status != ResultStatus.SUCCESS:
        return result
    if result.metadata.get("evaluation", {}).get("status") == "evaluated":
        validate_evaluated_scores(result.metadata, result.scores, result.primary_metric)
        return result
    item = result.input
    fields = [key for key in REFERENCE_FIELDS if isinstance(item, dict) and key in item]
    if not fields:
        return replace(
            result,
            metadata={
                **result.metadata,
                "evaluation": {
                    "status": "not_evaluated",
                    "reason": "missing_reference",
                },
            },
        )
    if len(fields) > 1:
        raise ValueError("Use only one of 'reference_answer' and 'reference' per dataset item.")
    evaluation = probe.evaluate_single(result.output, item[fields[0]], input_data=item)
    if isinstance(evaluation, ProbeResult):
        if evaluation.status != ResultStatus.SUCCESS:
            raise ValueError(evaluation.error or "ScoredProbe evaluation did not succeed.")
        details = dict(evaluation.metadata)
        if evaluation.scores:
            details["scores"] = evaluation.scores
        if evaluation.primary_metric is not None:
            details["primary_metric"] = evaluation.primary_metric
    elif isinstance(evaluation, dict):
        details = dict(evaluation)
    else:
        raise ValueError("ScoredProbe evaluation must return a dict or ProbeResult.")
    scores, primary_metric = _normalized_scores(details)
    return replace(
        result,
        scores=scores,
        primary_metric=primary_metric,
        metadata={
            **result.metadata,
            **details,
            "evaluation": {"status": "evaluated", "reference_field": fields[0]},
        },
    )


def evaluate_batch_result(probe: Probe, result: ProbeResult, item: Any) -> ProbeResult:
    """Normalize custom batch implementations through the same scoring contract."""
    result = replace(result, input=item).attach_original_error(result.original_error)
    try:
        return evaluate_probe_result(probe, result)
    except Exception as exc:
        return replace(
            result,
            status=ResultStatus.ERROR,
            error=str(exc),
            metadata={**result.metadata, "error_type": type(exc).__name__},
            scores={},
            primary_metric=None,
        ).attach_original_error(exc)


def validate_scored_resume_record(probe: Probe, record: dict[str, Any]) -> None:
    """Old labelled runs cannot be reused without their measured evaluation."""
    from insideLLMs.probes.base import ScoredProbe

    item = record.get("input")
    if (
        isinstance(probe, ScoredProbe)
        and record.get("status") == "success"
        and isinstance(item, dict)
        and any(key in item for key in REFERENCE_FIELDS)
    ):
        custom = record.get("custom")
        metadata = custom.get("evaluation") if isinstance(custom, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        evaluation = metadata.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        if evaluation.get("status") != "evaluated" or not record.get("scores"):
            raise ValueError(
                "Cannot resume a labelled ScoredProbe run without persisted evaluation. "
                "Re-run it into a new run directory to calculate behavioural scores."
            )
        validate_evaluated_scores(metadata, record.get("scores"), record.get("primary_metric"))


def validate_evaluated_scores(metadata: dict[str, Any], scores: Any, primary_metric: Any) -> None:
    """Reused evaluation must satisfy the same contract as freshly measured data."""
    if not isinstance(scores, dict) or not scores:
        raise ValueError("Persisted ScoredProbe evaluation requires nonempty finite scores.")
    normalized, primary = _normalized_scores(
        {**metadata, "scores": scores, "primary_metric": primary_metric}
    )
    if normalized != scores or primary != primary_metric:
        raise ValueError("Persisted ScoredProbe scores do not match their evaluation metadata.")
