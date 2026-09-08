"""Expose aggregate evidence after an interrupted runner invocation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional

from insideLLMs.analysis.statistics import generate_summary_report
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.runtime._run_health import assess_run_health, describe_run_abort
from insideLLMs.types import ExperimentResult

if TYPE_CHECKING:
    from insideLLMs.models.base import Model
    from insideLLMs.probes.base import Probe
    from insideLLMs.schemas import OutputValidator


def finalize_runner_results(
    model: Model,
    probe: Probe,
    results: list[dict[str, Any]],
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    config: dict[str, Any],
    strict_serialization: bool,
    expected_count: int,
    schema_version: str,
    validator: Optional[OutputValidator],
    validation_mode: Literal["strict", "warn"],
    error: Optional[RunnerExecutionError],
) -> tuple[ExperimentResult, Optional[RunnerExecutionError], Optional[dict[str, Any]]]:
    """Validate, score once and attach partial diagnostics before artifact finalization."""
    # High-level APIs also import interruption helpers, so defer this import.
    from insideLLMs.runtime._high_level import create_experiment_result
    from insideLLMs.schemas import SchemaRegistry

    try:
        if validator is not None:
            validator.validate(
                SchemaRegistry.RUNNER_OUTPUT,
                {"schema_version": schema_version, "results": results},
                schema_version=schema_version,
                mode=validation_mode,
            )
    except Exception as exc:
        if error is None:
            error = RunnerExecutionError(
                "Output validation failed", run_id=run_id, original_error=exc
            )
        else:
            error.secondary_diagnostics.append(
                {
                    "stage": "output_validation",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    create_experiment = partial(
        create_experiment_result,
        model,
        probe,
        results,
        config=config,
        experiment_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        strict_serialization=strict_serialization,
    )
    try:
        experiment = create_experiment(_abort_error=error)
    except Exception as exc:
        error = RunnerExecutionError("Aggregate scoring failed", run_id=run_id, original_error=exc)
        experiment = create_experiment(_abort_error=error, _skip_scoring=True)

    summary_payload = None
    if error is not None:
        summary_payload = build_partial_summary(
            error,
            experiment,
            results,
            expected_count=expected_count,
            schema_version=schema_version,
            generated_at=completed_at,
            config=config,
        )
    return experiment, error, summary_payload


@contextmanager
def capture_secondary_error(error: Optional[RunnerExecutionError], stage: str) -> Iterator[None]:
    """Keep diagnostic failures subordinate to the run's original failure."""
    try:
        yield
    except Exception as exc:
        if error is None:
            raise
        error.secondary_diagnostics.append(
            {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}
        )
        if error.partial_result is not None:
            error.partial_result["summary"]["secondary_diagnostics"] = list(
                error.secondary_diagnostics
            )


def write_scored_summary(
    run_dir: Path,
    experiment: ExperimentResult,
    *,
    schema_version: str,
    generated_at: datetime,
    config: dict[str, Any],
    strict_serialization: bool,
    validator: Optional[OutputValidator],
    validation_mode: Literal["strict", "warn"],
) -> None:
    """Persist the already-scored successful experiment before ultimate seals its bytes."""
    from insideLLMs._serialization import stable_json_dumps
    from insideLLMs.runtime._artifact_utils import _atomic_write_text
    from insideLLMs.schemas import SchemaRegistry

    payload = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "summary": generate_summary_report([experiment]),
        "config": config,
    }
    if validator is not None:
        validator.validate(
            SchemaRegistry.HARNESS_SUMMARY,
            payload,
            schema_version=schema_version,
            mode=validation_mode,
        )
    _atomic_write_text(
        run_dir / "summary.json", stable_json_dumps(payload, strict=strict_serialization)
    )


def abort_summary_metadata(
    error: RunnerExecutionError,
    results: list[dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    """Shared incomplete-health and secondary-finalization diagnostics."""
    metadata: dict[str, Any] = {
        "run_completed": False,
        "abort": describe_run_abort(error),
        "health": assess_run_health(results, expected_count=expected_count, run_completed=False),
    }
    if error.secondary_diagnostics:
        metadata["secondary_diagnostics"] = list(error.secondary_diagnostics)
    return metadata


def build_partial_summary(
    error: RunnerExecutionError,
    experiment: ExperimentResult,
    results: list[dict[str, Any]],
    *,
    expected_count: int,
    schema_version: str,
    generated_at: str | datetime,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Attach API diagnostics and return the schema-compatible summary payload."""
    summary = generate_summary_report([experiment])
    summary.update(abort_summary_metadata(error, results, expected_count))
    error.partial_results = results
    error.partial_result = {
        "results": results,
        "experiments": [experiment],
        "summary": summary,
        "config": config,
        "run_completed": False,
    }
    return {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "summary": summary,
        "config": config,
    }
