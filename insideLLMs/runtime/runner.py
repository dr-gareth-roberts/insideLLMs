"""Experiment runner with YAML/JSON config support.

This module provides tools for running LLM experiments, either programmatically
or from configuration files. Supports async execution for parallel processing.

The module contains two main runner classes:

- :class:`ProbeRunner`: Synchronous runner for sequential probe execution
- :class:`AsyncProbeRunner`: Asynchronous runner for concurrent probe execution

It also provides convenience functions for running experiments directly from
configuration files:

- :func:`run_probe`: Simple function to run a probe on a model
- :func:`run_probe_async`: Async version of run_probe
- :func:`run_experiment_from_config`: Run an experiment defined in YAML/JSON
- :func:`run_experiment_from_config_async`: Async version
- :func:`run_harness_from_config`: Run multi-model/multi-probe harness

Examples
--------
Basic probe execution with ProbeRunner:

    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.probes import FactualityProbe
    >>> from insideLLMs.runtime.runner import ProbeRunner
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> probe = FactualityProbe()
    >>> runner = ProbeRunner(model, probe)
    >>>
    >>> prompts = [
    ...     {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    ...     {"messages": [{"role": "user", "content": "Who wrote Hamlet?"}]},
    ... ]
    >>> results = runner.run(prompts)
    >>> print(f"Success rate: {runner.success_rate:.1%}")
    Success rate: 100.0%

Using AsyncProbeRunner for concurrent execution:

    >>> import asyncio
    >>> from insideLLMs.runtime.runner import AsyncProbeRunner
    >>>
    >>> async def run_concurrent():
    ...     model = OpenAIModel(model_name="gpt-4")
    ...     probe = FactualityProbe()
    ...     runner = AsyncProbeRunner(model, probe)
    ...     results = await runner.run(prompts, concurrency=10)
    ...     return results
    >>>
    >>> results = asyncio.run(run_concurrent())

Running from a configuration file:

    >>> from insideLLMs.runtime.runner import run_experiment_from_config
    >>>
    >>> # config.yaml contains model, probe, and dataset definitions
    >>> results = run_experiment_from_config("config.yaml")

Getting ExperimentResult with full metadata:

    >>> from insideLLMs.config_types import RunConfig
    >>>
    >>> config = RunConfig(return_experiment=True)
    >>> experiment = runner.run(prompts, config=config)
    >>> print(f"Experiment ID: {experiment.experiment_id}")
    >>> print(f"Score: {experiment.score}")

See Also
--------
insideLLMs.config_types.RunConfig : Configuration options for runners
insideLLMs.types.ExperimentResult : Structured experiment results
insideLLMs.types.ProbeResult : Individual probe execution results
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import platform
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml

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
from insideLLMs.config_types import ProgressInfo, RunConfig
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models.base import Model
from insideLLMs.probes.base import Probe
from insideLLMs.registry import (
    NotFoundError,
    dataset_registry,
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.runtime.timeout_wrapper import run_with_timeout
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.statistics import generate_summary_report
from insideLLMs.types import (
    ConfigDict,
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ResultStatus,
)
from insideLLMs.validation import validate_prompt_set

logger = logging.getLogger(__name__)

# Type alias for progress callbacks - supports both simple (current, total) and rich ProgressInfo
ProgressCallback = Union[Callable[[int, int], None], Callable[["ProgressInfo"], None]]


def _invoke_progress_callback(
    callback: Optional[ProgressCallback],
    current: int,
    total: int,
    start_time: float,
    current_item: Optional[Any] = None,
    current_index: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Invoke a progress callback with the appropriate signature.

    Supports both legacy (current, total) callbacks and new ProgressInfo callbacks.
    Detects the callback signature and invokes accordingly.

    Args:
        callback: The progress callback to invoke.
        current: Number of items completed.
        total: Total number of items.
        start_time: time.perf_counter() value at start.
        current_item: The item currently being processed.
        current_index: Index of current item.
        status: Current status message.
    """
    if callback is None:
        return

    callback_style = getattr(callback, "_insidellms_progress_style", None)
    if callback_style not in {"legacy", "info"}:
        try:
            sig = inspect.signature(callback)
            params = list(sig.parameters.values())
            callback_style = "legacy" if len(params) == 2 else "info"
        except (TypeError, ValueError):
            callback_style = "legacy"
        try:
            setattr(callback, "_insidellms_progress_style", callback_style)
        except Exception:
            pass

    if callback_style == "legacy":
        callback(current, total)  # type: ignore
        return

    # Use new ProgressInfo style
    info = ProgressInfo.create(
        current=current,
        total=total,
        start_time=start_time,
        current_item=current_item,
        current_index=current_index,
        status=status,
    )
    callback(info)  # type: ignore


def _normalize_validation_mode(mode: Optional[str]) -> str:
    """Normalize validation mode to the validator's accepted values.

    The schema validator accepts "strict" or "warn". Historically, runner
    configuration exposes "lenient" to mean "warn". This helper preserves
    backward compatibility while keeping validator behavior explicit.
    """
    if mode is None:
        return "strict"
    normalized = str(mode).strip().lower()
    if normalized in {"lenient", "warn"}:
        return "warn"
    return normalized


class _RunnerBase:
    """Base class for ProbeRunner and AsyncProbeRunner with shared functionality.

    This class provides common initialization, properties, and helper methods
    used by both synchronous and asynchronous runners. It should not be
    instantiated directly; use :class:`ProbeRunner` or :class:`AsyncProbeRunner`
    instead.

    Attributes
    ----------
    model : Model
        The model instance to test.
    probe : Probe
        The probe instance to run.
    last_run_id : Optional[str]
        The unique identifier of the last completed run.
    last_run_dir : Optional[Path]
        The directory where the last run's artifacts were stored.
    last_experiment : Optional[ExperimentResult]
        The ExperimentResult from the last completed run.

    Examples
    --------
    Accessing run statistics after execution:

        >>> runner = ProbeRunner(model, probe)
        >>> results = runner.run(prompts)
        >>> print(f"Success rate: {runner.success_rate:.1%}")
        >>> print(f"Error count: {runner.error_count}")
        >>> print(f"Run ID: {runner.last_run_id}")

    Accessing the last experiment result:

        >>> if runner.last_experiment:
        ...     print(f"Score: {runner.last_experiment.score}")
        ...     print(f"Started: {runner.last_experiment.started_at}")
    """

    def __init__(self, model: Model, probe: Probe):
        """Initialize the runner with a model and probe.

        Parameters
        ----------
        model : Model
            The model to test. Must implement the Model interface.
        probe : Probe
            The probe to run. Must implement the Probe interface.

        Examples
        --------
        Creating a runner for testing:

            >>> from insideLLMs.models import OpenAIModel
            >>> from insideLLMs.probes import FactualityProbe
            >>>
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> probe = FactualityProbe()
            >>> runner = ProbeRunner(model, probe)

        Using a custom model and probe:

            >>> from insideLLMs.models.base import Model
            >>> from insideLLMs.probes.base import Probe
            >>>
            >>> class MyModel(Model):
            ...     def generate(self, messages):
            ...         return "response"
            >>>
            >>> class MyProbe(Probe):
            ...     name = "my_probe"
            ...     def run(self, model, input_data):
            ...         return model.generate(input_data)
            >>>
            >>> runner = ProbeRunner(MyModel(), MyProbe())
        """
        self.model = model
        self.probe = probe
        self._results: list[dict[str, Any]] = []
        self.last_run_id: Optional[str] = None
        self.last_run_dir: Optional[Path] = None
        self.last_experiment: Optional[ExperimentResult] = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the last run.

        Returns the proportion of successful results from the most recent
        execution. Returns 0.0 if no results are available.

        Returns
        -------
        float
            Success rate as a decimal between 0.0 and 1.0.

        Examples
        --------
        Checking success rate after a run:

            >>> runner = ProbeRunner(model, probe)
            >>> results = runner.run(prompts)
            >>> print(f"Success rate: {runner.success_rate:.1%}")
            Success rate: 95.0%

        Handling empty results:

            >>> runner = ProbeRunner(model, probe)
            >>> # Before running any prompts
            >>> print(runner.success_rate)
            0.0
        """
        if not self._results:
            return 0.0
        successes = sum(1 for r in self._results if r.get("status") == "success")
        return successes / len(self._results)

    @property
    def error_count(self) -> int:
        """Count errors from the last run.

        Returns the total number of results with "error" status from the
        most recent execution.

        Returns
        -------
        int
            Number of errors encountered during the last run.

        Examples
        --------
        Checking error count:

            >>> runner = ProbeRunner(model, probe)
            >>> results = runner.run(prompts)
            >>> if runner.error_count > 0:
            ...     print(f"Encountered {runner.error_count} errors")
        """
        return sum(1 for r in self._results if r.get("status") == "error")


class ProbeRunner(_RunnerBase):
    """Synchronous runner for executing probes against a model.

    The ProbeRunner orchestrates the execution of probes, handling error
    recovery, progress tracking, result aggregation, and artifact emission.
    It processes prompts sequentially, making it suitable for scenarios where
    order matters or when API rate limits are a concern.

    For concurrent execution, use :class:`AsyncProbeRunner` instead.

    Attributes
    ----------
    model : Model
        The model instance being tested.
    probe : Probe
        The probe instance being run.
    last_run_id : Optional[str]
        Unique identifier of the last completed run, generated deterministically
        from the input configuration.
    last_run_dir : Optional[Path]
        Directory where the last run's artifacts were stored (records.jsonl,
        manifest.json, etc.).
    last_experiment : Optional[ExperimentResult]
        The ExperimentResult from the last completed run.
    success_rate : float
        Proportion of successful results from the last run (0.0 to 1.0).
    error_count : int
        Number of errors encountered in the last run.

    Examples
    --------
    Basic usage with a list of prompts:

        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.probes import FactualityProbe
        >>> from insideLLMs.runtime.runner import ProbeRunner
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> probe = FactualityProbe()
        >>> runner = ProbeRunner(model, probe)
        >>>
        >>> prompts = [
        ...     {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        ...     {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
        ... ]
        >>> results = runner.run(prompts)
        >>> print(f"Processed {len(results)} prompts")
        >>> print(f"Success rate: {runner.success_rate:.1%}")

    Using RunConfig for advanced options:

        >>> from insideLLMs.config_types import RunConfig
        >>>
        >>> config = RunConfig(
        ...     emit_run_artifacts=True,
        ...     run_root="./experiments",
        ...     stop_on_error=False,
        ...     return_experiment=True,
        ... )
        >>> experiment = runner.run(prompts, config=config)
        >>> print(f"Experiment ID: {experiment.experiment_id}")
        >>> print(f"Score: {experiment.score}")

    With progress tracking:

        >>> def on_progress(current, total):
        ...     print(f"Progress: {current}/{total}")
        >>>
        >>> results = runner.run(prompts, progress_callback=on_progress)

    Resuming an interrupted run:

        >>> # First run gets interrupted
        >>> try:
        ...     results = runner.run(prompts, emit_run_artifacts=True)
        ... except KeyboardInterrupt:
        ...     pass
        >>>
        >>> # Resume from where we left off
        >>> results = runner.run(prompts, resume=True, run_dir=runner.last_run_dir)

    See Also
    --------
    AsyncProbeRunner : Asynchronous runner for concurrent execution.
    run_probe : Convenience function for simple probe execution.
    run_experiment_from_config : Run experiments from YAML/JSON config files.
    """

    def run(
        self,
        prompt_set: list[Any],
        *,
        config: Optional[RunConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
        stop_on_error: Optional[bool] = None,
        validate_output: Optional[bool] = None,
        schema_version: Optional[str] = None,
        validation_mode: Optional[str] = None,
        emit_run_artifacts: Optional[bool] = None,
        run_dir: Optional[Union[str, Path]] = None,
        run_root: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        overwrite: Optional[bool] = None,
        dataset_info: Optional[dict[str, Any]] = None,
        config_snapshot: Optional[dict[str, Any]] = None,
        store_messages: Optional[bool] = None,
        strict_serialization: Optional[bool] = None,
        deterministic_artifacts: Optional[bool] = None,
        resume: Optional[bool] = None,
        use_probe_batch: Optional[bool] = None,
        batch_workers: Optional[int] = None,
        return_experiment: Optional[bool] = None,
        **probe_kwargs: Any,
    ) -> Union[list[dict[str, Any]], ExperimentResult]:
        """Run the probe on the model for each item in the prompt set.

        Executes the probe sequentially on each input, collecting results and
        optionally emitting run artifacts (records.jsonl, manifest.json) to disk.
        Supports progress tracking, error handling, and resumable runs.

        Parameters
        ----------
        prompt_set : list[Any]
            List of inputs to test. Each item is passed to the probe's run method.
            Typically a list of dictionaries with "messages" keys.
        config : Optional[RunConfig], default None
            RunConfig with all run settings. Individual kwargs override config
            values if both are provided.
        progress_callback : Optional[ProgressCallback], default None
            Callback for progress updates. Supports two signatures:
            - Legacy: ``callback(current: int, total: int)``
            - Rich: ``callback(info: ProgressInfo)``
        stop_on_error : Optional[bool], default None
            If True, stop execution on first error. If None, uses config value.
        validate_output : Optional[bool], default None
            If True, validate outputs against the schema.
        schema_version : Optional[str], default None
            Schema version for validation (e.g., "1.0.0").
        validation_mode : Optional[str], default None
            Validation mode: "strict" or "lenient" (alias: "warn").
        emit_run_artifacts : Optional[bool], default None
            If True, write records.jsonl and manifest.json to run_dir.
        run_dir : Optional[Union[str, Path]], default None
            Explicit directory for run artifacts.
        run_root : Optional[Union[str, Path]], default None
            Root directory under which run directories are created.
        run_id : Optional[str], default None
            Explicit run ID. If None, generated deterministically from inputs.
        overwrite : Optional[bool], default None
            If True, overwrite existing run directory.
        dataset_info : Optional[dict[str, Any]], default None
            Metadata about the dataset for the manifest.
        config_snapshot : Optional[dict[str, Any]], default None
            Configuration snapshot for reproducibility.
        store_messages : Optional[bool], default None
            If True, store full message content in records.
        strict_serialization : Optional[bool], default None
            If True, fail fast on non-deterministic values during hashing.
        deterministic_artifacts : Optional[bool], default None
            If True, omit host-dependent manifest fields (platform/python).
        resume : Optional[bool], default None
            If True, resume from existing records.jsonl, skipping completed items.
        use_probe_batch : Optional[bool], default None
            If True, use Probe.run_batch for potentially faster execution.
        batch_workers : Optional[int], default None
            Number of workers for Probe.run_batch.
        return_experiment : Optional[bool], default None
            If True, return ExperimentResult instead of list of dicts.
        **probe_kwargs : Any
            Additional keyword arguments passed to the probe's run method.

        Returns
        -------
        Union[list[dict[str, Any]], ExperimentResult]
            If return_experiment is False (default): list of result dictionaries,
            each containing "input", "output", "status", "latency_ms", etc.
            If return_experiment is True: ExperimentResult with aggregated metrics.

        Raises
        ------
        RunnerExecutionError
            If stop_on_error is True and a probe execution fails.
        ValueError
            If prompt_set is empty or invalid.
        FileExistsError
            If run_dir exists and is not empty, unless overwrite=True.

        Examples
        --------
        Basic execution returning result dictionaries:

            >>> runner = ProbeRunner(model, probe)
            >>> results = runner.run([
            ...     {"messages": [{"role": "user", "content": "Hello"}]},
            ...     {"messages": [{"role": "user", "content": "World"}]},
            ... ])
            >>> for r in results:
            ...     print(f"Status: {r['status']}, Output: {r['output'][:50]}")

        Using RunConfig for all options:

            >>> from insideLLMs.config_types import RunConfig
            >>> config = RunConfig(
            ...     emit_run_artifacts=True,
            ...     run_root="./experiments",
            ...     stop_on_error=False,
            ...     validate_output=True,
            ...     return_experiment=True,
            ... )
            >>> experiment = runner.run(prompts, config=config)

        With progress callback and error handling:

            >>> def progress(current, total):
            ...     pct = current / total * 100
            ...     print(f"\\rProgress: {pct:.1f}%", end="")
            >>>
            >>> try:
            ...     results = runner.run(
            ...         prompts,
            ...         progress_callback=progress,
            ...         stop_on_error=True
            ...     )
            ... except RunnerExecutionError as e:
            ...     print(f"Failed at prompt {e.prompt_index}: {e.reason}")

        Resuming an interrupted run:

            >>> # The run directory will be reused
            >>> results = runner.run(
            ...     prompts,
            ...     resume=True,
            ...     run_dir="./experiments/abc123",
            ...     emit_run_artifacts=True,
            ... )
            >>> # Only unprocessed items will be executed

        See Also
        --------
        AsyncProbeRunner.run : Async version with concurrency support.
        run_probe : Convenience function wrapping ProbeRunner.
        """
        from insideLLMs.schemas import OutputValidator, SchemaRegistry

        # Resolve config: use provided config or create default
        if config is None:
            config = RunConfig()

        # Override config with explicit kwargs (backward compatibility)
        stop_on_error = stop_on_error if stop_on_error is not None else config.stop_on_error
        validate_output = validate_output if validate_output is not None else config.validate_output
        schema_version = schema_version if schema_version is not None else config.schema_version
        validation_mode = validation_mode if validation_mode is not None else config.validation_mode
        validator_mode = _normalize_validation_mode(validation_mode)
        emit_run_artifacts = (
            emit_run_artifacts if emit_run_artifacts is not None else config.emit_run_artifacts
        )
        run_dir = run_dir if run_dir is not None else config.run_dir
        run_root = run_root if run_root is not None else config.run_root
        run_id = run_id if run_id is not None else config.run_id
        overwrite = overwrite if overwrite is not None else config.overwrite
        dataset_info = dataset_info if dataset_info is not None else config.dataset_info
        config_snapshot = config_snapshot if config_snapshot is not None else config.config_snapshot
        store_messages = store_messages if store_messages is not None else config.store_messages
        strict_serialization = (
            strict_serialization
            if strict_serialization is not None
            else config.strict_serialization
        )
        deterministic_artifacts = (
            deterministic_artifacts
            if deterministic_artifacts is not None
            else config.deterministic_artifacts
        )
        if deterministic_artifacts is None:
            deterministic_artifacts = strict_serialization
        resume = resume if resume is not None else config.resume
        use_probe_batch = use_probe_batch if use_probe_batch is not None else config.use_probe_batch
        batch_workers = batch_workers if batch_workers is not None else config.batch_workers
        return_experiment = (
            return_experiment if return_experiment is not None else config.return_experiment
        )

        # Validate prompt set before execution
        validate_prompt_set(prompt_set, field_name="prompt_set", allow_empty_set=False)

        registry = SchemaRegistry()
        validator = OutputValidator(registry)

        model_spec = _build_model_spec(self.model)
        probe_spec = _build_probe_spec(self.probe)
        dataset_spec = _build_dataset_spec(dataset_info)
        # Add params for backward compatibility
        dataset_spec["params"] = dataset_info or {}

        if run_id is None:
            try:
                if config_snapshot is not None:
                    resolved_run_id = _deterministic_run_id_from_config_snapshot(
                        config_snapshot,
                        schema_version=schema_version,
                        strict_serialization=strict_serialization,
                    )
                else:
                    resolved_run_id = _deterministic_run_id_from_inputs(
                        schema_version=schema_version,
                        model_spec=model_spec,
                        probe_spec=probe_spec,
                        dataset_spec=dataset_spec,
                        prompt_set=prompt_set,
                        probe_kwargs=probe_kwargs,
                        strict_serialization=strict_serialization,
                    )
            except StrictSerializationError as exc:
                raise ValueError(
                    "strict_serialization requires JSON-stable values for run_id derivation."
                ) from exc
        else:
            resolved_run_id = run_id

        self.last_run_id = resolved_run_id
        logger.info(
            "Starting sync probe run",
            extra={
                "run_id": resolved_run_id,
                "prompt_count": len(prompt_set),
                "emit_artifacts": emit_run_artifacts,
                "stop_on_error": stop_on_error,
            },
        )
        run_base_time = _deterministic_base_time(resolved_run_id)
        run_started_at, _ = _deterministic_run_times(run_base_time, len(prompt_set))

        root = Path(run_root) if run_root is not None else _default_run_root()
        if run_dir is not None:
            resolved_run_dir = Path(run_dir)
        else:
            resolved_run_dir = root / resolved_run_id

        # Create the run directory up-front so we can emit JSONL incrementally.
        # Policy: fail if non-empty unless overwrite=True.
        if emit_run_artifacts:
            if resume:
                _prepare_run_dir_for_resume(resolved_run_dir, run_root=root)
            else:
                _prepare_run_dir(resolved_run_dir, overwrite=overwrite, run_root=root)
            logger.debug(f"Prepared run directory: {resolved_run_dir}")
            self.last_run_dir = resolved_run_dir
            _ensure_run_sentinel(resolved_run_dir)

            # Snapshot of the fully resolved config (best-effort) for reproducibility.
            if config_snapshot is not None:
                _atomic_write_yaml(
                    resolved_run_dir / "config.resolved.yaml",
                    config_snapshot,
                    strict_serialization=strict_serialization,
                )
        else:
            self.last_run_dir = None

        records_path = resolved_run_dir / "records.jsonl"
        manifest_path = resolved_run_dir / "manifest.json"

        results: list[Optional[dict[str, Any]]] = [None] * len(prompt_set)
        completed = 0
        total = len(prompt_set)
        run_start_time = time.perf_counter()  # For progress tracking

        if emit_run_artifacts and resume:
            existing_records = _read_jsonl_records(records_path, truncate_incomplete=True)
            if existing_records:
                if len(existing_records) > len(prompt_set):
                    raise ValueError(
                        "Existing records.jsonl has more entries than the current prompt_set."
                    )
                for line_index, record in enumerate(existing_records):
                    _validate_resume_record(
                        record,
                        expected_index=line_index,
                        expected_item=prompt_set[line_index],
                        run_id=resolved_run_id,
                        strict_serialization=strict_serialization,
                    )
                    results[line_index] = _result_dict_from_record(
                        record,
                        schema_version=schema_version,
                    )
                completed = len(existing_records)

        # Use ExitStack for deterministic resource cleanup
        with ExitStack() as stack:
            records_fp = None
            if emit_run_artifacts:
                mode = "a" if resume else "x"
                records_fp = stack.enter_context(open(records_path, mode, encoding="utf-8"))

            if use_probe_batch:
                remaining_items = prompt_set[completed:]
                if remaining_items:
                    resolved_batch_workers = batch_workers if batch_workers is not None else 1

                    def batch_progress(current: int, total_batch: int) -> None:
                        _invoke_progress_callback(
                            progress_callback,
                            current=completed + current,
                            total=total,
                            start_time=run_start_time,
                            status="processing",
                        )

                    probe_results = self.probe.run_batch(
                        self.model,
                        remaining_items,
                        max_workers=resolved_batch_workers,
                        progress_callback=batch_progress if progress_callback else None,
                        **probe_kwargs,
                    )

                    stop_error: Optional[RunnerExecutionError] = None
                    for offset, probe_result in enumerate(probe_results):
                        i = completed + offset
                        error_type = None
                        if isinstance(probe_result.metadata, dict):
                            error_type = probe_result.metadata.get("error_type")
                        result_obj = _result_dict_from_probe_result(
                            probe_result,
                            schema_version=schema_version,
                            error_type=error_type,
                        )
                        result_obj["latency_ms"] = None
                        results[i] = result_obj

                        if emit_run_artifacts and records_fp is not None:
                            item_started_at, item_completed_at = _deterministic_item_times(
                                run_base_time,
                                i,
                            )
                            record = _build_result_record(
                                schema_version=schema_version,
                                run_id=resolved_run_id,
                                started_at=item_started_at,
                                completed_at=item_completed_at,
                                model=model_spec,
                                probe=probe_spec,
                                dataset=dataset_spec,
                                item=prompt_set[i],
                                output=probe_result.output,
                                latency_ms=None,
                                store_messages=store_messages,
                                index=i,
                                status=_normalize_status(probe_result.status),
                                error=probe_result.error,
                                error_type=error_type,
                                strict_serialization=strict_serialization,
                            )
                            if validate_output:
                                validator.validate(
                                    registry.RESULT_RECORD,
                                    record,
                                    schema_version=schema_version,
                                    mode=validator_mode,
                                )
                            records_fp.write(
                                _stable_json_dumps(record, strict=strict_serialization) + "\n"
                            )
                            records_fp.flush()

                        if stop_on_error and stop_error is None:
                            status = _normalize_status(probe_result.status)
                            if status != "success":
                                prompt_str = str(prompt_set[i])
                                stop_error = RunnerExecutionError(
                                    reason=probe_result.error or status,
                                    model_id=model_spec.get("model_id"),
                                    probe_id=probe_spec.get("probe_id"),
                                    prompt=prompt_str,
                                    prompt_index=i,
                                    run_id=resolved_run_id,
                                    elapsed_seconds=None,
                                    suggestions=[
                                        "Check the model API credentials and connectivity",
                                        "Verify the prompt format is valid for this model",
                                        "Review the original error message above",
                                    ],
                                )

                    if stop_error is not None:
                        raise stop_error
            else:
                for i in range(completed, len(prompt_set)):
                    item = prompt_set[i]
                    _invoke_progress_callback(
                        progress_callback,
                        current=i,
                        total=total,
                        start_time=run_start_time,
                        current_item=item,
                        current_index=i,
                        status="processing",
                    )

                    item_started_at, item_completed_at = _deterministic_item_times(run_base_time, i)
                    try:
                        output = self.probe.run(self.model, item, **probe_kwargs)
                        probe_result = ProbeResult(
                            input=item,
                            output=output,
                            status=ResultStatus.SUCCESS,
                            latency_ms=None,
                            metadata={},
                        )
                        result_obj = _result_dict_from_probe_result(
                            probe_result,
                            schema_version=schema_version,
                        )
                        results[i] = result_obj

                        if emit_run_artifacts and records_fp is not None:
                            record = _build_result_record(
                                schema_version=schema_version,
                                run_id=resolved_run_id,
                                started_at=item_started_at,
                                completed_at=item_completed_at,
                                model=model_spec,
                                probe=probe_spec,
                                dataset=dataset_spec,
                                item=item,
                                output=output,
                                latency_ms=None,
                                store_messages=store_messages,
                                index=i,
                                status="success",
                                error=None,
                                strict_serialization=strict_serialization,
                            )
                            if validate_output:
                                validator.validate(
                                    registry.RESULT_RECORD,
                                    record,
                                    schema_version=schema_version,
                                    mode=validator_mode,
                                )
                            records_fp.write(
                                _stable_json_dumps(record, strict=strict_serialization) + "\n"
                            )
                            records_fp.flush()

                    except Exception as e:
                        logger.warning(
                            "Probe execution failed",
                            extra={
                                "run_id": resolved_run_id,
                                "index": i,
                                "error_type": type(e).__name__,
                                "error": str(e),
                            },
                            exc_info=True,
                        )
                        probe_result = ProbeResult(
                            input=item,
                            status=ResultStatus.ERROR,
                            error=str(e),
                            latency_ms=None,
                            metadata={"error_type": type(e).__name__},
                        )
                        result_obj = _result_dict_from_probe_result(
                            probe_result,
                            schema_version=schema_version,
                            error_type=type(e).__name__,
                        )
                        results[i] = result_obj

                        if emit_run_artifacts and records_fp is not None:
                            record = _build_result_record(
                                schema_version=schema_version,
                                run_id=resolved_run_id,
                                started_at=item_started_at,
                                completed_at=item_completed_at,
                                model=model_spec,
                                probe=probe_spec,
                                dataset=dataset_spec,
                                item=item,
                                output=None,
                                latency_ms=None,
                                store_messages=store_messages,
                                index=i,
                                status="error",
                                error=e,
                                strict_serialization=strict_serialization,
                            )
                            if validate_output:
                                validator.validate(
                                    registry.RESULT_RECORD,
                                    record,
                                    schema_version=schema_version,
                                    mode=validator_mode,
                                )
                            records_fp.write(
                                _stable_json_dumps(record, strict=strict_serialization) + "\n"
                            )
                            records_fp.flush()

                        if stop_on_error:
                            # Raise enhanced exception with full context
                            prompt_str = str(item) if not isinstance(item, str) else item
                            raise RunnerExecutionError(
                                reason=str(e),
                                model_id=model_spec.get("model_id"),
                                probe_id=probe_spec.get("probe_id"),
                                prompt=prompt_str,
                                prompt_index=i,
                                run_id=resolved_run_id,
                                elapsed_seconds=None,
                                original_error=e,
                                suggestions=[
                                    "Check the model API credentials and connectivity",
                                    "Verify the prompt format is valid for this model",
                                    "Review the original error message above",
                                ],
                            ) from e
            # ExitStack automatically closes records_fp here

        _invoke_progress_callback(
            progress_callback,
            current=total,
            total=total,
            start_time=run_start_time,
            status="complete",
        )

        if any(result is None for result in results):
            raise RuntimeError("Runner did not produce results for all items.")
        final_results = [result for result in results if result is not None]

        self._results = final_results

        _, run_completed_at = _deterministic_run_times(run_base_time, len(final_results))

        if emit_run_artifacts:
            python_version = None if deterministic_artifacts else sys.version.split()[0]
            platform_info = None if deterministic_artifacts else platform.platform()

            def _serialize_manifest(value: Any) -> Any:
                return _serialize_value(value, strict=strict_serialization)

            manifest = {
                "schema_version": schema_version,
                "run_id": resolved_run_id,
                "created_at": run_started_at,
                "started_at": run_started_at,
                "completed_at": run_completed_at,
                "library_version": None,
                "python_version": python_version,
                "platform": platform_info,
                "command": None,
                "model": model_spec,
                "probe": probe_spec,
                "dataset": dataset_spec,
                "record_count": len(final_results),
                "success_count": sum(1 for r in final_results if r.get("status") == "success"),
                "error_count": sum(1 for r in final_results if r.get("status") == "error"),
                "records_file": "records.jsonl",
                "schemas": {
                    registry.RESULT_RECORD: schema_version,
                    registry.RUN_MANIFEST: schema_version,
                    registry.RUNNER_ITEM: schema_version,
                },
                "custom": {},
            }

            # Explicit completion marker was added in schema v1.0.1.
            if _semver_tuple(schema_version) >= (1, 0, 1):
                manifest["run_completed"] = True

            # Fill library version if available
            try:
                import insideLLMs

                manifest["library_version"] = getattr(insideLLMs, "__version__", None)
            except (ImportError, AttributeError):
                pass

            if validate_output:
                validator.validate(
                    registry.RUN_MANIFEST,
                    manifest,
                    schema_version=schema_version,
                    mode=validator_mode,
                )

            _atomic_write_text(
                manifest_path,
                json.dumps(
                    _serialize_manifest(manifest),
                    sort_keys=True,
                    indent=2,
                    default=_serialize_manifest,
                ),
            )

        if validate_output:
            validator.validate(
                registry.RUNNER_OUTPUT,
                {"schema_version": schema_version, "results": final_results},
                schema_version=schema_version,
                mode=validator_mode,
            )

        self.last_experiment = create_experiment_result(
            self.model,
            self.probe,
            final_results,
            config=config_snapshot or {},
            experiment_id=resolved_run_id,
            started_at=run_started_at,
            completed_at=run_completed_at,
            strict_serialization=strict_serialization,
        )

        success_count = sum(1 for r in final_results if r.get("status") == "success")
        error_count = len(final_results) - success_count
        logger.info(
            "Sync probe run completed",
            extra={
                "run_id": resolved_run_id,
                "total_items": len(final_results),
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / len(final_results) if final_results else 0,
            },
        )

        return self.last_experiment if return_experiment else final_results


def _normalize_status(status: Any) -> str:
    """Normalize a status value to a string representation.

    Converts various status representations (ResultStatus enum, other Enums,
    None, or raw strings) into a consistent string format for storage.

    Parameters
    ----------
    status : Any
        The status value to normalize. Can be ResultStatus, other Enum,
        None, or string.

    Returns
    -------
    str
        Normalized status string. Returns "error" for None values,
        the enum value for Enum types, or str(status) otherwise.

    Examples
    --------
    Normalizing ResultStatus enum:

        >>> from insideLLMs.types import ResultStatus
        >>> _normalize_status(ResultStatus.SUCCESS)
        'success'
        >>> _normalize_status(ResultStatus.ERROR)
        'error'

    Handling None (treated as error):

        >>> _normalize_status(None)
        'error'

    String passthrough:

        >>> _normalize_status("success")
        'success'
        >>> _normalize_status("custom_status")
        'custom_status'
    """
    if isinstance(status, ResultStatus):
        return status.value
    if isinstance(status, Enum):
        return str(status.value)
    if status is None:
        return "error"
    return str(status)


def _result_dict_from_probe_result(
    result: ProbeResult,
    *,
    schema_version: str,
    error_type: Optional[str] = None,
) -> dict[str, Any]:
    """Convert a ProbeResult into a dictionary for storage and serialization.

    Transforms a ProbeResult dataclass into a flat dictionary suitable for
    JSONL emission and result aggregation. Handles status normalization
    and optional error type extraction from metadata.

    Parameters
    ----------
    result : ProbeResult
        The probe result to convert. Contains input, output, status,
        latency_ms, error, and metadata fields.
    schema_version : str
        Schema version string to include in the output dictionary.
    error_type : Optional[str], default None
        Explicit error type to include. If None and result has an error,
        attempts to extract error_type from metadata.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - schema_version: The provided schema version
        - input: The original input from the result
        - output: The probe output
        - latency_ms: Execution time in milliseconds
        - status: Normalized status string
        - metadata: Additional metadata dict
        - error: Error message (if present)
        - error_type: Error type string (if present)

    Examples
    --------
    Converting a successful result:

        >>> from insideLLMs.types import ProbeResult, ResultStatus
        >>> result = ProbeResult(
        ...     input={"messages": [{"role": "user", "content": "Hi"}]},
        ...     output="Hello!",
        ...     status=ResultStatus.SUCCESS,
        ...     latency_ms=150.5,
        ...     metadata={"model": "gpt-4"},
        ... )
        >>> d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        >>> d["status"]
        'success'
        >>> d["latency_ms"]
        150.5

    Converting an error result:

        >>> result = ProbeResult(
        ...     input="test",
        ...     status=ResultStatus.ERROR,
        ...     error="API timeout",
        ...     metadata={"error_type": "TimeoutError"},
        ... )
        >>> d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        >>> d["error"]
        'API timeout'
        >>> d["error_type"]
        'TimeoutError'

    See Also
    --------
    _result_dict_from_record : Convert a stored record back to result dict.
    _build_result_record : Build a full JSONL record with additional context.
    """
    status = _normalize_status(result.status)
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if error_type is None:
        error_type = metadata.get("error_type")

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "input": result.input,
        "output": result.output,
        "latency_ms": result.latency_ms,
        "status": status,
        "metadata": metadata,
    }
    if result.error is not None:
        payload["error"] = result.error
    if error_type:
        payload["error_type"] = error_type
    return payload


def _record_index_from_record(
    record: dict[str, Any], default: Optional[int] = None
) -> Optional[int]:
    """Extract the record index from a stored JSONL record.

    Retrieves the record_index from a record's custom metadata. This is used
    during resume operations to verify that existing records form a contiguous
    prefix of the original prompt set.

    Parameters
    ----------
    record : dict[str, Any]
        A JSONL record dictionary, typically read from records.jsonl.
        Expected to have a "custom" key containing a dict with "record_index".
    default : Optional[int], default None
        Value to return if record_index cannot be extracted or parsed.

    Returns
    -------
    Optional[int]
        The record index as an integer, or the default value if not found
        or not parseable.

    Examples
    --------
    Extracting index from a valid record:

        >>> record = {"custom": {"record_index": 5, "replicate_key": "abc"}}
        >>> _record_index_from_record(record)
        5

    Handling missing custom field:

        >>> record = {"input": "test", "output": "result"}
        >>> _record_index_from_record(record, default=0)
        0

    Handling invalid index value:

        >>> record = {"custom": {"record_index": "not_a_number"}}
        >>> _record_index_from_record(record, default=-1)
        -1

    See Also
    --------
    _result_dict_from_record : Extract result data from a stored record.
    """
    custom = record.get("custom")
    if isinstance(custom, dict):
        index = custom.get("record_index")
        if index is not None:
            try:
                return int(index)
            except (TypeError, ValueError):
                return default
    return default


def _result_dict_from_record(
    record: dict[str, Any],
    *,
    schema_version: str,
) -> dict[str, Any]:
    """Convert a stored JSONL record back into a result dictionary.

    Extracts the core result fields from a full JSONL record, producing a
    result dictionary compatible with the runner's internal result format.
    Used during resume operations to reconstruct results from existing records.

    Parameters
    ----------
    record : dict[str, Any]
        A JSONL record dictionary read from records.jsonl. Expected to
        contain input, output, latency_ms, status, and optionally error fields.
    schema_version : str
        Fallback schema version if not present in the record.

    Returns
    -------
    dict[str, Any]
        A result dictionary containing:

        - schema_version: From record or fallback
        - input: Original input data
        - output: Probe output
        - latency_ms: Execution time
        - status: Status string
        - metadata: Empty dict (not preserved in records)
        - error: Error message (if present)
        - error_type: Error type (if present)

    Examples
    --------
    Converting a success record:

        >>> record = {
        ...     "schema_version": "1.0.0",
        ...     "input": {"messages": [{"role": "user", "content": "Hi"}]},
        ...     "output": "Hello!",
        ...     "latency_ms": 150.0,
        ...     "status": "success",
        ... }
        >>> result = _result_dict_from_record(record, schema_version="1.0.0")
        >>> result["status"]
        'success'

    Converting an error record:

        >>> record = {
        ...     "input": "test",
        ...     "status": "error",
        ...     "error": "Connection failed",
        ...     "error_type": "ConnectionError",
        ... }
        >>> result = _result_dict_from_record(record, schema_version="1.0.0")
        >>> result["error"]
        'Connection failed'

    See Also
    --------
    _result_dict_from_probe_result : Convert ProbeResult to result dict.
    _record_index_from_record : Extract index from a record.
    """
    payload: dict[str, Any] = {
        "schema_version": record.get("schema_version", schema_version),
        "input": record.get("input"),
        "output": record.get("output"),
        "latency_ms": record.get("latency_ms"),
        "status": record.get("status"),
        "metadata": {},
    }
    if record.get("error") is not None:
        payload["error"] = record.get("error")
    if record.get("error_type"):
        payload["error_type"] = record.get("error_type")
    return payload


def _truncate_incomplete_jsonl(path: Path) -> None:
    """Truncate a JSONL file to remove any incomplete final line.

    When a run is interrupted, the final line of records.jsonl may be
    incomplete (not terminated with a newline). This function removes
    any such incomplete line to ensure the file contains only valid
    JSON records for safe resume.

    Parameters
    ----------
    path : Path
        Path to the JSONL file to truncate. File must exist.

    Returns
    -------
    None
        The file is modified in place.

    Examples
    --------
    Truncating a file with incomplete final line:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        ...     f.write(b'{"line": 1}\\n{"line": 2}\\n{"incomplete":')
        ...     path = Path(f.name)
        >>> _truncate_incomplete_jsonl(path)
        >>> path.read_text()
        '{"line": 1}\\n{"line": 2}\\n'

    File already complete (no change):

        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        ...     f.write(b'{"line": 1}\\n{"line": 2}\\n')
        ...     path = Path(f.name)
        >>> _truncate_incomplete_jsonl(path)
        >>> path.read_text()
        '{"line": 1}\\n{"line": 2}\\n'

    Notes
    -----
    This function operates at the byte level for efficiency and handles
    files without any newlines by truncating to empty.

    See Also
    --------
    _read_jsonl_records : Read records with optional truncation.
    """
    data = path.read_bytes()
    if not data:
        return
    if data.endswith(b"\n"):
        return
    cutoff = data.rfind(b"\n")
    if cutoff == -1:
        path.write_bytes(b"")
        return
    path.write_bytes(data[: cutoff + 1])


def _read_jsonl_records(path: Path, *, truncate_incomplete: bool = False) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Parses a JSONL (JSON Lines) file and returns a list of record dictionaries.
    Optionally truncates incomplete final lines before reading, which is useful
    for resuming interrupted runs.

    Parameters
    ----------
    path : Path
        Path to the JSONL file. If the file doesn't exist, returns empty list.
    truncate_incomplete : bool, default False
        If True, truncate any incomplete final line before reading. This
        ensures safe parsing after an interrupted write operation.

    Returns
    -------
    list[dict[str, Any]]
        List of parsed record dictionaries. Empty lines are skipped.
        Non-dict JSON values are skipped.

    Raises
    ------
    ValueError
        If any non-empty line contains invalid JSON.

    Examples
    --------
    Reading a valid JSONL file:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        ...     f.write('{"id": 1, "status": "success"}\\n')
        ...     f.write('{"id": 2, "status": "error"}\\n')
        ...     path = Path(f.name)
        >>> records = _read_jsonl_records(path)
        >>> len(records)
        2
        >>> records[0]["id"]
        1

    Reading non-existent file:

        >>> _read_jsonl_records(Path("/nonexistent/file.jsonl"))
        []

    Reading with truncation (safe resume):

        >>> records = _read_jsonl_records(path, truncate_incomplete=True)

    Notes
    -----
    Empty lines are silently skipped. This allows JSONL files with trailing
    newlines or accidental blank lines to be parsed without error.

    See Also
    --------
    _truncate_incomplete_jsonl : Truncate incomplete final line.
    _build_result_record : Build records for writing to JSONL.
    """
    if not path.exists():
        return []
    if truncate_incomplete:
        _truncate_incomplete_jsonl(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record on line {line_no} in {path}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _validate_resume_record(
    record: dict[str, Any],
    *,
    expected_index: int,
    expected_item: Any,
    run_id: Optional[str],
    strict_serialization: bool = False,
) -> None:
    """Validate a stored record matches the expected prompt item.

    Ensures resume only proceeds when the existing records are a contiguous
    prefix and inputs match the current prompt_set. This prevents silently
    mixing results from different inputs.
    """
    record_index = _record_index_from_record(record, default=expected_index)
    if record_index != expected_index:
        raise ValueError("Existing records are not a contiguous prefix; cannot resume safely.")

    if run_id is not None:
        record_run_id = record.get("run_id")
        if record_run_id is not None and record_run_id != run_id:
            raise ValueError(
                f"Existing record run_id '{record_run_id}' does not match current run_id '{run_id}'."
            )

    try:
        expected_fp = _fingerprint_value(expected_item, strict=strict_serialization)
        record_fp = _fingerprint_value(record.get("input"), strict=strict_serialization)
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable prompts when validating resume records."
        ) from exc
    if expected_fp != record_fp:
        raise ValueError(
            f"Existing record input mismatch at index {expected_index}; cannot resume safely."
        )


# Module-level constants for deterministic timestamp generation.
# These are used to produce reproducible timestamps from run IDs,
# ensuring that the same configuration always produces the same
# artifact timestamps for testing and caching purposes.
_DETERMINISTIC_TIME_BASE = datetime(2000, 1, 1, tzinfo=timezone.utc)
_DETERMINISTIC_TIME_RANGE_SECONDS = 10 * 365 * 24 * 60 * 60  # 10 years in seconds


def _deterministic_hash(payload: Any, *, strict: bool = False) -> str:
    """Generate a deterministic SHA-256 hash of any payload.

    Produces a consistent hash for equivalent data structures by using
    stable JSON serialization. Used for generating run IDs, experiment IDs,
    and replicate keys.

    Parameters
    ----------
    payload : Any
        Data to hash. Can be any JSON-serializable structure.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 hash.

    Examples
    --------
    Hashing a configuration dict:

        >>> config = {"model": "gpt-4", "probe": "factuality"}
        >>> hash1 = _deterministic_hash(config)
        >>> hash2 = _deterministic_hash(config)
        >>> hash1 == hash2
        True
        >>> len(hash1)
        64

    Same data, different order, same hash:

        >>> _deterministic_hash({"a": 1, "b": 2}) == _deterministic_hash({"b": 2, "a": 1})
        True

    See Also
    --------
    _stable_json_dumps : Stable JSON serialization.
    _deterministic_run_id_from_inputs : Generate run IDs using this hash.
    """
    return hashlib.sha256(_stable_json_dumps(payload, strict=strict).encode("utf-8")).hexdigest()


def _default_run_root() -> Path:
    """Get the default root directory for run artifacts.

    Returns the directory where run artifacts (records.jsonl, manifest.json)
    are stored by default. Can be overridden via the INSIDELLMS_RUN_ROOT
    environment variable.

    Returns
    -------
    Path
        Default run root directory. Falls back to ~/.insidellms/runs.

    Examples
    --------
    Default location:

        >>> import os
        >>> if "INSIDELLMS_RUN_ROOT" not in os.environ:
        ...     root = _default_run_root()
        ...     str(root).endswith(".insidellms/runs")
        True

    With environment variable:

        >>> import os
        >>> os.environ["INSIDELLMS_RUN_ROOT"] = "/custom/runs"
        >>> _default_run_root()
        PosixPath('/custom/runs')
        >>> del os.environ["INSIDELLMS_RUN_ROOT"]

    See Also
    --------
    _prepare_run_dir : Create and validate a run directory.
    """
    env_root = os.environ.get("INSIDELLMS_RUN_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / ".insidellms" / "runs"


def _semver_tuple(version: str) -> tuple[int, int, int]:
    """Convert a semver string to a tuple for comparison.

    Parses a semantic version string (e.g., "1.2.3") into a tuple of integers
    for numeric comparison. Delegates to insideLLMs.schemas.registry.semver_tuple.

    Parameters
    ----------
    version : str
        Semantic version string in "major.minor.patch" format.

    Returns
    -------
    tuple[int, int, int]
        Tuple of (major, minor, patch) version numbers.

    Examples
    --------
    Parsing version strings:

        >>> _semver_tuple("1.0.0")
        (1, 0, 0)
        >>> _semver_tuple("2.3.4")
        (2, 3, 4)

    Version comparison:

        >>> _semver_tuple("1.0.1") >= (1, 0, 1)
        True
        >>> _semver_tuple("1.0.0") < (1, 0, 1)
        True

    See Also
    --------
    insideLLMs.schemas.registry.semver_tuple : The underlying implementation.
    """
    from insideLLMs.schemas.registry import semver_tuple

    return semver_tuple(version)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically using a temp file and rename.

    Ensures that the file is either fully written or not modified at all,
    preventing partial writes from interruptions. Delegates to
    insideLLMs.resources.atomic_write_text.

    Parameters
    ----------
    path : Path
        Destination file path.
    text : str
        Text content to write.

    Examples
    --------
    Writing a manifest atomically:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = Path(d) / "manifest.json"
        ...     _atomic_write_text(path, '{"key": "value"}')
        ...     path.read_text()
        '{"key": "value"}'

    Notes
    -----
    Uses a temporary file with rename to achieve atomicity. On POSIX systems,
    rename is atomic when source and destination are on the same filesystem.

    See Also
    --------
    _atomic_write_yaml : Atomic YAML file writing.
    insideLLMs.resources.atomic_write_text : The underlying implementation.
    """
    from insideLLMs.resources import atomic_write_text

    atomic_write_text(path, text)


def _atomic_write_yaml(path: Path, data: Any, *, strict_serialization: bool = False) -> None:
    """Write data to a YAML file atomically.

    Serializes data to YAML format and writes it atomically to ensure
    the file is either fully written or not modified. Used for writing
    config.resolved.yaml files for reproducibility.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : Any
        Data to serialize as YAML. Passed through _serialize_value first.

    Examples
    --------
    Writing configuration atomically:

        >>> from pathlib import Path
        >>> import tempfile
        >>> config = {"model": {"type": "openai", "args": {"model_name": "gpt-4"}}}
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = Path(d) / "config.yaml"
        ...     _atomic_write_yaml(path, config)
        ...     "model:" in path.read_text()
        True

    See Also
    --------
    _atomic_write_text : Atomic text file writing.
    _serialize_value : Value serialization for YAML compatibility.
    """
    content = yaml.safe_dump(
        _serialize_value(data, strict=strict_serialization),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )
    _atomic_write_text(path, content)


def _ensure_run_sentinel(run_dir_path: Path) -> None:
    """Create a marker file to identify run directories.

    Creates a .insidellms_run sentinel file in the run directory. This marker
    is used to verify that a directory is a legitimate insideLLMs run directory
    before allowing destructive operations like overwrite.

    Parameters
    ----------
    run_dir_path : Path
        Path to the run directory.

    Examples
    --------
    Creating a sentinel:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     run_dir = Path(d) / "test_run"
        ...     run_dir.mkdir()
        ...     _ensure_run_sentinel(run_dir)
        ...     (run_dir / ".insidellms_run").exists()
        True

    Notes
    -----
    This function never raises exceptions. If the sentinel cannot be created
    (e.g., due to permissions), the run continues without the marker.

    See Also
    --------
    _prepare_run_dir : Uses sentinel for overwrite safety.
    """
    marker = run_dir_path / ".insidellms_run"
    if not marker.exists():
        try:
            marker.write_text("insideLLMs run directory\n", encoding="utf-8")
        except (IOError, OSError):
            # Don't fail the run due to marker write issues.
            pass


def _normalize_info_obj_to_dict(info_obj: Any) -> dict[str, Any]:
    """Normalize model/probe info object to a dictionary.

    Converts various info object formats (dict, dataclass, pydantic model)
    to a standard dictionary representation. Used for extracting model and
    probe metadata in a format-agnostic way.

    Parameters
    ----------
    info_obj : Any
        Info object to normalize. Can be dict, dataclass, pydantic v1/v2
        model, or any other type.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the info object. Returns empty dict
        for None or unrecognized types.

    Examples
    --------
    Normalizing a dataclass:

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class ModelInfo:
        ...     name: str
        ...     version: str
        >>> info = ModelInfo(name="gpt-4", version="1.0")
        >>> _normalize_info_obj_to_dict(info)
        {'name': 'gpt-4', 'version': '1.0'}

    Passthrough for dict:

        >>> _normalize_info_obj_to_dict({"key": "value"})
        {'key': 'value'}

    Handling None:

        >>> _normalize_info_obj_to_dict(None)
        {}

    Notes
    -----
    Supports both pydantic v1 (.dict()) and pydantic v2 (.model_dump())
    for backward compatibility across different pydantic versions.

    See Also
    --------
    _build_model_spec : Uses this to extract model info.
    _build_probe_spec : Uses this to extract probe info.
    """
    if info_obj is None:
        return {}
    if isinstance(info_obj, dict):
        return info_obj
    if is_dataclass(info_obj) and not isinstance(info_obj, type):
        return asdict(info_obj)
    if hasattr(info_obj, "dict") and callable(getattr(info_obj, "dict")):
        # pydantic v1 compatibility
        try:
            return info_obj.dict()
        except Exception:
            return {}
    if hasattr(info_obj, "model_dump") and callable(getattr(info_obj, "model_dump")):
        # pydantic v2 compatibility
        try:
            return info_obj.model_dump()
        except Exception:
            return {}
    return {}


def _build_model_spec(model: Any) -> dict[str, Any]:
    """Build a model specification dictionary from a model object.

    Extracts key identifying information from a model to create a standardized
    specification dictionary for manifest generation and result records.

    Parameters
    ----------
    model : Any
        Model object with optional info() method or attributes.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - model_id: Identifier for the model
        - provider: Model provider/type (e.g., "openai", "anthropic")
        - params: Full info dictionary as parameters

    Examples
    --------
    Building spec from a model:

        >>> class MockModel:
        ...     def info(self):
        ...         return {"model_id": "gpt-4", "provider": "openai"}
        >>> spec = _build_model_spec(MockModel())
        >>> spec["model_id"]
        'gpt-4'
        >>> spec["provider"]
        'openai'

    Fallback to class name:

        >>> class CustomModel:
        ...     pass
        >>> spec = _build_model_spec(CustomModel())
        >>> spec["model_id"]
        'CustomModel'

    See Also
    --------
    _build_probe_spec : Build probe specification.
    _normalize_info_obj_to_dict : Normalize info objects.
    """
    info_obj: Any = {}
    try:
        info_obj = model.info() if hasattr(model, "info") else {}
    except (AttributeError, TypeError):
        info_obj = {}

    info = _normalize_info_obj_to_dict(info_obj)

    model_id = (
        info.get("model_id")
        or info.get("id")
        or info.get("name")
        or info.get("model_name")
        or getattr(model, "model_id", None)
        or getattr(model, "name", None)
        or model.__class__.__name__
    )

    provider = info.get("provider") or info.get("type") or info.get("model_type")
    provider = str(provider) if provider is not None else None
    return {"model_id": str(model_id), "provider": provider, "params": info}


def _build_probe_spec(probe: Any) -> dict[str, Any]:
    """Build a probe specification dictionary from a probe object.

    Extracts key identifying information from a probe to create a standardized
    specification dictionary for manifest generation and result records.

    Parameters
    ----------
    probe : Any
        Probe object with optional name, probe_id, version attributes.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - probe_id: Identifier for the probe
        - probe_version: Version string if available
        - params: Empty dict (reserved for future use)

    Examples
    --------
    Building spec from a probe:

        >>> class MockProbe:
        ...     name = "factuality"
        ...     version = "1.0.0"
        >>> spec = _build_probe_spec(MockProbe())
        >>> spec["probe_id"]
        'factuality'
        >>> spec["probe_version"]
        '1.0.0'

    Fallback to class name:

        >>> class CustomProbe:
        ...     pass
        >>> spec = _build_probe_spec(CustomProbe())
        >>> spec["probe_id"]
        'CustomProbe'

    See Also
    --------
    _build_model_spec : Build model specification.
    _build_dataset_spec : Build dataset specification.
    """
    probe_id = (
        getattr(probe, "name", None) or getattr(probe, "probe_id", None) or probe.__class__.__name__
    )
    probe_version = getattr(probe, "version", None) or getattr(probe, "probe_version", None)
    return {"probe_id": str(probe_id), "probe_version": probe_version, "params": {}}


def _build_dataset_spec(dataset_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build a dataset specification dictionary from dataset info.

    Extracts key identifying information from dataset configuration to create
    a standardized specification dictionary for manifest generation and
    result records.

    Parameters
    ----------
    dataset_info : Optional[dict[str, Any]]
        Dataset configuration dictionary. Can contain name, path, version,
        hash, provenance, format, etc.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - dataset_id: Identifier for the dataset (from name, dataset, or path)
        - dataset_version: Version string if available
        - dataset_hash: Content hash if available
        - provenance: Source information (from provenance, source, or format)

    Examples
    --------
    Building spec from HuggingFace dataset config:

        >>> config = {"name": "cais/mmlu", "version": "1.0", "format": "hf"}
        >>> spec = _build_dataset_spec(config)
        >>> spec["dataset_id"]
        'cais/mmlu'
        >>> spec["provenance"]
        'hf'

    Building spec from local file config:

        >>> config = {"path": "/data/test.jsonl", "format": "jsonl"}
        >>> spec = _build_dataset_spec(config)
        >>> spec["dataset_id"]
        '/data/test.jsonl'

    Handling None:

        >>> spec = _build_dataset_spec(None)
        >>> spec["dataset_id"] is None
        True

    See Also
    --------
    _build_model_spec : Build model specification.
    _build_probe_spec : Build probe specification.
    """
    di = dataset_info or {}
    dataset_id = di.get("name") or di.get("dataset") or di.get("path")
    dataset_version = di.get("version") or di.get("dataset_version")
    if di.get("revision"):
        if dataset_version:
            dataset_version = f"{dataset_version}@{di.get('revision')}"
        else:
            dataset_version = str(di.get("revision"))
    if di.get("split"):
        if dataset_version:
            dataset_version = f"{dataset_version}::{di.get('split')}"
        else:
            dataset_version = str(di.get("split"))
    dataset_hash = di.get("hash") or di.get("dataset_hash")
    provenance = di.get("provenance") or di.get("source") or di.get("format")
    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "dataset_version": str(dataset_version) if dataset_version is not None else None,
        "dataset_hash": str(dataset_hash) if dataset_hash is not None else None,
        "provenance": str(provenance) if provenance is not None else None,
    }


def _hash_prompt_set(prompt_set: list[Any], *, strict: bool = False) -> str:
    """Generate a deterministic hash of a prompt set.

    Creates a content-based hash of all prompts in a set, used for
    generating run IDs and detecting duplicate datasets.

    Parameters
    ----------
    prompt_set : list[Any]
        List of prompts to hash. Each prompt is serialized to stable JSON.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 hash of the concatenated prompts.

    Examples
    --------
    Hashing a prompt set:

        >>> prompts = [
        ...     {"messages": [{"role": "user", "content": "Hello"}]},
        ...     {"messages": [{"role": "user", "content": "World"}]},
        ... ]
        >>> hash1 = _hash_prompt_set(prompts)
        >>> hash2 = _hash_prompt_set(prompts)
        >>> hash1 == hash2
        True
        >>> len(hash1)
        64

    Order matters:

        >>> prompts_reversed = list(reversed(prompts))
        >>> _hash_prompt_set(prompts) != _hash_prompt_set(prompts_reversed)
        True

    See Also
    --------
    _deterministic_hash : General-purpose hashing.
    _deterministic_run_id_from_inputs : Uses this for run ID generation.
    """
    hasher = hashlib.sha256()
    for index, item in enumerate(prompt_set):
        try:
            hasher.update(_stable_json_dumps(item, strict=strict).encode("utf-8"))
        except StrictSerializationError as exc:
            raise StrictSerializationError(
                f"Non-deterministic prompt at index {index}: {exc}"
            ) from exc
        hasher.update(b"\n")
    return hasher.hexdigest()


def _replicate_key(
    *,
    model_spec: dict[str, Any],
    probe_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    example_id: str,
    record_index: int,
    input_hash: Optional[str],
    strict_serialization: bool = False,
) -> str:
    """Generate a unique replicate key for a result record.

    Creates a deterministic key that uniquely identifies a specific example
    within a specific run configuration. Used for deduplication and result
    tracking across multiple runs.

    Parameters
    ----------
    model_spec : dict[str, Any]
        Model specification dictionary.
    probe_spec : dict[str, Any]
        Probe specification dictionary.
    dataset_spec : dict[str, Any]
        Dataset specification dictionary.
    example_id : str
        Unique identifier for the example within the dataset.
    record_index : int
        Index position of this record in the result set.
    input_hash : Optional[str]
        Hash of the input data (messages hash or input fingerprint).

    Returns
    -------
    str
        16-character hexadecimal replicate key.

    Examples
    --------
    Generating a replicate key:

        >>> model_spec = {"model_id": "gpt-4", "provider": "openai"}
        >>> probe_spec = {"probe_id": "factuality"}
        >>> dataset_spec = {"dataset_id": "test_data"}
        >>> key = _replicate_key(
        ...     model_spec=model_spec,
        ...     probe_spec=probe_spec,
        ...     dataset_spec=dataset_spec,
        ...     example_id="ex_001",
        ...     record_index=0,
        ...     input_hash="abc123def456",
        ... )
        >>> len(key)
        16

    See Also
    --------
    _deterministic_hash : Hash generation.
    _build_result_record : Uses replicate keys in records.
    """
    payload = {
        "model_id": model_spec.get("model_id"),
        "probe_id": probe_spec.get("probe_id"),
        "dataset_id": dataset_spec.get("dataset_id"),
        "dataset_version": dataset_spec.get("dataset_version"),
        "dataset_hash": dataset_spec.get("dataset_hash"),
        "example_id": example_id,
        "record_index": record_index,
        "input_hash": input_hash,
    }
    return _deterministic_hash(payload, strict=strict_serialization)[:16]


def _deterministic_run_id_from_config_snapshot(
    config_snapshot: dict[str, Any],
    *,
    schema_version: str,
    strict_serialization: bool = False,
) -> str:
    """Generate a deterministic run ID from a configuration snapshot.

    Creates a unique run ID based on the full configuration, ensuring that
    identical configurations produce identical run IDs. This enables
    reproducibility and caching of results.

    Parameters
    ----------
    config_snapshot : dict[str, Any]
        Full configuration dictionary including model, probe, and dataset.
    schema_version : str
        Schema version to include in the hash.

    Returns
    -------
    str
        32-character hexadecimal run ID.

    Examples
    --------
    Generating a run ID:

        >>> config = {
        ...     "model": {"type": "openai", "args": {"model_name": "gpt-4"}},
        ...     "probe": {"type": "factuality"},
        ...     "dataset": {"format": "jsonl", "path": "test.jsonl"},
        ... }
        >>> run_id = _deterministic_run_id_from_config_snapshot(
        ...     config, schema_version="1.0.0"
        ... )
        >>> len(run_id)
        32

    Same config produces same ID:

        >>> run_id2 = _deterministic_run_id_from_config_snapshot(
        ...     config, schema_version="1.0.0"
        ... )
        >>> run_id == run_id2
        True

    See Also
    --------
    _deterministic_run_id_from_inputs : Generate from individual components.
    _deterministic_hash : Hash generation.
    """
    payload = {"schema_version": schema_version, "config": config_snapshot}
    return _deterministic_hash(payload, strict=strict_serialization)[:32]


def _deterministic_run_id_from_inputs(
    *,
    schema_version: str,
    model_spec: dict[str, Any],
    probe_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    prompt_set: list[Any],
    probe_kwargs: dict[str, Any],
    strict_serialization: bool = False,
) -> str:
    """Generate a deterministic run ID from individual input components.

    Creates a unique run ID based on model, probe, dataset, prompts, and
    additional arguments. Used when running without a config file.

    Parameters
    ----------
    schema_version : str
        Schema version to include in the hash.
    model_spec : dict[str, Any]
        Model specification dictionary.
    probe_spec : dict[str, Any]
        Probe specification dictionary.
    dataset_spec : dict[str, Any]
        Dataset specification dictionary.
    prompt_set : list[Any]
        List of prompts to be processed.
    probe_kwargs : dict[str, Any]
        Additional keyword arguments passed to the probe.

    Returns
    -------
    str
        32-character hexadecimal run ID.

    Examples
    --------
    Generating a run ID from components:

        >>> run_id = _deterministic_run_id_from_inputs(
        ...     schema_version="1.0.0",
        ...     model_spec={"model_id": "gpt-4"},
        ...     probe_spec={"probe_id": "factuality"},
        ...     dataset_spec={"dataset_id": "test"},
        ...     prompt_set=[{"messages": [{"role": "user", "content": "Hi"}]}],
        ...     probe_kwargs={},
        ... )
        >>> len(run_id)
        32

    See Also
    --------
    _deterministic_run_id_from_config_snapshot : Generate from config dict.
    _hash_prompt_set : Hash prompt set for inclusion.
    """
    payload = {
        "schema_version": schema_version,
        "model": model_spec,
        "probe": probe_spec,
        "dataset": dataset_spec,
        "prompt_set_hash": _hash_prompt_set(prompt_set, strict=strict_serialization),
        "probe_kwargs": probe_kwargs,
    }
    return _deterministic_hash(payload, strict=strict_serialization)[:32]


def _deterministic_base_time(run_id: str) -> datetime:
    """Generate a deterministic base timestamp from a run ID.

    Creates a reproducible timestamp derived from the run ID hash. This ensures
    that result records have consistent timestamps across repeated runs with
    the same configuration, enabling deterministic testing.

    Parameters
    ----------
    run_id : str
        The run identifier to derive the timestamp from.

    Returns
    -------
    datetime
        A UTC datetime within a 10-year range starting from 2000-01-01.

    Examples
    --------
    Generating deterministic timestamps:

        >>> from datetime import timezone
        >>> base = _deterministic_base_time("abc123")
        >>> base.tzinfo == timezone.utc
        True
        >>> base.year >= 2000 and base.year <= 2010
        True

    Same run ID produces same timestamp:

        >>> _deterministic_base_time("abc123") == _deterministic_base_time("abc123")
        True

    See Also
    --------
    _deterministic_item_times : Generate timestamps for individual items.
    _deterministic_run_times : Generate start/end timestamps for runs.
    """
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    seconds = int.from_bytes(digest[:8], "big") % _DETERMINISTIC_TIME_RANGE_SECONDS
    return _DETERMINISTIC_TIME_BASE + timedelta(seconds=seconds)


def _deterministic_item_times(base_time: datetime, index: int) -> tuple[datetime, datetime]:
    """Generate deterministic timestamps for an individual result item.

    Creates reproducible start and completion timestamps for a single item
    based on its index. Each item gets a unique timestamp pair offset by
    microseconds from the base time.

    Parameters
    ----------
    base_time : datetime
        The base timestamp for the run (from _deterministic_base_time).
    index : int
        Zero-based index of the item in the result set.

    Returns
    -------
    tuple[datetime, datetime]
        Tuple of (started_at, completed_at) timestamps.

    Examples
    --------
    Generating item timestamps:

        >>> from datetime import datetime, timezone
        >>> base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> started, completed = _deterministic_item_times(base, 0)
        >>> started == base
        True
        >>> (completed - started).microseconds
        1

    Sequential items have sequential timestamps:

        >>> t0_start, t0_end = _deterministic_item_times(base, 0)
        >>> t1_start, t1_end = _deterministic_item_times(base, 1)
        >>> t1_start > t0_end
        True

    See Also
    --------
    _deterministic_base_time : Generate the base timestamp.
    _deterministic_run_times : Generate run-level timestamps.
    """
    offset = index * 2
    started_at = base_time + timedelta(microseconds=offset)
    completed_at = base_time + timedelta(microseconds=offset + 1)
    return started_at, completed_at


def _deterministic_run_times(base_time: datetime, total: int) -> tuple[datetime, datetime]:
    """Generate deterministic start and end timestamps for a run.

    Creates reproducible run-level timestamps based on the base time and
    total number of items. The run end time is set to encompass all item
    timestamps.

    Parameters
    ----------
    base_time : datetime
        The base timestamp for the run (from _deterministic_base_time).
    total : int
        Total number of items in the run.

    Returns
    -------
    tuple[datetime, datetime]
        Tuple of (started_at, completed_at) timestamps for the run.

    Examples
    --------
    Generating run timestamps:

        >>> from datetime import datetime, timezone
        >>> base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> started, completed = _deterministic_run_times(base, 100)
        >>> started == base
        True
        >>> completed > started
        True

    Empty run:

        >>> started, completed = _deterministic_run_times(base, 0)
        >>> (completed - started).microseconds
        1

    See Also
    --------
    _deterministic_base_time : Generate the base timestamp.
    _deterministic_item_times : Generate item-level timestamps.
    """
    if total < 0:
        total = 0
    completed_offset = total * 2 + 1
    started_at = base_time
    completed_at = base_time + timedelta(microseconds=completed_offset)
    return started_at, completed_at


def _coerce_model_info(model: Model) -> ModelInfo:
    """Coerce model info to a standardized ModelInfo dataclass.

    Converts various model info formats into the canonical ModelInfo type.
    Handles models that return dicts, dataclasses, pydantic models, or
    already-canonical ModelInfo instances.

    Parameters
    ----------
    model : Model
        Model object with optional info() method.

    Returns
    -------
    ModelInfo
        Standardized ModelInfo instance with all fields populated.

    Examples
    --------
    Coercing from a model with info():

        >>> class MockModel:
        ...     def info(self):
        ...         return {"name": "gpt-4", "provider": "openai"}
        >>> info = _coerce_model_info(MockModel())
        >>> info.name
        'gpt-4'
        >>> info.provider
        'openai'

    Fallback for models without info():

        >>> class SimpleModel:
        ...     pass
        >>> info = _coerce_model_info(SimpleModel())
        >>> info.name
        'SimpleModel'

    See Also
    --------
    ModelInfo : The target dataclass type.
    _normalize_info_obj_to_dict : Convert info objects to dicts.
    """

    info_obj: Any = {}
    try:
        info_obj = model.info() or {}
    except Exception:
        info_obj = {}

    # If it's already the canonical type, keep it.
    if isinstance(info_obj, ModelInfo):
        return info_obj

    info = _normalize_info_obj_to_dict(info_obj)

    name = info.get("name") or getattr(model, "name", None) or model.__class__.__name__
    provider = (
        info.get("provider")
        or info.get("type")
        or info.get("model_type")
        or model.__class__.__name__.replace("Model", "")
    )
    model_id = (
        info.get("model_id")
        or info.get("id")
        or info.get("model_name")
        or getattr(model, "model_id", None)
        or name
    )

    extra: dict[str, Any] = {}
    if isinstance(info.get("extra"), dict):
        extra.update(info.get("extra") or {})
    for k, v in info.items():
        if k in {
            "name",
            "provider",
            "model_id",
            "max_tokens",
            "supports_streaming",
            "supports_chat",
            "extra",
        }:
            continue
        extra[k] = v

    return ModelInfo(
        name=str(name),
        provider=str(provider),
        model_id=str(model_id),
        max_tokens=info.get("max_tokens"),
        supports_streaming=bool(info.get("supports_streaming", False)),
        supports_chat=bool(info.get("supports_chat", True)),
        extra=extra,
    )


def _build_result_record(
    *,
    schema_version: str,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    model: dict[str, Any],
    probe: dict[str, Any],
    dataset: dict[str, Any],
    item: Any,
    output: Any,
    latency_ms: Optional[float],
    store_messages: bool,
    index: int,
    status: str,
    error: Optional[Union[BaseException, str]],
    error_type: Optional[str] = None,
    strict_serialization: bool = False,
) -> dict[str, Any]:
    """Build a ResultRecord-shaped dict for JSONL emission.

    This is intentionally best-effort and tolerant of arbitrary input/output
    structures so the runner can act as a harness.
    """

    import hashlib

    # Identify example
    example_id = None
    if isinstance(item, dict):
        example_id = item.get("example_id") or item.get("id")
    example_id = str(example_id) if example_id is not None else str(index)

    # Normalize messages for storage/hashing
    messages_raw = None
    if isinstance(item, dict) and isinstance(item.get("messages"), list):
        messages_raw = item.get("messages")

    normalized_messages: Optional[list[dict[str, Any]]] = None
    if messages_raw is not None:
        normalized_messages = []
        for m in messages_raw:
            if isinstance(m, dict):
                role = m.get("role") or "user"
                normalized_messages.append({"role": str(role), "content": m.get("content")})
            else:
                normalized_messages.append({"role": "user", "content": m})

    messages_hash = None
    if normalized_messages is not None:
        try:
            messages_hash = hashlib.sha256(
                _stable_json_dumps(normalized_messages, strict=strict_serialization).encode("utf-8")
            ).hexdigest()
        except StrictSerializationError as exc:
            if strict_serialization:
                raise ValueError(
                    "strict_serialization requires JSON-stable message content."
                ) from exc
            messages_hash = None
        except (ValueError, TypeError):
            messages_hash = None

    try:
        input_fingerprint = _fingerprint_value(item, strict=strict_serialization)
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable inputs for fingerprinting."
        ) from exc
    replicate_key = _replicate_key(
        model_spec=model,
        probe_spec=probe,
        dataset_spec=dataset,
        example_id=example_id,
        record_index=index,
        input_hash=messages_hash or input_fingerprint,
        strict_serialization=strict_serialization,
    )

    # Derive output_text / scoring fields (best-effort)
    output_text = None
    if isinstance(output, str):
        output_text = output
    elif isinstance(output, dict):
        output_text = output.get("output_text") or output.get("text")

    output_fingerprint = None
    if output is not None and not isinstance(output, str):
        try:
            output_fingerprint = _fingerprint_value(output, strict=strict_serialization)
        except StrictSerializationError as exc:
            raise ValueError(
                "strict_serialization requires JSON-stable structured outputs."
            ) from exc

    scores: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    primary_metric = None
    if isinstance(output, dict):
        if isinstance(output.get("scores"), dict):
            scores = output.get("scores")
        elif output.get("score") is not None:
            scores = {"score": output.get("score")}
        if isinstance(output.get("usage"), dict):
            usage = output.get("usage")
        primary_metric = output.get("primary_metric")

    err_str: Optional[str] = None
    err_type: Optional[str] = None
    if isinstance(error, BaseException):
        err_str = str(error)
        err_type = type(error).__name__
    elif isinstance(error, str):
        err_str = error
        err_type = error_type
    else:
        err_type = error_type

    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "model": model,
        "probe": probe,
        "dataset": dataset,
        "example_id": example_id,
        "input": item,
        "messages": normalized_messages
        if (store_messages and normalized_messages is not None)
        else None,
        "messages_hash": messages_hash,
        "messages_storage": None,
        "output": output,
        "output_text": output_text,
        "scores": scores,
        "primary_metric": primary_metric,
        "usage": usage,
        "latency_ms": latency_ms,
        "status": status,
        "error": err_str,
        "error_type": err_type,
        "custom": {
            "replicate_key": replicate_key,
            "record_index": index,
            "output_fingerprint": output_fingerprint,
        },
    }


class AsyncProbeRunner(_RunnerBase):
    """Asynchronous runner for concurrent probe execution.

    Use this when you need to run probes concurrently, which is especially
    useful for API-based models that can handle multiple requests in parallel.
    The runner uses asyncio to manage concurrent execution with a configurable
    concurrency limit.

    This class inherits from :class:`_RunnerBase` and provides the same
    properties for accessing run statistics and results.

    Attributes
    ----------
    model : Model
        The model instance being tested.
    probe : Probe
        The probe instance being run.
    last_run_id : Optional[str]
        Unique identifier of the last completed run.
    last_run_dir : Optional[Path]
        Directory where the last run's artifacts were stored.
    last_experiment : Optional[ExperimentResult]
        The ExperimentResult from the last completed run.
    success_rate : float
        Proportion of successful results from the last run.
    error_count : int
        Number of errors encountered in the last run.

    Examples
    --------
    Basic async execution:

        >>> import asyncio
        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.probes import FactualityProbe
        >>> from insideLLMs.runtime.runner import AsyncProbeRunner
        >>>
        >>> async def main():
        ...     model = OpenAIModel(model_name="gpt-4")
        ...     probe = FactualityProbe()
        ...     runner = AsyncProbeRunner(model, probe)
        ...
        ...     prompts = [{"messages": [{"role": "user", "content": f"Q{i}"}]}
        ...                for i in range(100)]
        ...     results = await runner.run(prompts, concurrency=10)
        ...     print(f"Success rate: {runner.success_rate:.1%}")
        ...     return results
        >>>
        >>> results = asyncio.run(main())

    With progress tracking:

        >>> async def main_with_progress():
        ...     runner = AsyncProbeRunner(model, probe)
        ...
        ...     def on_progress(current, total):
        ...         print(f"\\rProgress: {current}/{total}", end="")
        ...
        ...     results = await runner.run(
        ...         prompts,
        ...         concurrency=5,
        ...         progress_callback=on_progress
        ...     )
        ...     return results

    Using RunConfig:

        >>> from insideLLMs.config_types import RunConfig
        >>>
        >>> async def main_with_config():
        ...     config = RunConfig(
        ...         concurrency=10,
        ...         emit_run_artifacts=True,
        ...         return_experiment=True,
        ...     )
        ...     runner = AsyncProbeRunner(model, probe)
        ...     experiment = await runner.run(prompts, config=config)
        ...     return experiment

    Comparing sync vs async performance:

        >>> # For 100 API calls with ~1s latency each:
        >>> # Sync ProbeRunner: ~100 seconds
        >>> # AsyncProbeRunner (concurrency=10): ~10 seconds

    See Also
    --------
    ProbeRunner : Synchronous runner for sequential execution.
    run_probe_async : Convenience function wrapping AsyncProbeRunner.
    """

    async def run(
        self,
        prompt_set: list[Any],
        *,
        config: Optional[RunConfig] = None,
        stop_on_error: Optional[bool] = None,
        concurrency: Optional[int] = None,
        timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
        validate_output: Optional[bool] = None,
        schema_version: Optional[str] = None,
        validation_mode: Optional[str] = None,
        emit_run_artifacts: Optional[bool] = None,
        run_dir: Optional[Union[str, Path]] = None,
        run_root: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        overwrite: Optional[bool] = None,
        dataset_info: Optional[dict[str, Any]] = None,
        config_snapshot: Optional[dict[str, Any]] = None,
        store_messages: Optional[bool] = None,
        strict_serialization: Optional[bool] = None,
        deterministic_artifacts: Optional[bool] = None,
        resume: Optional[bool] = None,
        use_probe_batch: Optional[bool] = None,
        batch_workers: Optional[int] = None,
        return_experiment: Optional[bool] = None,
        **probe_kwargs: Any,
    ) -> Union[list[dict[str, Any]], ExperimentResult]:
        """Run the probe on all items with controlled concurrency.

        Executes the probe concurrently on all inputs using asyncio, with a
        semaphore to limit the maximum number of simultaneous executions.
        Results are written to disk in order as they complete.

        Parameters
        ----------
        prompt_set : list[Any]
            List of inputs to test. Each item is passed to the probe's run method.
        config : Optional[RunConfig], default None
            RunConfig with all run settings. Individual kwargs override config
            values if both are provided.
        stop_on_error : Optional[bool], default None
            If True, stop execution on first error. For determinism, async
            runs will enforce concurrency=1 when stop_on_error is enabled.
        concurrency : Optional[int], default None
            Maximum number of concurrent executions. Higher values speed up
            execution but may hit API rate limits.
        timeout : Optional[float], default None
            Per-item timeout in seconds. If None, no timeout is applied.
        progress_callback : Optional[ProgressCallback], default None
            Callback for progress updates. Called after each item completes.
            Supports both legacy and rich ProgressInfo signatures.
        validate_output : Optional[bool], default None
            If True, validate outputs against the schema.
        schema_version : Optional[str], default None
            Schema version for validation.
        validation_mode : Optional[str], default None
            Validation mode: "strict" or "lenient" (alias: "warn").
        emit_run_artifacts : Optional[bool], default None
            If True, write records.jsonl and manifest.json to run_dir.
        run_dir : Optional[Union[str, Path]], default None
            Explicit directory for run artifacts.
        run_root : Optional[Union[str, Path]], default None
            Root directory under which run directories are created.
        run_id : Optional[str], default None
            Explicit run ID. If None, generated deterministically from inputs.
        overwrite : Optional[bool], default None
            If True, overwrite existing run directory.
        dataset_info : Optional[dict[str, Any]], default None
            Metadata about the dataset for the manifest.
        config_snapshot : Optional[dict[str, Any]], default None
            Configuration snapshot for reproducibility.
        store_messages : Optional[bool], default None
            If True, store full message content in records.
        strict_serialization : Optional[bool], default None
            If True, fail fast on non-deterministic values during hashing.
        deterministic_artifacts : Optional[bool], default None
            If True, omit host-dependent manifest fields (platform/python).
        resume : Optional[bool], default None
            If True, resume from existing records.jsonl.
        use_probe_batch : Optional[bool], default None
            If True, use Probe.run_batch (runs in thread pool).
        batch_workers : Optional[int], default None
            Number of workers for Probe.run_batch.
        return_experiment : Optional[bool], default None
            If True, return ExperimentResult instead of list of dicts.
        **probe_kwargs : Any
            Additional keyword arguments passed to the probe's run method.

        Returns
        -------
        Union[list[dict[str, Any]], ExperimentResult]
            If return_experiment is False: list of result dictionaries.
            If return_experiment is True: ExperimentResult with aggregated metrics.

        Raises
        ------
        ValueError
            If prompt_set is empty or invalid.
        FileExistsError
            If run_dir exists and is not empty, unless overwrite=True.
        RuntimeError
            If not all items produced results (internal error).

        Examples
        --------
        Basic concurrent execution:

            >>> import asyncio
            >>>
            >>> async def run_experiment():
            ...     runner = AsyncProbeRunner(model, probe)
            ...     results = await runner.run(
            ...         prompts,
            ...         concurrency=10  # 10 concurrent requests
            ...     )
            ...     return results
            >>>
            >>> results = asyncio.run(run_experiment())

        With all options via RunConfig:

            >>> from insideLLMs.config_types import RunConfig
            >>>
            >>> async def run_full_experiment():
            ...     config = RunConfig(
            ...         concurrency=20,
            ...         emit_run_artifacts=True,
            ...         run_root="./experiments",
            ...         validate_output=True,
            ...         return_experiment=True,
            ...     )
            ...     runner = AsyncProbeRunner(model, probe)
            ...     experiment = await runner.run(prompts, config=config)
            ...     print(f"Score: {experiment.score}")
            ...     print(f"Results saved to: {runner.last_run_dir}")
            ...     return experiment

        Progress tracking with estimated time:

            >>> from insideLLMs.config_types import ProgressInfo
            >>>
            >>> async def run_with_eta():
            ...     def on_progress(info: ProgressInfo):
            ...         eta = info.eta_seconds or 0
            ...         print(f"\\r{info.current}/{info.total} - ETA: {eta:.0f}s", end="")
            ...
            ...     runner = AsyncProbeRunner(model, probe)
            ...     results = await runner.run(
            ...         prompts,
            ...         concurrency=5,
            ...         progress_callback=on_progress
            ...     )
            ...     return results

        Resumable async execution:

            >>> async def resumable_run():
            ...     runner = AsyncProbeRunner(model, probe)
            ...     try:
            ...         results = await runner.run(
            ...             prompts,
            ...             concurrency=10,
            ...             emit_run_artifacts=True,
            ...         )
            ...     except Exception:
            ...         print(f"Failed, can resume from: {runner.last_run_dir}")
            ...         raise
            ...     return results

        See Also
        --------
        ProbeRunner.run : Synchronous version for sequential execution.
        run_probe_async : Convenience function wrapping this method.
        """
        from insideLLMs.schemas import OutputValidator, SchemaRegistry

        # Resolve config: use provided config or create default
        if config is None:
            config = RunConfig()

        # Override config with explicit kwargs (backward compatibility)
        stop_on_error = stop_on_error if stop_on_error is not None else config.stop_on_error
        concurrency = concurrency if concurrency is not None else config.concurrency
        timeout = timeout if timeout is not None else config.timeout
        validate_output = validate_output if validate_output is not None else config.validate_output
        schema_version = schema_version if schema_version is not None else config.schema_version
        validation_mode = validation_mode if validation_mode is not None else config.validation_mode
        validator_mode = _normalize_validation_mode(validation_mode)
        emit_run_artifacts = (
            emit_run_artifacts if emit_run_artifacts is not None else config.emit_run_artifacts
        )
        run_dir = run_dir if run_dir is not None else config.run_dir
        run_root = run_root if run_root is not None else config.run_root
        run_id = run_id if run_id is not None else config.run_id
        overwrite = overwrite if overwrite is not None else config.overwrite
        dataset_info = dataset_info if dataset_info is not None else config.dataset_info
        config_snapshot = config_snapshot if config_snapshot is not None else config.config_snapshot
        store_messages = store_messages if store_messages is not None else config.store_messages
        strict_serialization = (
            strict_serialization
            if strict_serialization is not None
            else config.strict_serialization
        )
        deterministic_artifacts = (
            deterministic_artifacts
            if deterministic_artifacts is not None
            else config.deterministic_artifacts
        )
        if deterministic_artifacts is None:
            deterministic_artifacts = strict_serialization
        resume = resume if resume is not None else config.resume
        use_probe_batch = use_probe_batch if use_probe_batch is not None else config.use_probe_batch
        batch_workers = batch_workers if batch_workers is not None else config.batch_workers
        return_experiment = (
            return_experiment if return_experiment is not None else config.return_experiment
        )

        # Validate prompt set before execution
        validate_prompt_set(prompt_set, field_name="prompt_set", allow_empty_set=False)

        # Validate concurrency and batch_workers
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if batch_workers is not None and batch_workers < 1:
            raise ValueError(f"batch_workers must be >= 1, got {batch_workers}")

        if stop_on_error and concurrency > 1:
            logger.warning(
                "stop_on_error with concurrency>1 is non-deterministic; forcing concurrency=1"
            )
            concurrency = 1

        registry = SchemaRegistry()
        validator = OutputValidator(registry)

        model_spec = _build_model_spec(self.model)
        probe_spec = _build_probe_spec(self.probe)
        dataset_spec = _build_dataset_spec(dataset_info)
        # Add params for backward compatibility
        dataset_spec["params"] = dataset_info or {}

        if run_id is None:
            try:
                if config_snapshot is not None:
                    resolved_run_id = _deterministic_run_id_from_config_snapshot(
                        config_snapshot,
                        schema_version=schema_version,
                        strict_serialization=strict_serialization,
                    )
                else:
                    resolved_run_id = _deterministic_run_id_from_inputs(
                        schema_version=schema_version,
                        model_spec=model_spec,
                        probe_spec=probe_spec,
                        dataset_spec=dataset_spec,
                        prompt_set=prompt_set,
                        probe_kwargs=probe_kwargs,
                        strict_serialization=strict_serialization,
                    )
            except StrictSerializationError as exc:
                raise ValueError(
                    "strict_serialization requires JSON-stable values for run_id derivation."
                ) from exc
        else:
            resolved_run_id = run_id

        self.last_run_id = resolved_run_id
        logger.info(
            "Starting async probe run",
            extra={
                "run_id": resolved_run_id,
                "model": model_spec.get("model_id"),
                "probe": probe_spec.get("probe_id"),
                "examples": len(prompt_set),
                "concurrency": concurrency,
            },
        )
        run_base_time = _deterministic_base_time(resolved_run_id)
        run_started_at, _ = _deterministic_run_times(run_base_time, len(prompt_set))

        root = Path(run_root) if run_root is not None else _default_run_root()
        if run_dir is not None:
            resolved_run_dir = Path(run_dir)
        else:
            resolved_run_dir = root / resolved_run_id

        if emit_run_artifacts:
            if resume:
                _prepare_run_dir_for_resume(resolved_run_dir, run_root=root)
            else:
                _prepare_run_dir(resolved_run_dir, overwrite=overwrite, run_root=root)
            logger.debug(f"Prepared run directory: {resolved_run_dir}")
            self.last_run_dir = resolved_run_dir
            _ensure_run_sentinel(resolved_run_dir)
            if config_snapshot is not None:
                _atomic_write_yaml(
                    resolved_run_dir / "config.resolved.yaml",
                    config_snapshot,
                    strict_serialization=strict_serialization,
                )
        else:
            self.last_run_dir = None

        records_path = resolved_run_dir / "records.jsonl"
        manifest_path = resolved_run_dir / "manifest.json"

        semaphore = asyncio.Semaphore(concurrency)
        results: list[Optional[dict[str, Any]]] = [None] * len(prompt_set)
        errors: list[Optional[BaseException]] = [None] * len(prompt_set)
        stop_event = asyncio.Event()
        stop_error: Optional[RunnerExecutionError] = None
        completed = 0
        completed_lock = asyncio.Lock()  # Protect completed counter from race conditions
        total = len(prompt_set)
        run_start_time = time.perf_counter()  # For progress tracking

        if emit_run_artifacts and resume:
            existing_records = _read_jsonl_records(records_path, truncate_incomplete=True)
            if existing_records:
                if len(existing_records) > len(prompt_set):
                    raise ValueError(
                        "Existing records.jsonl has more entries than the current prompt_set."
                    )
                for line_index, record in enumerate(existing_records):
                    _validate_resume_record(
                        record,
                        expected_index=line_index,
                        expected_item=prompt_set[line_index],
                        run_id=resolved_run_id,
                        strict_serialization=strict_serialization,
                    )
                    results[line_index] = _result_dict_from_record(
                        record,
                        schema_version=schema_version,
                    )
                completed = len(existing_records)

        records_fp = None
        write_lock = asyncio.Lock()
        next_write_index = completed

        async def write_ready_records() -> None:
            nonlocal next_write_index
            if not emit_run_artifacts or records_fp is None:
                return
            async with write_lock:
                while next_write_index < len(prompt_set):
                    result_obj = results[next_write_index]
                    if result_obj is None:
                        break
                    error_value = errors[next_write_index]
                    error_type = result_obj.get("error_type")
                    if error_value is None and result_obj.get("error") is not None:
                        error_value = str(result_obj.get("error"))
                    item_started_at, item_completed_at = _deterministic_item_times(
                        run_base_time,
                        next_write_index,
                    )
                    record = _build_result_record(
                        schema_version=schema_version,
                        run_id=resolved_run_id,
                        started_at=item_started_at,
                        completed_at=item_completed_at,
                        model=model_spec,
                        probe=probe_spec,
                        dataset=dataset_spec,
                        item=prompt_set[next_write_index],
                        output=result_obj.get("output"),
                        latency_ms=result_obj.get("latency_ms"),
                        store_messages=store_messages,
                        index=next_write_index,
                        status=str(result_obj.get("status") or "error"),
                        error=error_value,
                        error_type=error_type,
                        strict_serialization=strict_serialization,
                    )
                    if validate_output:
                        validator.validate(
                            registry.RESULT_RECORD,
                            record,
                            schema_version=schema_version,
                            mode=validator_mode,
                        )
                    # Use executor to avoid blocking event loop on file I/O
                    record_line = _stable_json_dumps(record, strict=strict_serialization) + "\n"
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda line=record_line: (records_fp.write(line), records_fp.flush()),
                    )
                    next_write_index += 1

        async def run_single(index: int, item: Any) -> None:
            nonlocal completed, stop_error
            async with semaphore:
                if stop_event.is_set():
                    return
                try:
                    # Run in thread pool for sync models with optional timeout
                    loop = asyncio.get_running_loop()

                    async def execute_probe() -> Any:
                        return await loop.run_in_executor(
                            None,
                            lambda: self.probe.run(self.model, item, **probe_kwargs),
                        )

                    output = await run_with_timeout(
                        execute_probe,
                        timeout=timeout,
                        context={"index": index, "item_type": type(item).__name__},
                    )
                    probe_result = ProbeResult(
                        input=item,
                        output=output,
                        status=ResultStatus.SUCCESS,
                        latency_ms=None,
                        metadata={},
                    )
                    results[index] = _result_dict_from_probe_result(
                        probe_result,
                        schema_version=schema_version,
                    )
                except Exception as e:
                    logger.warning(
                        "Async probe execution failed",
                        extra={
                            "run_id": resolved_run_id,
                            "index": index,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        },
                        exc_info=True,
                    )
                    errors[index] = e
                    probe_result = ProbeResult(
                        input=item,
                        status=ResultStatus.ERROR,
                        error=str(e),
                        latency_ms=None,
                        metadata={"error_type": type(e).__name__},
                    )
                    results[index] = _result_dict_from_probe_result(
                        probe_result,
                        schema_version=schema_version,
                        error_type=type(e).__name__,
                    )
                    if stop_on_error and stop_error is None:
                        prompt_str = str(item) if not isinstance(item, str) else item
                        stop_error = RunnerExecutionError(
                            reason=str(e),
                            model_id=model_spec.get("model_id"),
                            probe_id=probe_spec.get("probe_id"),
                            prompt=prompt_str,
                            prompt_index=index,
                            run_id=resolved_run_id,
                            elapsed_seconds=None,
                            original_error=e,
                            suggestions=[
                                "Check the model API credentials and connectivity",
                                "Verify the prompt format is valid for this model",
                                "Review the original error message above",
                            ],
                        )
                        stop_event.set()
                finally:
                    async with completed_lock:
                        completed += 1
                        current_completed = completed
                    _invoke_progress_callback(
                        progress_callback,
                        current=current_completed,
                        total=total,
                        start_time=run_start_time,
                        current_item=item,
                        current_index=index,
                        status="processing",
                    )
                    await write_ready_records()

        with ExitStack() as stack:
            if emit_run_artifacts:
                mode = "a" if resume else "x"
                records_fp = stack.enter_context(open(records_path, mode, encoding="utf-8"))

            if use_probe_batch:
                remaining_items = prompt_set[completed:]
                if remaining_items:
                    resolved_batch_workers = (
                        batch_workers if batch_workers is not None else max(1, concurrency)
                    )

                    def batch_progress(current: int, total_batch: int) -> None:
                        _invoke_progress_callback(
                            progress_callback,
                            current=completed + current,
                            total=total,
                            start_time=run_start_time,
                            status="processing",
                        )

                    loop = asyncio.get_running_loop()
                    probe_results = await loop.run_in_executor(
                        None,
                        lambda: self.probe.run_batch(
                            self.model,
                            remaining_items,
                            max_workers=resolved_batch_workers,
                            progress_callback=batch_progress if progress_callback else None,
                            **probe_kwargs,
                        ),
                    )

                    for offset, probe_result in enumerate(probe_results):
                        index = completed + offset
                        error_type = None
                        if isinstance(probe_result.metadata, dict):
                            error_type = probe_result.metadata.get("error_type")
                        result_obj = _result_dict_from_probe_result(
                            probe_result,
                            schema_version=schema_version,
                            error_type=error_type,
                        )
                        result_obj["latency_ms"] = None
                        results[index] = result_obj
                        if stop_on_error and stop_error is None:
                            status = _normalize_status(probe_result.status)
                            if status != "success":
                                prompt_str = str(remaining_items[offset])
                                stop_error = RunnerExecutionError(
                                    reason=probe_result.error or status,
                                    model_id=model_spec.get("model_id"),
                                    probe_id=probe_spec.get("probe_id"),
                                    prompt=prompt_str,
                                    prompt_index=index,
                                    run_id=resolved_run_id,
                                    elapsed_seconds=None,
                                    suggestions=[
                                        "Check the model API credentials and connectivity",
                                        "Verify the prompt format is valid for this model",
                                        "Review the original error message above",
                                    ],
                                )
                    completed += len(remaining_items)
                    await write_ready_records()
            else:
                tasks = [run_single(i, item) for i, item in enumerate(prompt_set) if i >= completed]
                if tasks:
                    await asyncio.gather(*tasks)
                await write_ready_records()

        if stop_error is not None:
            raise stop_error

        _invoke_progress_callback(
            progress_callback,
            current=total,
            total=total,
            start_time=run_start_time,
            status="complete",
        )

        if any(result is None for result in results):
            raise RuntimeError("Runner did not produce results for all items.")
        final_results = [result for result in results if result is not None]

        self._results = final_results

        _, run_completed_at = _deterministic_run_times(run_base_time, len(final_results))

        if emit_run_artifacts:
            python_version = None if deterministic_artifacts else sys.version.split()[0]
            platform_info = None if deterministic_artifacts else platform.platform()

            def _serialize_manifest(value: Any) -> Any:
                return _serialize_value(value, strict=strict_serialization)

            manifest = {
                "schema_version": schema_version,
                "run_id": resolved_run_id,
                "created_at": run_started_at,
                "started_at": run_started_at,
                "completed_at": run_completed_at,
                "library_version": None,
                "python_version": python_version,
                "platform": platform_info,
                "command": None,
                "model": model_spec,
                "probe": probe_spec,
                "dataset": dataset_spec,
                "record_count": len(final_results),
                "success_count": sum(1 for r in final_results if r.get("status") == "success"),
                "error_count": sum(1 for r in final_results if r.get("status") == "error"),
                "records_file": "records.jsonl",
                "schemas": {
                    registry.RESULT_RECORD: schema_version,
                    registry.RUN_MANIFEST: schema_version,
                    registry.RUNNER_ITEM: schema_version,
                },
                "custom": {},
            }

            if _semver_tuple(schema_version) >= (1, 0, 1):
                manifest["run_completed"] = True

            try:
                import insideLLMs

                manifest["library_version"] = getattr(insideLLMs, "__version__", None)
            except (ImportError, AttributeError):
                pass

            if validate_output:
                validator.validate(
                    registry.RUN_MANIFEST,
                    manifest,
                    schema_version=schema_version,
                    mode=validator_mode,
                )

            _atomic_write_text(
                manifest_path,
                json.dumps(
                    _serialize_manifest(manifest),
                    sort_keys=True,
                    indent=2,
                    default=_serialize_manifest,
                ),
            )

        if validate_output:
            validator.validate(
                registry.RUNNER_OUTPUT,
                {"schema_version": schema_version, "results": final_results},
                schema_version=schema_version,
                mode=validator_mode,
            )

        self.last_experiment = create_experiment_result(
            self.model,
            self.probe,
            final_results,
            config=config_snapshot or {},
            experiment_id=resolved_run_id,
            started_at=run_started_at,
            completed_at=run_completed_at,
            strict_serialization=strict_serialization,
        )

        return self.last_experiment if return_experiment else final_results


def _prepare_run_dir(path: Path, *, overwrite: bool, run_root: Optional[Path] = None) -> None:
    """Prepare a directory for run artifact emission.

    Policy:
    - If directory doesn't exist: create it.
    - If directory exists and is empty: use it.
    - If directory exists and is non-empty: fail unless overwrite=True.
      If overwrite=True, remove and recreate.
    """

    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"run_dir '{path}' exists and is not a directory")

        try:
            is_empty = not any(path.iterdir())
        except (OSError, PermissionError):
            is_empty = False

        if is_empty:
            return

        if not overwrite:
            raise FileExistsError(
                f"run_dir '{path}' already exists and is not empty. "
                "Use overwrite=True (or CLI --overwrite) to replace it."
            )

        resolved = path.resolve()

        # Critical safety guards for overwrite.
        cwd_resolved = Path.cwd().resolve()
        home_resolved = Path.home().resolve()
        if resolved == cwd_resolved:
            raise ValueError(f"Refusing to overwrite current working directory: '{resolved}'")
        if resolved == home_resolved:
            raise ValueError(f"Refusing to overwrite home directory: '{resolved}'")

        # Filesystem root (POSIX '/' or Windows drive root like 'C:\\').
        if resolved.anchor and Path(resolved.anchor) == resolved:
            raise ValueError(f"Refusing to overwrite filesystem root: '{resolved}'")

        if run_root is not None:
            try:
                run_root_resolved = run_root.resolve()
            except (OSError, ValueError):
                run_root_resolved = run_root
            if resolved == run_root_resolved:
                raise ValueError(f"Refusing to overwrite run_root directory itself: '{resolved}'")

        # Refuse very short paths (e.g. '/tmp', 'C:\\tmp') that are easy to fat-finger.
        if len(resolved.parts) <= 2:
            raise ValueError(f"Refusing to overwrite high-risk short path: '{resolved}'")

        # Sentinel requirement: only overwrite if this looks like an insideLLMs run dir.
        sentinel_ok = (
            (path / "manifest.json").exists()
            or (path / ".insidellms_run").exists()
            or (path / "records.jsonl").exists()
            or (path / "config.resolved.yaml").exists()
        )
        if not sentinel_ok:
            raise ValueError(
                f"Refusing to overwrite '{resolved}': directory is not empty and does not look like an "
                "insideLLMs run (missing manifest.json or .insidellms_run)."
            )

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def _prepare_run_dir_for_resume(path: Path, *, run_root: Optional[Path] = None) -> None:
    """Prepare a directory for resumable run artifact emission.

    Policy:
    - If directory doesn't exist: create it.
    - If directory exists and is empty: use it.
    - If directory exists and is non-empty: require insideLLMs sentinel.
    """
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"run_dir '{path}' exists and is not a directory")

        try:
            is_empty = not any(path.iterdir())
        except (OSError, PermissionError):
            is_empty = False

        if is_empty:
            return

        sentinel_ok = (path / "manifest.json").exists() or (path / ".insidellms_run").exists()
        if not sentinel_ok:
            raise ValueError(
                f"Refusing to resume in '{path}': directory is not empty and does not look like an "
                "insideLLMs run (missing manifest.json or .insidellms_run)."
            )
        return

    path.mkdir(parents=True, exist_ok=True)


def run_probe(
    model: Model,
    probe: Probe,
    prompt_set: list[Any],
    *,
    config: Optional[RunConfig] = None,
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
    **probe_kwargs: Any,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run a probe on a model with a single function call.

    This is a convenience function that creates a :class:`ProbeRunner` and
    executes the probe in one step. For more control over execution or to
    access run statistics, use ProbeRunner directly.

    Parameters
    ----------
    model : Model
        The model to test. Must implement the Model interface.
    probe : Probe
        The probe to run. Must implement the Probe interface.
    prompt_set : list[Any]
        List of inputs to test. Each item is passed to the probe's run method.
    config : Optional[RunConfig], default None
        RunConfig with all run settings including emit_run_artifacts,
        stop_on_error, return_experiment, etc.
    **probe_kwargs : Any
        Additional keyword arguments passed to the probe's run method.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        If config.return_experiment is False (default): list of result dicts.
        If config.return_experiment is True: ExperimentResult.

    Examples
    --------
    Simple probe execution:

        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.probes import FactualityProbe
        >>> from insideLLMs.runtime.runner import run_probe
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> probe = FactualityProbe()
        >>> prompts = [{"messages": [{"role": "user", "content": "Hello"}]}]
        >>> results = run_probe(model, probe, prompts)

    With RunConfig options:

        >>> from insideLLMs.config_types import RunConfig
        >>>
        >>> config = RunConfig(
        ...     emit_run_artifacts=True,
        ...     return_experiment=True,
        ... )
        >>> experiment = run_probe(model, probe, prompts, config=config)
        >>> print(f"Score: {experiment.score}")

    Passing probe-specific arguments:

        >>> results = run_probe(
        ...     model, probe, prompts,
        ...     temperature=0.7,  # Passed to probe
        ...     max_tokens=100,   # Passed to probe
        ... )

    See Also
    --------
    ProbeRunner : Class for more control over execution.
    run_probe_async : Async version with concurrency support.
    """
    runner = ProbeRunner(model, probe)
    return runner.run(
        prompt_set,
        config=config,
        strict_serialization=strict_serialization,
        deterministic_artifacts=deterministic_artifacts,
        **probe_kwargs,
    )


async def run_probe_async(
    model: Model,
    probe: Probe,
    prompt_set: list[Any],
    *,
    config: Optional[RunConfig] = None,
    concurrency: Optional[int] = None,
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
    **probe_kwargs: Any,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run a probe asynchronously with concurrent execution.

    This is a convenience function that creates an :class:`AsyncProbeRunner`
    and executes the probe in one step. For more control over execution or
    to access run statistics, use AsyncProbeRunner directly.

    Parameters
    ----------
    model : Model
        The model to test. Must implement the Model interface.
    probe : Probe
        The probe to run. Must implement the Probe interface.
    prompt_set : list[Any]
        List of inputs to test. Each item is passed to the probe's run method.
    config : Optional[RunConfig], default None
        RunConfig with all run settings.
    concurrency : Optional[int], default None
        Maximum number of concurrent executions. Overrides config.concurrency.
    **probe_kwargs : Any
        Additional keyword arguments passed to the probe's run method.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        If config.return_experiment is False (default): list of result dicts.
        If config.return_experiment is True: ExperimentResult.

    Examples
    --------
    Basic async execution:

        >>> import asyncio
        >>> from insideLLMs.runtime.runner import run_probe_async
        >>>
        >>> async def main():
        ...     results = await run_probe_async(
        ...         model, probe, prompts,
        ...         concurrency=10
        ...     )
        ...     return results
        >>>
        >>> results = asyncio.run(main())

    With RunConfig:

        >>> from insideLLMs.config_types import RunConfig
        >>>
        >>> async def main_with_config():
        ...     config = RunConfig(
        ...         concurrency=20,
        ...         emit_run_artifacts=True,
        ...         return_experiment=True,
        ...     )
        ...     experiment = await run_probe_async(model, probe, prompts, config=config)
        ...     print(f"Score: {experiment.score}")
        ...     return experiment

    Inline in an existing async context:

        >>> async def process_batch():
        ...     # Run multiple probes concurrently
        ...     factuality_task = run_probe_async(model, factuality_probe, prompts)
        ...     bias_task = run_probe_async(model, bias_probe, prompts)
        ...     factuality_results, bias_results = await asyncio.gather(
        ...         factuality_task, bias_task
        ...     )
        ...     return factuality_results, bias_results

    See Also
    --------
    AsyncProbeRunner : Class for more control over async execution.
    run_probe : Synchronous version for sequential execution.
    """
    runner = AsyncProbeRunner(model, probe)
    return await runner.run(
        prompt_set,
        config=config,
        concurrency=concurrency,
        strict_serialization=strict_serialization,
        deterministic_artifacts=deterministic_artifacts,
        **probe_kwargs,
    )


def load_config(path: Union[str, Path]) -> ConfigDict:
    """Load an experiment configuration file.

    Reads and parses a YAML or JSON configuration file that defines model,
    probe, and dataset settings for an experiment.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the configuration file. Must have .yaml, .yml, or .json extension.

    Returns
    -------
    ConfigDict
        The parsed configuration dictionary containing model, probe, and
        dataset definitions.

    Raises
    ------
    ValueError
        If the file extension is not supported (.yaml, .yml, or .json).
    FileNotFoundError
        If the configuration file doesn't exist.

    Examples
    --------
    Load a YAML configuration:

        >>> from insideLLMs.runtime.runner import load_config
        >>>
        >>> config = load_config("experiment.yaml")
        >>> print(config["model"]["type"])
        'openai'

    Load a JSON configuration:

        >>> config = load_config("experiment.json")
        >>> print(config["probe"]["type"])
        'factuality'

    Check configuration contents:

        >>> config = load_config("config.yaml")
        >>> for key in config:
        ...     print(f"{key}: {type(config[key])}")
        model: <class 'dict'>
        probe: <class 'dict'>
        dataset: <class 'dict'>

    See Also
    --------
    run_experiment_from_config : Load and run an experiment in one step.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}: expected a mapping at top level.")
    return data


def _resolve_path(path: str, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    p = Path(expanded)
    if p.is_absolute():
        return p
    return base_dir / p


def _build_resolved_config_snapshot(config: ConfigDict, base_dir: Path) -> dict[str, Any]:
    """Build a reproducibility snapshot of the config actually used for the run.

    This is intended for writing to `config.resolved.yaml` in a run directory.
    We keep the structure of the original config, but resolve any relative
    filesystem paths that insideLLMs resolves at runtime (e.g. CSV/JSONL dataset
    paths).
    """
    import copy
    import posixpath

    snapshot: dict[str, Any] = copy.deepcopy(config)  # type: ignore[arg-type]

    dataset = snapshot.get("dataset") if isinstance(snapshot, dict) else None
    if isinstance(dataset, dict):
        fmt = dataset.get("format")
        if fmt in ("csv", "jsonl") and isinstance(dataset.get("path"), str):
            # Keep paths deterministic and portable: avoid baking absolute checkout paths
            # into the snapshot (and therefore the run_id). Only normalize relative paths
            # to a canonical posix form unless content hashing succeeds.
            raw_path = dataset["path"]
            normalized = posixpath.normpath(str(raw_path).replace("\\", "/"))
            resolved_path = _resolve_path(normalized, base_dir)

            # Prefer content-addressing over path-addressing for run_id stability.
            # If the user didn't provide an explicit hash, compute a deterministic
            # SHA-256 over the dataset file bytes.
            if dataset.get("hash") is None and dataset.get("dataset_hash") is None:
                try:
                    hasher = hashlib.sha256()
                    with open(resolved_path, "rb") as fp:
                        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    dataset["dataset_hash"] = f"sha256:{hasher.hexdigest()}"
                except (IOError, OSError) as e:
                    logger.warning(
                        f"Failed to hash dataset file '{dataset.get('path')}': {e}. "
                        "Run ID will not include dataset content hash."
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error hashing dataset '{dataset.get('path')}': {e}",
                        exc_info=True,
                    )

            dataset_hash = dataset.get("hash") or dataset.get("dataset_hash")
            path_value = normalized
            if Path(normalized).is_absolute():
                try:
                    rel = resolved_path.resolve().relative_to(base_dir.resolve())
                    path_value = rel.as_posix()
                except Exception:
                    if dataset_hash:
                        path_value = Path(normalized).name
            dataset["path"] = posixpath.normpath(path_value)
        elif fmt == "hf":
            if (
                dataset.get("dataset_hash") is None
                and dataset.get("hash") is None
                and dataset.get("revision") is None
            ):
                logger.warning(
                    "HuggingFace dataset config missing revision or dataset_hash; "
                    "run_id may not be stable if the dataset changes."
                )

    return snapshot


def _resolve_determinism_options(
    config: Any,
    *,
    strict_override: Optional[bool],
    deterministic_artifacts_override: Optional[bool],
) -> tuple[bool, bool]:
    """Resolve determinism controls from config plus explicit overrides."""
    cfg_strict = True
    cfg_artifacts: Optional[bool] = None

    if isinstance(config, dict):
        det = config.get("determinism")
        if isinstance(det, dict):
            if "strict_serialization" in det:
                raw_strict = det.get("strict_serialization")
                if raw_strict is not None and not isinstance(raw_strict, bool):
                    raise ValueError(
                        "determinism.strict_serialization must be a bool or null/None, "
                        f"got {type(raw_strict).__name__}"
                    )
                if isinstance(raw_strict, bool):
                    cfg_strict = raw_strict
            if "deterministic_artifacts" in det:
                raw_artifacts = det.get("deterministic_artifacts")
                if raw_artifacts is not None and not isinstance(raw_artifacts, bool):
                    raise ValueError(
                        "determinism.deterministic_artifacts must be a bool or null/None, "
                        f"got {type(raw_artifacts).__name__}"
                    )
                cfg_artifacts = raw_artifacts

    strict_serialization = strict_override if strict_override is not None else cfg_strict
    deterministic_artifacts = (
        deterministic_artifacts_override
        if deterministic_artifacts_override is not None
        else cfg_artifacts
    )
    if deterministic_artifacts is None:
        deterministic_artifacts = strict_serialization
    return strict_serialization, deterministic_artifacts


def _extract_probe_kwargs_from_config(config: Any) -> dict[str, Any]:
    """Extract `probe.run(...)` kwargs from a config dict.

    We support a few synonymous keys so users can express generation settings in a
    more natural way:

    - `generation`: preferred name for model generation parameters
    - `probe_kwargs`: explicit name matching ProbeRunner.run(**probe_kwargs)
    - `run_kwargs`: legacy/alternate name

    If multiple keys are present, later keys override earlier ones.
    """
    if not isinstance(config, dict):
        return {}

    merged: dict[str, Any] = {}
    for key in ("generation", "probe_kwargs", "run_kwargs"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a mapping/dict, got {type(value).__name__}")
        merged.update(value)
    return merged


def _create_middlewares_from_config(items: Any) -> list[Any]:
    if not items:
        return []
    if not isinstance(items, list):
        raise ValueError("pipeline.middlewares must be a list")

    from insideLLMs.pipeline import (
        CacheMiddleware,
        CostTrackingMiddleware,
        PassthroughMiddleware,
        RateLimitMiddleware,
        RetryMiddleware,
        TraceMiddleware,
    )

    def normalize(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    middleware_map = {
        "cache": CacheMiddleware,
        "cachemiddleware": CacheMiddleware,
        "rate_limit": RateLimitMiddleware,
        "ratelimit": RateLimitMiddleware,
        "ratelimitmiddleware": RateLimitMiddleware,
        "retry": RetryMiddleware,
        "retrymiddleware": RetryMiddleware,
        "cost": CostTrackingMiddleware,
        "cost_tracking": CostTrackingMiddleware,
        "costtracking": CostTrackingMiddleware,
        "costtrackingmiddleware": CostTrackingMiddleware,
        "trace": TraceMiddleware,
        "tracemiddleware": TraceMiddleware,
        "passthrough": PassthroughMiddleware,
        "passthroughmiddleware": PassthroughMiddleware,
    }

    middlewares: list[Any] = []
    for item in items:
        if isinstance(item, str):
            mw_type = normalize(item)
            mw_args: dict[str, Any] = {}
        elif isinstance(item, dict):
            raw_type = item.get("type") or item.get("name")
            if not raw_type:
                raise ValueError("Middleware config missing 'type'")
            mw_type = normalize(str(raw_type))
            mw_args = item.get("args") or {}
        else:
            raise ValueError(f"Unsupported middleware config entry: {item!r}")

        mw_cls = middleware_map.get(mw_type)
        if mw_cls is None:
            raise ValueError(f"Unknown middleware type: {mw_type}")
        middlewares.append(mw_cls(**mw_args))

    return middlewares


def _create_model_from_config(
    config: ConfigDict,
    *,
    prefer_async_pipeline: bool = False,
) -> Model:
    """Create a model instance from configuration.

    Uses the registry if available, falls back to direct imports.
    """
    ensure_builtins_registered()

    model_type = config["type"]
    model_args = config.get("args", {})

    try:
        base_model = model_registry.get(model_type, **model_args)
    except NotFoundError:
        # Fallback to direct import for backwards compatibility.
        # Only import the specific model type to avoid ImportError for
        # optional dependencies that aren't installed.
        known_types = {
            "dummy",
            "openai",
            "huggingface",
            "anthropic",
            "gemini",
            "cohere",
            "llamacpp",
            "ollama",
            "vllm",
        }

        if model_type not in known_types:
            raise ValueError(f"Unknown model type: {model_type}")

        # Import only the requested model class to avoid triggering
        # ImportError for other models with missing dependencies.
        try:
            if model_type == "dummy":
                from insideLLMs.models.dummy import DummyModel

                base_model = DummyModel(**model_args)
            elif model_type == "openai":
                from insideLLMs.models.openai import OpenAIModel

                base_model = OpenAIModel(**model_args)
            elif model_type == "anthropic":
                from insideLLMs.models.anthropic import AnthropicModel

                base_model = AnthropicModel(**model_args)
            elif model_type == "huggingface":
                from insideLLMs.models.huggingface import HuggingFaceModel

                base_model = HuggingFaceModel(**model_args)
            elif model_type == "gemini":
                from insideLLMs.models.gemini import GeminiModel

                base_model = GeminiModel(**model_args)
            elif model_type == "cohere":
                from insideLLMs.models.cohere import CohereModel

                base_model = CohereModel(**model_args)
            elif model_type == "llamacpp":
                from insideLLMs.models.llamacpp import LlamaCppModel

                base_model = LlamaCppModel(**model_args)
            elif model_type == "ollama":
                from insideLLMs.models.ollama import OllamaModel

                base_model = OllamaModel(**model_args)
            elif model_type == "vllm":
                from insideLLMs.models.vllm import VLLMModel

                base_model = VLLMModel(**model_args)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except ImportError as e:
            raise ImportError(
                f"Model type '{model_type}' requires additional dependencies. "
                f"Install with: pip install insidellms[{model_type}]. "
                f"Original error: {e}"
            ) from e

    pipeline_cfg = config.get("pipeline") if isinstance(config, dict) else None
    if isinstance(pipeline_cfg, dict):
        from insideLLMs.pipeline import AsyncModelPipeline, ModelPipeline

        middlewares_cfg = pipeline_cfg.get("middlewares")
        if middlewares_cfg is None:
            middlewares_cfg = pipeline_cfg.get("middleware", [])
        middlewares = _create_middlewares_from_config(middlewares_cfg)
        if middlewares:
            pipeline_async = pipeline_cfg.get("async")
            if pipeline_async is None:
                pipeline_async = prefer_async_pipeline
            pipeline_name = pipeline_cfg.get("name")
            if pipeline_async:
                return AsyncModelPipeline(base_model, middlewares=middlewares, name=pipeline_name)
            return ModelPipeline(base_model, middlewares=middlewares, name=pipeline_name)

    return base_model


def _create_probe_from_config(config: ConfigDict) -> Probe:
    """Create a probe instance from configuration.

    Uses the registry if available, falls back to direct imports.
    """
    ensure_builtins_registered()

    probe_type = config["type"]
    probe_args = config.get("args", {})

    try:
        return probe_registry.get(probe_type, **probe_args)
    except NotFoundError:
        # Fallback to direct import for backwards compatibility
        from insideLLMs.probes import (
            AttackProbe,
            BiasProbe,
            CodeDebugProbe,
            CodeExplanationProbe,
            CodeGenerationProbe,
            ConstraintComplianceProbe,
            FactualityProbe,
            InstructionFollowingProbe,
            JailbreakProbe,
            LogicProbe,
            MultiStepTaskProbe,
            PromptInjectionProbe,
        )

        probe_map = {
            "logic": LogicProbe,
            "bias": BiasProbe,
            "attack": AttackProbe,
            "factuality": FactualityProbe,
            "prompt_injection": PromptInjectionProbe,
            "jailbreak": JailbreakProbe,
            "code_generation": CodeGenerationProbe,
            "code_explanation": CodeExplanationProbe,
            "code_debug": CodeDebugProbe,
            "instruction_following": InstructionFollowingProbe,
            "multi_step_task": MultiStepTaskProbe,
            "constraint_compliance": ConstraintComplianceProbe,
        }

        if probe_type not in probe_map:
            raise ValueError(f"Unknown probe type: {probe_type}") from None

        return probe_map[probe_type](**probe_args)


def _load_dataset_from_config(config: ConfigDict, base_dir: Path) -> list[Any]:
    """Load a dataset from configuration."""
    ensure_builtins_registered()

    format_type = config["format"]

    if format_type in ("csv", "jsonl"):
        path = _resolve_path(config["path"], base_dir)
        try:
            loader = dataset_registry.get_factory(format_type)
            return loader(str(path))
        except NotFoundError:
            from insideLLMs.dataset_utils import load_csv_dataset, load_jsonl_dataset

            if format_type == "csv":
                return load_csv_dataset(str(path))
            else:
                return load_jsonl_dataset(str(path))

    elif format_type == "hf":
        excluded_keys = {
            "format",
            "name",
            "split",
            "path",
            "dataset",
            "hash",
            "dataset_hash",
            "version",
            "dataset_version",
            "provenance",
            "source",
        }
        extra_kwargs = {k: v for k, v in config.items() if k not in excluded_keys}
        try:
            loader = dataset_registry.get_factory("hf")
            return loader(config["name"], split=config.get("split", "test"), **extra_kwargs)
        except NotFoundError:
            from insideLLMs.dataset_utils import load_hf_dataset

            return load_hf_dataset(
                config["name"], split=config.get("split", "test"), **extra_kwargs
            )

    else:
        raise ValueError(f"Unknown dataset format: {format_type}")


def run_experiment_from_config(
    config_path: Union[str, Path],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    validate_output: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
    emit_run_artifacts: bool = True,
    run_dir: Optional[Union[str, Path]] = None,
    run_root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = False,
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
    use_probe_batch: bool = False,
    batch_workers: Optional[int] = None,
    return_experiment: bool = False,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run an experiment defined in a YAML or JSON configuration file.

    Loads the configuration file, creates the model and probe instances,
    loads the dataset, and runs the experiment. This is the primary entry
    point for configuration-driven experimentation.

    The configuration file must specify:

    - ``model``: Model type and arguments
    - ``probe``: Probe type and arguments
    - ``dataset``: Dataset format, path/name, and optional split

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML (.yaml, .yml) or JSON (.json) configuration file.
    progress_callback : Optional[Callable[[int, int], None]], default None
        Callback for progress updates with signature (current, total).
    validate_output : bool, default False
        If True, validate outputs against the schema.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version for validation.
    validation_mode : str, default "strict"
        Validation mode: "strict" or "lenient" (alias: "warn").
    emit_run_artifacts : bool, default True
        If True, write records.jsonl and manifest.json to run directory.
    run_dir : Optional[Union[str, Path]], default None
        Explicit directory for run artifacts.
    run_root : Optional[Union[str, Path]], default None
        Root directory under which run directories are created.
    run_id : Optional[str], default None
        Explicit run ID. If None, generated deterministically from config.
    overwrite : bool, default False
        If True, overwrite existing run directory.
    resume : bool, default False
        If True, resume from existing records.jsonl.
    strict_serialization : Optional[bool], default None
        If True, fail fast on non-deterministic values during hashing.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields (platform/python).
    use_probe_batch : bool, default False
        If True, use Probe.run_batch for potentially faster execution.
    batch_workers : Optional[int], default None
        Number of workers for Probe.run_batch.
    return_experiment : bool, default False
        If True, return ExperimentResult instead of list of dicts.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        If return_experiment is False: list of result dictionaries.
        If return_experiment is True: ExperimentResult with aggregated metrics.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    ValueError
        If the configuration is invalid or has unknown model/probe types.

    Examples
    --------
    Basic usage with a YAML config file:

        >>> from insideLLMs.runtime.runner import run_experiment_from_config
        >>>
        >>> # experiment.yaml:
        >>> # model:
        >>> #   type: openai
        >>> #   args:
        >>> #     model_name: gpt-4
        >>> # probe:
        >>> #   type: factuality
        >>> #   args: {}
        >>> # dataset:
        >>> #   format: jsonl
        >>> #   path: data/questions.jsonl
        >>>
        >>> results = run_experiment_from_config("experiment.yaml")
        >>> print(f"Processed {len(results)} examples")

    With progress tracking:

        >>> def on_progress(current, total):
        ...     print(f"\\rProgress: {current}/{total}", end="")
        >>>
        >>> results = run_experiment_from_config(
        ...     "experiment.yaml",
        ...     progress_callback=on_progress
        ... )

    Getting ExperimentResult with full metadata:

        >>> experiment = run_experiment_from_config(
        ...     "experiment.yaml",
        ...     return_experiment=True
        ... )
        >>> print(f"Experiment ID: {experiment.experiment_id}")
        >>> print(f"Score: {experiment.score}")

    Saving artifacts to a specific directory:

        >>> results = run_experiment_from_config(
        ...     "experiment.yaml",
        ...     emit_run_artifacts=True,
        ...     run_root="./my_experiments",
        ... )

    Configuration with HuggingFace dataset:

        >>> # hf_experiment.yaml:
        >>> # model:
        >>> #   type: anthropic
        >>> #   args:
        >>> #     model_name: claude-3-opus-20240229
        >>> # probe:
        >>> #   type: bias
        >>> #   args: {}
        >>> # dataset:
        >>> #   format: hf
        >>> #   name: cais/mmlu
        >>> #   split: test
        >>>
        >>> results = run_experiment_from_config("hf_experiment.yaml")

    See Also
    --------
    run_experiment_from_config_async : Async version with concurrency support.
    run_harness_from_config : Run multi-model/multi-probe harness.
    load_config : Load configuration file without running.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    config_snapshot = _build_resolved_config_snapshot(config, base_dir)
    strict_serialization, deterministic_artifacts = _resolve_determinism_options(
        config_snapshot,
        strict_override=strict_serialization,
        deterministic_artifacts_override=deterministic_artifacts,
    )
    try:
        resolved_run_id = run_id or _deterministic_run_id_from_config_snapshot(
            config_snapshot,
            schema_version=schema_version,
            strict_serialization=strict_serialization,
        )
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable values in the resolved config snapshot."
        ) from exc

    model = _create_model_from_config(config["model"], prefer_async_pipeline=True)
    probe = _create_probe_from_config(config["probe"])
    dataset = _load_dataset_from_config(config["dataset"], base_dir)

    probe_kwargs = _extract_probe_kwargs_from_config(config_snapshot)
    probe_kwargs.update(_extract_probe_kwargs_from_config(config_snapshot.get("probe")))

    runner = ProbeRunner(model, probe)
    return runner.run(
        dataset,
        progress_callback=progress_callback,
        validate_output=validate_output,
        schema_version=schema_version,
        validation_mode=validation_mode,
        emit_run_artifacts=emit_run_artifacts,
        run_dir=run_dir,
        run_root=run_root,
        run_id=resolved_run_id,
        overwrite=overwrite,
        dataset_info=config_snapshot.get("dataset"),
        config_snapshot=config_snapshot,
        resume=resume,
        strict_serialization=strict_serialization,
        deterministic_artifacts=deterministic_artifacts,
        use_probe_batch=use_probe_batch,
        batch_workers=batch_workers,
        return_experiment=return_experiment,
        **probe_kwargs,
    )


def run_harness_from_config(
    config_path: Union[str, Path],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    validate_output: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
) -> dict[str, Any]:
    """Run a cross-model probe harness from a configuration file.

    Executes all combinations of models and probes defined in the configuration
    against a shared dataset. This is useful for comparing multiple models
    across multiple probes in a single run.

    The configuration file must specify:

    - ``models``: List of model configurations (type + args)
    - ``probes``: List of probe configurations (type + args)
    - ``dataset``: Dataset format, path/name, and optional split
    - ``max_examples``: Optional limit on number of examples

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML (.yaml, .yml) or JSON (.json) configuration file.
    progress_callback : Optional[Callable[[int, int], None]], default None
        Callback for progress updates. Called with (current, total) where
        total = len(models) * len(probes) * len(dataset).
    validate_output : bool, default False
        If True, validate outputs against the schema.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version for validation.
    validation_mode : str, default "strict"
        Validation mode: "strict" or "lenient" (alias: "warn").
    strict_serialization : Optional[bool], default None
        If True, fail fast on non-deterministic values during hashing.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields (platform/python).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - ``records``: List of per-example result records
        - ``experiments``: List of ExperimentResult objects
        - ``summary``: Statistical summary report
        - ``config``: The loaded configuration
        - ``run_id``: Unique harness run identifier
        - ``generated_at``: Timestamp of run completion

    Raises
    ------
    ValueError
        If configuration is missing required fields (models, probes, dataset).
    FileNotFoundError
        If the configuration file doesn't exist.

    Examples
    --------
    Basic harness execution:

        >>> from insideLLMs.runtime.runner import run_harness_from_config
        >>>
        >>> # harness.yaml:
        >>> # models:
        >>> #   - type: openai
        >>> #     args:
        >>> #       model_name: gpt-4
        >>> #   - type: anthropic
        >>> #     args:
        >>> #       model_name: claude-3-opus-20240229
        >>> # probes:
        >>> #   - type: factuality
        >>> #     args: {}
        >>> #   - type: bias
        >>> #     args: {}
        >>> # dataset:
        >>> #   format: jsonl
        >>> #   path: data/questions.jsonl
        >>> # max_examples: 100
        >>>
        >>> results = run_harness_from_config("harness.yaml")
        >>> print(f"Total records: {len(results['records'])}")
        >>> print(f"Experiments: {len(results['experiments'])}")

    Accessing the summary report:

        >>> results = run_harness_from_config("harness.yaml")
        >>> summary = results["summary"]
        >>> for model_name, scores in summary.items():
        ...     print(f"{model_name}: {scores}")

    With progress tracking:

        >>> def on_progress(current, total):
        ...     pct = current / total * 100
        ...     print(f"\\rHarness progress: {pct:.1f}%", end="")
        >>>
        >>> results = run_harness_from_config(
        ...     "harness.yaml",
        ...     progress_callback=on_progress
        ... )

    Analyzing individual experiments:

        >>> results = run_harness_from_config("harness.yaml")
        >>> for experiment in results["experiments"]:
        ...     print(f"Model: {experiment.model_info.name}")
        ...     print(f"Probe: {experiment.probe_name}")
        ...     print(f"Score: {experiment.score}")
        ...     print()

    See Also
    --------
    run_experiment_from_config : Run single model/probe experiment.
    run_experiment_from_config_async : Async version of single experiment.
    """
    validator_mode = _normalize_validation_mode(validation_mode)
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    config_snapshot = _build_resolved_config_snapshot(config, base_dir)
    strict_serialization, deterministic_artifacts = _resolve_determinism_options(
        config_snapshot,
        strict_override=strict_serialization,
        deterministic_artifacts_override=deterministic_artifacts,
    )
    try:
        harness_run_id = _deterministic_run_id_from_config_snapshot(
            config_snapshot,
            schema_version=schema_version,
            strict_serialization=strict_serialization,
        )
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable values in the resolved harness config."
        ) from exc
    harness_base_time = _deterministic_base_time(harness_run_id)
    harness_generated_at, _ = _deterministic_run_times(harness_base_time, 0)

    models_config = config_snapshot.get("models", [])
    probes_config = config_snapshot.get("probes", [])
    dataset_config = config_snapshot.get("dataset")

    if not models_config:
        raise ValueError("Harness config requires at least one model in 'models'.")
    if not probes_config:
        raise ValueError("Harness config requires at least one probe in 'probes'.")
    if not dataset_config:
        raise ValueError("Harness config requires a 'dataset' section.")

    dataset = _load_dataset_from_config(dataset_config, base_dir)
    max_examples = config.get("max_examples")
    if max_examples:
        dataset = dataset[:max_examples]

    global_probe_kwargs = _extract_probe_kwargs_from_config(config_snapshot)

    experiments: list[ExperimentResult] = []
    records: list[dict[str, Any]] = []

    total_items = len(dataset) * len(models_config) * len(probes_config)
    dataset_ref = dataset_config.get("name") or dataset_config.get("path") or "dataset"
    dataset_format = dataset_config.get("format")

    def _model_spec_for_harness(model_obj: Model, model_cfg: dict[str, Any]) -> dict[str, Any]:
        info_obj: Any = {}
        try:
            info_obj = getattr(model_obj, "info", lambda: {})() or {}
        except (AttributeError, TypeError):
            info_obj = {}
        info = _normalize_info_obj_to_dict(info_obj)
        model_id = (
            info.get("model_id")
            or info.get("id")
            or info.get("name")
            or info.get("model_name")
            or getattr(model_obj, "model_id", None)
            or getattr(model_obj, "name", None)
            or model_obj.__class__.__name__
        )
        provider = (
            info.get("provider")
            or model_cfg.get("type")
            or info.get("type")
            or info.get("model_type")
        )
        provider = str(provider) if provider is not None else None
        return {"model_id": str(model_id), "provider": provider, "params": info}

    def _probe_spec_for_harness(probe_obj: Probe, probe_cfg: dict[str, Any]) -> dict[str, Any]:
        probe_id = (
            getattr(probe_obj, "name", None)
            or probe_cfg.get("type")
            or probe_obj.__class__.__name__
        )
        probe_version = getattr(probe_obj, "version", None) or getattr(
            probe_obj, "probe_version", None
        )
        return {"probe_id": str(probe_id), "probe_version": probe_version, "params": {}}

    def _dataset_spec_for_harness(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
        dataset_id = (
            dataset_cfg.get("name") or dataset_cfg.get("dataset") or dataset_cfg.get("path")
        )
        dataset_version = dataset_cfg.get("version") or dataset_cfg.get("dataset_version")
        if dataset_cfg.get("revision"):
            if dataset_version:
                dataset_version = f"{dataset_version}@{dataset_cfg.get('revision')}"
            else:
                dataset_version = str(dataset_cfg.get("revision"))
        if dataset_cfg.get("split"):
            if dataset_version:
                dataset_version = f"{dataset_version}::{dataset_cfg.get('split')}"
            else:
                dataset_version = str(dataset_cfg.get("split"))
        dataset_hash = dataset_cfg.get("hash") or dataset_cfg.get("dataset_hash")
        provenance = (
            dataset_cfg.get("provenance") or dataset_cfg.get("source") or dataset_cfg.get("format")
        )
        return {
            "dataset_id": str(dataset_id) if dataset_id is not None else None,
            "dataset_version": str(dataset_version) if dataset_version is not None else None,
            "dataset_hash": str(dataset_hash) if dataset_hash is not None else None,
            "provenance": str(provenance) if provenance is not None else None,
            "params": dataset_cfg,
        }

    for model_idx, model_config in enumerate(models_config):
        model = _create_model_from_config(model_config)
        model_info = _coerce_model_info(model)
        model_spec = _model_spec_for_harness(model, model_config)

        for probe_idx, probe_config in enumerate(probes_config):
            probe = _create_probe_from_config(probe_config)
            probe_spec = _probe_spec_for_harness(probe, probe_config)
            dataset_spec = _dataset_spec_for_harness(dataset_config)
            run_offset = (model_idx * len(probes_config) + probe_idx) * len(dataset)
            experiment_id = _deterministic_harness_experiment_id(
                model_config=model_config,
                probe_config=probe_config,
                dataset_config=dataset_config,
                model_index=model_idx,
                probe_index=probe_idx,
                max_examples=max_examples,
                schema_version=schema_version,
                harness_run_id=harness_run_id,
                strict_serialization=strict_serialization,
            )
            experiment_base_time = _deterministic_base_time(experiment_id)
            experiment_started_at, experiment_completed_at = _deterministic_run_times(
                experiment_base_time,
                len(dataset),
            )

            def local_progress(current: int, total: int) -> None:
                if progress_callback and total_items:
                    progress_callback(run_offset + current, total_items)

            probe_kwargs = dict(global_probe_kwargs)
            probe_kwargs.update(_extract_probe_kwargs_from_config(probe_config))

            results = ProbeRunner(model, probe).run(
                dataset,
                progress_callback=local_progress if progress_callback else None,
                validate_output=validate_output,
                schema_version=schema_version,
                validation_mode=validation_mode,
                emit_run_artifacts=False,
                run_id=experiment_id,
                strict_serialization=strict_serialization,
                deterministic_artifacts=deterministic_artifacts,
                **probe_kwargs,
            )

            experiment_config = {
                "model": model_config,
                "probe": probe_config,
                "dataset": dataset_config,
            }

            experiment = create_experiment_result(
                model,
                probe,
                results,
                config=experiment_config,
                experiment_id=experiment_id,
                started_at=experiment_started_at,
                completed_at=experiment_completed_at,
                strict_serialization=strict_serialization,
            )
            experiments.append(experiment)

            probe_category = getattr(probe, "category", ProbeCategory.CUSTOM)
            if isinstance(probe_category, ProbeCategory):
                probe_category_str = probe_category.value
            else:
                probe_category_str = str(probe_category)

            for example_index, result in enumerate(results):
                record_started_at, record_completed_at = _deterministic_item_times(
                    experiment_base_time,
                    example_index,
                )
                record = _build_result_record(
                    schema_version=schema_version,
                    run_id=harness_run_id,
                    started_at=record_started_at,
                    completed_at=record_completed_at,
                    model=model_spec,
                    probe=probe_spec,
                    dataset=dataset_spec,
                    item=result.get("input"),
                    output=result.get("output"),
                    latency_ms=result.get("latency_ms"),
                    store_messages=False,
                    index=example_index,
                    status=str(result.get("status")),
                    error=None,
                    strict_serialization=strict_serialization,
                )

                # Preserve any error details coming from ProbeRunner.run()'s raw results.
                record["error"] = result.get("error")
                record["error_type"] = result.get("error_type")

                # Keep harness-specific grouping fields, but nest them under `custom`
                # so the top-level record matches the standard ResultRecord schema.
                record_custom = (
                    record.get("custom") if isinstance(record.get("custom"), dict) else {}
                )
                record_custom["harness"] = {
                    "experiment_id": experiment.experiment_id,
                    "model_type": model_config.get("type"),
                    "model_name": model_info.name,
                    "model_id": model_info.model_id,
                    "probe_type": probe_config.get("type"),
                    "probe_name": probe.name,
                    "probe_category": probe_category_str,
                    "dataset": dataset_ref,
                    "dataset_format": dataset_format,
                    "example_index": example_index,
                }
                record["custom"] = record_custom
                records.append(record)

    summary = generate_summary_report(
        experiments,
        include_ci=True,
        confidence_level=config.get("confidence_level", 0.95),
    )

    if validate_output:
        from insideLLMs.schemas import OutputValidator, SchemaRegistry

        registry = SchemaRegistry()
        validator = OutputValidator()
        for record in records:
            validator.validate(
                registry.RESULT_RECORD,
                record,
                schema_version=schema_version,
                mode=validator_mode,
            )

    return {
        "records": records,
        "experiments": experiments,
        "summary": summary,
        "config": config,
        "config_snapshot": config_snapshot,
        "run_id": harness_run_id,
        "generated_at": harness_generated_at,
        "strict_serialization": strict_serialization,
        "deterministic_artifacts": deterministic_artifacts,
    }


async def run_experiment_from_config_async(
    config_path: Union[str, Path],
    *,
    concurrency: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    validate_output: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
    emit_run_artifacts: bool = True,
    run_dir: Optional[Union[str, Path]] = None,
    run_root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    overwrite: bool = False,
    resume: bool = False,
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
    use_probe_batch: bool = False,
    batch_workers: Optional[int] = None,
    return_experiment: bool = False,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run an experiment asynchronously from a configuration file.

    Like :func:`run_experiment_from_config`, but uses async execution for
    concurrent processing, which can significantly speed up API-based workloads.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML (.yaml, .yml) or JSON (.json) configuration file.
    concurrency : int, default 5
        Maximum number of concurrent executions.
    progress_callback : Optional[Callable[[int, int], None]], default None
        Callback for progress updates with signature (current, total).
    validate_output : bool, default False
        If True, validate outputs against the schema.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version for validation.
    validation_mode : str, default "strict"
        Validation mode: "strict" or "lenient" (alias: "warn").
    emit_run_artifacts : bool, default True
        If True, write records.jsonl and manifest.json to run directory.
    run_dir : Optional[Union[str, Path]], default None
        Explicit directory for run artifacts.
    run_root : Optional[Union[str, Path]], default None
        Root directory under which run directories are created.
    run_id : Optional[str], default None
        Explicit run ID. If None, generated deterministically from config.
    overwrite : bool, default False
        If True, overwrite existing run directory.
    resume : bool, default False
        If True, resume from existing records.jsonl.
    strict_serialization : Optional[bool], default None
        If True, fail fast on non-deterministic values during hashing.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields (platform/python).
    use_probe_batch : bool, default False
        If True, use Probe.run_batch (runs in thread pool).
    batch_workers : Optional[int], default None
        Number of workers for Probe.run_batch.
    return_experiment : bool, default False
        If True, return ExperimentResult instead of list of dicts.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        If return_experiment is False: list of result dictionaries.
        If return_experiment is True: ExperimentResult with aggregated metrics.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    ValueError
        If the configuration is invalid or has unknown model/probe types.

    Examples
    --------
    Basic async execution from config:

        >>> import asyncio
        >>> from insideLLMs.runtime.runner import run_experiment_from_config_async
        >>>
        >>> async def main():
        ...     results = await run_experiment_from_config_async(
        ...         "experiment.yaml",
        ...         concurrency=10
        ...     )
        ...     return results
        >>>
        >>> results = asyncio.run(main())

    With progress tracking:

        >>> async def main_with_progress():
        ...     def on_progress(current, total):
        ...         print(f"\\rProgress: {current}/{total}", end="")
        ...
        ...     results = await run_experiment_from_config_async(
        ...         "experiment.yaml",
        ...         concurrency=20,
        ...         progress_callback=on_progress
        ...     )
        ...     return results

    Getting ExperimentResult:

        >>> async def main_experiment():
        ...     experiment = await run_experiment_from_config_async(
        ...         "experiment.yaml",
        ...         concurrency=10,
        ...         return_experiment=True
        ...     )
        ...     print(f"Score: {experiment.score}")
        ...     return experiment

    Comparing timing with sync version:

        >>> import time
        >>>
        >>> # Sync version (for reference)
        >>> start = time.time()
        >>> results_sync = run_experiment_from_config("experiment.yaml")
        >>> print(f"Sync: {time.time() - start:.1f}s")
        >>>
        >>> # Async version with concurrency
        >>> async def timed_async():
        ...     start = time.time()
        ...     results = await run_experiment_from_config_async(
        ...         "experiment.yaml",
        ...         concurrency=10
        ...     )
        ...     print(f"Async: {time.time() - start:.1f}s")
        ...     return results

    See Also
    --------
    run_experiment_from_config : Synchronous version.
    run_harness_from_config : Run multi-model/multi-probe harness.
    AsyncProbeRunner : Direct access to async runner.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    config_snapshot = _build_resolved_config_snapshot(config, base_dir)
    strict_serialization, deterministic_artifacts = _resolve_determinism_options(
        config_snapshot,
        strict_override=strict_serialization,
        deterministic_artifacts_override=deterministic_artifacts,
    )
    try:
        resolved_run_id = run_id or _deterministic_run_id_from_config_snapshot(
            config_snapshot,
            schema_version=schema_version,
            strict_serialization=strict_serialization,
        )
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable values in the resolved config snapshot."
        ) from exc

    model = _create_model_from_config(config["model"])
    probe = _create_probe_from_config(config["probe"])
    dataset = _load_dataset_from_config(config["dataset"], base_dir)

    probe_kwargs = _extract_probe_kwargs_from_config(config_snapshot)
    probe_kwargs.update(_extract_probe_kwargs_from_config(config_snapshot.get("probe")))

    runner = AsyncProbeRunner(model, probe)
    return await runner.run(
        dataset,
        concurrency=concurrency,
        progress_callback=progress_callback,
        validate_output=validate_output,
        schema_version=schema_version,
        validation_mode=validation_mode,
        emit_run_artifacts=emit_run_artifacts,
        run_dir=run_dir,
        run_root=run_root,
        run_id=resolved_run_id,
        overwrite=overwrite,
        dataset_info=config_snapshot.get("dataset"),
        config_snapshot=config_snapshot,
        resume=resume,
        strict_serialization=strict_serialization,
        deterministic_artifacts=deterministic_artifacts,
        use_probe_batch=use_probe_batch,
        batch_workers=batch_workers,
        return_experiment=return_experiment,
        **probe_kwargs,
    )


def create_experiment_result(
    model: Model,
    probe: Probe,
    results: Union[list[dict[str, Any]], list[ProbeResult]],
    config: Optional[ConfigDict] = None,
    *,
    experiment_id: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    strict_serialization: bool = False,
) -> ExperimentResult:
    """Create a structured ExperimentResult from raw results.

    Converts raw result dictionaries or ProbeResult objects into a structured
    ExperimentResult that includes aggregated scores, model/probe metadata,
    and timing information.

    Parameters
    ----------
    model : Model
        The model that was tested.
    probe : Probe
        The probe that was run.
    results : Union[list[dict[str, Any]], list[ProbeResult]]
        The raw results, either as dictionaries with "input", "output",
        "status", etc., or as ProbeResult objects.
    config : Optional[ConfigDict], default None
        Optional configuration dictionary used for the experiment.
    experiment_id : Optional[str], default None
        Unique experiment identifier. If None, generated deterministically
        from inputs.
    started_at : Optional[datetime], default None
        Experiment start time. If None, generated deterministically.
    completed_at : Optional[datetime], default None
        Experiment completion time. If None, generated deterministically.

    Returns
    -------
    ExperimentResult
        Structured result containing:

        - ``experiment_id``: Unique identifier
        - ``model_info``: ModelInfo with model metadata
        - ``probe_name``: Name of the probe
        - ``probe_category``: Category of the probe
        - ``results``: List of ProbeResult objects
        - ``score``: Aggregated score from probe.score()
        - ``started_at``: Start timestamp
        - ``completed_at``: Completion timestamp
        - ``config``: Configuration dictionary

    Examples
    --------
    Creating from raw result dictionaries:

        >>> from insideLLMs.runtime.runner import create_experiment_result
        >>>
        >>> results = [
        ...     {"input": "Q1", "output": "A1", "status": "success"},
        ...     {"input": "Q2", "output": "A2", "status": "success"},
        ... ]
        >>> experiment = create_experiment_result(model, probe, results)
        >>> print(f"Score: {experiment.score}")
        >>> print(f"Success count: {sum(1 for r in experiment.results if r.status.value == 'success')}")

    Creating from ProbeResult objects:

        >>> from insideLLMs.types import ProbeResult, ResultStatus
        >>>
        >>> probe_results = [
        ...     ProbeResult(input="Q1", output="A1", status=ResultStatus.SUCCESS),
        ...     ProbeResult(input="Q2", output="A2", status=ResultStatus.ERROR, error="Failed"),
        ... ]
        >>> experiment = create_experiment_result(model, probe, probe_results)

    With explicit experiment ID and timestamps:

        >>> from datetime import datetime
        >>>
        >>> experiment = create_experiment_result(
        ...     model, probe, results,
        ...     experiment_id="exp_001",
        ...     started_at=datetime(2024, 1, 1, 12, 0),
        ...     completed_at=datetime(2024, 1, 1, 12, 30),
        ... )

    Including configuration:

        >>> config = {
        ...     "model": {"type": "openai", "args": {"model_name": "gpt-4"}},
        ...     "probe": {"type": "factuality"},
        ... }
        >>> experiment = create_experiment_result(model, probe, results, config=config)
        >>> print(experiment.config)

    See Also
    --------
    ExperimentResult : The structured result type.
    ProbeRunner.run : Returns ExperimentResult when return_experiment=True.
    """

    def _coerce_status(value: Any) -> ResultStatus:
        if isinstance(value, ResultStatus):
            return value
        if isinstance(value, Enum):
            try:
                return ResultStatus(str(value.value))
            except Exception:
                return ResultStatus.ERROR
        try:
            return ResultStatus(str(value))
        except Exception:
            return ResultStatus.ERROR

    probe_results: list[ProbeResult] = []
    if results and isinstance(results[0], ProbeResult):
        probe_results = list(results)  # type: ignore[list-item]
    else:
        for r in results:
            status = _coerce_status(r.get("status"))
            probe_results.append(
                ProbeResult(
                    input=r.get("input"),
                    output=r.get("output"),
                    status=status,
                    error=r.get("error"),
                    latency_ms=r.get("latency_ms"),
                    metadata=r.get("metadata") or {},
                )
            )

    # Calculate scores
    score = probe.score(probe_results) if hasattr(probe, "score") else None

    # Determine category
    category = getattr(probe, "category", ProbeCategory.CUSTOM)
    if not isinstance(category, ProbeCategory):
        try:
            category = ProbeCategory(str(category))
        except Exception:
            category = ProbeCategory.CUSTOM

    resolved_experiment_id = experiment_id
    if resolved_experiment_id is None:
        if probe_results:
            inputs = [r.input for r in probe_results]
        else:
            inputs = []
        try:
            resolved_experiment_id = _deterministic_hash(
                {
                    "model": _serialize_value(
                        _coerce_model_info(model),
                        strict=strict_serialization,
                    ),
                    "probe": {
                        "name": probe.name,
                        "version": getattr(probe, "version", None),
                    },
                    "inputs_hash": _hash_prompt_set(inputs, strict=strict_serialization),
                },
                strict=strict_serialization,
            )[:12]
        except StrictSerializationError as exc:
            raise ValueError(
                "strict_serialization requires JSON-stable inputs for experiment_id derivation."
            ) from exc

    if started_at is None or completed_at is None:
        base_time = _deterministic_base_time(resolved_experiment_id)
        deterministic_started, deterministic_completed = _deterministic_run_times(
            base_time,
            len(results),
        )
        if started_at is None:
            started_at = deterministic_started
        if completed_at is None:
            completed_at = deterministic_completed

    return ExperimentResult(
        experiment_id=resolved_experiment_id,
        model_info=_coerce_model_info(model),
        probe_name=probe.name,
        probe_category=category,
        results=probe_results,
        score=score,
        started_at=started_at,
        completed_at=completed_at,
        config=config or {},
    )


def _deterministic_harness_experiment_id(
    *,
    model_config: dict[str, Any],
    probe_config: dict[str, Any],
    dataset_config: dict[str, Any],
    model_index: int,
    probe_index: int,
    max_examples: Optional[int],
    schema_version: str,
    harness_run_id: str,
    strict_serialization: bool = False,
) -> str:
    payload = {
        "schema_version": schema_version,
        "harness_run_id": harness_run_id,
        "model": model_config,
        "probe": probe_config,
        "dataset": dataset_config,
        "max_examples": max_examples,
        "model_index": model_index,
        "probe_index": probe_index,
    }
    return _deterministic_hash(payload, strict=strict_serialization)[:16]


def derive_run_id_from_config_path(
    config_path: Union[str, Path],
    *,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    strict_serialization: Optional[bool] = None,
) -> str:
    """Derive a deterministic run ID from a configuration file.

    Generates a unique run ID based on the contents of the configuration file.
    The same configuration will always produce the same run ID, enabling
    reproducibility and caching.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML or JSON configuration file.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version to include in the hash.

    Returns
    -------
    str
        A 32-character hexadecimal run ID derived from the configuration.

    Examples
    --------
    Get the run ID before running an experiment:

        >>> from insideLLMs.runtime.runner import derive_run_id_from_config_path
        >>>
        >>> run_id = derive_run_id_from_config_path("experiment.yaml")
        >>> print(f"Run ID: {run_id}")
        Run ID: a1b2c3d4e5f6789012345678901234ab

    Check if a run already exists:

        >>> from pathlib import Path
        >>>
        >>> run_id = derive_run_id_from_config_path("experiment.yaml")
        >>> run_dir = Path.home() / ".insidellms" / "runs" / run_id
        >>> if run_dir.exists():
        ...     print("Run already exists, loading results...")
        ... else:
        ...     print("New run, executing experiment...")

    Using with run_experiment_from_config:

        >>> run_id = derive_run_id_from_config_path("experiment.yaml")
        >>> results = run_experiment_from_config(
        ...     "experiment.yaml",
        ...     run_id=run_id  # Optional, would be the same anyway
        ... )

    See Also
    --------
    run_experiment_from_config : Runs an experiment from config.
    load_config : Loads a configuration file.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    config_snapshot = _build_resolved_config_snapshot(config, config_path.parent)
    strict_mode, _ = _resolve_determinism_options(
        config_snapshot,
        strict_override=strict_serialization,
        deterministic_artifacts_override=None,
    )
    try:
        return _deterministic_run_id_from_config_snapshot(
            config_snapshot,
            schema_version=schema_version,
            strict_serialization=strict_mode,
        )
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable values in the config snapshot."
        ) from exc
