"""Synchronous probe runner implementation.

This module provides the ProbeRunner class for sequential probe execution.
"""

import json
import logging
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional, Union

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    serialize_value as _serialize_value,
)
from insideLLMs._serialization import (
    stable_json_dumps as _stable_json_dumps,
)
from insideLLMs.config_types import RunConfig
from insideLLMs.exceptions import ProbeExecutionError, RunnerExecutionError
from insideLLMs.runtime._artifact_utils import (
    _atomic_write_text,
    _atomic_write_yaml,
    _default_run_root,
    _ensure_run_sentinel,
    _prepare_run_dir,
    _prepare_run_dir_for_resume,
    _read_jsonl_records,
    _semver_tuple,
    _validate_resume_record,
)
from insideLLMs.runtime._base import (
    ProgressCallback,
    _invoke_progress_callback,
    _normalize_validation_mode,
    _RunnerBase,
)
from insideLLMs.runtime._determinism import (
    _deterministic_base_time,
    _deterministic_item_times,
    _deterministic_run_id_from_config_snapshot,
    _deterministic_run_id_from_inputs,
    _deterministic_run_times,
)
from insideLLMs.runtime._result_utils import (
    _build_dataset_spec,
    _build_model_spec,
    _build_probe_spec,
    _build_result_record,
    _normalize_status,
    _result_dict_from_probe_result,
    _result_dict_from_record,
)
from insideLLMs.types import (
    ExperimentResult,
    ProbeResult,
    ResultStatus,
)
from insideLLMs.validation import validate_prompt_set

logger = logging.getLogger(__name__)


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
        config : Optional[RunConfig], default None
            RunConfig with all run settings. Individual kwargs override config.
        progress_callback : Optional[ProgressCallback], default None
            Callback for progress updates.
        stop_on_error : Optional[bool], default None
            If True, stop execution on first error.
        validate_output : Optional[bool], default None
            If True, validate outputs against the schema.
        schema_version : Optional[str], default None
            Schema version for validation.
        validation_mode : Optional[str], default None
            Validation mode: "strict" or "lenient".
        emit_run_artifacts : Optional[bool], default None
            If True, write records.jsonl and manifest.json.
        run_dir : Optional[Union[str, Path]], default None
            Explicit directory for run artifacts.
        run_root : Optional[Union[str, Path]], default None
            Root directory for run directories.
        run_id : Optional[str], default None
            Explicit run ID.
        overwrite : Optional[bool], default None
            If True, overwrite existing run directory.
        dataset_info : Optional[dict[str, Any]], default None
            Dataset metadata for the manifest.
        config_snapshot : Optional[dict[str, Any]], default None
            Configuration snapshot for reproducibility.
        store_messages : Optional[bool], default None
            If True, store full message content in records.
        strict_serialization : Optional[bool], default None
            If True, fail on non-deterministic values.
        deterministic_artifacts : Optional[bool], default None
            If True, omit host-dependent manifest fields.
        resume : Optional[bool], default None
            If True, resume from existing records.
        use_probe_batch : Optional[bool], default None
            If True, use Probe.run_batch.
        batch_workers : Optional[int], default None
            Number of workers for Probe.run_batch.
        return_experiment : Optional[bool], default None
            If True, return ExperimentResult instead of list.
        **probe_kwargs : Any
            Additional kwargs passed to the probe.

        Returns
        -------
        Union[list[dict[str, Any]], ExperimentResult]
            List of result dicts or ExperimentResult if return_experiment=True.
        """
        # Import here to avoid circular import
        from insideLLMs.runtime._high_level import create_experiment_result
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
        run_mode = getattr(config, "run_mode", "default")

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

        # Ultimate mode: receipt sink and wrap model so every call is logged
        effective_model = self.model
        if emit_run_artifacts and run_mode == "ultimate":
            receipts_dir = resolved_run_dir / "receipts"
            receipts_dir.mkdir(parents=True, exist_ok=True)
            receipt_sink = receipts_dir / "calls.jsonl"
            from insideLLMs.runtime.pipeline import ModelPipeline
            from insideLLMs.runtime.receipt import ReceiptMiddleware

            effective_model = ModelPipeline(
                self.model,
                middlewares=[ReceiptMiddleware(receipt_sink=receipt_sink)],
            )

        results: list[Optional[dict[str, Any]]] = [None] * len(prompt_set)
        completed = 0
        total = len(prompt_set)
        run_start_time = time.perf_counter()

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
                        effective_model,
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
                            if isinstance(probe_result.metadata, dict) and isinstance(
                                record.get("custom"), dict
                            ):
                                timeout_seconds = probe_result.metadata.get("timeout_seconds")
                                if isinstance(timeout_seconds, (int, float)):
                                    record["custom"]["timeout_seconds"] = float(timeout_seconds)
                                if _normalize_status(probe_result.status) == "timeout":
                                    record["custom"]["timeout"] = True
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
                        output = self.probe.run(effective_model, item, **probe_kwargs)
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
                        is_timeout = False
                        if isinstance(e, ProbeExecutionError):
                            reason = str(getattr(e, "details", {}).get("reason", "")).lower()
                            is_timeout = "timed out" in reason

                        probe_result = ProbeResult(
                            input=item,
                            status=ResultStatus.TIMEOUT if is_timeout else ResultStatus.ERROR,
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
                                status="timeout" if is_timeout else "error",
                                error=e,
                                strict_serialization=strict_serialization,
                            )
                            if is_timeout and isinstance(record.get("custom"), dict):
                                record["custom"]["timeout"] = True
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
            status_counts = {
                "success": sum(1 for r in final_results if r.get("status") == "success"),
                "error": sum(1 for r in final_results if r.get("status") == "error"),
                "timeout": sum(1 for r in final_results if r.get("status") == "timeout"),
            }

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
                "success_count": status_counts["success"],
                "error_count": status_counts["error"],
                "records_file": "records.jsonl",
                "schemas": {
                    registry.RESULT_RECORD: schema_version,
                    registry.RUN_MANIFEST: schema_version,
                    registry.RUNNER_ITEM: schema_version,
                },
                "custom": {
                    "status_counts": status_counts,
                    "timeout_count": status_counts["timeout"],
                },
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

            if run_mode == "ultimate":
                from insideLLMs.runtime._ultimate import run_ultimate_post_artifact

                try:
                    import insideLLMs as _pkg

                    _ver = getattr(_pkg, "__version__", None)
                except ImportError:
                    _ver = None
                run_ultimate_post_artifact(
                    resolved_run_dir,
                    dataset_spec=dataset_spec,
                    config_snapshot=config_snapshot,
                    insidellms_version=_ver,
                    publish_oci_ref=config.publish_oci_ref,
                    scitt_service_url=config.scitt_service_url,
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
        error_count = sum(1 for r in final_results if r.get("status") == "error")
        timeout_count = sum(1 for r in final_results if r.get("status") == "timeout")
        logger.info(
            "Sync probe run completed",
            extra={
                "run_id": resolved_run_id,
                "total_items": len(final_results),
                "success_count": success_count,
                "error_count": error_count,
                "timeout_count": timeout_count,
                "success_rate": success_count / len(final_results) if final_results else 0,
            },
        )

        return self.last_experiment if return_experiment else final_results


__all__ = ["ProbeRunner"]
