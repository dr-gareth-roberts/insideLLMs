"""Asynchronous probe runner implementation.

This module provides the AsyncProbeRunner class for concurrent probe execution.
"""

import asyncio
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
from insideLLMs.runtime.timeout_wrapper import run_with_timeout
from insideLLMs.types import (
    ExperimentResult,
    ProbeResult,
    ResultStatus,
)
from insideLLMs.validation import validate_prompt_set

logger = logging.getLogger(__name__)


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

        Parameters
        ----------
        prompt_set : list[Any]
            List of inputs to test.
        config : Optional[RunConfig], default None
            RunConfig with all run settings.
        stop_on_error : Optional[bool], default None
            If True, stop execution on first error.
        concurrency : Optional[int], default None
            Maximum number of concurrent executions.
        timeout : Optional[float], default None
            Per-item timeout in seconds.
        progress_callback : Optional[ProgressCallback], default None
            Callback for progress updates.
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
            Dataset metadata.
        config_snapshot : Optional[dict[str, Any]], default None
            Configuration snapshot.
        store_messages : Optional[bool], default None
            If True, store full message content.
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
            If True, return ExperimentResult.
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

        # Resolve config
        if config is None:
            config = RunConfig()

        # Override config with explicit kwargs
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
        run_mode = getattr(config, "run_mode", "default")

        # Validate prompt set
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

        effective_model = self.model
        if emit_run_artifacts and run_mode == "ultimate":
            receipts_dir = resolved_run_dir / "receipts"
            receipts_dir.mkdir(parents=True, exist_ok=True)
            receipt_sink = receipts_dir / "calls.jsonl"
            from insideLLMs.runtime.pipeline import AsyncModelPipeline
            from insideLLMs.runtime.receipt import ReceiptMiddleware

            effective_model = AsyncModelPipeline(
                self.model,
                middlewares=[ReceiptMiddleware(receipt_sink=receipt_sink)],
            )

        semaphore = asyncio.Semaphore(concurrency)
        results: list[Optional[dict[str, Any]]] = [None] * len(prompt_set)
        errors: list[Optional[BaseException]] = [None] * len(prompt_set)
        stop_event = asyncio.Event()
        stop_error: Optional[RunnerExecutionError] = None
        completed = 0
        completed_lock = asyncio.Lock()
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
                    record_metadata = result_obj.get("metadata")
                    if isinstance(record_metadata, dict) and isinstance(record.get("custom"), dict):
                        timeout_seconds = record_metadata.get("timeout_seconds")
                        if isinstance(timeout_seconds, (int, float)):
                            record["custom"]["timeout_seconds"] = float(timeout_seconds)
                        if str(result_obj.get("status") or "") == "timeout":
                            record["custom"]["timeout"] = True
                    if validate_output:
                        validator.validate(
                            registry.RESULT_RECORD,
                            record,
                            schema_version=schema_version,
                            mode=validator_mode,
                        )
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
                    loop = asyncio.get_running_loop()

                    async def execute_probe() -> Any:
                        return await loop.run_in_executor(
                            None,
                            lambda: self.probe.run(effective_model, item, **probe_kwargs),
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
                    is_timeout = False
                    if isinstance(e, ProbeExecutionError):
                        reason = str(getattr(e, "details", {}).get("reason", "")).lower()
                        is_timeout = "timed out" in reason

                    metadata: dict[str, Any] = {"error_type": type(e).__name__}
                    if is_timeout and timeout is not None:
                        metadata["timeout_seconds"] = float(timeout)

                    probe_result = ProbeResult(
                        input=item,
                        status=ResultStatus.TIMEOUT if is_timeout else ResultStatus.ERROR,
                        error=str(e),
                        latency_ms=None,
                        metadata=metadata,
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
                            effective_model,
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

        return self.last_experiment if return_experiment else final_results


__all__ = ["AsyncProbeRunner"]
