"""Experiment runner with YAML/JSON config support.

This module provides tools for running LLM experiments, either programmatically
or from configuration files. Supports async execution for parallel processing.
"""

import asyncio
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml

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

    # Check callback signature to determine which style to use
    sig = inspect.signature(callback)
    params = list(sig.parameters.values())

    # If callback takes 2 positional args, use legacy (current, total) style
    if len(params) == 2:
        callback(current, total)  # type: ignore
    else:
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


class _RunnerBase:
    """Base class for ProbeRunner and AsyncProbeRunner with shared functionality.

    This class provides common initialization, properties, and helper methods
    used by both synchronous and asynchronous runners.
    """

    def __init__(self, model: Model, probe: Probe):
        """Initialize the runner.

        Args:
            model: The model to test.
            probe: The probe to run.
        """
        self.model = model
        self.probe = probe
        self._results: list[dict[str, Any]] = []
        self.last_run_id: Optional[str] = None
        self.last_run_dir: Optional[Path] = None
        self.last_experiment: Optional[ExperimentResult] = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the last run."""
        if not self._results:
            return 0.0
        successes = sum(1 for r in self._results if r.get("status") == "success")
        return successes / len(self._results)

    @property
    def error_count(self) -> int:
        """Count errors from the last run."""
        return sum(1 for r in self._results if r.get("status") == "error")


class ProbeRunner(_RunnerBase):
    """Runs a probe against a model on a dataset.

    The ProbeRunner orchestrates the execution of probes, handling error
    recovery, progress tracking, and result aggregation.

    Attributes:
        model: The model to test.
        probe: The probe to run.

    Example:
        >>> runner = ProbeRunner(model, probe)
        >>> results = runner.run(dataset)
        >>> print(f"Success rate: {runner.success_rate:.1%}")
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
        resume: Optional[bool] = None,
        use_probe_batch: Optional[bool] = None,
        batch_workers: Optional[int] = None,
        return_experiment: Optional[bool] = None,
        **probe_kwargs: Any,
    ) -> Union[list[dict[str, Any]], ExperimentResult]:
        """Run the probe on the model for each item in the prompt set.

        Args:
            prompt_set: List of inputs to test.
            config: Optional RunConfig with all run settings. Individual kwargs
                override config values if both are provided.
            progress_callback: Optional callback for progress updates. Supports both
                legacy (current, total) signature and new ProgressInfo signature.
            stop_on_error: If True, stop execution on first error.
            resume: If True, resume from existing records.jsonl.
            use_probe_batch: If True, execute using Probe.run_batch.
            batch_workers: Worker count for Probe.run_batch.
            return_experiment: If True, return ExperimentResult.
            **probe_kwargs: Additional arguments passed to the probe.

        Returns:
            A list of result dictionaries or an ExperimentResult.
        """
        import time

        from insideLLMs.schemas import OutputValidator, SchemaRegistry

        # Resolve config: use provided config or create default
        if config is None:
            config = RunConfig()

        # Override config with explicit kwargs (backward compatibility)
        stop_on_error = stop_on_error if stop_on_error is not None else config.stop_on_error
        validate_output = validate_output if validate_output is not None else config.validate_output
        schema_version = schema_version if schema_version is not None else config.schema_version
        validation_mode = validation_mode if validation_mode is not None else config.validation_mode
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
            if config_snapshot is not None:
                resolved_run_id = _deterministic_run_id_from_config_snapshot(
                    config_snapshot,
                    schema_version=schema_version,
                )
            else:
                resolved_run_id = _deterministic_run_id_from_inputs(
                    schema_version=schema_version,
                    model_spec=model_spec,
                    probe_spec=probe_spec,
                    dataset_spec=dataset_spec,
                    prompt_set=prompt_set,
                    probe_kwargs=probe_kwargs,
                )
        else:
            resolved_run_id = run_id

        self.last_run_id = resolved_run_id
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
            self.last_run_dir = resolved_run_dir
            _ensure_run_sentinel(resolved_run_dir)

            # Snapshot of the fully resolved config (best-effort) for reproducibility.
            if config_snapshot is not None:
                _atomic_write_yaml(resolved_run_dir / "config.resolved.yaml", config_snapshot)
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
                    record_index = _record_index_from_record(record, default=line_index)
                    if record_index != line_index:
                        raise ValueError(
                            "Existing records are not a contiguous prefix; cannot resume safely."
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
                                latency_ms=probe_result.latency_ms,
                                store_messages=store_messages,
                                index=i,
                                status=_normalize_status(probe_result.status),
                                error=probe_result.error,
                                error_type=error_type,
                            )
                            if validate_output:
                                validator.validate(
                                    registry.RESULT_RECORD,
                                    record,
                                    schema_version=schema_version,
                                    mode=validation_mode,
                                )
                            records_fp.write(json.dumps(record, default=_serialize_value) + "\n")
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
                                    elapsed_seconds=(
                                        (probe_result.latency_ms or 0) / 1000
                                        if probe_result.latency_ms is not None
                                        else None
                                    ),
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
                    start_time = time.perf_counter()
                    try:
                        output = self.probe.run(self.model, item, **probe_kwargs)
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        probe_result = ProbeResult(
                            input=item,
                            output=output,
                            status=ResultStatus.SUCCESS,
                            latency_ms=latency_ms,
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
                                latency_ms=latency_ms,
                                store_messages=store_messages,
                                index=i,
                                status="success",
                                error=None,
                            )
                            if validate_output:
                                validator.validate(
                                    registry.RESULT_RECORD,
                                    record,
                                    schema_version=schema_version,
                                    mode=validation_mode,
                                )
                            records_fp.write(json.dumps(record, default=_serialize_value) + "\n")
                            records_fp.flush()

                    except Exception as e:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        probe_result = ProbeResult(
                            input=item,
                            status=ResultStatus.ERROR,
                            error=str(e),
                            latency_ms=latency_ms,
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
                                latency_ms=latency_ms,
                                store_messages=store_messages,
                                index=i,
                                status="error",
                                error=e,
                            )
                            if validate_output:
                                validator.validate(
                                    registry.RESULT_RECORD,
                                    record,
                                    schema_version=schema_version,
                                    mode=validation_mode,
                                )
                            records_fp.write(json.dumps(record, default=_serialize_value) + "\n")
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
                                elapsed_seconds=latency_ms / 1000,
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
            manifest = {
                "schema_version": schema_version,
                "run_id": resolved_run_id,
                "created_at": run_started_at,
                "started_at": run_started_at,
                "completed_at": run_completed_at,
                "library_version": None,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
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
            except Exception:
                pass

            if validate_output:
                validator.validate(
                    registry.RUN_MANIFEST,
                    manifest,
                    schema_version=schema_version,
                    mode=validation_mode,
                )

            _atomic_write_text(
                manifest_path,
                json.dumps(_serialize_value(manifest), indent=2, default=_serialize_value),
            )

        if validate_output:
            validator.validate(
                registry.RUNNER_OUTPUT,
                {"schema_version": schema_version, "results": final_results},
                schema_version=schema_version,
                mode=validation_mode,
            )

        self.last_experiment = create_experiment_result(
            self.model,
            self.probe,
            final_results,
            config=config_snapshot or {},
            experiment_id=resolved_run_id,
            started_at=run_started_at,
            completed_at=run_completed_at,
        )

        return self.last_experiment if return_experiment else final_results


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_serialize_value(v) for v in value]
        try:
            return sorted(normalized)
        except TypeError:
            return sorted(
                normalized,
                key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), default=str),
            )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Last-resort: keep JSONL/manifest emission resilient even for exotic objects.
    return str(value)


def _normalize_status(status: Any) -> str:
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
    if not path.exists():
        return []
    if truncate_incomplete:
        _truncate_incomplete_jsonl(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {path}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


_DETERMINISTIC_TIME_BASE = datetime(2000, 1, 1, tzinfo=timezone.utc)
_DETERMINISTIC_TIME_RANGE_SECONDS = 10 * 365 * 24 * 60 * 60


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(
        _serialize_value(data),
        sort_keys=True,
        separators=(",", ":"),
        default=_serialize_value,
    )


def _deterministic_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _fingerprint_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha256(_stable_json_dumps(value).encode("utf-8")).hexdigest()[:12]


def _default_run_root() -> Path:
    """Get the default root directory for run artifacts."""
    env_root = os.environ.get("INSIDELLMS_RUN_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / ".insidellms" / "runs"


def _semver_tuple(version: str) -> tuple[int, int, int]:
    """Convert a semver string to a tuple for comparison."""
    from insideLLMs.schemas.registry import normalize_semver

    v = normalize_semver(version)
    parts = v.split(".")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically using a temp file and rename."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def _atomic_write_yaml(path: Path, data: Any) -> None:
    """Write data to a YAML file atomically."""
    content = yaml.safe_dump(
        _serialize_value(data),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    _atomic_write_text(path, content)


def _ensure_run_sentinel(run_dir_path: Path) -> None:
    """Create a marker file to identify run directories."""
    marker = run_dir_path / ".insidellms_run"
    if not marker.exists():
        try:
            marker.write_text("insideLLMs run directory\n", encoding="utf-8")
        except Exception:
            # Don't fail the run due to marker write issues.
            pass


def _normalize_info_obj_to_dict(info_obj: Any) -> dict[str, Any]:
    """Normalize model/probe info object to a dict."""
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
    """Build model specification dict from a model object."""
    info_obj: Any = {}
    try:
        info_obj = model.info() if hasattr(model, "info") else {}
    except Exception:
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
    """Build probe specification dict from a probe object."""
    probe_id = (
        getattr(probe, "name", None) or getattr(probe, "probe_id", None) or probe.__class__.__name__
    )
    probe_version = getattr(probe, "version", None) or getattr(probe, "probe_version", None)
    return {"probe_id": str(probe_id), "probe_version": probe_version, "params": {}}


def _build_dataset_spec(dataset_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build dataset specification dict from dataset info."""
    di = dataset_info or {}
    dataset_id = di.get("name") or di.get("dataset") or di.get("path")
    dataset_version = di.get("version") or di.get("dataset_version")
    dataset_hash = di.get("hash") or di.get("dataset_hash")
    provenance = di.get("provenance") or di.get("source") or di.get("format")
    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "dataset_version": str(dataset_version) if dataset_version is not None else None,
        "dataset_hash": str(dataset_hash) if dataset_hash is not None else None,
        "provenance": str(provenance) if provenance is not None else None,
    }


def _hash_prompt_set(prompt_set: list[Any]) -> str:
    hasher = hashlib.sha256()
    for item in prompt_set:
        hasher.update(_stable_json_dumps(item).encode("utf-8"))
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
) -> str:
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
    return _deterministic_hash(payload)[:16]


def _deterministic_run_id_from_config_snapshot(
    config_snapshot: dict[str, Any],
    *,
    schema_version: str,
) -> str:
    payload = {"schema_version": schema_version, "config": config_snapshot}
    return _deterministic_hash(payload)[:32]


def _deterministic_run_id_from_inputs(
    *,
    schema_version: str,
    model_spec: dict[str, Any],
    probe_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    prompt_set: list[Any],
    probe_kwargs: dict[str, Any],
) -> str:
    payload = {
        "schema_version": schema_version,
        "model": model_spec,
        "probe": probe_spec,
        "dataset": dataset_spec,
        "prompt_set_hash": _hash_prompt_set(prompt_set),
        "probe_kwargs": probe_kwargs,
    }
    return _deterministic_hash(payload)[:32]


def _deterministic_base_time(run_id: str) -> datetime:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    seconds = int.from_bytes(digest[:8], "big") % _DETERMINISTIC_TIME_RANGE_SECONDS
    return _DETERMINISTIC_TIME_BASE + timedelta(seconds=seconds)


def _deterministic_item_times(base_time: datetime, index: int) -> tuple[datetime, datetime]:
    offset = index * 2
    started_at = base_time + timedelta(microseconds=offset)
    completed_at = base_time + timedelta(microseconds=offset + 1)
    return started_at, completed_at


def _deterministic_run_times(base_time: datetime, total: int) -> tuple[datetime, datetime]:
    if total < 0:
        total = 0
    completed_offset = total * 2 + 1
    started_at = base_time
    completed_at = base_time + timedelta(microseconds=completed_offset)
    return started_at, completed_at


def _coerce_model_info(model: Model) -> ModelInfo:
    """Return a ModelInfo instance even if model.info() returns dict-like data."""

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
                json.dumps(
                    normalized_messages,
                    sort_keys=True,
                    default=_serialize_value,
                ).encode("utf-8")
            ).hexdigest()
        except Exception:
            messages_hash = None

    input_fingerprint = _fingerprint_value(item)
    replicate_key = _replicate_key(
        model_spec=model,
        probe_spec=probe,
        dataset_spec=dataset,
        example_id=example_id,
        record_index=index,
        input_hash=messages_hash or input_fingerprint,
    )

    # Derive output_text / scoring fields (best-effort)
    output_text = None
    if isinstance(output, str):
        output_text = output
    elif isinstance(output, dict):
        output_text = output.get("output_text") or output.get("text")

    output_fingerprint = None
    if output is not None and not isinstance(output, str):
        output_fingerprint = _fingerprint_value(output)

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
    """Async version of ProbeRunner for parallel execution.

    Use this when you need to run probes concurrently, which is especially
    useful for API-based models that can handle multiple requests in parallel.

    Example:
        >>> runner = AsyncProbeRunner(model, probe)
        >>> results = await runner.run(dataset, concurrency=10)
    """

    async def run(
        self,
        prompt_set: list[Any],
        *,
        config: Optional[RunConfig] = None,
        concurrency: Optional[int] = None,
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
        resume: Optional[bool] = None,
        use_probe_batch: Optional[bool] = None,
        batch_workers: Optional[int] = None,
        return_experiment: Optional[bool] = None,
        **probe_kwargs: Any,
    ) -> Union[list[dict[str, Any]], ExperimentResult]:
        """Run the probe on all items with controlled concurrency.

        Args:
            prompt_set: List of inputs to test.
            config: Optional RunConfig with all run settings. Individual kwargs
                override config values if both are provided.
            concurrency: Maximum number of concurrent executions.
            progress_callback: Optional callback for progress updates. Supports both
                legacy (current, total) signature and new ProgressInfo signature.
            **probe_kwargs: Additional arguments passed to the probe.

        Returns:
            A list of result dictionaries or an ExperimentResult.
        """
        import time

        from insideLLMs.schemas import OutputValidator, SchemaRegistry

        # Resolve config: use provided config or create default
        if config is None:
            config = RunConfig()

        # Override config with explicit kwargs (backward compatibility)
        concurrency = concurrency if concurrency is not None else config.concurrency
        validate_output = validate_output if validate_output is not None else config.validate_output
        schema_version = schema_version if schema_version is not None else config.schema_version
        validation_mode = validation_mode if validation_mode is not None else config.validation_mode
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
            if config_snapshot is not None:
                resolved_run_id = _deterministic_run_id_from_config_snapshot(
                    config_snapshot,
                    schema_version=schema_version,
                )
            else:
                resolved_run_id = _deterministic_run_id_from_inputs(
                    schema_version=schema_version,
                    model_spec=model_spec,
                    probe_spec=probe_spec,
                    dataset_spec=dataset_spec,
                    prompt_set=prompt_set,
                    probe_kwargs=probe_kwargs,
                )
        else:
            resolved_run_id = run_id

        self.last_run_id = resolved_run_id
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
            self.last_run_dir = resolved_run_dir
            _ensure_run_sentinel(resolved_run_dir)
            if config_snapshot is not None:
                _atomic_write_yaml(resolved_run_dir / "config.resolved.yaml", config_snapshot)
        else:
            self.last_run_dir = None

        records_path = resolved_run_dir / "records.jsonl"
        manifest_path = resolved_run_dir / "manifest.json"

        semaphore = asyncio.Semaphore(concurrency)
        results: list[Optional[dict[str, Any]]] = [None] * len(prompt_set)
        errors: list[Optional[BaseException]] = [None] * len(prompt_set)
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
                    record_index = _record_index_from_record(record, default=line_index)
                    if record_index != line_index:
                        raise ValueError(
                            "Existing records are not a contiguous prefix; cannot resume safely."
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
                    )
                    if validate_output:
                        validator.validate(
                            registry.RESULT_RECORD,
                            record,
                            schema_version=schema_version,
                            mode=validation_mode,
                        )
                    records_fp.write(json.dumps(record, default=_serialize_value) + "\n")
                    records_fp.flush()
                    next_write_index += 1

        async def run_single(index: int, item: Any) -> None:
            nonlocal completed
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    # Run in thread pool for sync models
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(
                        None,
                        lambda: self.probe.run(self.model, item, **probe_kwargs),
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    probe_result = ProbeResult(
                        input=item,
                        output=output,
                        status=ResultStatus.SUCCESS,
                        latency_ms=latency_ms,
                        metadata={},
                    )
                    results[index] = _result_dict_from_probe_result(
                        probe_result,
                        schema_version=schema_version,
                    )
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    errors[index] = e
                    probe_result = ProbeResult(
                        input=item,
                        status=ResultStatus.ERROR,
                        error=str(e),
                        latency_ms=latency_ms,
                        metadata={"error_type": type(e).__name__},
                    )
                    results[index] = _result_dict_from_probe_result(
                        probe_result,
                        schema_version=schema_version,
                        error_type=type(e).__name__,
                    )
                finally:
                    completed += 1
                    _invoke_progress_callback(
                        progress_callback,
                        current=completed,
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
                        results[index] = _result_dict_from_probe_result(
                            probe_result,
                            schema_version=schema_version,
                            error_type=error_type,
                        )
                    await write_ready_records()
            else:
                tasks = [run_single(i, item) for i, item in enumerate(prompt_set) if i >= completed]
                if tasks:
                    await asyncio.gather(*tasks)
                await write_ready_records()

        if any(result is None for result in results):
            raise RuntimeError("Runner did not produce results for all items.")
        final_results = [result for result in results if result is not None]

        self._results = final_results

        _, run_completed_at = _deterministic_run_times(run_base_time, len(final_results))

        if emit_run_artifacts:
            manifest = {
                "schema_version": schema_version,
                "run_id": resolved_run_id,
                "created_at": run_started_at,
                "started_at": run_started_at,
                "completed_at": run_completed_at,
                "library_version": None,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
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
            except Exception:
                pass

            if validate_output:
                validator.validate(
                    registry.RUN_MANIFEST,
                    manifest,
                    schema_version=schema_version,
                    mode=validation_mode,
                )

            _atomic_write_text(
                manifest_path,
                json.dumps(_serialize_value(manifest), indent=2, default=_serialize_value),
            )

        if validate_output:
            validator.validate(
                registry.RUNNER_OUTPUT,
                {"schema_version": schema_version, "results": final_results},
                schema_version=schema_version,
                mode=validation_mode,
            )

        self.last_experiment = create_experiment_result(
            self.model,
            self.probe,
            final_results,
            config=config_snapshot or {},
            experiment_id=resolved_run_id,
            started_at=run_started_at,
            completed_at=run_completed_at,
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
        except Exception:
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
            except Exception:
                run_root_resolved = run_root
            if resolved == run_root_resolved:
                raise ValueError(f"Refusing to overwrite run_root directory itself: '{resolved}'")

        # Refuse very short paths (e.g. '/tmp', 'C:\\tmp') that are easy to fat-finger.
        if len(resolved.parts) <= 2:
            raise ValueError(f"Refusing to overwrite high-risk short path: '{resolved}'")

        # Sentinel requirement: only overwrite if this looks like an insideLLMs run dir.
        sentinel_ok = (path / "manifest.json").exists() or (path / ".insidellms_run").exists()
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
        except Exception:
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
    **probe_kwargs: Any,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Convenience function to run a probe on a model.

    Args:
        model: The model to test.
        probe: The probe to run.
        prompt_set: List of inputs to test.
        config: Optional RunConfig with run settings.
        **probe_kwargs: Additional arguments for the probe.

    Returns:
        A list of result dictionaries or an ExperimentResult.
    """
    runner = ProbeRunner(model, probe)
    return runner.run(prompt_set, config=config, **probe_kwargs)


async def run_probe_async(
    model: Model,
    probe: Probe,
    prompt_set: list[Any],
    *,
    config: Optional[RunConfig] = None,
    concurrency: Optional[int] = None,
    **probe_kwargs: Any,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Convenience function to run a probe asynchronously.

    Args:
        model: The model to test.
        probe: The probe to run.
        prompt_set: List of inputs to test.
        config: Optional RunConfig with run settings.
        concurrency: Maximum concurrent executions (overrides config).
        **probe_kwargs: Additional arguments for the probe.

    Returns:
        A list of result dictionaries or an ExperimentResult.
    """
    runner = AsyncProbeRunner(model, probe)
    return await runner.run(prompt_set, config=config, concurrency=concurrency, **probe_kwargs)


def load_config(path: Union[str, Path]) -> ConfigDict:
    """Load a configuration file.

    Supports YAML (.yaml, .yml) and JSON (.json) formats.

    Args:
        path: Path to the configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def _resolve_path(path: str, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory."""
    p = Path(path)
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

    snapshot: dict[str, Any] = copy.deepcopy(config)  # type: ignore[arg-type]

    dataset = snapshot.get("dataset") if isinstance(snapshot, dict) else None
    if isinstance(dataset, dict):
        fmt = dataset.get("format")
        if fmt in ("csv", "jsonl") and isinstance(dataset.get("path"), str):
            try:
                dataset["path"] = str(_resolve_path(dataset["path"], base_dir).resolve())
            except Exception:
                # Best-effort only; do not fail the run for a snapshot.
                pass

    return snapshot


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
        # Fallback to direct import for backwards compatibility
        from insideLLMs.models import (
            AnthropicModel,
            CohereModel,
            DummyModel,
            GeminiModel,
            HuggingFaceModel,
            LlamaCppModel,
            OllamaModel,
            OpenAIModel,
            VLLMModel,
        )

        model_map = {
            "dummy": DummyModel,
            "openai": OpenAIModel,
            "huggingface": HuggingFaceModel,
            "anthropic": AnthropicModel,
            "gemini": GeminiModel,
            "cohere": CohereModel,
            "llamacpp": LlamaCppModel,
            "ollama": OllamaModel,
            "vllm": VLLMModel,
        }

        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")

        base_model = model_map[model_type](**model_args)

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
        try:
            loader = dataset_registry.get_factory("hf")
            return loader(config["name"], split=config.get("split", "test"))
        except NotFoundError:
            from insideLLMs.dataset_utils import load_hf_dataset

            return load_hf_dataset(config["name"], split=config.get("split", "test"))

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
    use_probe_batch: bool = False,
    batch_workers: Optional[int] = None,
    return_experiment: bool = False,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run an experiment from a configuration file.

    The configuration file should specify:
    - model: type and args
    - probe: type and args
    - dataset: format, path/name, and optional split

    Args:
        config_path: Path to the YAML or JSON configuration file.
        progress_callback: Optional callback for progress updates.
        resume: Whether to resume from existing records.jsonl.
        use_probe_batch: Whether to use Probe.run_batch for execution.
        batch_workers: Worker count for probe batch execution.
        return_experiment: Whether to return an ExperimentResult.

    Returns:
        A list of result dictionaries or an ExperimentResult.

    Example config (YAML):
        ```yaml
        model:
          type: openai
          args:
            model_name: gpt-4
        probe:
          type: factuality
          args: {}
        dataset:
          format: jsonl
          path: data/questions.jsonl
        ```
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    config_snapshot = _build_resolved_config_snapshot(config, base_dir)
    resolved_run_id = run_id or _deterministic_run_id_from_config_snapshot(
        config_snapshot,
        schema_version=schema_version,
    )

    model = _create_model_from_config(config["model"], prefer_async_pipeline=True)
    probe = _create_probe_from_config(config["probe"])
    dataset = _load_dataset_from_config(config["dataset"], base_dir)

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
        dataset_info=config.get("dataset"),
        config_snapshot=config_snapshot,
        resume=resume,
        use_probe_batch=use_probe_batch,
        batch_workers=batch_workers,
        return_experiment=return_experiment,
    )


def run_harness_from_config(
    config_path: Union[str, Path],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    validate_output: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
) -> dict[str, Any]:
    """Run a cross-model probe harness from a configuration file.

    The configuration file should specify:
    - models: list of model configs (type + args)
    - probes: list of probe configs (type + args)
    - dataset: format, path/name, and optional split
    - max_examples: optional limit

    Args:
        config_path: Path to the YAML or JSON configuration file.
        progress_callback: Optional callback for progress updates.

    Returns:
        Dictionary with per-example records, experiments, and summary.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    config_snapshot = _build_resolved_config_snapshot(config, base_dir)
    harness_run_id = _deterministic_run_id_from_config_snapshot(
        config_snapshot,
        schema_version=schema_version,
    )
    harness_base_time = _deterministic_base_time(harness_run_id)
    harness_generated_at, _ = _deterministic_run_times(harness_base_time, 0)

    models_config = config.get("models", [])
    probes_config = config.get("probes", [])
    dataset_config = config.get("dataset")

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

    experiments: list[ExperimentResult] = []
    records: list[dict[str, Any]] = []

    total_items = len(dataset) * len(models_config) * len(probes_config)
    dataset_ref = dataset_config.get("name") or dataset_config.get("path") or "dataset"
    dataset_format = dataset_config.get("format")

    def _model_spec_for_harness(model_obj: Model, model_cfg: dict[str, Any]) -> dict[str, Any]:
        info_obj: Any = {}
        try:
            info_obj = getattr(model_obj, "info", lambda: {})() or {}
        except Exception:
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
            )
            experiment_base_time = _deterministic_base_time(experiment_id)
            experiment_started_at, experiment_completed_at = _deterministic_run_times(
                experiment_base_time,
                len(dataset),
            )

            def local_progress(current: int, total: int) -> None:
                if progress_callback and total_items:
                    progress_callback(run_offset + current, total_items)

            results = ProbeRunner(model, probe).run(
                dataset,
                progress_callback=local_progress if progress_callback else None,
                validate_output=validate_output,
                schema_version=schema_version,
                validation_mode=validation_mode,
                emit_run_artifacts=False,
                run_id=experiment_id,
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
                mode=validation_mode,
            )

    return {
        "records": records,
        "experiments": experiments,
        "summary": summary,
        "config": config,
        "run_id": harness_run_id,
        "generated_at": harness_generated_at,
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
    use_probe_batch: bool = False,
    batch_workers: Optional[int] = None,
    return_experiment: bool = False,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run an experiment asynchronously from a configuration file.

    Like run_experiment_from_config, but uses async execution for
    parallel processing.

    Args:
        config_path: Path to the configuration file.
        concurrency: Maximum number of concurrent executions.
        progress_callback: Optional callback for progress updates.
        resume: Whether to resume from existing records.jsonl.
        use_probe_batch: Whether to use Probe.run_batch for execution.
        batch_workers: Worker count for probe batch execution.
        return_experiment: Whether to return an ExperimentResult.

    Returns:
        A list of result dictionaries or an ExperimentResult.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    config_snapshot = _build_resolved_config_snapshot(config, base_dir)
    resolved_run_id = run_id or _deterministic_run_id_from_config_snapshot(
        config_snapshot,
        schema_version=schema_version,
    )

    model = _create_model_from_config(config["model"])
    probe = _create_probe_from_config(config["probe"])
    dataset = _load_dataset_from_config(config["dataset"], base_dir)

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
        dataset_info=config.get("dataset"),
        config_snapshot=config_snapshot,
        resume=resume,
        use_probe_batch=use_probe_batch,
        batch_workers=batch_workers,
        return_experiment=return_experiment,
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
) -> ExperimentResult:
    """Create a structured ExperimentResult from raw results.

    Args:
        model: The model that was tested.
        probe: The probe that was run.
        results: The raw result dictionaries.
        config: Optional configuration used for the experiment.

    Returns:
        An ExperimentResult with structured data and scores.
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
    if isinstance(category, str):
        category = ProbeCategory(category)

    resolved_experiment_id = experiment_id
    if resolved_experiment_id is None:
        if probe_results:
            inputs = [r.input for r in probe_results]
        else:
            inputs = []
        resolved_experiment_id = _deterministic_hash(
            {
                "model": _serialize_value(_coerce_model_info(model)),
                "probe": {
                    "name": probe.name,
                    "version": getattr(probe, "version", None),
                },
                "inputs_hash": _hash_prompt_set(inputs),
            }
        )[:12]

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
    return _deterministic_hash(payload)[:16]


def derive_run_id_from_config_path(
    config_path: Union[str, Path],
    *,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> str:
    config_path = Path(config_path)
    config = load_config(config_path)
    config_snapshot = _build_resolved_config_snapshot(config, config_path.parent)
    return _deterministic_run_id_from_config_snapshot(
        config_snapshot,
        schema_version=schema_version,
    )
