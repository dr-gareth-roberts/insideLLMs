"""High-level configuration-driven experiment functions.

This module provides high-level functions for running experiments from
configuration files, as well as convenience wrappers for simple probe execution.
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    serialize_value as _serialize_value,
)
from insideLLMs.analysis.statistics import generate_summary_report
from insideLLMs.models.base import Model
from insideLLMs.probes.base import Probe
from insideLLMs.runtime._base import _normalize_validation_mode
from insideLLMs.runtime._config_loader import (
    _build_resolved_config_snapshot,
    _create_model_from_config,
    _create_probe_from_config,
    _extract_probe_kwargs_from_config,
    _load_dataset_from_config,
    _resolve_determinism_options,
    load_config,
)
from insideLLMs.runtime._determinism import (
    _deterministic_base_time,
    _deterministic_harness_experiment_id,
    _deterministic_hash,
    _deterministic_item_times,
    _deterministic_run_id_from_config_snapshot,
    _deterministic_run_times,
    _hash_prompt_set,
)
from insideLLMs.runtime._result_utils import (
    _build_result_record,
    _coerce_model_info,
    _normalize_info_obj_to_dict,
)
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.types import (
    ConfigDict,
    ExperimentResult,
    ProbeCategory,
    ProbeResult,
    ResultStatus,
)

logger = logging.getLogger(__name__)


def run_probe(
    model: Model,
    probe: Probe,
    prompt_set: list[Any],
    *,
    config: Optional["RunConfig"] = None,
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
        List of inputs to test.
    config : Optional[RunConfig], default None
        RunConfig with all run settings.
    strict_serialization : Optional[bool], default None
        If True, fail on non-deterministic values.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields.
    **probe_kwargs : Any
        Additional keyword arguments passed to the probe.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        List of result dicts or ExperimentResult if config.return_experiment=True.
    """
    from insideLLMs.runtime._sync_runner import ProbeRunner

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
    config: Optional["RunConfig"] = None,
    concurrency: Optional[int] = None,
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
    **probe_kwargs: Any,
) -> Union[list[dict[str, Any]], ExperimentResult]:
    """Run a probe asynchronously with concurrent execution.

    This is a convenience function that creates an :class:`AsyncProbeRunner`
    and executes the probe in one step.

    Parameters
    ----------
    model : Model
        The model to test.
    probe : Probe
        The probe to run.
    prompt_set : list[Any]
        List of inputs to test.
    config : Optional[RunConfig], default None
        RunConfig with all run settings.
    concurrency : Optional[int], default None
        Maximum concurrent executions. Overrides config.concurrency.
    strict_serialization : Optional[bool], default None
        If True, fail on non-deterministic values.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields.
    **probe_kwargs : Any
        Additional kwargs passed to the probe.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        List of result dicts or ExperimentResult if config.return_experiment=True.
    """
    from insideLLMs.runtime._async_runner import AsyncProbeRunner

    runner = AsyncProbeRunner(model, probe)
    return await runner.run(
        prompt_set,
        config=config,
        concurrency=concurrency,
        strict_serialization=strict_serialization,
        deterministic_artifacts=deterministic_artifacts,
        **probe_kwargs,
    )


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
    loads the dataset, and runs the experiment.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the configuration file.
    progress_callback : Optional[Callable[[int, int], None]], default None
        Callback for progress updates.
    validate_output : bool, default False
        If True, validate outputs against the schema.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version for validation.
    validation_mode : str, default "strict"
        Validation mode: "strict" or "lenient".
    emit_run_artifacts : bool, default True
        If True, write records.jsonl and manifest.json.
    run_dir : Optional[Union[str, Path]], default None
        Explicit directory for run artifacts.
    run_root : Optional[Union[str, Path]], default None
        Root directory for run directories.
    run_id : Optional[str], default None
        Explicit run ID.
    overwrite : bool, default False
        If True, overwrite existing run directory.
    resume : bool, default False
        If True, resume from existing records.
    strict_serialization : Optional[bool], default None
        If True, fail on non-deterministic values.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields.
    use_probe_batch : bool, default False
        If True, use Probe.run_batch.
    batch_workers : Optional[int], default None
        Number of workers for Probe.run_batch.
    return_experiment : bool, default False
        If True, return ExperimentResult.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        List of result dicts or ExperimentResult if return_experiment=True.
    """
    from insideLLMs.runtime._sync_runner import ProbeRunner

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
    against a shared dataset.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the configuration file.
    progress_callback : Optional[Callable[[int, int], None]], default None
        Callback for progress updates.
    validate_output : bool, default False
        If True, validate outputs against the schema.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version for validation.
    validation_mode : str, default "strict"
        Validation mode: "strict" or "lenient".
    strict_serialization : Optional[bool], default None
        If True, fail on non-deterministic values.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent manifest fields.

    Returns
    -------
    dict[str, Any]
        Dictionary containing records, experiments, summary, config, run_id,
        and generated_at timestamp.
    """
    from insideLLMs.runtime._sync_runner import ProbeRunner

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

                record["error"] = result.get("error")
                record["error_type"] = result.get("error_type")

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
    concurrent processing.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the configuration file.
    concurrency : int, default 5
        Maximum concurrent executions.
    progress_callback : Optional[Callable[[int, int], None]], default None
        Callback for progress updates.
    validate_output : bool, default False
        If True, validate outputs.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version.
    validation_mode : str, default "strict"
        Validation mode.
    emit_run_artifacts : bool, default True
        If True, write artifacts.
    run_dir : Optional[Union[str, Path]], default None
        Explicit run directory.
    run_root : Optional[Union[str, Path]], default None
        Root directory.
    run_id : Optional[str], default None
        Explicit run ID.
    overwrite : bool, default False
        If True, overwrite existing.
    resume : bool, default False
        If True, resume from existing.
    strict_serialization : Optional[bool], default None
        If True, fail on non-deterministic.
    deterministic_artifacts : Optional[bool], default None
        If True, omit host-dependent fields.
    use_probe_batch : bool, default False
        If True, use Probe.run_batch.
    batch_workers : Optional[int], default None
        Number of batch workers.
    return_experiment : bool, default False
        If True, return ExperimentResult.

    Returns
    -------
    Union[list[dict[str, Any]], ExperimentResult]
        List of results or ExperimentResult.
    """
    from insideLLMs.runtime._async_runner import AsyncProbeRunner

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
        The raw results.
    config : Optional[ConfigDict], default None
        Optional configuration dictionary.
    experiment_id : Optional[str], default None
        Unique experiment identifier.
    started_at : Optional[datetime], default None
        Experiment start time.
    completed_at : Optional[datetime], default None
        Experiment completion time.
    strict_serialization : bool, default False
        If True, fail on non-deterministic values.

    Returns
    -------
    ExperimentResult
        Structured result with scores, metadata, and timing.
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
        probe_results = list(results)  # type: ignore[list-item]  # Runtime check confirms ProbeResult type
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


def derive_run_id_from_config_path(
    config_path: Union[str, Path],
    *,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    strict_serialization: Optional[bool] = None,
) -> str:
    """Derive a deterministic run ID from a configuration file.

    Generates a unique run ID based on the contents of the configuration file.
    The same configuration will always produce the same run ID.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the configuration file.
    schema_version : str, default DEFAULT_SCHEMA_VERSION
        Schema version to include in the hash.
    strict_serialization : Optional[bool], default None
        If True, fail on non-deterministic values.

    Returns
    -------
    str
        A 32-character hexadecimal run ID.
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


# Import RunConfig at module level for type hints
try:
    from insideLLMs.config_types import RunConfig
except ImportError:
    RunConfig = None  # type: ignore[misc, assignment]


__all__ = [
    "run_probe",
    "run_probe_async",
    "run_experiment_from_config",
    "run_harness_from_config",
    "run_experiment_from_config_async",
    "create_experiment_result",
    "derive_run_id_from_config_path",
]
