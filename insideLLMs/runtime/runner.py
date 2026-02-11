"""Probe runner module for executing probes against models.

This module provides the core runner classes and functions for executing
probes against LLM models. The implementation has been modularized into
focused submodules for maintainability:

- ``_base``: Base runner class and progress callback utilities
- ``_sync_runner``: Synchronous ProbeRunner implementation
- ``_async_runner``: Asynchronous AsyncProbeRunner implementation
- ``_determinism``: Deterministic run ID and timestamp generation
- ``_result_utils``: Result conversion and spec building utilities
- ``_artifact_utils``: Run directory and JSONL handling utilities
- ``_config_loader``: Configuration loading and object creation
- ``_high_level``: High-level experiment functions

This module re-exports all public APIs for backward compatibility.

Examples
--------
Basic synchronous usage:

    >>> from insideLLMs.runtime.runner import ProbeRunner
    >>> runner = ProbeRunner(model, probe)
    >>> results = runner.run(prompts)

Asynchronous with concurrency:

    >>> from insideLLMs.runtime.runner import AsyncProbeRunner
    >>> runner = AsyncProbeRunner(model, probe)
    >>> results = await runner.run(prompts, concurrency=10)

Running from configuration:

    >>> from insideLLMs.runtime.runner import run_experiment_from_config
    >>> results = run_experiment_from_config("experiment.yaml")
"""

# Re-export base classes and utilities
from insideLLMs._serialization import (
    StrictSerializationError,
)

# Re-export serialization utilities (for backward compatibility)
from insideLLMs._serialization import (
    fingerprint_value as _fingerprint_value,
)
from insideLLMs._serialization import (
    serialize_value as _serialize_value,
)
from insideLLMs._serialization import (
    stable_json_dumps as _stable_json_dumps,
)

# Re-export registry items (for backward compatibility with tests)
from insideLLMs.registry import (
    NotFoundError,
    dataset_registry,
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)

# Re-export artifact utilities
from insideLLMs.runtime._artifact_utils import (
    _atomic_write_text,
    _atomic_write_yaml,
    _default_run_root,
    _ensure_run_sentinel,
    _prepare_run_dir,
    _prepare_run_dir_for_resume,
    _read_jsonl_records,
    _semver_tuple,
    _truncate_incomplete_jsonl,
    _validate_resume_record,
)
from insideLLMs.runtime._async_runner import AsyncProbeRunner
from insideLLMs.runtime._base import (
    LegacyProgressCallback,
    ProgressCallback,
    RichProgressCallback,
    _invoke_progress_callback,
    _normalize_validation_mode,
    _RunnerBase,
)

# Re-export config loader utilities
from insideLLMs.runtime._config_loader import (
    _build_resolved_config_snapshot,
    _create_middlewares_from_config,
    _create_model_from_config,
    _create_probe_from_config,
    _extract_probe_kwargs_from_config,
    _load_dataset_from_config,
    _resolve_determinism_options,
    _resolve_path,
    load_config,
)

# Re-export determinism utilities
from insideLLMs.runtime._determinism import (
    _DETERMINISTIC_TIME_BASE,
    _DETERMINISTIC_TIME_RANGE_SECONDS,
    _deterministic_base_time,
    _deterministic_harness_experiment_id,
    _deterministic_hash,
    _deterministic_item_times,
    _deterministic_run_id_from_config_snapshot,
    _deterministic_run_id_from_inputs,
    _deterministic_run_times,
    _hash_prompt_set,
    _replicate_key,
)

# Re-export high-level functions
from insideLLMs.runtime._high_level import (
    create_experiment_result,
    derive_run_id_from_config_path,
    run_experiment_from_config,
    run_experiment_from_config_async,
    run_harness_from_config,
    run_probe,
    run_probe_async,
)

# Re-export result utilities
from insideLLMs.runtime._result_utils import (
    _build_dataset_spec,
    _build_model_spec,
    _build_probe_spec,
    _build_result_record,
    _coerce_model_info,
    _normalize_info_obj_to_dict,
    _normalize_status,
    _record_index_from_record,
    _result_dict_from_probe_result,
    _result_dict_from_record,
)

# Re-export runner classes
from insideLLMs.runtime._sync_runner import ProbeRunner
from insideLLMs.runtime.workflows import (
    diff_run_dirs,
    run_harness_to_dir,
)

# Re-export schema version constant
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION

__all__ = [
    # Base classes and utilities
    "LegacyProgressCallback",
    "ProgressCallback",
    "RichProgressCallback",
    "_invoke_progress_callback",
    "_normalize_validation_mode",
    "_RunnerBase",
    # Determinism utilities
    "_deterministic_base_time",
    "_deterministic_harness_experiment_id",
    "_deterministic_hash",
    "_deterministic_item_times",
    "_deterministic_run_id_from_config_snapshot",
    "_deterministic_run_id_from_inputs",
    "_deterministic_run_times",
    "_DETERMINISTIC_TIME_BASE",
    "_DETERMINISTIC_TIME_RANGE_SECONDS",
    "_hash_prompt_set",
    "_replicate_key",
    # Result utilities
    "_build_dataset_spec",
    "_build_model_spec",
    "_build_probe_spec",
    "_build_result_record",
    "_coerce_model_info",
    "_normalize_info_obj_to_dict",
    "_normalize_status",
    "_record_index_from_record",
    "_result_dict_from_probe_result",
    "_result_dict_from_record",
    # Artifact utilities
    "_atomic_write_text",
    "_atomic_write_yaml",
    "_default_run_root",
    "_ensure_run_sentinel",
    "_prepare_run_dir",
    "_prepare_run_dir_for_resume",
    "_read_jsonl_records",
    "_semver_tuple",
    "_truncate_incomplete_jsonl",
    "_validate_resume_record",
    # Config loader utilities
    "_build_resolved_config_snapshot",
    "_create_middlewares_from_config",
    "_create_model_from_config",
    "_create_probe_from_config",
    "_extract_probe_kwargs_from_config",
    "_load_dataset_from_config",
    "_resolve_determinism_options",
    "_resolve_path",
    "load_config",
    # Runner classes
    "ProbeRunner",
    "AsyncProbeRunner",
    # High-level functions
    "create_experiment_result",
    "derive_run_id_from_config_path",
    "run_experiment_from_config",
    "run_experiment_from_config_async",
    "run_harness_from_config",
    "run_harness_to_dir",
    "run_probe",
    "run_probe_async",
    "diff_run_dirs",
    # Constants
    "DEFAULT_SCHEMA_VERSION",
    # Serialization utilities (for backward compatibility)
    "_fingerprint_value",
    "_serialize_value",
    "_stable_json_dumps",
    "StrictSerializationError",
    # Registry items (for backward compatibility)
    "dataset_registry",
    "ensure_builtins_registered",
    "model_registry",
    "NotFoundError",
    "probe_registry",
]
