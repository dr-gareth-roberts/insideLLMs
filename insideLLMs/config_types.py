"""Configuration types for insideLLMs runner.

This module provides dataclasses for configuring probe runs, replacing
the many keyword arguments in ProbeRunner.run() and AsyncProbeRunner.run()
with a single, validated configuration object.

Example:
    >>> from insideLLMs.config_types import RunConfig
    >>> config = RunConfig(
    ...     emit_run_artifacts=True,
    ...     run_root="./runs",
    ...     validate_output=True,
    ... )
    >>> runner.run(prompts, config=config)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION


@dataclass
class RunConfig:
    """Configuration for a probe run.

    Groups all the configuration options for ProbeRunner.run() and
    AsyncProbeRunner.run() into a single, validated object.

    Attributes:
        stop_on_error: If True, stop execution on first error. Default False.
        validate_output: If True, validate output against schema. Default False.
        schema_version: Schema version for output validation.
        validation_mode: Validation strictness ("strict" or "lenient").
        emit_run_artifacts: If True, write records and manifest files.
        run_dir: Explicit run directory path. Overrides run_root/run_id.
        run_root: Root directory for runs. Default ~/.insidellms/runs.
        run_id: Unique identifier for this run. Auto-generated if not provided.
        overwrite: If True, overwrite existing run directory.
        dataset_info: Optional metadata about the dataset being used.
        config_snapshot: Optional snapshot of the experiment configuration.
        store_messages: If True, store full message history in records.
        concurrency: Maximum concurrent executions (async only). Default 5.
    """

    # Error handling
    stop_on_error: bool = False

    # Validation
    validate_output: bool = False
    schema_version: str = DEFAULT_SCHEMA_VERSION
    validation_mode: str = "strict"

    # Artifact output
    emit_run_artifacts: bool = True
    run_dir: Optional[Union[str, Path]] = None
    run_root: Optional[Union[str, Path]] = None
    run_id: Optional[str] = None
    overwrite: bool = False
    store_messages: bool = True

    # Metadata
    dataset_info: Optional[dict[str, Any]] = None
    config_snapshot: Optional[dict[str, Any]] = None

    # Async-specific
    concurrency: int = 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.validation_mode not in ("strict", "lenient"):
            raise ValueError(
                f"validation_mode must be 'strict' or 'lenient', got {self.validation_mode!r}"
            )
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "RunConfig":
        """Create a RunConfig from keyword arguments.

        This is used for backward compatibility with the old API.
        Unknown kwargs are ignored with a warning.

        Args:
            **kwargs: Keyword arguments matching RunConfig fields.

        Returns:
            A new RunConfig instance.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        config_kwargs = {}
        unknown_kwargs = []

        for key, value in kwargs.items():
            if key in valid_fields:
                config_kwargs[key] = value
            else:
                unknown_kwargs.append(key)

        if unknown_kwargs:
            warnings.warn(
                f"Unknown RunConfig fields ignored: {unknown_kwargs}",
                UserWarning,
                stacklevel=2,
            )

        return cls(**config_kwargs)


@dataclass
class RunContext:
    """Runtime context for a probe run.

    This holds the resolved runtime state during execution, including
    the actual run directory, run ID, and timing information.

    This is an internal class used by the runner implementations.
    """

    run_id: str
    run_dir: Path
    run_started_at: Any  # datetime
    run_base_time: float  # time.perf_counter() at start

    # Resolved specs for manifest
    model_spec: dict[str, Any] = field(default_factory=dict)
    probe_spec: dict[str, Any] = field(default_factory=dict)
    dataset_spec: dict[str, Any] = field(default_factory=dict)


__all__ = ["RunConfig", "RunContext"]

