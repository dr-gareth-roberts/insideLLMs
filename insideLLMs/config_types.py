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
from typing import Any, Optional, Union

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


class RunConfigBuilder:
    """Fluent builder for RunConfig.

    Provides a chainable API for constructing RunConfig objects,
    making configuration more readable and discoverable.

    Example:
        >>> config = (
        ...     RunConfigBuilder()
        ...     .with_validation(schema_version="1.0.0", mode="strict")
        ...     .with_artifacts(run_root="./runs", run_id="my-run")
        ...     .with_concurrency(10)
        ...     .build()
        ... )
        >>> runner.run(prompts, config=config)
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._stop_on_error: bool = False
        self._validate_output: bool = False
        self._schema_version: str = DEFAULT_SCHEMA_VERSION
        self._validation_mode: str = "strict"
        self._emit_run_artifacts: bool = True
        self._run_dir: Optional[Union[str, Path]] = None
        self._run_root: Optional[Union[str, Path]] = None
        self._run_id: Optional[str] = None
        self._overwrite: bool = False
        self._store_messages: bool = True
        self._dataset_info: Optional[dict[str, Any]] = None
        self._config_snapshot: Optional[dict[str, Any]] = None
        self._concurrency: int = 5

    def with_validation(
        self,
        enabled: bool = True,
        schema_version: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> "RunConfigBuilder":
        """Configure output validation.

        Args:
            enabled: Whether to validate output against schema.
            schema_version: Schema version for validation.
            mode: Validation strictness ("strict" or "lenient").

        Returns:
            Self for method chaining.
        """
        self._validate_output = enabled
        if schema_version is not None:
            self._schema_version = schema_version
        if mode is not None:
            self._validation_mode = mode
        return self

    def with_artifacts(
        self,
        enabled: bool = True,
        run_dir: Optional[Union[str, Path]] = None,
        run_root: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> "RunConfigBuilder":
        """Configure artifact output.

        Args:
            enabled: Whether to emit run artifacts (records, manifest).
            run_dir: Explicit run directory path.
            run_root: Root directory for runs.
            run_id: Unique identifier for this run.
            overwrite: Whether to overwrite existing run directory.

        Returns:
            Self for method chaining.
        """
        self._emit_run_artifacts = enabled
        if run_dir is not None:
            self._run_dir = run_dir
        if run_root is not None:
            self._run_root = run_root
        if run_id is not None:
            self._run_id = run_id
        self._overwrite = overwrite
        return self

    def with_concurrency(self, concurrency: int) -> "RunConfigBuilder":
        """Set concurrency level for async execution.

        Args:
            concurrency: Maximum concurrent executions.

        Returns:
            Self for method chaining.
        """
        self._concurrency = concurrency
        return self

    def with_error_handling(self, stop_on_error: bool = True) -> "RunConfigBuilder":
        """Configure error handling behavior.

        Args:
            stop_on_error: If True, stop execution on first error.

        Returns:
            Self for method chaining.
        """
        self._stop_on_error = stop_on_error
        return self

    def with_dataset_info(self, info: dict[str, Any]) -> "RunConfigBuilder":
        """Set dataset metadata.

        Args:
            info: Dictionary with dataset metadata (name, version, etc.).

        Returns:
            Self for method chaining.
        """
        self._dataset_info = info
        return self

    def with_config_snapshot(self, snapshot: dict[str, Any]) -> "RunConfigBuilder":
        """Set experiment configuration snapshot.

        Args:
            snapshot: Dictionary with experiment configuration.

        Returns:
            Self for method chaining.
        """
        self._config_snapshot = snapshot
        return self

    def with_message_storage(self, enabled: bool = True) -> "RunConfigBuilder":
        """Configure message history storage.

        Args:
            enabled: Whether to store full message history in records.

        Returns:
            Self for method chaining.
        """
        self._store_messages = enabled
        return self

    def build(self) -> RunConfig:
        """Build the RunConfig object.

        Returns:
            A new RunConfig instance with the configured values.

        Raises:
            ValueError: If configuration is invalid.
        """
        return RunConfig(
            stop_on_error=self._stop_on_error,
            validate_output=self._validate_output,
            schema_version=self._schema_version,
            validation_mode=self._validation_mode,
            emit_run_artifacts=self._emit_run_artifacts,
            run_dir=self._run_dir,
            run_root=self._run_root,
            run_id=self._run_id,
            overwrite=self._overwrite,
            store_messages=self._store_messages,
            dataset_info=self._dataset_info,
            config_snapshot=self._config_snapshot,
            concurrency=self._concurrency,
        )


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


@dataclass
class ProgressInfo:
    """Structured progress information for callbacks.

    Provides detailed progress information including timing, rate,
    and ETA estimates for better progress reporting.

    Example:
        >>> def progress_callback(info: ProgressInfo):
        ...     print(f"{info.current}/{info.total} ({info.percent:.1f}%)")
        ...     print(f"Rate: {info.rate:.1f}/s, ETA: {info.eta_seconds:.0f}s")
        >>> runner.run(prompts, progress_callback=progress_callback)

    Attributes:
        current: Number of items completed.
        total: Total number of items.
        elapsed_seconds: Time elapsed since start.
        rate: Items per second (0 if no items completed yet).
        eta_seconds: Estimated time remaining (None if cannot estimate).
        current_item: The item currently being processed (optional).
        current_index: Index of current item (optional).
        status: Current status message (optional).
    """

    current: int
    total: int
    elapsed_seconds: float
    rate: float = 0.0
    eta_seconds: Optional[float] = None
    current_item: Optional[Any] = None
    current_index: Optional[int] = None
    status: Optional[str] = None

    @property
    def percent(self) -> float:
        """Percentage complete (0-100)."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100

    @property
    def remaining(self) -> int:
        """Number of items remaining."""
        return self.total - self.current

    @property
    def is_complete(self) -> bool:
        """Whether all items are complete."""
        return self.current >= self.total

    @classmethod
    def create(
        cls,
        current: int,
        total: int,
        start_time: float,
        current_item: Optional[Any] = None,
        current_index: Optional[int] = None,
        status: Optional[str] = None,
    ) -> "ProgressInfo":
        """Create a ProgressInfo with calculated rate and ETA.

        Args:
            current: Number of items completed.
            total: Total number of items.
            start_time: time.perf_counter() value at start.
            current_item: The item currently being processed.
            current_index: Index of current item.
            status: Current status message.

        Returns:
            A new ProgressInfo instance with calculated fields.
        """
        import time

        elapsed = time.perf_counter() - start_time
        rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
        remaining = total - current
        eta = remaining / rate if rate > 0 else None

        return cls(
            current=current,
            total=total,
            elapsed_seconds=elapsed,
            rate=rate,
            eta_seconds=eta,
            current_item=current_item,
            current_index=current_index,
            status=status,
        )

    def __str__(self) -> str:
        """Human-readable progress string."""
        parts = [f"{self.current}/{self.total} ({self.percent:.1f}%)"]
        if self.rate > 0:
            parts.append(f"{self.rate:.1f}/s")
        if self.eta_seconds is not None:
            if self.eta_seconds < 60:
                parts.append(f"ETA: {self.eta_seconds:.0f}s")
            else:
                minutes = int(self.eta_seconds // 60)
                seconds = int(self.eta_seconds % 60)
                parts.append(f"ETA: {minutes}m{seconds}s")
        return " | ".join(parts)


__all__ = ["ProgressInfo", "RunConfig", "RunConfigBuilder", "RunContext"]
