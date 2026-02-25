"""Configuration types for insideLLMs runner.

This module provides dataclasses and builder patterns for configuring probe runs,
replacing the many keyword arguments in ProbeRunner.run() and AsyncProbeRunner.run()
with a single, validated configuration object. The configuration system supports
validation, artifact management, concurrency control, and progress tracking.

Overview
--------
The module provides four main types:

- ``RunConfig``: The primary configuration dataclass that holds all run settings
- ``RunConfigBuilder``: A fluent builder for constructing RunConfig objects
- ``RunContext``: Internal runtime state during execution (used by runners)
- ``ProgressInfo``: Structured progress information for callbacks

The configuration system supports:

- **Error handling**: Control whether to stop on first error or continue
- **Output validation**: Validate probe outputs against JSON schemas
- **Artifact management**: Control where and how run artifacts are stored
- **Concurrency**: Configure parallel execution for async runners
- **Resumption**: Resume interrupted runs from saved artifacts
- **Batch execution**: Use probe-native batch APIs when available

Basic Usage
-----------
Create a configuration using the dataclass directly:

    >>> from insideLLMs.config_types import RunConfig
    >>> config = RunConfig(
    ...     emit_run_artifacts=True,
    ...     run_root="./runs",
    ...     validate_output=True,
    ... )
    >>> runner.run(prompts, config=config)

Or use the fluent builder for better readability:

    >>> from insideLLMs.config_types import RunConfigBuilder
    >>> config = (
    ...     RunConfigBuilder()
    ...     .with_validation(enabled=True, mode="strict")
    ...     .with_artifacts(run_root="./experiments", run_id="exp-001")
    ...     .with_concurrency(10)
    ...     .build()
    ... )

Migration from Keyword Arguments
--------------------------------
The ``from_kwargs`` class method provides backward compatibility:

    >>> # Old style (deprecated)
    >>> runner.run(prompts, emit_run_artifacts=True, validate_output=True)

    >>> # New style via from_kwargs (for migration)
    >>> config = RunConfig.from_kwargs(emit_run_artifacts=True, validate_output=True)
    >>> runner.run(prompts, config=config)

    >>> # Preferred style
    >>> config = RunConfig(emit_run_artifacts=True, validate_output=True)
    >>> runner.run(prompts, config=config)

Progress Tracking
-----------------
Use ProgressInfo with callbacks for detailed progress reporting:

    >>> from insideLLMs.config_types import ProgressInfo
    >>>
    >>> def my_progress_callback(info: ProgressInfo) -> None:
    ...     print(f"Progress: {info.current}/{info.total} ({info.percent:.1f}%)")
    ...     if info.eta_seconds:
    ...         print(f"  ETA: {info.eta_seconds:.0f} seconds")
    ...     if info.status:
    ...         print(f"  Status: {info.status}")
    >>>
    >>> runner.run(prompts, progress_callback=my_progress_callback)

Configuration Patterns
----------------------
Development configuration with verbose output:

    >>> dev_config = RunConfig(
    ...     stop_on_error=True,      # Fail fast during development
    ...     validate_output=True,    # Catch schema issues early
    ...     validation_mode="strict",
    ...     emit_run_artifacts=True,
    ...     store_messages=True,     # Keep full conversation history
    ...     run_root="./dev-runs",
    ... )

Production configuration for large-scale experiments:

    >>> prod_config = RunConfig(
    ...     stop_on_error=False,     # Continue on errors
    ...     validate_output=True,
    ...     validation_mode="lenient",  # Warn but don't fail
    ...     emit_run_artifacts=True,
    ...     concurrency=20,          # High parallelism
    ...     resume=True,             # Enable resumption
    ...     use_probe_batch=True,    # Use batch APIs
    ...     batch_workers=4,
    ...     return_experiment=True,  # Get structured results
    ... )

Minimal configuration for quick tests:

    >>> test_config = RunConfig(
    ...     emit_run_artifacts=False,  # No file I/O
    ...     validate_output=False,
    ...     store_messages=False,
    ... )

Notes
-----
- RunConfig is immutable after creation; create a new instance to change settings
- The builder pattern is recommended for complex configurations
- Unknown kwargs passed to ``from_kwargs`` generate warnings but don't fail
- Validation happens in ``__post_init__`` and will raise ValueError for invalid values
- RunContext is an internal class and should not be instantiated directly

See Also
--------
insideLLMs.runtime.runner.ProbeRunner : Synchronous probe runner
insideLLMs.runtime.runner.AsyncProbeRunner : Asynchronous probe runner
insideLLMs.schemas.constants : Schema version constants

Examples
--------
Complete example with async runner:

    >>> import asyncio
    >>> from insideLLMs.config_types import RunConfig, RunConfigBuilder
    >>> from insideLLMs.runtime.runner import AsyncProbeRunner
    >>>
    >>> async def run_experiment():
    ...     config = (
    ...         RunConfigBuilder()
    ...         .with_validation(enabled=True)
    ...         .with_artifacts(
    ...             run_root="./experiments",
    ...             run_id="safety-eval-v1",
    ...         )
    ...         .with_concurrency(10)
    ...         .with_resume(enabled=True)
    ...         .with_output(return_experiment=True)
    ...         .build()
    ...     )
    ...
    ...     runner = AsyncProbeRunner(model=my_model, probe=my_probe)
    ...     result = await runner.run(prompts, config=config)
    ...     return result
    >>>
    >>> result = asyncio.run(run_experiment())
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION


@dataclass
class RunConfig:
    """Configuration for a probe run.

    Groups all the configuration options for ProbeRunner.run() and
    AsyncProbeRunner.run() into a single, validated object. This dataclass
    provides a type-safe, validated way to configure experiment runs with
    sensible defaults for common use cases.

    The configuration is validated during initialization via ``__post_init__``,
    ensuring that invalid configurations fail early with clear error messages.

    Parameters
    ----------
    stop_on_error : bool, default=False
        If True, stop execution immediately when the first error occurs.
        If False, continue processing remaining items and collect all errors.
        Use True during development to fail fast, False in production for
        maximum data collection.

    validate_output : bool, default=False
        If True, validate each probe output against the configured schema.
        Validation catches malformed outputs early and ensures data quality.

    schema_version : str, default=DEFAULT_SCHEMA_VERSION
        The schema version to use for output validation. Must match a
        registered schema version. See ``insideLLMs.schemas.constants``
        for available versions.

    validation_mode : {"strict", "lenient"}, default="strict"
        Controls validation strictness:
        - "strict": Raise ValidationError on any schema violation
        - "lenient": Log warnings but continue execution (alias: "warn")

    emit_run_artifacts : bool, default=True
        If True, write run artifacts to disk including:
        - ``records.jsonl``: Individual probe results
        - ``manifest.json``: Run metadata and configuration
        Set to False for testing or when artifacts aren't needed.

    run_dir : str or Path, optional
        Explicit path for the run directory. If provided, overrides
        ``run_root`` and ``run_id``. Use for precise control over
        output location.

    run_root : str or Path, optional
        Root directory under which run directories are created.
        Defaults to ``~/.insidellms/runs``. Each run creates a
        subdirectory named by ``run_id``.

    run_id : str, optional
        Unique identifier for this run. Used as the directory name
        under ``run_root``. Auto-generated as a UUID if not provided.
        Use descriptive IDs for easier organization (e.g., "safety-eval-2024-01").

    overwrite : bool, default=False
        If True, overwrite an existing run directory with the same ID.
        If False, raise an error if the directory already exists.
        Use with caution to avoid data loss.

    store_messages : bool, default=True
        If True, store the full message history (conversation) in each
        record. Set to False to reduce storage for large-scale experiments
        where only final outputs matter.

    dataset_info : dict, optional
        Metadata about the dataset being evaluated. Stored in the manifest
        for reproducibility. Typically includes name, version, source, etc.

    config_snapshot : dict, optional
        Snapshot of the experiment configuration. Stored in the manifest
        for full reproducibility of the experimental setup.

    concurrency : int, default=5
        Maximum number of concurrent probe executions for async runners.
        Higher values increase throughput but may hit rate limits.
        Ignored by synchronous runners.

    timeout : float, optional
        Timeout in seconds for each probe execution when using async runners.
        If None, no timeout is applied.

    resume : bool, default=False
        If True, attempt to resume from existing ``records.jsonl``.
        Already-processed items (by ID) are skipped. Useful for
        recovering from interrupted runs.

    use_probe_batch : bool, default=False
        If True, use the probe's native ``run_batch`` method when available.
        Batch APIs can be more efficient for some probe implementations.

    batch_workers : int, optional
        Number of workers for batch execution when ``use_probe_batch=True``.
        If None, uses the probe's default worker count.

    return_experiment : bool, default=False
        If True, return an ``ExperimentResult`` object instead of raw dicts.
        The structured result provides convenient methods for analysis.

    Attributes
    ----------
    stop_on_error : bool
        Whether to halt on first error.
    validate_output : bool
        Whether to validate outputs against schema.
    schema_version : str
        Schema version for validation.
    validation_mode : str
        Validation strictness level.
    emit_run_artifacts : bool
        Whether to write artifacts to disk.
    run_dir : Path or None
        Explicit run directory path.
    run_root : Path or None
        Root directory for runs.
    run_id : str or None
        Unique run identifier.
    overwrite : bool
        Whether to overwrite existing runs.
    store_messages : bool
        Whether to store message history.
    dataset_info : dict or None
        Dataset metadata.
    config_snapshot : dict or None
        Configuration snapshot.
    concurrency : int
        Maximum concurrent executions.
    timeout : float or None
        Per-item timeout in seconds for async runs.
    resume : bool
        Whether to resume from existing artifacts.
    use_probe_batch : bool
        Whether to use batch APIs.
    batch_workers : int or None
        Worker count for batch execution.
    return_experiment : bool
        Whether to return structured results.

    Raises
    ------
    ValueError
        If ``validation_mode`` is not "strict" or "lenient".
        If ``concurrency`` is less than 1.
        If ``batch_workers`` is provided and less than 1.

    See Also
    --------
    RunConfigBuilder : Fluent builder for RunConfig
    RunContext : Runtime context during execution
    ProgressInfo : Progress information for callbacks

    Examples
    --------
    Basic configuration with validation enabled:

        >>> config = RunConfig(
        ...     validate_output=True,
        ...     emit_run_artifacts=True,
        ...     run_root="./experiments",
        ... )

    Development configuration that fails fast:

        >>> dev_config = RunConfig(
        ...     stop_on_error=True,
        ...     validate_output=True,
        ...     validation_mode="strict",
        ...     emit_run_artifacts=True,
        ...     store_messages=True,
        ...     run_root="./dev-runs",
        ... )

    Production configuration for large-scale evaluation:

        >>> prod_config = RunConfig(
        ...     stop_on_error=False,
        ...     validate_output=True,
        ...     validation_mode="lenient",
        ...     emit_run_artifacts=True,
        ...     run_root="/data/experiments",
        ...     run_id="safety-eval-2024-01-15",
        ...     concurrency=20,
        ...     resume=True,
        ...     use_probe_batch=True,
        ...     batch_workers=4,
        ...     return_experiment=True,
        ... )

    Configuration with dataset metadata:

        >>> config = RunConfig(
        ...     emit_run_artifacts=True,
        ...     run_id="toxicity-eval-v2",
        ...     dataset_info={
        ...         "name": "toxicity-prompts",
        ...         "version": "2.1.0",
        ...         "source": "https://example.com/datasets/toxicity",
        ...         "num_samples": 10000,
        ...     },
        ...     config_snapshot={
        ...         "model": "gpt-4",
        ...         "temperature": 0.7,
        ...         "probe": "toxicity-classifier-v3",
        ...     },
        ... )

    Minimal configuration for unit tests:

        >>> test_config = RunConfig(
        ...     emit_run_artifacts=False,
        ...     validate_output=False,
        ...     store_messages=False,
        ... )

    Resumable configuration for long-running experiments:

        >>> resumable_config = RunConfig(
        ...     emit_run_artifacts=True,
        ...     run_root="./long-experiments",
        ...     run_id="week-long-eval",
        ...     resume=True,
        ...     overwrite=False,  # Don't accidentally overwrite
        ... )

    Invalid configurations raise ValueError:

        >>> try:
        ...     bad_config = RunConfig(validation_mode="invalid")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: validation_mode must be 'strict' or 'lenient', got 'invalid'

        >>> try:
        ...     bad_config = RunConfig(concurrency=0)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: concurrency must be >= 1, got 0
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

    # Determinism controls
    strict_serialization: bool = True
    deterministic_artifacts: Optional[bool] = None

    # Metadata
    dataset_info: Optional[dict[str, Any]] = None
    config_snapshot: Optional[dict[str, Any]] = None

    # Async-specific
    concurrency: int = 5
    timeout: Optional[float] = None  # Timeout in seconds for each probe execution

    # Resume/batch/output controls
    resume: bool = False
    use_probe_batch: bool = False
    batch_workers: Optional[int] = None
    return_experiment: bool = False

    # Verifiable evaluation (Ultimate mode)
    run_mode: Literal["default", "ultimate"] = "default"

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        This method is automatically called by the dataclass machinery after
        ``__init__`` completes. It performs validation of configuration values
        to ensure they are within acceptable ranges and combinations.

        The validation is intentionally strict to catch configuration errors
        early, before they cause issues during execution.

        Raises
        ------
        ValueError
            If ``validation_mode`` is not one of "strict" or "lenient".
            If ``concurrency`` is less than 1.
            If ``batch_workers`` is provided and is less than 1.

        Examples
        --------
        Valid configuration passes silently:

            >>> config = RunConfig(
            ...     validation_mode="strict",
            ...     concurrency=10,
            ...     batch_workers=4,
            ... )
            >>> # No error raised

        Invalid validation_mode raises ValueError:

            >>> try:
            ...     config = RunConfig(validation_mode="relaxed")
            ... except ValueError as e:
            ...     print("Caught:", e)
            Caught: validation_mode must be 'strict' or 'lenient', got 'relaxed'

        Zero concurrency is not allowed:

            >>> try:
            ...     config = RunConfig(concurrency=0)
            ... except ValueError as e:
            ...     print("Caught:", e)
            Caught: concurrency must be >= 1, got 0

        Negative batch_workers is not allowed:

            >>> try:
            ...     config = RunConfig(batch_workers=-1)
            ... except ValueError as e:
            ...     print("Caught:", e)
            Caught: batch_workers must be >= 1, got -1

        None is acceptable for batch_workers (uses probe default):

            >>> config = RunConfig(batch_workers=None)
            >>> # No error raised
        """
        if self.validation_mode == "warn":
            # Normalize CLI/validator terminology to RunConfig's "lenient" value.
            self.validation_mode = "lenient"
        if self.validation_mode not in ("strict", "lenient"):
            raise ValueError(
                f"validation_mode must be 'strict' or 'lenient', got {self.validation_mode!r}"
            )
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.batch_workers is not None and self.batch_workers < 1:
            raise ValueError(f"batch_workers must be >= 1, got {self.batch_workers}")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if not isinstance(self.strict_serialization, bool):
            raise ValueError(
                f"strict_serialization must be a bool, got {type(self.strict_serialization).__name__}"
            )
        if self.deterministic_artifacts is not None and not isinstance(
            self.deterministic_artifacts, bool
        ):
            raise ValueError(
                "deterministic_artifacts must be a bool or None, "
                f"got {type(self.deterministic_artifacts).__name__}"
            )

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "RunConfig":
        """Create a RunConfig from keyword arguments.

        This factory method provides backward compatibility with the old API
        where configuration was passed as individual keyword arguments to
        ``ProbeRunner.run()`` and ``AsyncProbeRunner.run()``. It also provides
        a convenient way to create configurations from dictionaries.

        Unknown keyword arguments are filtered out with a warning, making this
        method safe to use with dictionaries that may contain extra keys.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments matching RunConfig field names. Valid fields are:
            stop_on_error, validate_output, schema_version, validation_mode,
            emit_run_artifacts, run_dir, run_root, run_id, overwrite,
            store_messages, strict_serialization, deterministic_artifacts,
            dataset_info, config_snapshot, concurrency, timeout, resume,
            use_probe_batch, batch_workers, return_experiment.

        Returns
        -------
        RunConfig
            A new RunConfig instance with the specified values. Fields not
            provided use their default values.

        Warns
        -----
        UserWarning
            If any keyword arguments don't match RunConfig fields. The warning
            lists all unknown fields that were ignored.

        Raises
        ------
        ValueError
            If any provided values fail validation in ``__post_init__``.

        See Also
        --------
        RunConfigBuilder : Alternative fluent API for building configurations

        Notes
        -----
        This method is primarily intended for migration from the old API.
        For new code, prefer using the dataclass constructor directly or
        the ``RunConfigBuilder`` for complex configurations.

        The stacklevel for warnings is set to 2, so the warning points to
        the caller's code rather than this method.

        Examples
        --------
        Basic usage with valid fields:

            >>> config = RunConfig.from_kwargs(
            ...     validate_output=True,
            ...     concurrency=10,
            ...     run_root="./experiments",
            ... )
            >>> config.validate_output
            True
            >>> config.concurrency
            10

        Creating from a dictionary:

            >>> settings = {
            ...     "emit_run_artifacts": True,
            ...     "run_id": "my-experiment",
            ...     "concurrency": 5,
            ... }
            >>> config = RunConfig.from_kwargs(**settings)
            >>> config.run_id
            'my-experiment'

        Unknown fields are ignored with a warning:

            >>> import warnings
            >>> with warnings.catch_warnings(record=True) as w:
            ...     warnings.simplefilter("always")
            ...     config = RunConfig.from_kwargs(
            ...         validate_output=True,
            ...         unknown_field="ignored",
            ...         another_unknown=42,
            ...     )
            ...     if w:
            ...         print(f"Warning: {w[0].message}")
            Warning: Unknown RunConfig fields ignored: ['unknown_field', 'another_unknown']

        Migration from old API style:

            >>> # Old style (deprecated)
            >>> # runner.run(prompts, emit_run_artifacts=True, validate_output=True)

            >>> # New style using from_kwargs
            >>> legacy_kwargs = {"emit_run_artifacts": True, "validate_output": True}
            >>> config = RunConfig.from_kwargs(**legacy_kwargs)
            >>> # runner.run(prompts, config=config)

        Invalid values still raise ValueError:

            >>> try:
            ...     config = RunConfig.from_kwargs(concurrency=-1)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: concurrency must be >= 1, got -1

        Empty kwargs returns default configuration:

            >>> config = RunConfig.from_kwargs()
            >>> config.concurrency
            5
            >>> config.emit_run_artifacts
            True
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

    Provides a chainable (fluent) API for constructing RunConfig objects,
    making configuration more readable and discoverable. Each method returns
    ``self``, allowing method calls to be chained together.

    The builder pattern is particularly useful when:

    - Building configurations incrementally based on conditions
    - Creating configuration presets that can be further customized
    - Improving code readability for complex configurations

    Parameters
    ----------
    None
        The builder starts with default values matching RunConfig defaults.

    Attributes
    ----------
    All internal attributes are private (prefixed with ``_``) and should not
    be accessed directly. Use the ``with_*`` methods to set values and
    ``build()`` to create the final RunConfig.

    See Also
    --------
    RunConfig : The configuration class this builder creates
    RunConfig.from_kwargs : Alternative factory method for simple cases

    Notes
    -----
    The builder performs validation only when ``build()`` is called, not
    during the ``with_*`` method calls. This means invalid values won't
    raise errors until the final ``build()`` step.

    The builder is not reusable after ``build()`` is called. Create a new
    builder instance for each configuration you need to create.

    Examples
    --------
    Basic usage with method chaining:

        >>> config = (
        ...     RunConfigBuilder()
        ...     .with_validation(enabled=True, mode="strict")
        ...     .with_artifacts(run_root="./runs", run_id="my-run")
        ...     .with_concurrency(10)
        ...     .build()
        ... )
        >>> config.validate_output
        True
        >>> config.concurrency
        10

    Building a development configuration:

        >>> dev_config = (
        ...     RunConfigBuilder()
        ...     .with_error_handling(stop_on_error=True)
        ...     .with_validation(enabled=True, mode="strict")
        ...     .with_artifacts(run_root="./dev-runs")
        ...     .with_message_storage(enabled=True)
        ...     .build()
        ... )

    Building a production configuration:

        >>> prod_config = (
        ...     RunConfigBuilder()
        ...     .with_error_handling(stop_on_error=False)
        ...     .with_validation(enabled=True, mode="lenient")
        ...     .with_artifacts(
        ...         run_root="/data/experiments",
        ...         run_id="safety-eval-prod",
        ...     )
        ...     .with_concurrency(20)
        ...     .with_resume(enabled=True)
        ...     .with_probe_batch(enabled=True, batch_workers=4)
        ...     .with_output(return_experiment=True)
        ...     .build()
        ... )

    Conditional configuration building:

        >>> builder = RunConfigBuilder()
        >>> builder.with_artifacts(run_root="./runs")
        <insideLLMs.config_types.RunConfigBuilder object at ...>
        >>>
        >>> # Add validation only in debug mode
        >>> debug_mode = True
        >>> if debug_mode:
        ...     builder.with_validation(enabled=True, mode="strict")
        ...     builder.with_error_handling(stop_on_error=True)
        <insideLLMs.config_types.RunConfigBuilder object at ...>
        >>>
        >>> config = builder.build()

    Creating configuration presets:

        >>> def create_test_config_builder():
        ...     '''Create a pre-configured builder for testing.'''
        ...     return (
        ...         RunConfigBuilder()
        ...         .with_artifacts(enabled=False)
        ...         .with_validation(enabled=False)
        ...         .with_message_storage(enabled=False)
        ...     )
        >>>
        >>> # Use the preset and customize
        >>> test_config = (
        ...     create_test_config_builder()
        ...     .with_concurrency(1)  # Single-threaded for determinism
        ...     .build()
        ... )

    Using dataset and config metadata:

        >>> config = (
        ...     RunConfigBuilder()
        ...     .with_artifacts(run_id="toxicity-v2")
        ...     .with_dataset_info({
        ...         "name": "toxicity-prompts",
        ...         "version": "2.0.0",
        ...         "samples": 5000,
        ...     })
        ...     .with_config_snapshot({
        ...         "model": "gpt-4",
        ...         "temperature": 0.7,
        ...         "max_tokens": 1024,
        ...     })
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder with default values.

        Creates a new RunConfigBuilder with all configuration values set to
        their defaults, matching the defaults of RunConfig. The builder can
        then be customized using the ``with_*`` methods.

        Examples
        --------
        Create a builder and customize it:

            >>> builder = RunConfigBuilder()
            >>> config = builder.with_concurrency(10).build()
            >>> config.concurrency
            10

        Create a builder with defaults:

            >>> builder = RunConfigBuilder()
            >>> config = builder.build()
            >>> config.emit_run_artifacts  # Default is True
            True
            >>> config.concurrency  # Default is 5
            5
        """
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
        self._strict_serialization: bool = True
        self._deterministic_artifacts: Optional[bool] = None
        self._dataset_info: Optional[dict[str, Any]] = None
        self._config_snapshot: Optional[dict[str, Any]] = None
        self._concurrency: int = 5
        self._resume: bool = False
        self._use_probe_batch: bool = False
        self._batch_workers: Optional[int] = None
        self._return_experiment: bool = False
        self._run_mode: Literal["default", "ultimate"] = "default"

    def with_validation(
        self,
        enabled: bool = True,
        schema_version: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> "RunConfigBuilder":
        """Configure output validation settings.

        Enables or disables validation of probe outputs against a JSON schema,
        and configures the validation behavior. Validation is useful for
        ensuring data quality and catching malformed outputs early.

        Parameters
        ----------
        enabled : bool, default=True
            Whether to validate each probe output against the schema.
            When True, outputs are checked for schema compliance.

        schema_version : str, optional
            The schema version to use for validation. If not provided,
            keeps the current value (default: DEFAULT_SCHEMA_VERSION).
            Must match a registered schema version.

        mode : {"strict", "lenient"}, optional
            Validation strictness level. If not provided, keeps the
            current value (default: "strict").
            - "strict": Raise ValidationError on schema violations
            - "lenient": Log warnings but continue execution (alias: "warn")

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.validate_output : The underlying configuration field
        RunConfig.schema_version : Schema version field
        RunConfig.validation_mode : Validation mode field

        Examples
        --------
        Enable strict validation:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_validation(enabled=True, mode="strict")
            ...     .build()
            ... )
            >>> config.validate_output
            True
            >>> config.validation_mode
            'strict'

        Enable lenient validation with specific schema:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_validation(
            ...         enabled=True,
            ...         schema_version="1.0.0",
            ...         mode="lenient",
            ...     )
            ...     .build()
            ... )
            >>> config.validation_mode
            'lenient'

        Disable validation:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_validation(enabled=False)
            ...     .build()
            ... )
            >>> config.validate_output
            False

        Only change the mode, keep other settings:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_validation(enabled=True)  # Enable first
            ...     .with_validation(mode="lenient")  # Then adjust mode
            ...     .build()
            ... )
            >>> config.validate_output
            True
            >>> config.validation_mode
            'lenient'
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
        """Configure artifact output settings.

        Controls where and how run artifacts are stored. Artifacts include
        ``records.jsonl`` (individual probe results) and ``manifest.json``
        (run metadata and configuration).

        The output directory is determined by:

        1. If ``run_dir`` is provided, use it directly
        2. Otherwise, create ``{run_root}/{run_id}/``
        3. If neither is provided, use defaults (``~/.insidellms/runs/{uuid}/``)

        Parameters
        ----------
        enabled : bool, default=True
            Whether to emit run artifacts to disk. Set to False for
            testing or when artifacts aren't needed.

        run_dir : str or Path, optional
            Explicit path for the run directory. If provided, overrides
            ``run_root`` and ``run_id``. Use for precise control over
            the output location.

        run_root : str or Path, optional
            Root directory under which run directories are created.
            Each run creates a subdirectory named by ``run_id``.
            Defaults to ``~/.insidellms/runs``.

        run_id : str, optional
            Unique identifier for this run. Used as the directory name
            under ``run_root``. Auto-generated as a UUID if not provided.
            Use descriptive IDs for organization (e.g., "safety-eval-v1").

        overwrite : bool, default=False
            If True, overwrite an existing run directory with the same ID.
            If False (default), raise an error if the directory exists.
            Use with caution to avoid accidental data loss.

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.emit_run_artifacts : Enable/disable artifact output
        RunConfig.run_dir : Explicit directory path
        RunConfig.run_root : Root directory for runs

        Examples
        --------
        Basic artifact configuration:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(run_root="./experiments", run_id="exp-001")
            ...     .build()
            ... )
            >>> config.emit_run_artifacts
            True
            >>> config.run_id
            'exp-001'

        Explicit run directory:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(run_dir="/data/experiments/my-specific-run")
            ...     .build()
            ... )
            >>> str(config.run_dir)
            '/data/experiments/my-specific-run'

        Disable artifacts for testing:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(enabled=False)
            ...     .build()
            ... )
            >>> config.emit_run_artifacts
            False

        Allow overwriting existing runs:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(
            ...         run_root="./runs",
            ...         run_id="rerun-experiment",
            ...         overwrite=True,
            ...     )
            ...     .build()
            ... )
            >>> config.overwrite
            True

        Using Path objects:

            >>> from pathlib import Path
            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(
            ...         run_root=Path.home() / "experiments",
            ...         run_id="home-experiment",
            ...     )
            ...     .build()
            ... )
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

        Configures the maximum number of probe executions that can run
        concurrently when using ``AsyncProbeRunner``. Higher values
        increase throughput but may hit API rate limits or consume
        more memory.

        This setting is ignored by synchronous runners.

        Parameters
        ----------
        concurrency : int
            Maximum number of concurrent probe executions. Must be >= 1.
            Typical values range from 1 (sequential) to 50+ (high parallelism).

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.concurrency : The underlying configuration field

        Notes
        -----
        The optimal concurrency depends on:

        - API rate limits of the target model
        - Available memory and CPU resources
        - Network bandwidth and latency
        - Whether the probe does additional processing

        Start with lower values (5-10) and increase based on observed
        performance and error rates.

        Examples
        --------
        Low concurrency for rate-limited APIs:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_concurrency(3)
            ...     .build()
            ... )
            >>> config.concurrency
            3

        High concurrency for batch processing:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_concurrency(50)
            ...     .build()
            ... )
            >>> config.concurrency
            50

        Sequential execution:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_concurrency(1)
            ...     .build()
            ... )
            >>> config.concurrency
            1
        """
        self._concurrency = concurrency
        return self

    def with_error_handling(self, stop_on_error: bool = True) -> "RunConfigBuilder":
        """Configure error handling behavior.

        Controls whether the runner stops immediately when an error occurs
        or continues processing remaining items. This affects both individual
        probe errors and validation errors.

        Parameters
        ----------
        stop_on_error : bool, default=True
            If True, stop execution immediately on the first error.
            If False, continue processing and collect all errors.

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.stop_on_error : The underlying configuration field

        Notes
        -----
        Use ``stop_on_error=True`` during development to fail fast and
        identify issues quickly. Use ``stop_on_error=False`` in production
        to maximize data collection even when some items fail.

        When ``stop_on_error=False``, errors are recorded in the output
        records and can be analyzed after the run completes.

        Examples
        --------
        Fail fast for development:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_error_handling(stop_on_error=True)
            ...     .build()
            ... )
            >>> config.stop_on_error
            True

        Continue on errors for production:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_error_handling(stop_on_error=False)
            ...     .build()
            ... )
            >>> config.stop_on_error
            False
        """
        self._stop_on_error = stop_on_error
        return self

    def with_stop_on_error(self, stop_on_error: bool = True) -> "RunConfigBuilder":
        """Alias for with_error_handling (backward compatibility)."""
        return self.with_error_handling(stop_on_error=stop_on_error)

    def with_dataset_info(self, info: dict[str, Any]) -> "RunConfigBuilder":
        """Set dataset metadata for the run manifest.

        Stores metadata about the dataset being evaluated. This information
        is included in the run manifest (``manifest.json``) for reproducibility
        and documentation purposes.

        Parameters
        ----------
        info : dict
            Dictionary containing dataset metadata. Common fields include:

            - ``name``: Dataset name or identifier
            - ``version``: Dataset version
            - ``source``: URL or path to the dataset
            - ``num_samples``: Number of samples in the dataset
            - ``description``: Brief description
            - ``split``: Train/test/validation split used
            - ``hash``: Content hash for integrity verification

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.dataset_info : The underlying configuration field
        with_config_snapshot : For experiment configuration metadata

        Examples
        --------
        Basic dataset info:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_dataset_info({
            ...         "name": "safety-prompts",
            ...         "version": "1.0.0",
            ...         "num_samples": 1000,
            ...     })
            ...     .build()
            ... )
            >>> config.dataset_info["name"]
            'safety-prompts'

        Comprehensive dataset metadata:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_dataset_info({
            ...         "name": "toxicity-benchmark",
            ...         "version": "2.1.0",
            ...         "source": "https://huggingface.co/datasets/toxicity",
            ...         "num_samples": 50000,
            ...         "split": "test",
            ...         "description": "Toxicity detection benchmark",
            ...         "hash": "sha256:abc123...",
            ...         "created_at": "2024-01-15",
            ...     })
            ...     .build()
            ... )
        """
        self._dataset_info = info
        return self

    def with_config_snapshot(self, snapshot: dict[str, Any]) -> "RunConfigBuilder":
        """Set experiment configuration snapshot for the run manifest.

        Stores a snapshot of the experiment configuration at the time of
        the run. This is included in the manifest for full reproducibility
        of the experimental setup.

        Parameters
        ----------
        snapshot : dict
            Dictionary containing experiment configuration. Common fields:

            - ``model``: Model name or identifier
            - ``temperature``: Sampling temperature
            - ``max_tokens``: Maximum tokens to generate
            - ``top_p``: Nucleus sampling parameter
            - ``probe``: Probe name or configuration
            - ``prompt_template``: Template used for prompts
            - ``system_prompt``: System prompt if used

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.config_snapshot : The underlying configuration field
        with_dataset_info : For dataset metadata

        Examples
        --------
        Model configuration snapshot:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_config_snapshot({
            ...         "model": "gpt-4-turbo",
            ...         "temperature": 0.7,
            ...         "max_tokens": 1024,
            ...     })
            ...     .build()
            ... )
            >>> config.config_snapshot["model"]
            'gpt-4-turbo'

        Full experiment configuration:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_config_snapshot({
            ...         "model": "claude-3-opus",
            ...         "temperature": 0.5,
            ...         "max_tokens": 2048,
            ...         "top_p": 0.95,
            ...         "probe": "safety-classifier-v2",
            ...         "system_prompt": "You are a helpful assistant.",
            ...         "experiment_name": "safety-eval-2024",
            ...         "hypothesis": "Testing refusal rates on edge cases",
            ...     })
            ...     .build()
            ... )
        """
        self._config_snapshot = snapshot
        return self

    def with_message_storage(self, enabled: bool = True) -> "RunConfigBuilder":
        """Configure message history storage in records.

        Controls whether the full conversation history (all messages exchanged
        with the model) is stored in each output record. Disabling this can
        significantly reduce storage requirements for large-scale experiments.

        Parameters
        ----------
        enabled : bool, default=True
            If True, store the complete message history in each record.
            If False, omit message history to save storage space.

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.store_messages : The underlying configuration field

        Notes
        -----
        Message history is useful for:

        - Debugging probe behavior
        - Analyzing conversation patterns
        - Reproducing specific interactions

        Disable message storage when:

        - Running large-scale experiments (millions of items)
        - Only final outputs matter for analysis
        - Storage space is limited

        Examples
        --------
        Enable message storage (default):

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_message_storage(enabled=True)
            ...     .build()
            ... )
            >>> config.store_messages
            True

        Disable for large experiments:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_message_storage(enabled=False)
            ...     .build()
            ... )
            >>> config.store_messages
            False
        """
        self._store_messages = enabled
        return self

    def with_determinism(
        self,
        *,
        strict_serialization: bool = True,
        deterministic_artifacts: Optional[bool] = None,
    ) -> "RunConfigBuilder":
        """Configure determinism hardening controls.

        Parameters
        ----------
        strict_serialization : bool, default=True
            If True, hashing/fingerprinting fails fast on values that would
            otherwise fall back to ``str(value)``.
        deterministic_artifacts : Optional[bool], default=None
            If True, omit host-dependent manifest fields (platform/python).
            If None, follows ``strict_serialization`` at runtime.
        """
        self._strict_serialization = strict_serialization
        self._deterministic_artifacts = deterministic_artifacts
        return self

    def with_resume(self, enabled: bool = True) -> "RunConfigBuilder":
        """Enable resumable runs.

        When enabled, the runner will check for existing ``records.jsonl``
        in the run directory and skip items that have already been processed.
        This is useful for recovering from interrupted runs without
        reprocessing completed items.

        Parameters
        ----------
        enabled : bool, default=True
            If True, resume from existing artifacts when available.
            If False, start fresh (may fail if directory exists and
            ``overwrite=False``).

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.resume : The underlying configuration field
        with_artifacts : For configuring output directory and overwrite

        Notes
        -----
        For resume to work correctly:

        - Use the same ``run_id`` as the interrupted run
        - Items are matched by their unique ID field
        - Already-processed items are loaded from ``records.jsonl``

        This is particularly useful for:

        - Long-running experiments that may be interrupted
        - Incremental dataset updates
        - Recovering from temporary API failures

        Examples
        --------
        Enable resume for long experiments:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(run_id="long-experiment")
            ...     .with_resume(enabled=True)
            ...     .build()
            ... )
            >>> config.resume
            True

        Disable resume for fresh runs:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_resume(enabled=False)
            ...     .build()
            ... )
            >>> config.resume
            False

        Combined with overwrite for clean restarts:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_artifacts(run_id="experiment", overwrite=True)
            ...     .with_resume(enabled=False)
            ...     .build()
            ... )
        """
        self._resume = enabled
        return self

    def with_probe_batch(
        self,
        enabled: bool = True,
        batch_workers: Optional[int] = None,
    ) -> "RunConfigBuilder":
        """Enable probe batch execution.

        When enabled, the runner will use the probe's native ``run_batch``
        method instead of processing items individually. This can be more
        efficient for probes that support batched operations.

        Parameters
        ----------
        enabled : bool, default=True
            If True, use ``Probe.run_batch`` when the probe supports it.
            If False, process items individually.

        batch_workers : int, optional
            Number of workers for batch execution. If None, uses the
            probe's default worker count. Must be >= 1 if provided.

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.use_probe_batch : The underlying configuration field
        RunConfig.batch_workers : Worker count field

        Notes
        -----
        Batch execution is beneficial when:

        - The probe has efficient batch processing
        - API supports batched requests
        - Overhead per-item is significant

        Not all probes support batch execution. When ``use_probe_batch=True``
        but the probe doesn't support it, the runner falls back to
        individual processing.

        Examples
        --------
        Enable batch execution:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_probe_batch(enabled=True)
            ...     .build()
            ... )
            >>> config.use_probe_batch
            True

        Enable with specific worker count:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_probe_batch(enabled=True, batch_workers=4)
            ...     .build()
            ... )
            >>> config.batch_workers
            4

        Disable batch execution:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_probe_batch(enabled=False)
            ...     .build()
            ... )
            >>> config.use_probe_batch
            False
        """
        self._use_probe_batch = enabled
        if batch_workers is not None:
            self._batch_workers = batch_workers
        return self

    def with_output(self, return_experiment: bool = True) -> "RunConfigBuilder":
        """Configure output format for run results.

        Controls whether the runner returns raw dictionaries or a structured
        ``ExperimentResult`` object. The structured result provides convenient
        methods for analysis and aggregation.

        Parameters
        ----------
        return_experiment : bool, default=True
            If True, return an ``ExperimentResult`` object with analysis methods.
            If False, return a list of raw result dictionaries.

        Returns
        -------
        RunConfigBuilder
            Self, for method chaining.

        See Also
        --------
        RunConfig.return_experiment : The underlying configuration field

        Notes
        -----
        The ``ExperimentResult`` object provides:

        - Aggregation methods (e.g., success rate, error counts)
        - Filtering and grouping utilities
        - Export to various formats (DataFrame, JSON, CSV)
        - Statistical summaries

        Use raw dictionaries when:

        - Integrating with existing pipelines expecting dicts
        - Custom processing that doesn't need ExperimentResult features
        - Minimal memory footprint is important

        Examples
        --------
        Return structured result:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_output(return_experiment=True)
            ...     .build()
            ... )
            >>> config.return_experiment
            True

        Return raw dictionaries:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_output(return_experiment=False)
            ...     .build()
            ... )
            >>> config.return_experiment
            False
        """
        self._return_experiment = return_experiment
        return self

    def build(self) -> RunConfig:
        """Build the RunConfig object from the builder state.

        Creates and returns a new ``RunConfig`` instance with all the values
        configured through the builder methods. This method performs validation
        via ``RunConfig.__post_init__``.

        Returns
        -------
        RunConfig
            A new RunConfig instance with the configured values.

        Raises
        ------
        ValueError
            If any configured values are invalid. Common causes:

            - ``validation_mode`` not in {"strict", "lenient"}
            - ``concurrency`` < 1
            - ``batch_workers`` < 1 (when provided)

        See Also
        --------
        RunConfig : The configuration class created by this method
        RunConfig.__post_init__ : Validation performed on creation

        Notes
        -----
        The builder should not be reused after calling ``build()``.
        Create a new builder instance for each configuration.

        Validation errors are raised at build time, not during the
        ``with_*`` method calls. This means invalid values won't be
        caught until ``build()`` is called.

        Examples
        --------
        Build a valid configuration:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_validation(enabled=True)
            ...     .with_concurrency(10)
            ...     .build()
            ... )
            >>> isinstance(config, RunConfig)
            True
            >>> config.validate_output
            True

        Build with defaults:

            >>> config = RunConfigBuilder().build()
            >>> config.concurrency
            5
            >>> config.emit_run_artifacts
            True

        Invalid configuration raises on build:

            >>> try:
            ...     config = (
            ...         RunConfigBuilder()
            ...         .with_concurrency(0)
            ...         .build()
            ...     )
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: concurrency must be >= 1, got 0

        Complete configuration:

            >>> config = (
            ...     RunConfigBuilder()
            ...     .with_error_handling(stop_on_error=False)
            ...     .with_validation(enabled=True, mode="lenient")
            ...     .with_artifacts(
            ...         run_root="./experiments",
            ...         run_id="full-eval",
            ...     )
            ...     .with_concurrency(20)
            ...     .with_resume(enabled=True)
            ...     .with_probe_batch(enabled=True, batch_workers=4)
            ...     .with_output(return_experiment=True)
            ...     .with_dataset_info({"name": "test-dataset"})
            ...     .with_config_snapshot({"model": "gpt-4"})
            ...     .build()
            ... )
            >>> config.concurrency
            20
            >>> config.resume
            True
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
            strict_serialization=self._strict_serialization,
            deterministic_artifacts=self._deterministic_artifacts,
            dataset_info=self._dataset_info,
            config_snapshot=self._config_snapshot,
            concurrency=self._concurrency,
            resume=self._resume,
            use_probe_batch=self._use_probe_batch,
            batch_workers=self._batch_workers,
            return_experiment=self._return_experiment,
            run_mode=self._run_mode,
        )


@dataclass
class RunContext:
    """Runtime context for a probe run.

    This dataclass holds the resolved runtime state during probe execution,
    including the actual run directory, run ID, timing information, and
    resolved specifications for the manifest. It is created by the runner
    at the start of execution and passed through to various internal methods.

    This is an **internal class** used by the runner implementations
    (``ProbeRunner`` and ``AsyncProbeRunner``). External code should not
    need to instantiate or manipulate ``RunContext`` directly.

    Parameters
    ----------
    run_id : str
        The unique identifier for this run. Either user-provided via
        ``RunConfig.run_id`` or auto-generated as a UUID.

    run_dir : Path
        The resolved path to the run directory where artifacts are stored.
        This is computed from ``RunConfig.run_dir`` or
        ``{run_root}/{run_id}/``.

    run_started_at : datetime
        The datetime when the run started, used for the manifest and
        timing calculations. This is a ``datetime.datetime`` object.

    run_base_time : float
        The ``time.perf_counter()`` value at run start, used for
        calculating elapsed time and progress estimates. This provides
        high-resolution timing independent of system clock changes.

    model_spec : dict, default={}
        Resolved specification for the model being used. Included in
        the manifest for reproducibility. Typically contains model name,
        version, and configuration.

    probe_spec : dict, default={}
        Resolved specification for the probe being used. Included in
        the manifest. Contains probe name, version, and parameters.

    dataset_spec : dict, default={}
        Resolved specification for the dataset. Derived from
        ``RunConfig.dataset_info`` if provided.

    Attributes
    ----------
    run_id : str
        Unique run identifier.
    run_dir : Path
        Path to run directory.
    run_started_at : datetime
        Run start timestamp.
    run_base_time : float
        High-resolution start time.
    model_spec : dict
        Model specification.
    probe_spec : dict
        Probe specification.
    dataset_spec : dict
        Dataset specification.

    See Also
    --------
    RunConfig : User-facing configuration (resolved into RunContext)
    ProgressInfo : Progress tracking information

    Notes
    -----
    The separation between ``RunConfig`` (user-facing configuration) and
    ``RunContext`` (runtime state) allows the configuration to remain
    immutable while the context can hold mutable state during execution.

    The ``run_started_at`` and ``run_base_time`` fields serve different
    purposes:

    - ``run_started_at``: Human-readable timestamp for logs and manifest
    - ``run_base_time``: High-precision timing for progress calculations

    Examples
    --------
    Internal usage by runners (not for external use):

        >>> from pathlib import Path
        >>> from datetime import datetime
        >>> import time
        >>>
        >>> # This is how runners create RunContext internally
        >>> context = RunContext(
        ...     run_id="safety-eval-001",
        ...     run_dir=Path("/data/runs/safety-eval-001"),
        ...     run_started_at=datetime.now(),
        ...     run_base_time=time.perf_counter(),
        ...     model_spec={"name": "gpt-4", "version": "turbo"},
        ...     probe_spec={"name": "safety-classifier", "version": "2.0"},
        ...     dataset_spec={"name": "safety-prompts", "count": 1000},
        ... )

    Accessing context fields:

        >>> context.run_id
        'safety-eval-001'
        >>> context.run_dir
        PosixPath('/data/runs/safety-eval-001')
        >>> context.model_spec["name"]
        'gpt-4'

    Computing elapsed time:

        >>> import time
        >>> elapsed = time.perf_counter() - context.run_base_time
        >>> print(f"Elapsed: {elapsed:.2f} seconds")
        Elapsed: 0.00 seconds
    """

    run_id: str
    run_dir: Path = field(default_factory=lambda: Path("."))
    run_started_at: datetime = field(default_factory=datetime.now)
    run_base_time: float = field(default_factory=time.perf_counter)

    # Resolved specs for manifest
    model_spec: dict[str, Any] = field(default_factory=dict)
    probe_spec: dict[str, Any] = field(default_factory=dict)
    dataset_spec: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressInfo:
    """Structured progress information for callbacks.

    Provides detailed progress information including timing, completion rate,
    and ETA estimates for implementing progress reporting in probe runs.
    This class is passed to progress callbacks registered with the runner.

    The class includes both raw data (current, total, elapsed) and computed
    properties (percent, remaining, is_complete) for convenience in
    formatting progress displays.

    Parameters
    ----------
    current : int
        Number of items that have been completed. Starts at 0 and
        increments as items are processed.

    total : int
        Total number of items to process. Known at the start of the run.

    elapsed_seconds : float
        Time elapsed since the start of the run, in seconds. Computed
        from ``time.perf_counter()`` for high precision.

    rate : float, default=0.0
        Processing rate in items per second. Calculated as
        ``current / elapsed_seconds`` when both are positive.

    eta_seconds : float, optional
        Estimated time remaining in seconds, calculated as
        ``remaining / rate``. None if rate is 0 or cannot be estimated.

    current_item : Any, optional
        The item currently being processed. Useful for debugging or
        displaying item-specific progress information.

    current_index : int, optional
        Zero-based index of the current item in the input sequence.
        May differ from ``current`` during parallel processing.

    status : str, optional
        Current status message (e.g., "Processing", "Validating",
        "Writing artifacts"). Useful for multi-phase operations.

    Attributes
    ----------
    current : int
        Completed items count.
    total : int
        Total items count.
    elapsed_seconds : float
        Time elapsed since start.
    rate : float
        Items per second.
    eta_seconds : float or None
        Estimated time remaining.
    current_item : Any
        Current item being processed.
    current_index : int or None
        Current item index.
    status : str or None
        Current status message.

    See Also
    --------
    RunConfig : Configuration for probe runs
    RunContext : Runtime context during execution

    Notes
    -----
    The ``rate`` and ``eta_seconds`` fields are estimates based on
    average processing time. Actual completion time may vary due to:

    - Variable item complexity
    - API rate limiting
    - Network latency variations
    - System load changes

    For more accurate estimates in production, consider implementing
    a rolling average or exponential smoothing in your callback.

    Examples
    --------
    Basic progress callback:

        >>> def progress_callback(info: ProgressInfo) -> None:
        ...     print(f"Progress: {info.current}/{info.total} ({info.percent:.1f}%)")
        ...
        >>> # Register with runner
        >>> # runner.run(prompts, progress_callback=progress_callback)

    Progress bar with ETA:

        >>> def progress_with_eta(info: ProgressInfo) -> None:
        ...     bar_width = 40
        ...     filled = int(bar_width * info.percent / 100)
        ...     bar = "=" * filled + "-" * (bar_width - filled)
        ...     eta_str = f"{info.eta_seconds:.0f}s" if info.eta_seconds else "N/A"
        ...     print(f"[{bar}] {info.percent:.1f}% | ETA: {eta_str}")

    Verbose progress with rate:

        >>> def verbose_progress(info: ProgressInfo) -> None:
        ...     print(f"Completed: {info.current}/{info.total}")
        ...     print(f"Elapsed: {info.elapsed_seconds:.1f}s")
        ...     print(f"Rate: {info.rate:.2f} items/sec")
        ...     print(f"Remaining: {info.remaining} items")
        ...     if info.eta_seconds:
        ...         print(f"ETA: {info.eta_seconds:.0f} seconds")
        ...     if info.status:
        ...         print(f"Status: {info.status}")

    Using the string representation:

        >>> info = ProgressInfo(
        ...     current=50,
        ...     total=100,
        ...     elapsed_seconds=25.0,
        ...     rate=2.0,
        ...     eta_seconds=25.0,
        ... )
        >>> print(info)
        50/100 (50.0%) | 2.0/s | ETA: 25s

    Checking completion status:

        >>> info = ProgressInfo(current=100, total=100, elapsed_seconds=50.0)
        >>> info.is_complete
        True
        >>> info.remaining
        0

    Creating from timing data (factory method):

        >>> import time
        >>> start_time = time.perf_counter()
        >>> # ... process 10 items ...
        >>> info = ProgressInfo.create(
        ...     current=10,
        ...     total=100,
        ...     start_time=start_time,
        ...     status="Processing batch 1",
        ... )
        >>> info.current
        10
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
        """Calculate percentage of items completed.

        Returns the completion percentage as a float between 0 and 100.
        Returns 0.0 if total is 0.

        Returns
        -------
        float
            Percentage complete, in range [0.0, 100.0].

        Examples
        --------
        Normal calculation:

            >>> info = ProgressInfo(current=25, total=100, elapsed_seconds=10.0)
            >>> info.percent
            25.0

        Empty input returns 0%:

            >>> info = ProgressInfo(current=0, total=0, elapsed_seconds=0.0)
            >>> info.percent
            0.0

        Complete:

            >>> info = ProgressInfo(current=100, total=100, elapsed_seconds=50.0)
            >>> info.percent
            100.0
        """
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    @property
    def percentage(self) -> float:
        """Alias for percent (backward compatibility)."""
        return self.percent

    @property
    def remaining(self) -> int:
        """Calculate number of items remaining to process.

        Returns
        -------
        int
            Number of items not yet completed. Always >= 0.

        Examples
        --------
            >>> info = ProgressInfo(current=30, total=100, elapsed_seconds=15.0)
            >>> info.remaining
            70

            >>> info = ProgressInfo(current=100, total=100, elapsed_seconds=50.0)
            >>> info.remaining
            0
        """
        return self.total - self.current

    @property
    def is_complete(self) -> bool:
        """Check whether all items have been processed.

        Returns
        -------
        bool
            True if current >= total, False otherwise.

        Examples
        --------
            >>> info = ProgressInfo(current=50, total=100, elapsed_seconds=25.0)
            >>> info.is_complete
            False

            >>> info = ProgressInfo(current=100, total=100, elapsed_seconds=50.0)
            >>> info.is_complete
            True

            >>> info = ProgressInfo(current=0, total=0, elapsed_seconds=0.0)
            >>> info.is_complete
            True
        """
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
        """Create a ProgressInfo with automatically calculated rate and ETA.

        Factory method that creates a ProgressInfo instance with computed
        ``elapsed_seconds``, ``rate``, and ``eta_seconds`` fields based on
        the provided start time. This is the recommended way to create
        ProgressInfo instances during execution.

        Parameters
        ----------
        current : int
            Number of items that have been completed so far.

        total : int
            Total number of items to process.

        start_time : float
            The ``time.perf_counter()`` value captured at the start of
            processing. Used to calculate elapsed time.

        current_item : Any, optional
            The item currently being processed. Passed through to the
            created instance.

        current_index : int, optional
            Zero-based index of the current item. Passed through to the
            created instance.

        status : str, optional
            Current status message. Passed through to the created instance.

        Returns
        -------
        ProgressInfo
            A new ProgressInfo instance with all fields populated,
            including computed ``elapsed_seconds``, ``rate``, and
            ``eta_seconds``.

        See Also
        --------
        ProgressInfo : The class this method creates instances of

        Notes
        -----
        The method calculates:

        - ``elapsed_seconds``: Current time minus ``start_time``
        - ``rate``: ``current / elapsed_seconds`` (0 if no items completed)
        - ``eta_seconds``: ``remaining / rate`` (None if rate is 0)

        The ``time`` module is imported inside this method to avoid
        circular import issues and minimize import overhead when the
        factory isn't used.

        Examples
        --------
        Basic usage during processing:

            >>> import time
            >>> start = time.perf_counter()
            >>> # ... process some items ...
            >>> time.sleep(0.1)  # Simulate processing
            >>> info = ProgressInfo.create(
            ...     current=10,
            ...     total=100,
            ...     start_time=start,
            ... )
            >>> info.current
            10
            >>> info.total
            100
            >>> info.elapsed_seconds > 0
            True

        With status message:

            >>> import time
            >>> start = time.perf_counter()
            >>> info = ProgressInfo.create(
            ...     current=5,
            ...     total=50,
            ...     start_time=start,
            ...     status="Validating outputs",
            ... )
            >>> info.status
            'Validating outputs'

        With current item tracking:

            >>> import time
            >>> start = time.perf_counter()
            >>> current_prompt = {"id": "prompt-42", "text": "Hello, world!"}
            >>> info = ProgressInfo.create(
            ...     current=42,
            ...     total=100,
            ...     start_time=start,
            ...     current_item=current_prompt,
            ...     current_index=41,
            ... )
            >>> info.current_item["id"]
            'prompt-42'
            >>> info.current_index
            41

        Zero items completed (rate is 0, ETA is None):

            >>> import time
            >>> start = time.perf_counter()
            >>> info = ProgressInfo.create(
            ...     current=0,
            ...     total=100,
            ...     start_time=start,
            ... )
            >>> info.rate
            0.0
            >>> info.eta_seconds is None
            True
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
        """Return a human-readable progress string.

        Formats the progress information as a compact, human-readable string
        suitable for logging or display. The format includes completion count,
        percentage, rate (if available), and ETA (if available).

        Returns
        -------
        str
            A formatted progress string. Format varies based on available data:
            - Basic: "10/100 (10.0%)"
            - With rate: "10/100 (10.0%) | 2.0/s"
            - With ETA < 60s: "10/100 (10.0%) | 2.0/s | ETA: 45s"
            - With ETA >= 60s: "10/100 (10.0%) | 2.0/s | ETA: 2m30s"

        Examples
        --------
        Basic progress (no rate/ETA):

            >>> info = ProgressInfo(current=10, total=100, elapsed_seconds=0.0)
            >>> str(info)
            '10/100 (10.0%)'

        With rate:

            >>> info = ProgressInfo(
            ...     current=10,
            ...     total=100,
            ...     elapsed_seconds=5.0,
            ...     rate=2.0,
            ... )
            >>> str(info)
            '10/100 (10.0%) | 2.0/s'

        With rate and short ETA:

            >>> info = ProgressInfo(
            ...     current=10,
            ...     total=100,
            ...     elapsed_seconds=5.0,
            ...     rate=2.0,
            ...     eta_seconds=45.0,
            ... )
            >>> str(info)
            '10/100 (10.0%) | 2.0/s | ETA: 45s'

        With rate and long ETA:

            >>> info = ProgressInfo(
            ...     current=10,
            ...     total=100,
            ...     elapsed_seconds=5.0,
            ...     rate=2.0,
            ...     eta_seconds=150.0,
            ... )
            >>> str(info)
            '10/100 (10.0%) | 2.0/s | ETA: 2m30s'

        Complete (100%):

            >>> info = ProgressInfo(
            ...     current=100,
            ...     total=100,
            ...     elapsed_seconds=50.0,
            ...     rate=2.0,
            ... )
            >>> str(info)
            '100/100 (100.0%) | 2.0/s'
        """
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
