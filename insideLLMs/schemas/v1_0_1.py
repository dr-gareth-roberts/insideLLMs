"""Schema definitions for the output contract v1.0.1.

This module provides versioned Pydantic schema models for insideLLMs output
contracts, specifically version 1.0.1. This version is a backward-compatible
extension of v1.0.0, adding the ``run_completed`` field to ``RunManifest``
for explicit run completion tracking.

The schemas in this module are used to validate and serialize outputs from
probe runs, benchmarks, and other insideLLMs operations. They ensure consistent,
machine-checkable data formats across different library versions and enable
reliable data exchange between systems.

Key Features
------------
- Strict field validation (extra fields forbidden) to prevent schema drift
- Full backward compatibility with v1.0.0 for all unchanged schemas
- Explicit run completion status tracking via ``run_completed`` boolean
- Automatic schema version tagging for migration support

Version Differences from v1.0.0
-------------------------------
- **RunManifest**: Added ``run_completed: bool`` field (default: False) to
  explicitly mark when a probe run completed successfully versus crashed
  or was interrupted.

All other schemas (ProbeResult, ResultRecord, etc.) are identical to v1.0.0
and are accessed via delegation to the v1.0.0 module.

Examples
--------
Creating a RunManifest with completion status:

    >>> from datetime import datetime
    >>> from insideLLMs.schemas.v1_0_1 import RunManifest
    >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
    >>> manifest = RunManifest(
    ...     run_id="run-abc-123",
    ...     created_at=datetime(2024, 1, 15, 10, 0, 0),
    ...     started_at=datetime(2024, 1, 15, 10, 0, 1),
    ...     completed_at=datetime(2024, 1, 15, 10, 5, 30),
    ...     run_completed=True,
    ...     model=ModelSpec(model_id="gpt-4"),
    ...     probe=ProbeSpec(probe_id="truthfulness-v1"),
    ...     record_count=100,
    ...     success_count=98,
    ...     error_count=2
    ... )
    >>> manifest.schema_version
    '1.0.1'
    >>> manifest.run_completed
    True

Looking up schemas by name:

    >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
    >>> RunManifestClass = get_schema_model("RunManifest")
    >>> RunManifestClass.__name__
    'RunManifest'

Accessing unchanged v1.0.0 schemas through delegation:

    >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
    >>> ProbeResult = get_schema_model("ProbeResult")
    >>> result = ProbeResult(input="What is 2+2?", status="success", output="4")
    >>> result.status
    'success'

Serializing a manifest to JSON:

    >>> from datetime import datetime
    >>> from insideLLMs.schemas.v1_0_1 import RunManifest
    >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
    >>> manifest = RunManifest(
    ...     run_id="run-xyz-789",
    ...     created_at=datetime(2024, 2, 20, 14, 0, 0),
    ...     started_at=datetime(2024, 2, 20, 14, 0, 0),
    ...     completed_at=datetime(2024, 2, 20, 14, 30, 0),
    ...     run_completed=True,
    ...     model=ModelSpec(model_id="claude-3-opus"),
    ...     probe=ProbeSpec(probe_id="safety-probe-v2")
    ... )
    >>> json_data = manifest.model_dump()  # Pydantic v2
    >>> json_data["run_completed"]
    True
    >>> json_data["schema_version"]
    '1.0.1'

Notes
-----
- This module requires Pydantic (v1 or v2) to be installed. If Pydantic is
  not available, importing this module will raise an ImportError with
  installation instructions.

- The ``run_completed`` field defaults to False for safety. This means that
  if a run crashes before setting this flag, the manifest will correctly
  indicate incomplete status.

- Schema validation is strict: attempting to add fields not defined in the
  schema will raise a validation error. This prevents accidental schema
  drift and ensures data consistency.

- For cross-version compatibility, use the ``SchemaRegistry`` from
  ``insideLLMs.schemas.registry`` to look up schemas by name and version.

See Also
--------
insideLLMs.schemas.v1_0_0 : Base schema definitions for v1.0.0
insideLLMs.schemas.registry : Schema registry for version-aware lookups
insideLLMs.schemas.registry.SchemaRegistry : Main registry class for schema access

Attributes
----------
SCHEMA_VERSION : str
    The semantic version string for this schema module ("1.0.1").
RunManifest : class
    The v1.0.1 RunManifest schema with ``run_completed`` field.
get_schema_model : function
    Function to retrieve schema classes by name.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from insideLLMs.schemas import v1_0_0

SCHEMA_VERSION = "1.0.1"


class RunManifest(v1_0_0._BaseSchema):
    """Run directory manifest emitted alongside records.jsonl (v1.0.1).

    This schema represents the metadata manifest for a probe run directory.
    It contains comprehensive information about the run including timing,
    environment details, model and probe specifications, and aggregate
    statistics about the results.

    Version 1.0.1 extends the v1.0.0 RunManifest by adding the ``run_completed``
    field, which explicitly tracks whether a run completed successfully or was
    interrupted (e.g., due to a crash, timeout, or user cancellation).

    The manifest is typically written as ``manifest.json`` in the run output
    directory, alongside the ``records.jsonl`` file containing individual
    result records.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run. Typically a UUID or timestamp-based
        identifier that distinguishes this run from others.
    created_at : datetime
        Timestamp when the run directory was created.
    started_at : datetime
        Timestamp when the actual probe execution began.
    completed_at : datetime
        Timestamp when the run finished (regardless of success/failure).
    run_completed : bool, optional
        Whether the run completed successfully without interruption.
        Default is False for safety (assumes incomplete until explicitly
        marked complete).
    model : ModelSpec
        Specification of the model being probed, including model_id,
        provider, and any configuration parameters.
    probe : ProbeSpec
        Specification of the probe being run, including probe_id,
        version, and configuration parameters.
    dataset : DatasetSpec, optional
        Specification of the dataset used for the probe run, including
        provenance and versioning information.
    library_version : str, optional
        Version of the insideLLMs library used for the run.
    python_version : str, optional
        Python version used for the run (e.g., "3.11.4").
    platform : str, optional
        Platform information (e.g., "linux-x86_64", "darwin-arm64").
    command : list[str], optional
        Command line arguments used to invoke the run.
    record_count : int, optional
        Total number of records processed. Default is 0.
    success_count : int, optional
        Number of records that completed successfully. Default is 0.
    error_count : int, optional
        Number of records that resulted in errors. Default is 0.
    records_file : str, optional
        Filename of the JSONL records file. Default is "records.jsonl".
    schemas : dict[str, Any], optional
        Embedded JSON schemas for validation. Default is empty dict.
    custom : dict[str, Any], optional
        User-defined custom metadata. Default is empty dict.

    Attributes
    ----------
    schema_version : str
        The schema version identifier, automatically set to "1.0.1".
    run_id : str
        The unique run identifier.
    created_at : datetime
        When the run was created.
    started_at : datetime
        When execution started.
    completed_at : datetime
        When execution ended.
    run_completed : bool
        Whether the run completed successfully (v1.0.1 addition).
    library_version : str or None
        insideLLMs library version.
    python_version : str or None
        Python interpreter version.
    platform : str or None
        Operating system and architecture.
    command : list[str] or None
        CLI command used to start the run.
    model : ModelSpec
        Model specification.
    probe : ProbeSpec
        Probe specification.
    dataset : DatasetSpec
        Dataset specification.
    record_count : int
        Total records processed.
    success_count : int
        Records with success status.
    error_count : int
        Records with error status.
    records_file : str
        Name of the records JSONL file.
    schemas : dict[str, Any]
        Embedded schema definitions.
    custom : dict[str, Any]
        Custom user metadata.

    Examples
    --------
    Creating a basic manifest for a completed run:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_1 import RunManifest
        >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
        >>> manifest = RunManifest(
        ...     run_id="run-2024-01-15-abc123",
        ...     created_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     started_at=datetime(2024, 1, 15, 10, 0, 5),
        ...     completed_at=datetime(2024, 1, 15, 10, 15, 30),
        ...     run_completed=True,
        ...     model=ModelSpec(model_id="gpt-4-turbo", provider="openai"),
        ...     probe=ProbeSpec(probe_id="factual-accuracy-v2")
        ... )
        >>> manifest.run_completed
        True
        >>> manifest.schema_version
        '1.0.1'

    Creating a manifest with full metadata:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_1 import RunManifest
        >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec, DatasetSpec
        >>> manifest = RunManifest(
        ...     run_id="run-prod-20240115-001",
        ...     created_at=datetime(2024, 1, 15, 8, 0, 0),
        ...     started_at=datetime(2024, 1, 15, 8, 0, 1),
        ...     completed_at=datetime(2024, 1, 15, 9, 30, 45),
        ...     run_completed=True,
        ...     library_version="0.5.2",
        ...     python_version="3.11.4",
        ...     platform="linux-x86_64",
        ...     command=["insidellms", "run", "--probe", "safety-v1"],
        ...     model=ModelSpec(
        ...         model_id="claude-3-sonnet",
        ...         provider="anthropic",
        ...         params={"temperature": 0.7, "max_tokens": 1024}
        ...     ),
        ...     probe=ProbeSpec(
        ...         probe_id="safety-alignment-v1",
        ...         probe_version="1.2.0",
        ...         params={"strict_mode": True}
        ...     ),
        ...     dataset=DatasetSpec(
        ...         dataset_id="safety-prompts-v3",
        ...         dataset_version="3.0.0",
        ...         dataset_hash="sha256:abc123...",
        ...         provenance="huggingface"
        ...     ),
        ...     record_count=1000,
        ...     success_count=985,
        ...     error_count=15
        ... )
        >>> manifest.record_count
        1000
        >>> manifest.success_count + manifest.error_count <= manifest.record_count
        True

    Detecting interrupted runs via run_completed flag:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_1 import RunManifest
        >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
        >>> # Manifest from a crashed run (run_completed defaults to False)
        >>> crashed_manifest = RunManifest(
        ...     run_id="run-crashed-001",
        ...     created_at=datetime(2024, 1, 15, 12, 0, 0),
        ...     started_at=datetime(2024, 1, 15, 12, 0, 0),
        ...     completed_at=datetime(2024, 1, 15, 12, 5, 0),  # partial
        ...     model=ModelSpec(model_id="gpt-4"),
        ...     probe=ProbeSpec(probe_id="test-probe"),
        ...     record_count=50  # only 50 of expected 100
        ... )
        >>> crashed_manifest.run_completed
        False

    Serializing to JSON for storage:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_1 import RunManifest
        >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
        >>> manifest = RunManifest(
        ...     run_id="run-serialize-test",
        ...     created_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     started_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     completed_at=datetime(2024, 1, 15, 10, 5, 0),
        ...     run_completed=True,
        ...     model=ModelSpec(model_id="gpt-4"),
        ...     probe=ProbeSpec(probe_id="test")
        ... )
        >>> data = manifest.model_dump(mode='json')  # Pydantic v2
        >>> data["run_completed"]
        True
        >>> data["schema_version"]
        '1.0.1'
        >>> "model" in data and "probe" in data
        True

    Adding custom metadata:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_1 import RunManifest
        >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
        >>> manifest = RunManifest(
        ...     run_id="run-with-custom",
        ...     created_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     started_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     completed_at=datetime(2024, 1, 15, 10, 5, 0),
        ...     run_completed=True,
        ...     model=ModelSpec(model_id="gpt-4"),
        ...     probe=ProbeSpec(probe_id="test"),
        ...     custom={
        ...         "experiment_group": "ablation-study-A",
        ...         "researcher": "alice",
        ...         "notes": "Testing new prompt format"
        ...     }
        ... )
        >>> manifest.custom["experiment_group"]
        'ablation-study-A'

    Raises
    ------
    pydantic.ValidationError
        If required fields are missing or field values don't match their
        expected types. Also raised if extra fields are provided (strict
        validation is enabled).

    See Also
    --------
    insideLLMs.schemas.v1_0_0.RunManifest : Previous version without run_completed
    insideLLMs.schemas.v1_0_0.ModelSpec : Model specification schema
    insideLLMs.schemas.v1_0_0.ProbeSpec : Probe specification schema
    insideLLMs.schemas.v1_0_0.DatasetSpec : Dataset specification schema
    insideLLMs.schemas.v1_0_0.ResultRecord : Individual result record schema

    Notes
    -----
    The ``run_completed`` field was added in v1.0.1 to address a limitation
    in v1.0.0 where it was impossible to distinguish between:

    1. A run that completed successfully
    2. A run that crashed or was interrupted before completion
    3. A run that is still in progress

    By defaulting to False, the field provides fail-safe behavior: if the
    run crashes before setting ``run_completed=True``, the manifest will
    correctly indicate that the run did not complete.

    The computed statistics (``record_count``, ``success_count``, ``error_count``)
    should be updated incrementally during the run and finalized when setting
    ``run_completed=True``.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)

    run_id: str
    created_at: datetime
    started_at: datetime
    completed_at: datetime
    run_completed: bool = False

    library_version: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    command: Optional[list[str]] = None

    model: v1_0_0.ModelSpec
    probe: v1_0_0.ProbeSpec
    dataset: v1_0_0.DatasetSpec = Field(default_factory=v1_0_0.DatasetSpec)

    record_count: int = 0
    success_count: int = 0
    error_count: int = 0

    records_file: str = "records.jsonl"
    schemas: dict[str, Any] = Field(default_factory=dict)
    custom: dict[str, Any] = Field(default_factory=dict)


def get_schema_model(schema_name: str):
    """Retrieve a schema model class by its canonical name.

    This function provides name-based lookup of Pydantic schema model classes
    for the v1.0.1 schema version. It returns the v1.0.1 ``RunManifest`` for
    that name and delegates all other schema lookups to v1.0.0 (since those
    schemas are unchanged in this version).

    This function is the primary mechanism for dynamic schema access when
    the schema name is determined at runtime (e.g., from configuration files
    or command-line arguments).

    Parameters
    ----------
    schema_name : str
        The canonical name of the schema to retrieve. Valid names include:

        - "RunManifest" - Returns v1.0.1 RunManifest (with run_completed)
        - "ProbeResult" - Returns v1.0.0 ProbeResult (delegated)
        - "ResultRecord" - Returns v1.0.0 ResultRecord (delegated)
        - "RunnerOutput" - Returns v1.0.0 RunnerOutput (delegated)
        - "HarnessRecord" - Returns v1.0.0 HarnessRecord (delegated)
        - "HarnessSummary" - Returns v1.0.0 HarnessSummary (delegated)
        - "BenchmarkSummary" - Returns v1.0.0 BenchmarkSummary (delegated)
        - "ComparisonReport" - Returns v1.0.0 ComparisonReport (delegated)
        - "DiffReport" - Returns v1.0.0 DiffReport (delegated)
        - "ExportMetadata" - Returns v1.0.0 ExportMetadata (delegated)

    Returns
    -------
    type
        The Pydantic model class corresponding to the schema name. This can
        be used to instantiate new instances, validate data, or generate
        JSON schemas.

    Raises
    ------
    KeyError
        If ``schema_name`` does not match any known schema name. The error
        message includes the unrecognized name for debugging.

    Examples
    --------
    Getting the v1.0.1 RunManifest:

        >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
        >>> RunManifest = get_schema_model("RunManifest")
        >>> RunManifest.__name__
        'RunManifest'
        >>> hasattr(RunManifest, 'model_fields')  # Pydantic v2
        True

    Creating an instance from the retrieved model:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
        >>> from insideLLMs.schemas.v1_0_0 import ModelSpec, ProbeSpec
        >>> RunManifest = get_schema_model("RunManifest")
        >>> manifest = RunManifest(
        ...     run_id="dynamic-run-001",
        ...     created_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     started_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     completed_at=datetime(2024, 1, 15, 10, 5, 0),
        ...     run_completed=True,
        ...     model=ModelSpec(model_id="gpt-4"),
        ...     probe=ProbeSpec(probe_id="test")
        ... )
        >>> manifest.schema_version
        '1.0.1'

    Accessing delegated v1.0.0 schemas:

        >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
        >>> ProbeResult = get_schema_model("ProbeResult")
        >>> result = ProbeResult(input="test", status="success")
        >>> result.schema_version
        '1.0.0'

    Using for dynamic validation:

        >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
        >>> ResultRecord = get_schema_model("ResultRecord")
        >>> from datetime import datetime
        >>> record = ResultRecord(
        ...     run_id="run-123",
        ...     started_at=datetime(2024, 1, 15, 10, 0, 0),
        ...     completed_at=datetime(2024, 1, 15, 10, 0, 1),
        ...     model={"model_id": "claude-3"},
        ...     probe={"probe_id": "safety-v1"},
        ...     example_id="ex-001",
        ...     status="success"
        ... )
        >>> record.status
        'success'

    Handling unknown schema names:

        >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
        >>> get_schema_model("NonExistentSchema")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        KeyError: 'Unknown schema name: NonExistentSchema'

    Runtime schema selection from configuration:

        >>> from insideLLMs.schemas.v1_0_1 import get_schema_model
        >>> config = {"output_schema": "ProbeResult"}  # From config file
        >>> Schema = get_schema_model(config["output_schema"])
        >>> instance = Schema(input="prompt", status="success", output="response")
        >>> instance.status
        'success'

    See Also
    --------
    insideLLMs.schemas.v1_0_0.get_schema_model : Base version schema lookup
    insideLLMs.schemas.registry.SchemaRegistry : Version-aware schema registry
    insideLLMs.schemas.registry.SchemaRegistry.get_model : Versioned schema lookup

    Notes
    -----
    This function is designed for use within the v1.0.1 schema context. For
    applications that need to work with multiple schema versions dynamically,
    use the ``SchemaRegistry`` class from ``insideLLMs.schemas.registry``,
    which provides version-aware lookups and caching.

    The delegation pattern ensures that v1.0.1 maintains full backward
    compatibility with v1.0.0 for all schemas except ``RunManifest``.
    """
    if schema_name == "RunManifest":
        return RunManifest
    # Delegate everything else to v1.0.0 (unchanged in 1.0.1).
    return v1_0_0.get_schema_model(schema_name)
