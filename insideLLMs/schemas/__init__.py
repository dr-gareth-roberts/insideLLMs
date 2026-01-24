"""Versioned output schemas and validation utilities for insideLLMs.

This package provides **versioned** (SemVer) Pydantic schemas for the serialized
outputs produced by insideLLMs probe runs, benchmarks, comparisons, and exports.
The schema system ensures stable, machine-checkable data formats across different
library versions and enables reliable data exchange between systems.

Overview
--------
The schemas package implements a comprehensive schema versioning system that:

    - **Validates outputs**: Ensures probe results, manifests, and reports conform
      to expected structures using strict Pydantic models
    - **Supports multiple versions**: Maintains backward compatibility by keeping
      older schema versions available for parsing historical data
    - **Enables migration**: Provides utilities for converting data between schema
      versions as the library evolves
    - **Exports JSON Schema**: Generates JSON Schema representations for external
      tool integration and API documentation

Pydantic is intentionally treated as an **optional dependency**: importing this
package is safe even without Pydantic installed, but using schema models or
validation features requires Pydantic (v1 or v2).

Available Schema Versions
-------------------------
The package currently provides the following schema versions:

**Version 1.0.0** (``insideLLMs.schemas.v1_0_0``)
    Initial stable release with core schemas for all output types.

**Version 1.0.1** (``insideLLMs.schemas.v1_0_1``) - Current Default
    Backward-compatible extension adding ``run_completed`` field to RunManifest
    for explicit run completion tracking.

**Custom Trace Schema** (``insideLLMs.schemas.custom_trace_v1``)
    Specialized schema for trace bundles stored in ``ResultRecord.custom["trace"]``.
    Uses version identifier ``"insideLLMs.custom.trace@1"``.

Available Schemas
-----------------
The following schema types are available in both v1.0.0 and v1.0.1:

**Core Output Schemas**
    - ``ProbeResult``: Per-item output produced by ``ProbeRunner.run()``
    - ``RunnerOutput``: Batch output wrapper for probe execution
    - ``ResultRecord``: JSONL record for wind-tunnel logging
    - ``RunManifest``: Run directory manifest (``manifest.json``)

**Harness Schemas**
    - ``HarnessRecord``: Per-line JSONL record from harness runs
    - ``HarnessSummary``: Summary payload (``summary.json``)

**Benchmark and Comparison Schemas**
    - ``BenchmarkSummary``: Output for benchmark runs
    - ``ComparisonReport``: Benchmark comparison output
    - ``DiffReport``: Diff output (``diff.json``)

**Export Schemas**
    - ``ExportMetadata``: Metadata for export bundles

**Trace Schemas**
    - ``CustomTrace``: Trace bundle schema (``TraceBundleV1``)

Module Components
-----------------
DEFAULT_SCHEMA_VERSION : str
    The current default schema version (``"1.0.1"``). Used when no explicit
    version is specified during validation.

OutputValidationError : exception
    Raised when output validation fails in strict mode. Contains schema name,
    version, and detailed error information.

SchemaRegistry : class
    Central registry for looking up versioned schema Pydantic models, generating
    JSON Schema, and performing migrations between versions.

OutputValidator : class
    High-level validation interface supporting strict and warn modes.

ValidationMode : type alias
    Literal type for validation modes: ``"strict"`` or ``"warn"``.

Examples
--------
**Importing the package**

The recommended way to use schemas is through the public API::

    >>> from insideLLMs.schemas import (
    ...     DEFAULT_SCHEMA_VERSION,
    ...     SchemaRegistry,
    ...     OutputValidator,
    ...     ValidationMode,
    ...     OutputValidationError,
    ... )
    >>> DEFAULT_SCHEMA_VERSION
    '1.0.1'

**Basic output validation**

Validate probe results using the OutputValidator::

    >>> from insideLLMs.schemas import OutputValidator
    >>> validator = OutputValidator()
    >>> result = validator.validate(
    ...     "ProbeResult",
    ...     {
    ...         "input": "What is 2+2?",
    ...         "output": "4",
    ...         "status": "success",
    ...         "latency_ms": 150.5
    ...     }
    ... )
    >>> result.status
    'success'
    >>> result.latency_ms
    150.5

**Using the SchemaRegistry directly**

Look up and instantiate schema models::

    >>> from insideLLMs.schemas import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
    >>> result = ProbeResult(
    ...     input="Test prompt",
    ...     output="Model response",
    ...     status="success"
    ... )
    >>> result.input
    'Test prompt'

**Checking available schema versions**

Discover what versions are registered for a schema::

    >>> from insideLLMs.schemas import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> versions = registry.available_versions("RunManifest")
    >>> "1.0.0" in versions
    True
    >>> "1.0.1" in versions
    True

**Validating with a specific schema version**

Explicitly specify the schema version for validation::

    >>> from insideLLMs.schemas import OutputValidator
    >>> from datetime import datetime
    >>> validator = OutputValidator()
    >>> manifest_data = {
    ...     "run_id": "run-abc-123",
    ...     "created_at": datetime.now().isoformat(),
    ...     "started_at": datetime.now().isoformat(),
    ...     "completed_at": datetime.now().isoformat(),
    ...     "run_completed": True,  # New in v1.0.1
    ...     "model": {"model_id": "gpt-4"},
    ...     "probe": {"probe_id": "truthfulness-v1"},
    ...     "record_count": 100
    ... }
    >>> manifest = validator.validate(
    ...     "RunManifest",
    ...     manifest_data,
    ...     schema_version="1.0.1"
    ... )
    >>> manifest.run_completed
    True

**Handling validation errors**

Catch and inspect validation failures::

    >>> from insideLLMs.schemas import OutputValidator, OutputValidationError
    >>> validator = OutputValidator()
    >>> try:
    ...     validator.validate("ProbeResult", {"invalid": "data"})
    ... except OutputValidationError as e:
    ...     print(f"Schema: {e.schema_name}")
    ...     print(f"Version: {e.schema_version}")
    ...     print(f"Errors: {len(e.errors)} validation error(s)")
    Schema: ProbeResult
    Version: 1.0.0
    Errors: 1 validation error(s)

**Using warn mode for lenient validation**

Get warnings instead of exceptions on validation failure::

    >>> import warnings
    >>> from insideLLMs.schemas import OutputValidator
    >>> validator = OutputValidator()
    >>> invalid_data = {"not": "valid"}
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     result = validator.validate(
    ...         "ProbeResult",
    ...         invalid_data,
    ...         mode="warn"
    ...     )
    ...     result is invalid_data  # Returns original on failure
    True

**Generating JSON Schema for external tools**

Export JSON Schema for API documentation or external validation::

    >>> from insideLLMs.schemas import SchemaRegistry
    >>> import json
    >>> registry = SchemaRegistry()
    >>> json_schema = registry.get_json_schema("ProbeResult", "1.0.0")
    >>> json_schema["title"]
    'ProbeResult@1.0.0'
    >>> "properties" in json_schema
    True
    >>> # Export for OpenAPI or other tools
    >>> schema_str = json.dumps(json_schema, indent=2)
    >>> "input" in schema_str
    True

**Batch validation with error collection**

Validate multiple records and collect errors::

    >>> from insideLLMs.schemas import OutputValidator, OutputValidationError
    >>> validator = OutputValidator()
    >>> records = [
    ...     {"input": "test1", "status": "success"},
    ...     {"input": "test2", "status": "error", "error": "timeout"},
    ...     {"bad": "data"},  # Invalid record
    ... ]
    >>> valid_records = []
    >>> failed_records = []
    >>> for i, record in enumerate(records):
    ...     try:
    ...         validated = validator.validate("ProbeResult", record)
    ...         valid_records.append(validated)
    ...     except OutputValidationError as e:
    ...         failed_records.append((i, record, e))
    >>> len(valid_records)
    2
    >>> len(failed_records)
    1

**Using schema registry constants**

Access schema names via constants for type safety::

    >>> from insideLLMs.schemas import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> # Use constants instead of string literals
    >>> model = registry.get_model(
    ...     SchemaRegistry.RUNNER_ITEM,  # "ProbeResult"
    ...     "1.0.0"
    ... )
    >>> model.__name__
    'SchemaProbeResult'

**Direct schema module access**

For advanced use cases, access schema modules directly::

    >>> from insideLLMs.schemas.v1_0_0 import ProbeResult, ResultRecord, RunManifest
    >>> from insideLLMs.schemas.v1_0_1 import RunManifest as RunManifestV101
    >>> # v1.0.1 RunManifest has run_completed field
    >>> hasattr(RunManifestV101, "model_fields") or hasattr(RunManifestV101, "__fields__")
    True

**Working with trace schemas**

Access the custom trace bundle schema::

    >>> from insideLLMs.schemas import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> trace_versions = registry.available_versions(SchemaRegistry.CUSTOM_TRACE)
    >>> "insideLLMs.custom.trace@1" in trace_versions
    True
    >>> TraceBundleV1 = registry.get_model(
    ...     SchemaRegistry.CUSTOM_TRACE,
    ...     "insideLLMs.custom.trace@1"
    ... )

**Version comparison for migration decisions**

Check if data needs migration to a newer schema::

    >>> from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
    >>> from insideLLMs.schemas.registry import semver_tuple
    >>> def needs_migration(old_version: str) -> bool:
    ...     '''Check if major version changed (breaking change).'''
    ...     old = semver_tuple(old_version)
    ...     new = semver_tuple(DEFAULT_SCHEMA_VERSION)
    ...     return old[0] < new[0]
    >>> needs_migration("0.9.0")  # Old major version
    True
    >>> needs_migration("1.0.0")  # Same major version
    False

**Embedding version in output for traceability**

Include schema version in serialized outputs::

    >>> from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
    >>> import json
    >>> output_record = {
    ...     "schema_version": DEFAULT_SCHEMA_VERSION,
    ...     "input": "What is AI?",
    ...     "output": "Artificial Intelligence is...",
    ...     "status": "success",
    ... }
    >>> serialized = json.dumps(output_record)
    >>> loaded = json.loads(serialized)
    >>> loaded["schema_version"]
    '1.0.1'

Notes
-----
- **Strict validation**: Schemas are intentionally strict (extra fields forbidden
  by default). This prevents silent schema drift and ensures data consistency.

- **Pydantic compatibility**: The package supports both Pydantic v1 and v2. The
  validation code automatically detects and uses the appropriate API.

- **Optional dependency**: Pydantic is only imported when schema features are
  actually used. The package can be imported without Pydantic for basic library
  functionality that doesn't require validation.

- **Thread safety**: Schema lookups are cached and the cache is populated lazily.
  SchemaHandle instances are frozen (immutable) for thread-safe access.

- **Migration support**: The ``SchemaRegistry.migrate()`` method provides basic
  migration support. For complex migrations, use the ``custom_migration`` callback
  parameter.

See Also
--------
insideLLMs.schemas.registry : Schema registry with version-aware lookups.
insideLLMs.schemas.validator : Output validation using versioned schemas.
insideLLMs.schemas.exceptions : Custom exceptions for validation errors.
insideLLMs.schemas.constants : Version constants and SemVer utilities.
insideLLMs.schemas.v1_0_0 : Schema definitions for v1.0.0.
insideLLMs.schemas.v1_0_1 : Schema definitions for v1.0.1.
insideLLMs.schemas.custom_trace_v1 : Trace bundle schema.

Version History
---------------
1.0.0 : 2024-01-15
    Initial stable release with core schemas:

    - ProbeResult: Individual probe output records
    - RunnerOutput: Batch result wrapper
    - ResultRecord: JSONL records for wind-tunnel logging
    - RunManifest: Run directory metadata
    - HarnessRecord/HarnessSummary: Harness execution schemas
    - BenchmarkSummary: Benchmark run output
    - ComparisonReport: Benchmark comparison output
    - DiffReport: Diff analysis output
    - ExportMetadata: Export bundle metadata

1.0.1 : 2024-02-01
    Backward-compatible enhancement:

    - Added ``run_completed`` field to RunManifest
    - Enables explicit tracking of run completion vs crashes/interruptions
    - Defaults to False for safety (incomplete until explicitly marked)

Custom Trace v1 : 2024-03-01
    Specialized schema for execution traces:

    - TraceBundleV1: Root trace bundle with mode (compact/full)
    - TraceCounts: Event counting with per-kind breakdown
    - TraceFingerprint: SHA256 fingerprinting support
    - TraceNormaliser: Normaliser configuration tracking
    - TraceContractsSummary: Contract violation summaries
    - TraceViolation: Individual violation records
    - TraceEventStored: Stored event records
    - TraceTruncation: Truncation policy tracking
    - TraceDerived: Derived analytics (tool calls, etc.)
"""

from __future__ import annotations

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.schemas.exceptions import OutputValidationError
from insideLLMs.schemas.registry import SchemaRegistry
from insideLLMs.schemas.validator import OutputValidator, ValidationMode

__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "OutputValidationError",
    "SchemaRegistry",
    "OutputValidator",
    "ValidationMode",
]
