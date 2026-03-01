"""Schema registry for mapping (schema_name, version) -> Pydantic model.

This module provides the central registry for versioned output schemas in the
insideLLMs library. The primary goal is to make serialized outputs stable and
machine-checkable across different library versions.

Overview
--------
The schema registry is the cornerstone of insideLLMs' data contract system. It
provides a single source of truth for all versioned Pydantic models used throughout
the library, ensuring that:

1. **Output stability**: Serialized data can be validated against specific schema
   versions, enabling reliable data interchange between different library versions.

2. **Machine-checkable contracts**: JSON Schema export allows external tools (CI/CD
   pipelines, API validators, documentation generators) to validate data without
   importing Python code.

3. **Backward compatibility**: Legacy version strings (e.g., "1.0") are normalized
   to full SemVer format ("1.0.0"), maintaining compatibility with older data.

4. **Migration support**: The registry provides an extensible framework for
   transforming data between schema versions.

The registry provides:
    - Lookup of versioned schema Pydantic models
    - JSON Schema export for external tooling integration
    - Schema version normalization (e.g., "1.0" -> "1.0.0")
    - Migration support between schema versions (extensible)
    - Lazy loading and caching for performance

Pydantic remains an optional dependency: schema models are only imported
when explicitly requested, allowing the library to function without Pydantic
for users who don't need schema validation.

Architecture
------------
The registry follows a lazy-loading pattern::

    User Request          Registry            Version Modules
    ───────────────      ────────────        ────────────────
    get_model("X","1.0") ─> normalize_semver ─> "1.0.0"
                          ─> check cache
                          ─> import v1_0_0 module (lazy)
                          ─> cache SchemaHandle
                          <─ return model class

This design ensures:
    - Minimal startup overhead (no schema imports until needed)
    - Fast repeated lookups (cached after first access)
    - Thread-safe immutable handles (frozen dataclasses)

Available Schemas
-----------------
The registry exposes the following schema types:

- **ProbeResult** (``RUNNER_ITEM``): Per-item result from a single probe execution.
  Contains input, output, status, timing, and optional metadata.

- **RunnerOutput** (``RUNNER_OUTPUT``): Wrapper for a batch of probe results,
  including aggregate statistics and run metadata.

- **ResultRecord** (``RESULT_RECORD``): JSONL-compatible record format for
  streaming results to disk during long-running probe sessions.

- **RunManifest** (``RUN_MANIFEST``): Metadata file (manifest.json) describing
  a complete probe run, including configuration and completion status.

- **HarnessRecord** (``HARNESS_RECORD``): Per-line record format for evaluation
  harness output, used in benchmark pipelines.

- **HarnessSummary** (``HARNESS_SUMMARY``): Aggregate summary (summary.json) for
  an evaluation harness run.

- **BenchmarkSummary** (``BENCHMARK_SUMMARY``): Output format for benchmark runs,
  including metrics and comparison data.

- **ComparisonReport** (``COMPARISON_REPORT``): Structured comparison between two
  or more probe runs or model outputs.

- **DiffReport** (``DIFF_REPORT``): Detailed diff output (diff.json) showing
  differences between runs.

- **ExportMetadata** (``EXPORT_METADATA``): Metadata for exported data bundles,
  used in data portability features.

- **CustomTrace** (``CUSTOM_TRACE``): Trace bundle format for storing detailed
  execution traces with custom metadata.

Examples
--------
Basic schema lookup:

    >>> from insideLLMs.schemas.registry import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
    >>> result = ProbeResult(input="test", status="success")
    >>> result.status
    'success'

Getting available versions for a schema:

    >>> from insideLLMs.schemas.registry import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> versions = registry.available_versions("RunManifest")
    >>> "1.0.0" in versions
    True
    >>> "1.0.1" in versions
    True

Exporting JSON Schema for external tools:

    >>> from insideLLMs.schemas.registry import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> json_schema = registry.get_json_schema("ProbeResult", "1.0.0")
    >>> json_schema["title"]
    'ProbeResult@1.0.0'
    >>> "properties" in json_schema
    True

Version normalization:

    >>> from insideLLMs.schemas.registry import normalize_semver, semver_tuple
    >>> normalize_semver("1.0")
    '1.0.0'
    >>> normalize_semver("1.0.0")
    '1.0.0'
    >>> semver_tuple("1.2.3")
    (1, 2, 3)

Validating data from external sources:

    >>> import json
    >>> from insideLLMs.schemas.registry import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> ResultRecord = registry.get_model("ResultRecord", "1.0.0")
    >>> # Simulate loading from JSONL file
    >>> raw_data = {
    ...     "run_id": "run-abc123",
    ...     "started_at": "2024-01-15T10:30:00Z",
    ...     "completed_at": "2024-01-15T10:30:05Z",
    ...     "model": {"model_id": "gpt-4"},
    ...     "probe": {"probe_id": "safety-probe"},
    ...     "example_id": "ex-001",
    ...     "status": "success"
    ... }
    >>> record = ResultRecord.model_validate(raw_data)
    >>> record.run_id
    'run-abc123'

Using schema constants for type-safe lookups:

    >>> from insideLLMs.schemas.registry import SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> # Use constants instead of string literals
    >>> model = registry.get_model(
    ...     SchemaRegistry.RUNNER_ITEM,  # "ProbeResult"
    ...     "1.0.1"
    ... )
    >>> hasattr(model, "model_fields")
    True

Comparing schema versions:

    >>> from insideLLMs.schemas.registry import semver_tuple
    >>> # Determine which version is newer
    >>> semver_tuple("1.0.1") > semver_tuple("1.0.0")
    True
    >>> semver_tuple("2.0.0") > semver_tuple("1.9.9")
    True

Attributes
----------
SchemaHandle : dataclass
    A frozen dataclass representing a resolved schema with its name, version,
    and Pydantic model class. Used internally by the registry for caching.

SchemaRegistry : class
    The main registry class for looking up and managing versioned schemas.
    Provides methods for model lookup, JSON Schema generation, and migration.

normalize_semver : function
    Utility function to normalize version strings to full SemVer format.

semver_tuple : function
    Utility function to convert version strings to comparable tuples.

Notes
-----
- Schema models are cached after first access to improve performance when
  validating many records against the same schema.

- The registry is designed to be instantiated per-use; it does not maintain
  global state. Create a new registry for each independent validation context.

- For best performance in loops, retrieve the model once and reuse it::

      # Good: retrieve once, use many times
      registry = SchemaRegistry()
      ProbeResult = registry.get_model("ProbeResult", "1.0.0")
      for data in large_dataset:
          result = ProbeResult.model_validate(data)

      # Avoid: retrieving in loop (though cached, adds overhead)
      for data in large_dataset:
          model = registry.get_model("ProbeResult", "1.0.0")
          result = model.model_validate(data)

- When migrating data between versions, prefer explicit migration functions
  over implicit field mapping to ensure data integrity.

See Also
--------
insideLLMs.schemas.v1_0_0 : Version 1.0.0 schema definitions
insideLLMs.schemas.v1_0_1 : Version 1.0.1 schema definitions
insideLLMs.schemas.custom_trace_v1 : Custom trace bundle schema
insideLLMs.structured : JSON Schema generation utilities

References
----------
.. [1] Semantic Versioning 2.0.0: https://semver.org/
.. [2] JSON Schema: https://json-schema.org/
.. [3] Pydantic V2 Documentation: https://docs.pydantic.dev/latest/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type


def normalize_semver(version: str) -> str:
    """Normalize a version string into SemVer form.

    Accepts legacy forms like "1.0" and normalizes them to full SemVer format
    "1.0.0". This ensures consistent version comparison and lookup throughout
    the schema system.

    This function is essential for maintaining backward compatibility with older
    data files that may have been serialized with short-form version strings.
    It guarantees that version lookups work consistently regardless of the input
    format.

    Parameters
    ----------
    version : str
        A version string in either short form ("1.0") or full SemVer form
        ("1.0.0"). Non-numeric versions are returned unchanged.

        Common input formats:
            - Short form: "1.0", "2.1"
            - Full SemVer: "1.0.0", "2.3.4"
            - Pre-release: "1.0.0-beta", "2.0.0-rc.1"
            - Special: "latest", "dev"

    Returns
    -------
    str
        The normalized version string in "MAJOR.MINOR.PATCH" format. If the
        input doesn't match expected patterns (short or full SemVer), it's
        returned unchanged.

    Examples
    --------
    Normalizing short versions:

        >>> normalize_semver("1.0")
        '1.0.0'
        >>> normalize_semver("2.1")
        '2.1.0'
        >>> normalize_semver("0.9")
        '0.9.0'

    Full versions pass through unchanged:

        >>> normalize_semver("1.0.0")
        '1.0.0'
        >>> normalize_semver("2.3.4")
        '2.3.4'
        >>> normalize_semver("10.20.30")
        '10.20.30'

    Pre-release and build metadata versions are preserved:

        >>> normalize_semver("1.0.0-beta")
        '1.0.0-beta'
        >>> normalize_semver("2.0.0-rc.1")
        '2.0.0-rc.1'
        >>> normalize_semver("1.0.0+build.123")
        '1.0.0+build.123'

    Special version strings are returned as-is:

        >>> normalize_semver("latest")
        'latest'
        >>> normalize_semver("dev")
        'dev'
        >>> normalize_semver("")
        ''

    Edge cases with unusual input:

        >>> normalize_semver("1")
        '1'
        >>> normalize_semver("1.0.0.0")
        '1.0.0.0'

    Practical usage with registry:

        >>> from insideLLMs.schemas.registry import SchemaRegistry, normalize_semver
        >>> registry = SchemaRegistry()
        >>> # Both lookups return the same model due to normalization
        >>> model_short = registry.get_model("ProbeResult", "1.0")
        >>> model_full = registry.get_model("ProbeResult", "1.0.0")
        >>> model_short is model_full
        True

    Notes
    -----
    The function uses strict regex matching:

    - Full SemVer: ``^[0-9]+\\.[0-9]+\\.[0-9]+$``
    - Short form: ``^[0-9]+\\.[0-9]+$``

    Versions with pre-release identifiers (e.g., "-beta") or build metadata
    (e.g., "+build") are not normalized because they don't match the strict
    numeric patterns.

    See Also
    --------
    semver_tuple : Convert version string to comparable tuple
    SchemaRegistry.get_model : Look up schema by name and version
    """
    if re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
        return version
    if re.match(r"^[0-9]+\.[0-9]+$", version):
        return f"{version}.0"
    return version


def semver_tuple(version: str) -> tuple[int, int, int]:
    """Convert a semver string to a tuple for comparison.

    This function parses a semantic version string and returns a tuple of
    integers that can be used for version comparison operations. It handles
    both short-form ("1.0") and full-form ("1.0.0") version strings by
    first normalizing them via ``normalize_semver``.

    The resulting tuple enables direct comparison using Python's built-in
    comparison operators, making it easy to determine version ordering,
    find the latest version, or filter versions within a range.

    Parameters
    ----------
    version : str
        Version string like "1.2.3" or "1.2". Short-form versions are
        normalized before parsing. Pre-release suffixes (e.g., "-beta")
        are included in the split but will cause parsing to fail, returning
        the fallback tuple.

    Returns
    -------
    tuple[int, int, int]
        Tuple of (major, minor, patch) integers. Returns (0, 0, 0) if the
        version string cannot be parsed as three integers.

        The tuple ordering follows SemVer semantics:
            - major: Breaking changes
            - minor: New features (backward compatible)
            - patch: Bug fixes (backward compatible)

    Examples
    --------
    Standard version parsing:

        >>> semver_tuple("1.2.3")
        (1, 2, 3)
        >>> semver_tuple("2.0.0")
        (2, 0, 0)
        >>> semver_tuple("0.1.0")
        (0, 1, 0)

    Short-form versions are normalized first:

        >>> semver_tuple("1.0")
        (1, 0, 0)
        >>> semver_tuple("3.5")
        (3, 5, 0)
        >>> semver_tuple("10.20")
        (10, 20, 0)

    Version comparison using comparison operators:

        >>> semver_tuple("2.0.0") > semver_tuple("1.9.9")
        True
        >>> semver_tuple("1.0.1") > semver_tuple("1.0.0")
        True
        >>> semver_tuple("1.1.0") > semver_tuple("1.0.99")
        True
        >>> semver_tuple("1.0.0") == semver_tuple("1.0")
        True

    Finding the latest version from a list:

        >>> versions = ["1.0.0", "1.0.1", "0.9.0", "1.1.0"]
        >>> max(versions, key=semver_tuple)
        '1.1.0'
        >>> min(versions, key=semver_tuple)
        '0.9.0'

    Sorting versions:

        >>> versions = ["2.0.0", "1.9.9", "1.10.0", "1.2.0"]
        >>> sorted(versions, key=semver_tuple)
        ['1.2.0', '1.9.9', '1.10.0', '2.0.0']

    Filtering versions within a range:

        >>> versions = ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]
        >>> [v for v in versions if semver_tuple("1.0.0") <= semver_tuple(v) < semver_tuple("2.0.0")]
        ['1.0.0', '1.0.1', '1.1.0']

    Invalid versions return zeros (safe fallback):

        >>> semver_tuple("invalid")
        (0, 0, 0)
        >>> semver_tuple("1.x.y")
        (0, 0, 0)
        >>> semver_tuple("")
        (0, 0, 0)
        >>> semver_tuple("1.0.0-beta")  # Pre-release suffix causes parse failure
        (0, 0, 0)

    Practical usage with registry:

        >>> from insideLLMs.schemas.registry import SchemaRegistry, semver_tuple
        >>> registry = SchemaRegistry()
        >>> # Get latest version for a schema
        >>> versions = registry.available_versions("ProbeResult")
        >>> latest = max(versions, key=semver_tuple)
        >>> latest
        '1.0.1'

    Raises
    ------
    This function does not raise exceptions. Invalid input returns (0, 0, 0).

    Notes
    -----
    - The function calls ``normalize_semver`` internally, so "1.0" and "1.0.0"
      produce the same tuple.

    - Pre-release versions (e.g., "1.0.0-beta") are NOT properly handled and
      will return (0, 0, 0). If you need pre-release comparison, use a
      dedicated SemVer library.

    - The fallback (0, 0, 0) ensures that invalid versions sort before all
      valid versions, which may or may not be desired behavior for your
      use case.

    See Also
    --------
    normalize_semver : Normalize version strings to full SemVer format
    SchemaRegistry.available_versions : Get all versions for a schema
    """
    v = normalize_semver(version)
    parts = v.split(".")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def _require_pydantic() -> None:
    """Ensure Pydantic is installed before using schema features.

    This internal function checks for Pydantic availability and raises a
    helpful error message if it's not installed. It's called lazily when
    schema features are first accessed, implementing the "optional dependency"
    pattern for Pydantic.

    The lazy check allows the insideLLMs library to be imported and used
    without Pydantic for features that don't require schema validation. Only
    when schema-specific functionality is accessed will the dependency be
    enforced.

    Parameters
    ----------
    None
        This function takes no parameters.

    Returns
    -------
    None
        Returns None if Pydantic is available.

    Raises
    ------
    ImportError
        If Pydantic is not installed. The error message includes installation
        instructions: "Install with: pip install pydantic"

    Examples
    --------
    Normal usage (Pydantic installed):

        >>> _require_pydantic()  # No error if pydantic is available

    When Pydantic is missing (hypothetical):

        >>> _require_pydantic()  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        ImportError: Pydantic is required for output schema validation.
        Install with: pip install pydantic

    Internal usage pattern:

        >>> from insideLLMs.schemas.registry import SchemaRegistry
        >>> registry = SchemaRegistry()
        >>> # _require_pydantic is called internally by get_model
        >>> model = registry.get_model("ProbeResult", "1.0.0")  # Checks pydantic

    Notes
    -----
    This is a private function (prefixed with underscore) and should not be
    called directly by external code. It's automatically invoked by:

    - ``SchemaRegistry.get_model()``
    - ``SchemaRegistry.get_json_schema()``

    The function uses a simple try/except pattern rather than checking
    ``importlib.util.find_spec`` because the latter can have edge cases
    with certain package managers and virtual environments.

    See Also
    --------
    SchemaRegistry.get_model : Primary method that uses this check
    SchemaRegistry.get_json_schema : Another method that uses this check
    """
    try:
        import pydantic  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Pydantic is required for output schema validation. Install with: pip install pydantic"
        ) from e


@dataclass(frozen=True)
class SchemaHandle:
    """A resolved schema reference containing name, version, and model class.

    This frozen dataclass represents a fully-resolved schema lookup result.
    It's used internally by the SchemaRegistry to cache resolved schemas
    and provide a structured reference to schema metadata.

    Being frozen (immutable) ensures that cached schema handles cannot be
    accidentally modified, providing thread-safety for the cache. Multiple
    threads can safely share the same SchemaHandle instance without risk
    of data corruption.

    The SchemaHandle acts as a value object that bundles together all the
    information needed to work with a specific schema version, making it
    convenient for passing around in code that needs to know both the
    identity and the implementation of a schema.

    Parameters
    ----------
    schema_name : str
        The canonical name of the schema. Should match one of the schema
        constants defined in SchemaRegistry (e.g., RUNNER_ITEM, RESULT_RECORD).

    schema_version : str
        The normalized semantic version string. Should always be in full
        "MAJOR.MINOR.PATCH" format (e.g., "1.0.0", not "1.0").

    model : Type[Any]
        The Pydantic model class that implements the schema. This is the
        actual class, not an instance.

    Attributes
    ----------
    schema_name : str
        The canonical name of the schema (e.g., "ProbeResult", "ResultRecord",
        "RunManifest"). This matches the names exposed by SchemaRegistry
        constants.

    schema_version : str
        The normalized semantic version string (e.g., "1.0.0", "1.0.1").
        Always in full SemVer format after registry resolution.

    model : Type[Any]
        The Pydantic model class that implements the schema. This can be
        used directly to instantiate or validate data:
        - ``model(**data)`` - create instance from keyword arguments
        - ``model.model_validate(data)`` - validate dict (Pydantic v2)
        - ``model.model_json_schema()`` - get JSON schema (Pydantic v2)

    Examples
    --------
    Creating a schema handle manually:

        >>> from insideLLMs.schemas.registry import SchemaHandle
        >>> from insideLLMs.schemas.v1_0_0 import ProbeResult
        >>> handle = SchemaHandle(
        ...     schema_name="ProbeResult",
        ...     schema_version="1.0.0",
        ...     model=ProbeResult
        ... )
        >>> handle.schema_name
        'ProbeResult'
        >>> handle.schema_version
        '1.0.0'

    Using positional arguments:

        >>> from insideLLMs.schemas.registry import SchemaHandle
        >>> from insideLLMs.schemas.v1_0_0 import ProbeResult
        >>> handle = SchemaHandle("ProbeResult", "1.0.0", ProbeResult)
        >>> handle.schema_name
        'ProbeResult'

    Using the model from a handle to create instances:

        >>> from insideLLMs.schemas.registry import SchemaRegistry
        >>> registry = SchemaRegistry()
        >>> model = registry.get_model("ProbeResult", "1.0.0")
        >>> result = model(input="test", status="success")
        >>> result.status
        'success'

    Validating data with the model:

        >>> from insideLLMs.schemas.registry import SchemaRegistry
        >>> registry = SchemaRegistry()
        >>> model = registry.get_model("ProbeResult", "1.0.0")
        >>> data = {"input": "Hello", "status": "success", "output": "World"}
        >>> result = model.model_validate(data)
        >>> result.input
        'Hello'

    Schema handles are immutable (frozen):

        >>> from insideLLMs.schemas.registry import SchemaHandle
        >>> from insideLLMs.schemas.v1_0_0 import ProbeResult
        >>> handle = SchemaHandle("ProbeResult", "1.0.0", ProbeResult)
        >>> handle.schema_version = "1.0.1"  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        FrozenInstanceError: cannot assign to field 'schema_version'

    Handles can be used as dictionary keys (hashable):

        >>> from insideLLMs.schemas.registry import SchemaHandle
        >>> from insideLLMs.schemas.v1_0_0 import ProbeResult, ResultRecord
        >>> handle1 = SchemaHandle("ProbeResult", "1.0.0", ProbeResult)
        >>> handle2 = SchemaHandle("ResultRecord", "1.0.0", ResultRecord)
        >>> cache = {handle1: "cached_data_1", handle2: "cached_data_2"}
        >>> cache[handle1]
        'cached_data_1'

    Equality comparison:

        >>> from insideLLMs.schemas.registry import SchemaHandle
        >>> from insideLLMs.schemas.v1_0_0 import ProbeResult
        >>> h1 = SchemaHandle("ProbeResult", "1.0.0", ProbeResult)
        >>> h2 = SchemaHandle("ProbeResult", "1.0.0", ProbeResult)
        >>> h1 == h2
        True

    Accessing internal cache (advanced usage):

        >>> from insideLLMs.schemas.registry import SchemaRegistry
        >>> registry = SchemaRegistry()
        >>> _ = registry.get_model("ProbeResult", "1.0.0")
        >>> ("ProbeResult", "1.0.0") in registry._cache
        True
        >>> handle = registry._cache[("ProbeResult", "1.0.0")]
        >>> handle.schema_name
        'ProbeResult'

    Notes
    -----
    - SchemaHandle is primarily for internal use by SchemaRegistry. External
      code typically interacts with model classes directly via ``get_model()``.

    - The frozen nature of the dataclass means that after creation, no
      attributes can be modified. This is intentional for cache safety.

    - The ``model`` attribute stores the class itself, not an instance.
      Use ``model(**kwargs)`` or ``model.model_validate(data)`` to create
      instances.

    - While SchemaHandle can be created manually, it's typically obtained
      from the registry's internal cache for debugging or introspection.

    See Also
    --------
    SchemaRegistry : Main class for schema lookup and management
    SchemaRegistry.get_model : Returns the model from a resolved handle
    dataclasses.dataclass : Python dataclass decorator documentation
    """

    schema_name: str
    schema_version: str
    model: Type[Any]


class SchemaRegistry:
    """Registry for versioned output schemas.

    The SchemaRegistry is the central lookup mechanism for insideLLMs output
    schemas. It provides versioned access to Pydantic model classes, JSON Schema
    generation, and basic migration support between schema versions.

    The registry uses lazy loading and caching to minimize import overhead.
    Schema models are only loaded when first requested and cached for subsequent
    lookups. This design allows the library to start quickly even when many
    schema versions are available.

    Each SchemaRegistry instance maintains its own cache. For most use cases,
    creating a single registry instance and reusing it is recommended. However,
    creating multiple instances is safe and each will independently cache
    its resolved schemas.

    Parameters
    ----------
    None
        The constructor takes no parameters. All configuration is internal.

    Attributes
    ----------
    _cache : dict[tuple[str, str], SchemaHandle]
        Internal cache mapping (schema_name, schema_version) tuples to resolved
        SchemaHandle instances. This is a private attribute but can be accessed
        for debugging or introspection.

    Class Attributes
    ----------------
    RUNNER_ITEM : str
        Schema name for per-item probe results ("ProbeResult"). This is the
        primary output type for individual probe executions.

    RUNNER_OUTPUT : str
        Schema name for batch result wrapper ("RunnerOutput"). Contains a
        collection of ProbeResult items along with aggregate metadata.

    RESULT_RECORD : str
        Schema name for JSONL records ("ResultRecord"). Used for streaming
        results to disk in JSONL format during long-running probe sessions.

    RUN_MANIFEST : str
        Schema name for run manifests ("RunManifest"). The manifest.json file
        that describes a complete probe run's configuration and status.

    HARNESS_RECORD : str
        Schema name for harness records ("HarnessRecord"). Per-line record
        format for evaluation harness output.

    HARNESS_SUMMARY : str
        Schema name for harness summaries ("HarnessSummary"). Aggregate summary
        file (summary.json) for an evaluation harness run.

    BENCHMARK_SUMMARY : str
        Schema name for benchmark output ("BenchmarkSummary"). Contains metrics
        and comparison data for benchmark runs.

    COMPARISON_REPORT : str
        Schema name for comparisons ("ComparisonReport"). Structured comparison
        between two or more probe runs or model outputs.

    DIFF_REPORT : str
        Schema name for diff output ("DiffReport"). Detailed diff.json showing
        differences between runs.

    EXPORT_METADATA : str
        Schema name for export bundles ("ExportMetadata"). Metadata for
        exported data bundles used in data portability features.

    CUSTOM_TRACE : str
        Schema name for trace bundles ("CustomTrace"). Special format for
        storing detailed execution traces with custom metadata.

    Examples
    --------
    Basic registry usage:

        >>> from insideLLMs.schemas.registry import SchemaRegistry
        >>> registry = SchemaRegistry()
        >>> ProbeResult = registry.get_model(
        ...     SchemaRegistry.RUNNER_ITEM, "1.0.0"
        ... )
        >>> result = ProbeResult(input="Hello", status="success")
        >>> result.input
        'Hello'

    Checking available versions for a schema:

        >>> registry = SchemaRegistry()
        >>> versions = registry.available_versions("ProbeResult")
        >>> len(versions) >= 2
        True
        >>> "1.0.0" in versions and "1.0.1" in versions
        True

    Generating JSON Schema for API documentation:

        >>> registry = SchemaRegistry()
        >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
        >>> "properties" in schema
        True
        >>> "input" in schema["properties"]
        True

    Using schema constants for type safety:

        >>> registry = SchemaRegistry()
        >>> model = registry.get_model(
        ...     SchemaRegistry.RESULT_RECORD, "1.0.1"
        ... )
        >>> hasattr(model, "model_fields") or hasattr(model, "__fields__")
        True

    Validating data from external sources (e.g., JSONL files):

        >>> import json
        >>> registry = SchemaRegistry()
        >>> ResultRecord = registry.get_model(SchemaRegistry.RESULT_RECORD, "1.0.0")
        >>> # Simulated JSONL line
        >>> raw_json = '''{"run_id": "run-123", "started_at": "2024-01-15T10:00:00Z",
        ...     "completed_at": "2024-01-15T10:00:05Z", "model": {"model_id": "gpt-4"},
        ...     "probe": {"probe_id": "safety"}, "example_id": "ex-1", "status": "success"}'''
        >>> data = json.loads(raw_json)
        >>> record = ResultRecord.model_validate(data)
        >>> record.run_id
        'run-123'

    Working with multiple schema versions:

        >>> registry = SchemaRegistry()
        >>> # Load both versions for comparison or migration
        >>> v1_0_0 = registry.get_model("ProbeResult", "1.0.0")
        >>> v1_0_1 = registry.get_model("ProbeResult", "1.0.1")
        >>> # Each version may have different fields or validation rules
        >>> v1_0_0 is not v1_0_1
        True

    Generating schemas for OpenAPI/Swagger documentation:

        >>> import json
        >>> registry = SchemaRegistry()
        >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
        >>> # Schema can be embedded in OpenAPI spec
        >>> openapi_components = {
        ...     "components": {
        ...         "schemas": {
        ...             "ProbeResult": schema
        ...         }
        ...     }
        ... }
        >>> "ProbeResult" in openapi_components["components"]["schemas"]
        True

    Batch validation with caching benefits:

        >>> registry = SchemaRegistry()
        >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
        >>> # Model is cached - subsequent calls return same class
        >>> ProbeResult2 = registry.get_model("ProbeResult", "1.0.0")
        >>> ProbeResult is ProbeResult2
        True
        >>> # Validate many items efficiently
        >>> items = [
        ...     {"input": "test1", "status": "success"},
        ...     {"input": "test2", "status": "error", "error": "timeout"},
        ... ]
        >>> results = [ProbeResult.model_validate(item) for item in items]
        >>> len(results)
        2

    Error handling for unknown schemas:

        >>> registry = SchemaRegistry()
        >>> try:
        ...     registry.get_model("NonExistentSchema", "1.0.0")
        ... except KeyError as e:
        ...     "Unknown schema" in str(e)
        True

    Notes
    -----
    - **Thread Safety**: The registry is safe for concurrent read access.
      Multiple threads can call ``get_model()`` simultaneously. However,
      the cache is not protected by locks, so there may be redundant
      imports if multiple threads request the same schema simultaneously
      before it's cached. This is benign (no data corruption).

    - **Memory**: Cached schema models remain in memory for the lifetime
      of the registry instance. For long-running applications with many
      schemas, this is typically desirable for performance.

    - **Pydantic Dependency**: Pydantic is checked lazily. The registry
      can be instantiated without Pydantic, but ``get_model()`` and
      ``get_json_schema()`` will raise ImportError if Pydantic is missing.

    - **Version Normalization**: All version strings are normalized via
      ``normalize_semver()`` before lookup. This means "1.0" and "1.0.0"
      are treated as equivalent.

    - **Extensibility**: To add new schema versions, create a new version
      module (e.g., ``v1_0_2.py``) and update the version dispatch in
      ``get_model()``.

    See Also
    --------
    SchemaHandle : The cached schema reference dataclass
    normalize_semver : Version string normalization
    semver_tuple : Version comparison utilities
    insideLLMs.schemas.v1_0_0 : Version 1.0.0 schema definitions
    insideLLMs.schemas.v1_0_1 : Version 1.0.1 schema definitions
    """

    # Canonical schema names exposed by insideLLMs.
    RUNNER_ITEM = "ProbeResult"  # per-item result from ProbeRunner.run
    RUNNER_OUTPUT = "RunnerOutput"  # wrapper for a batch of results
    RESULT_RECORD = "ResultRecord"  # JSONL record for ProbeRunner.run()
    RUN_MANIFEST = "RunManifest"  # manifest.json for run directories
    HARNESS_RECORD = "HarnessRecord"  # per-line JSONL record
    HARNESS_SUMMARY = "HarnessSummary"  # summary.json payload
    HARNESS_EXPLAIN = "HarnessExplain"  # explain.json payload
    BENCHMARK_SUMMARY = "BenchmarkSummary"  # benchmark run output
    COMPARISON_REPORT = "ComparisonReport"  # comparison output
    DIFF_REPORT = "DiffReport"  # diff.json output
    EXPORT_METADATA = "ExportMetadata"  # export bundle metadata
    CUSTOM_TRACE = "CustomTrace"  # ResultRecord.custom["trace"] bundle

    def __init__(self) -> None:
        """Initialize a new SchemaRegistry instance.

        Creates an empty cache for storing resolved schema handles. The cache
        is populated lazily as schemas are requested via ``get_model()`` or
        ``get_json_schema()``.

        The constructor does not import any schema modules or check for
        Pydantic availability. This allows the registry to be instantiated
        quickly and without side effects, even in environments where Pydantic
        is not installed.

        Parameters
        ----------
        None
            This method takes no parameters.

        Returns
        -------
        None
            The constructor returns None (implicitly).

        Examples
        --------
        Basic instantiation:

            >>> registry = SchemaRegistry()
            >>> isinstance(registry._cache, dict)
            True
            >>> len(registry._cache)
            0

        Multiple independent registries:

            >>> registry1 = SchemaRegistry()
            >>> registry2 = SchemaRegistry()
            >>> registry1 is not registry2
            True
            >>> # Each has its own cache
            >>> registry1._cache is not registry2._cache
            True

        Cache is populated on first use:

            >>> registry = SchemaRegistry()
            >>> len(registry._cache)  # Empty initially
            0
            >>> _ = registry.get_model("ProbeResult", "1.0.0")
            >>> len(registry._cache)  # Now has one entry
            1
            >>> ("ProbeResult", "1.0.0") in registry._cache
            True

        Registry can be created without Pydantic installed:

            >>> # This works even if pydantic is not installed
            >>> registry = SchemaRegistry()
            >>> # Only get_model/get_json_schema require pydantic

        Notes
        -----
        The cache uses ``(schema_name, schema_version)`` tuples as keys.
        Both components are strings, and the version is always in normalized
        SemVer format (e.g., "1.0.0", not "1.0").

        See Also
        --------
        get_model : Method that populates the cache
        SchemaHandle : The type stored in the cache
        """
        self._cache: dict[tuple[str, str], SchemaHandle] = {}

    def available_versions(self, schema_name: str) -> list[str]:
        """Get all available versions for a given schema name.

        Returns a list of version strings that are registered for the specified
        schema. This is useful for discovering what versions are available,
        for validation tools that need to support multiple versions, and for
        implementing version selection UI or CLI commands.

        This method does not require Pydantic and does not import any schema
        modules. It returns hardcoded version lists based on what versions
        are implemented in the codebase.

        Parameters
        ----------
        schema_name : str
            The canonical name of the schema (e.g., "ProbeResult", "ResultRecord").
            Use class constants like ``SchemaRegistry.RUNNER_ITEM`` for type safety
            and to avoid typos.

            Valid schema names:
                - SchemaRegistry.RUNNER_ITEM ("ProbeResult")
                - SchemaRegistry.RUNNER_OUTPUT ("RunnerOutput")
                - SchemaRegistry.RESULT_RECORD ("ResultRecord")
                - SchemaRegistry.RUN_MANIFEST ("RunManifest")
                - SchemaRegistry.HARNESS_RECORD ("HarnessRecord")
                - SchemaRegistry.HARNESS_SUMMARY ("HarnessSummary")
                - SchemaRegistry.BENCHMARK_SUMMARY ("BenchmarkSummary")
                - SchemaRegistry.COMPARISON_REPORT ("ComparisonReport")
                - SchemaRegistry.DIFF_REPORT ("DiffReport")
                - SchemaRegistry.EXPORT_METADATA ("ExportMetadata")
                - SchemaRegistry.CUSTOM_TRACE ("CustomTrace")

        Returns
        -------
        list[str]
            A list of version strings in ascending order. Returns an empty list
            if the schema name is not recognized. For most schemas, this returns
            ["1.0.0", "1.0.1"]. The CustomTrace schema uses a special versioning
            scheme.

        Examples
        --------
        Getting versions for a known schema:

            >>> registry = SchemaRegistry()
            >>> versions = registry.available_versions("ProbeResult")
            >>> "1.0.0" in versions
            True
            >>> "1.0.1" in versions
            True
            >>> versions == ["1.0.0", "1.0.1"]
            True

        Using schema constants (recommended):

            >>> registry = SchemaRegistry()
            >>> versions = registry.available_versions(
            ...     SchemaRegistry.RUN_MANIFEST
            ... )
            >>> len(versions) >= 2
            True
            >>> versions[0]
            '1.0.0'

        Checking all standard schemas:

            >>> registry = SchemaRegistry()
            >>> for name in [
            ...     SchemaRegistry.RUNNER_ITEM,
            ...     SchemaRegistry.RESULT_RECORD,
            ...     SchemaRegistry.RUN_MANIFEST,
            ... ]:
            ...     versions = registry.available_versions(name)
            ...     assert len(versions) >= 2, f"{name} should have versions"

        Unknown schemas return empty list (no exception):

            >>> registry = SchemaRegistry()
            >>> registry.available_versions("UnknownSchema")
            []
            >>> registry.available_versions("")
            []
            >>> registry.available_versions("probe_result")  # Case sensitive!
            []

        Custom trace schema has special versioning:

            >>> registry = SchemaRegistry()
            >>> versions = registry.available_versions(
            ...     SchemaRegistry.CUSTOM_TRACE
            ... )
            >>> "insideLLMs.custom.trace@1" in versions
            True
            >>> len(versions)
            1

        Finding the latest version:

            >>> from insideLLMs.schemas.registry import SchemaRegistry, semver_tuple
            >>> registry = SchemaRegistry()
            >>> versions = registry.available_versions("ProbeResult")
            >>> latest = max(versions, key=semver_tuple)
            >>> latest
            '1.0.1'

        Checking if a specific version exists:

            >>> registry = SchemaRegistry()
            >>> "1.0.0" in registry.available_versions("ResultRecord")
            True
            >>> "2.0.0" in registry.available_versions("ResultRecord")
            False

        Building a version dropdown for UI:

            >>> registry = SchemaRegistry()
            >>> schema_name = "ProbeResult"
            >>> options = [
            ...     {"label": f"v{v}", "value": v}
            ...     for v in registry.available_versions(schema_name)
            ... ]
            >>> options[0]["label"]
            'v1.0.0'

        Notes
        -----
        - Schema names are case-sensitive. "ProbeResult" is valid, but
          "proberesult" or "PROBERESULT" will return an empty list.

        - This method returns a new list on each call. Modifying the returned
          list will not affect future calls.

        - The CustomTrace schema uses a URN-style version identifier
          ("insideLLMs.custom.trace@1") rather than standard SemVer. This is
          because trace formats may have different compatibility guarantees.

        - This method does not validate whether the versions can actually be
          loaded. Use ``get_model()`` to verify that a version is functional.

        See Also
        --------
        get_model : Load a specific schema version
        semver_tuple : Compare version strings
        """
        if schema_name == self.CUSTOM_TRACE:
            return ["insideLLMs.custom.trace@1"]
        if schema_name not in {
            self.RUNNER_ITEM,
            self.RUNNER_OUTPUT,
            self.RESULT_RECORD,
            self.RUN_MANIFEST,
            self.HARNESS_RECORD,
            self.HARNESS_SUMMARY,
            self.HARNESS_EXPLAIN,
            self.BENCHMARK_SUMMARY,
            self.COMPARISON_REPORT,
            self.DIFF_REPORT,
            self.EXPORT_METADATA,
        }:
            return []
        return ["1.0.0", "1.0.1"]

    def get_model(self, schema_name: str, schema_version: str) -> Type[Any]:
        """Return the Pydantic model class for a schema.

        Looks up and returns the Pydantic model class for the specified schema
        name and version. The model is cached after first lookup for performance,
        making repeated calls very fast.

        This is the primary method for obtaining schema models. The returned
        model class can be used to:
        - Create new instances: ``model(field1=value1, ...)``
        - Validate dictionaries: ``model.model_validate(data_dict)``
        - Validate JSON strings: ``model.model_validate_json(json_string)``
        - Generate JSON Schema: ``model.model_json_schema()``

        Parameters
        ----------
        schema_name : str
            The canonical name of the schema (e.g., "ProbeResult", "ResultRecord").
            Use class constants like ``SchemaRegistry.RUNNER_ITEM`` for type safety.

            Must be one of:
                - SchemaRegistry.RUNNER_ITEM ("ProbeResult")
                - SchemaRegistry.RUNNER_OUTPUT ("RunnerOutput")
                - SchemaRegistry.RESULT_RECORD ("ResultRecord")
                - SchemaRegistry.RUN_MANIFEST ("RunManifest")
                - SchemaRegistry.HARNESS_RECORD ("HarnessRecord")
                - SchemaRegistry.HARNESS_SUMMARY ("HarnessSummary")
                - SchemaRegistry.BENCHMARK_SUMMARY ("BenchmarkSummary")
                - SchemaRegistry.COMPARISON_REPORT ("ComparisonReport")
                - SchemaRegistry.DIFF_REPORT ("DiffReport")
                - SchemaRegistry.EXPORT_METADATA ("ExportMetadata")
                - SchemaRegistry.CUSTOM_TRACE ("CustomTrace")

        schema_version : str
            The semantic version string (e.g., "1.0.0", "1.0.1"). Short-form
            versions like "1.0" are normalized automatically to "1.0.0".

            For CustomTrace schemas, use the special version identifier
            "insideLLMs.custom.trace@1".

        Returns
        -------
        Type[Any]
            The Pydantic model class (not an instance) that implements the
            schema. This is a subclass of ``pydantic.BaseModel``.

        Raises
        ------
        ImportError
            If Pydantic is not installed. The error message includes
            installation instructions.

        KeyError
            If the schema name is not recognized, or if the version is not
            available for the given schema. The error message indicates
            which lookup failed.

        Examples
        --------
        Getting a model and creating an instance:

            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
            >>> result = ProbeResult(
            ...     input="What is 2+2?",
            ...     output="4",
            ...     status="success"
            ... )
            >>> result.status
            'success'

        Using schema constants (recommended):

            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model(
            ...     SchemaRegistry.RUNNER_ITEM,
            ...     "1.0.0"
            ... )
            >>> result = ProbeResult(input="test", status="success")

        Short-form versions are normalized:

            >>> registry = SchemaRegistry()
            >>> model1 = registry.get_model("ProbeResult", "1.0")
            >>> model2 = registry.get_model("ProbeResult", "1.0.0")
            >>> model1 is model2  # Same cached instance
            True

        Validating data with model_validate (Pydantic v2):

            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
            >>> data = {"input": "Hello", "status": "success", "output": "World"}
            >>> result = ProbeResult.model_validate(data)
            >>> result.input
            'Hello'

        Validating JSON strings directly:

            >>> import json
            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
            >>> json_str = '{"input": "test", "status": "success"}'
            >>> result = ProbeResult.model_validate_json(json_str)
            >>> result.status
            'success'

        Creating ResultRecord with required fields:

            >>> registry = SchemaRegistry()
            >>> ResultRecord = registry.get_model("ResultRecord", "1.0.0")
            >>> from datetime import datetime
            >>> record = ResultRecord(
            ...     run_id="run-123",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model={"model_id": "gpt-4"},
            ...     probe={"probe_id": "test-probe"},
            ...     example_id="ex-1",
            ...     status="success"
            ... )
            >>> record.run_id
            'run-123'

        Handling validation errors:

            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
            >>> try:
            ...     # Missing required 'status' field
            ...     result = ProbeResult(input="test")
            ... except Exception as e:
            ...     "status" in str(e).lower() or "validation" in str(e).lower()
            True

        Handling unknown schemas:

            >>> registry = SchemaRegistry()
            >>> try:
            ...     registry.get_model("InvalidSchema", "1.0.0")
            ... except KeyError as e:
            ...     "Unknown schema" in str(e)
            True

        Handling unknown versions:

            >>> registry = SchemaRegistry()
            >>> try:
            ...     registry.get_model("ProbeResult", "99.0.0")
            ... except KeyError as e:
            ...     "Unknown schema version" in str(e)
            True

        Caching behavior:

            >>> registry = SchemaRegistry()
            >>> model1 = registry.get_model("ProbeResult", "1.0.0")
            >>> model2 = registry.get_model("ProbeResult", "1.0.0")
            >>> model1 is model2  # Same object from cache
            True

        Working with CustomTrace schema:

            >>> registry = SchemaRegistry()
            >>> TraceBundleV1 = registry.get_model(
            ...     SchemaRegistry.CUSTOM_TRACE,
            ...     "insideLLMs.custom.trace@1"
            ... )
            >>> hasattr(TraceBundleV1, "model_fields")
            True

        Batch validation pattern:

            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
            >>> items = [
            ...     {"input": "test1", "status": "success"},
            ...     {"input": "test2", "status": "error", "error": "timeout"},
            ...     {"input": "test3", "status": "success", "output": "result"},
            ... ]
            >>> validated = [ProbeResult.model_validate(item) for item in items]
            >>> len(validated)
            3

        Notes
        -----
        - The returned model class is cached in the registry's ``_cache``
          dictionary. The cache key is ``(schema_name, normalized_version)``.

        - Version normalization happens before cache lookup, so "1.0" and
          "1.0.0" share the same cache entry.

        - This method imports schema modules lazily. The first call for a
          given version will import the module; subsequent calls use the cache.

        - The returned class is the actual Pydantic model class, not a wrapper.
          You can use all Pydantic features directly.

        - For best performance in loops, call ``get_model()`` once and reuse
          the returned class rather than calling ``get_model()`` repeatedly.

        See Also
        --------
        available_versions : List available versions for a schema
        get_json_schema : Generate JSON Schema for a model
        migrate : Transform data between schema versions
        SchemaHandle : Internal cache entry type
        """
        _require_pydantic()
        v = normalize_semver(schema_version)

        cache_key = (schema_name, v)
        if cache_key in self._cache:
            return self._cache[cache_key].model

        if schema_name == self.CUSTOM_TRACE:
            if v != "insideLLMs.custom.trace@1":
                raise KeyError(f"Unknown schema version: {schema_name}@{v}")
            from insideLLMs.schemas import custom_trace_v1

            model = custom_trace_v1.TraceBundleV1
        elif v == "1.0.0":
            from insideLLMs.schemas import v1_0_0

            model = v1_0_0.get_schema_model(schema_name)  # type: ignore[assignment]
        elif v == "1.0.1":
            from insideLLMs.schemas import v1_0_1

            model = v1_0_1.get_schema_model(schema_name)  # type: ignore[assignment]
        else:
            raise KeyError(f"Unknown schema version: {schema_name}@{v}")

        self._cache[cache_key] = SchemaHandle(
            schema_name=schema_name, schema_version=v, model=model
        )
        return model

    def get_json_schema(self, schema_name: str, schema_version: str) -> dict[str, Any]:
        """Generate JSON Schema for a schema model.

        Produces a JSON Schema (Draft 2020-12 for Pydantic v2, or Draft 7 for
        Pydantic v1) representation of the specified schema. This is useful for:

        - **API Documentation**: Embedding in OpenAPI/Swagger specifications
        - **External Validation**: Using JSON Schema validators in other languages
        - **Code Generation**: Generating types for TypeScript, Go, etc.
        - **Schema Comparison**: Comparing schema versions programmatically

        The generated schema includes all field definitions, types, constraints,
        default values, and descriptions from the Pydantic model.

        Parameters
        ----------
        schema_name : str
            The canonical name of the schema (e.g., "ProbeResult", "ResultRecord").
            Use class constants like ``SchemaRegistry.RUNNER_ITEM`` for type safety.

        schema_version : str
            The semantic version string (e.g., "1.0.0"). Short-form versions
            like "1.0" are normalized automatically.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the JSON Schema representation. The schema
            includes:

            - ``title``: Set to "SchemaName@version" (e.g., "ProbeResult@1.0.0")
            - ``type``: Usually "object" for model schemas
            - ``properties``: Field definitions with types and constraints
            - ``required``: List of required field names
            - ``$defs`` or ``definitions``: Nested type definitions (if any)

            The exact structure follows JSON Schema standards and may vary
            slightly based on the Pydantic version.

        Raises
        ------
        ImportError
            If Pydantic is not installed. The error message includes
            installation instructions.

        KeyError
            If the schema name or version is not recognized.

        Examples
        --------
        Generating a JSON Schema:

            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> schema["title"]
            'ProbeResult@1.0.0'
            >>> "properties" in schema
            True
            >>> schema["type"]
            'object'

        Inspecting schema properties:

            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> "input" in schema["properties"]
            True
            >>> "status" in schema["properties"]
            True
            >>> "output" in schema["properties"]
            True

        Checking required fields:

            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> required = schema.get("required", [])
            >>> "input" in required
            True
            >>> "status" in required
            True

        Using for OpenAPI/Swagger documentation:

            >>> import json
            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> # Build OpenAPI components section
            >>> openapi_spec = {
            ...     "openapi": "3.0.0",
            ...     "components": {
            ...         "schemas": {
            ...             "ProbeResult": schema
            ...         }
            ...     }
            ... }
            >>> "ProbeResult" in openapi_spec["components"]["schemas"]
            True

        Serializing to JSON string:

            >>> import json
            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ResultRecord", "1.0.1")
            >>> json_str = json.dumps(schema, indent=2)
            >>> "run_id" in json_str
            True
            >>> print(json_str[:100])  # doctest: +SKIP
            {
              "title": "ResultRecord@1.0.1",
              "type": "object",
              ...

        Writing schema to file for external tools:

            >>> import json
            >>> import tempfile
            >>> import os
            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ...     json.dump(schema, f, indent=2)
            ...     temp_path = f.name
            >>> os.path.exists(temp_path)
            True
            >>> os.unlink(temp_path)  # Cleanup

        Comparing schemas across versions:

            >>> registry = SchemaRegistry()
            >>> v1_schema = registry.get_json_schema("RunManifest", "1.0.0")
            >>> v1_1_schema = registry.get_json_schema("RunManifest", "1.0.1")
            >>> # Check for new fields in 1.0.1
            >>> v1_props = set(v1_schema.get("properties", {}).keys())
            >>> v1_1_props = set(v1_1_schema.get("properties", {}).keys())
            >>> new_fields = v1_1_props - v1_props
            >>> "run_completed" in new_fields
            True

        Getting field type information:

            >>> registry = SchemaRegistry()
            >>> schema = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> input_field = schema["properties"]["input"]
            >>> "type" in input_field or "anyOf" in input_field
            True

        Building a complete API schema:

            >>> registry = SchemaRegistry()
            >>> schemas = {}
            >>> for name in ["ProbeResult", "ResultRecord"]:
            ...     for version in registry.available_versions(name):
            ...         key = f"{name}_{version.replace('.', '_')}"
            ...         schemas[key] = registry.get_json_schema(name, version)
            >>> len(schemas) >= 4
            True

        Short-form versions work:

            >>> registry = SchemaRegistry()
            >>> schema1 = registry.get_json_schema("ProbeResult", "1.0")
            >>> schema2 = registry.get_json_schema("ProbeResult", "1.0.0")
            >>> schema1["title"] == schema2["title"]
            True

        Notes
        -----
        - The ``title`` field is automatically set to "SchemaName@version" if
          not already present in the schema. This helps identify which schema
          version generated the JSON Schema.

        - The generated schema is a fresh copy each time. Modifying the returned
          dictionary will not affect future calls.

        - For complex schemas with nested models, the JSON Schema will include
          ``$defs`` (Pydantic v2) or ``definitions`` (Pydantic v1) containing
          the nested type definitions.

        - Some Pydantic-specific features (like custom validators) cannot be
          represented in JSON Schema. The schema captures the structural
          constraints but not custom validation logic.

        - The schema is generated by Pydantic's built-in JSON Schema generation,
          which follows the JSON Schema specification closely.

        See Also
        --------
        get_model : Get the Pydantic model class directly
        insideLLMs.structured.get_json_schema : Low-level JSON Schema generation
        """
        _require_pydantic()
        from insideLLMs.structured import get_json_schema

        model = self.get_model(schema_name, schema_version)
        schema = get_json_schema(model)
        schema.setdefault("title", f"{schema_name}@{normalize_semver(schema_version)}")
        return schema

    def migrate(
        self,
        schema_name: str,
        data: Any,
        from_version: str,
        to_version: str,
        *,
        custom_migration: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        """Migrate data between schema versions.

        Transforms data from one schema version to another. This is useful when
        working with historical data that was serialized with an older schema
        version and needs to be processed with newer tooling, or when
        consolidating data from multiple schema versions.

        The migration system is designed to be extensible. Currently supported
        migrations:

        - **Legacy normalization**: "1.0" -> "1.0.0" (version string normalization)
        - **Identity migrations**: Same version, with optional custom transform
        - **Custom migrations**: User-provided functions for arbitrary transforms

        Future versions may include built-in migration paths between specific
        versions (e.g., 1.0.0 -> 1.0.1).

        Parameters
        ----------
        schema_name : str
            The canonical name of the schema being migrated (e.g., "ProbeResult",
            "ResultRecord"). This is used for error messages and future
            schema-specific migration logic.

        data : Any
            The data to migrate. Can be:
            - A dictionary (most common)
            - A Pydantic model instance
            - Any serializable object

            The data is passed to the migration function as-is. If you need
            to work with a dict, ensure you pass a dict or call ``.model_dump()``
            on Pydantic instances first.

        from_version : str
            The source schema version. This is the version the data was
            originally serialized with. Short-form versions (e.g., "1.0")
            are normalized automatically.

        to_version : str
            The target schema version. This is the version you want the data
            to conform to after migration. Short-form versions are normalized.

        custom_migration : Callable[[Any], Any], optional
            A callable that transforms the data. If provided, it's applied
            after version normalization (when from_version and to_version
            normalize to the same value).

            The callable:
            - Receives the data as its single argument
            - Should return the transformed data
            - Can modify the input in-place or return a new object
            - Is only called for identity migrations (same normalized version)

        Returns
        -------
        Any
            The migrated data. The return type depends on:
            - Identity migration without custom_migration: Returns input unchanged
            - Identity migration with custom_migration: Returns result of callable
            - Future cross-version migrations: Returns transformed data

        Raises
        ------
        NotImplementedError
            If no migration path exists between the specified versions.
            The error message includes the schema name and version range.

        Examples
        --------
        Identity migration (same version):

            >>> registry = SchemaRegistry()
            >>> data = {"input": "test", "status": "success"}
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0.0", "1.0.0"
            ... )
            >>> migrated == data
            True
            >>> migrated is data  # Returns same object for identity
            True

        Normalized version migration:

            >>> registry = SchemaRegistry()
            >>> data = {"input": "test", "status": "success"}
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0", "1.0.0"
            ... )
            >>> migrated == data
            True

        Using custom migration function to add fields:

            >>> registry = SchemaRegistry()
            >>> data = {"input": "test", "status": "success"}
            >>> def add_metadata(d):
            ...     d = d.copy()  # Don't modify original
            ...     d["metadata"] = {"migrated": True}
            ...     return d
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0.0", "1.0.0",
            ...     custom_migration=add_metadata
            ... )
            >>> migrated["metadata"]["migrated"]
            True
            >>> "metadata" not in data  # Original unchanged
            True

        Using custom migration to rename fields:

            >>> registry = SchemaRegistry()
            >>> data = {"old_field": "value", "status": "success"}
            >>> def rename_field(d):
            ...     d = d.copy()
            ...     d["new_field"] = d.pop("old_field", None)
            ...     return d
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0.0", "1.0.0",
            ...     custom_migration=rename_field
            ... )
            >>> migrated["new_field"]
            'value'
            >>> "old_field" in migrated
            False

        Using custom migration to transform values:

            >>> registry = SchemaRegistry()
            >>> data = {"input": "test", "status": "SUCCESS"}
            >>> def normalize_status(d):
            ...     d = d.copy()
            ...     d["status"] = d["status"].lower()
            ...     return d
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0.0", "1.0.0",
            ...     custom_migration=normalize_status
            ... )
            >>> migrated["status"]
            'success'

        Chaining custom migrations:

            >>> registry = SchemaRegistry()
            >>> data = {"input": "test", "status": "success"}
            >>> def migration_chain(d):
            ...     d = d.copy()
            ...     d["version"] = "1.0.0"
            ...     d["migrated_at"] = "2024-01-15"
            ...     return d
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0.0", "1.0.0",
            ...     custom_migration=migration_chain
            ... )
            >>> migrated["version"]
            '1.0.0'

        Migrating Pydantic model instances:

            >>> registry = SchemaRegistry()
            >>> ProbeResult = registry.get_model("ProbeResult", "1.0.0")
            >>> instance = ProbeResult(input="test", status="success")
            >>> # Convert to dict for migration
            >>> data = instance.model_dump()
            >>> def add_output(d):
            ...     d = d.copy()
            ...     d["output"] = "migrated output"
            ...     return d
            >>> migrated = registry.migrate(
            ...     "ProbeResult", data, "1.0.0", "1.0.0",
            ...     custom_migration=add_output
            ... )
            >>> migrated["output"]
            'migrated output'

        Batch migration:

            >>> registry = SchemaRegistry()
            >>> items = [
            ...     {"input": "test1", "status": "success"},
            ...     {"input": "test2", "status": "error"},
            ... ]
            >>> def add_index(d):
            ...     # Note: This example shows the pattern; use enumerate in practice
            ...     d = d.copy()
            ...     return d
            >>> migrated_items = [
            ...     registry.migrate("ProbeResult", item, "1.0", "1.0.0")
            ...     for item in items
            ... ]
            >>> len(migrated_items)
            2

        Unsupported version migration raises error:

            >>> registry = SchemaRegistry()
            >>> try:
            ...     registry.migrate("ProbeResult", {}, "1.0.0", "2.0.0")
            ... except NotImplementedError as e:
            ...     "No migration registered" in str(e)
            True

        Error message includes useful context:

            >>> registry = SchemaRegistry()
            >>> try:
            ...     registry.migrate("ProbeResult", {}, "1.0.0", "3.0.0")
            ... except NotImplementedError as e:
            ...     "ProbeResult" in str(e) and "1.0.0" in str(e) and "3.0.0" in str(e)
            True

        Notes
        -----
        - Version normalization is applied before comparison. "1.0" and "1.0.0"
          are considered the same version.

        - The current implementation only supports identity migrations (same
          version after normalization). Cross-version migrations will be added
          in future releases.

        - Custom migration functions are only called for identity migrations.
          For cross-version migrations (when implemented), built-in migration
          logic will be used.

        - This method does not validate the input or output data against the
          schema. Use ``get_model().model_validate()`` after migration if you
          need validation.

        - For large-scale migrations, consider using streaming approaches and
          writing results incrementally rather than loading all data into memory.

        - The ``schema_name`` parameter is currently used only for error messages
          but will be used for schema-specific migrations in the future.

        See Also
        --------
        get_model : Get schema model for validation after migration
        normalize_semver : Version normalization used by this method
        """
        f = normalize_semver(from_version)
        t = normalize_semver(to_version)
        if f == t:
            return custom_migration(data) if custom_migration else data
        raise NotImplementedError(f"No migration registered for {schema_name}: {f} -> {t}")
