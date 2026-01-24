"""Constants for output schema versioning.

This module defines version constants used throughout the insideLLMs schema system.
The schema version follows Semantic Versioning (SemVer) conventions to ensure
backward compatibility and clear version progression.

Overview
--------
The ``constants`` module serves as the single source of truth for schema versioning
in the insideLLMs library. All output validation, serialization, and deserialization
operations reference these constants to ensure consistency across the codebase.

Schema versioning is critical for:
    - **Data Integrity**: Ensuring outputs conform to expected structures
    - **Backward Compatibility**: Supporting migration paths between versions
    - **Reproducibility**: Tracking which schema version produced a given output
    - **Tooling Integration**: Enabling downstream tools to parse outputs correctly

Schema Versioning Philosophy
----------------------------
The library follows Semantic Versioning (SemVer) 2.0.0 specification:

    - **MAJOR version** (X.y.z): Incremented for breaking changes that require
      data migration or code updates. Examples include removing fields, changing
      field types, or modifying required/optional status.

    - **MINOR version** (x.Y.z): Incremented for backward-compatible additions.
      Examples include adding new optional fields with sensible defaults or
      introducing new schema types.

    - **PATCH version** (x.y.Z): Incremented for backward-compatible bug fixes
      and documentation updates. Examples include fixing validation regex
      patterns or clarifying field descriptions.

Module Constants
----------------
DEFAULT_SCHEMA_VERSION : str
    The current default schema version used for output validation.
    This version is applied when no explicit version is specified
    during validation. Currently set to "1.0.1".

Examples
--------
Accessing the default schema version:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> DEFAULT_SCHEMA_VERSION
    '1.0.1'

    >>> # The version follows SemVer format
    >>> parts = DEFAULT_SCHEMA_VERSION.split('.')
    >>> len(parts)
    3
    >>> all(part.isdigit() for part in parts)
    True

Using the version in validation workflows:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> from insideLLMs.schemas.validator import OutputValidator
    >>> validator = OutputValidator()
    >>> # Validate a ProbeResult with explicit version
    >>> result = validator.validate(
    ...     "ProbeResult",
    ...     {"input": "test", "status": "success"},
    ...     schema_version=DEFAULT_SCHEMA_VERSION
    ... )
    >>> result.is_valid
    True

Checking version compatibility programmatically:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> from insideLLMs.schemas.registry import semver_tuple
    >>> major, minor, patch = semver_tuple(DEFAULT_SCHEMA_VERSION)
    >>> major >= 1  # Ensure we're on a stable version
    True
    >>> # Check for minimum required version
    >>> (major, minor, patch) >= (1, 0, 0)
    True

Embedding version in output files for traceability:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> import json
    >>> output_record = {
    ...     "schema_version": DEFAULT_SCHEMA_VERSION,
    ...     "input": "What is 2+2?",
    ...     "output": "4",
    ...     "status": "success",
    ...     "latency_ms": 150.5
    ... }
    >>> # Serialize with version for later parsing
    >>> serialized = json.dumps(output_record)
    >>> # Later, deserialize and check version
    >>> loaded = json.loads(serialized)
    >>> loaded["schema_version"]
    '1.0.1'

Comparing versions for migration decisions:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> def needs_migration(old_version: str) -> bool:
    ...     '''Check if data needs migration to current schema.'''
    ...     old_parts = tuple(int(p) for p in old_version.split('.'))
    ...     new_parts = tuple(int(p) for p in DEFAULT_SCHEMA_VERSION.split('.'))
    ...     # Major version change requires migration
    ...     return old_parts[0] < new_parts[0]
    >>> needs_migration("0.9.0")
    True
    >>> needs_migration("1.0.0")
    False

Using version in configuration files:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> config = {
    ...     "probe_config": {
    ...         "model": "gpt-4",
    ...         "temperature": 0.7,
    ...         "schema_version": DEFAULT_SCHEMA_VERSION
    ...     }
    ... }
    >>> # The config can now be validated against the correct schema
    >>> config["probe_config"]["schema_version"]
    '1.0.1'

Logging version information for debugging:

    >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> # In practice, this logs version info at startup
    >>> log_msg = f"Using schema version: {DEFAULT_SCHEMA_VERSION}"
    >>> "1.0.1" in log_msg
    True

See Also
--------
insideLLMs.schemas.registry : Schema registry with version-aware lookups.
insideLLMs.schemas.validator : Output validation using versioned schemas.
insideLLMs.schemas.models : Pydantic models defining schema structures.
https://semver.org : Semantic Versioning 2.0.0 specification.

Notes
-----
- Schemas are intentionally strict (extra fields forbidden by default). Any
  additions to serialized outputs must bump this version and be reflected
  in the schema registry.

- When introducing new schema versions, ensure backward compatibility tests
  are added to verify that older data can still be loaded (with appropriate
  defaults for new fields).

- The version constant should only be modified as part of a deliberate
  release process, never in feature branches or bug fixes unless the fix
  specifically relates to schema structure.

- For custom schemas or extensions, consider using a separate namespace
  (e.g., "myext-1.0.0") to avoid conflicts with library versions.

Version History
---------------
1.0.0 : 2024-01-15
    Initial stable release with core schemas:
    - ProbeResult: Individual probe output records
    - RunManifest: Batch run metadata and configuration
    - TraceRecord: Execution trace for debugging

1.0.1 : 2024-02-01
    Added ``run_completed`` field to RunManifest:
    - New optional boolean field indicating run completion status
    - Defaults to None for backward compatibility
    - Enables better run state tracking in long-running probes

Warning
-------
Do not modify ``DEFAULT_SCHEMA_VERSION`` without updating the corresponding
schema definitions in ``insideLLMs.schemas.registry``. Mismatched versions
will cause validation failures.
"""

from __future__ import annotations

__all__ = ["DEFAULT_SCHEMA_VERSION"]


# =============================================================================
# Schema Version Constant
# =============================================================================

#: DEFAULT_SCHEMA_VERSION : str
#:     The default semantic version for the output contract(s) shipped by the library.
#:
#:     This version string follows SemVer format (MAJOR.MINOR.PATCH) and is used
#:     as the default when validating outputs without an explicit version. It serves
#:     as the canonical reference for all schema-related operations in the library.
#:
#:     The version is used by:
#:         - :class:`~insideLLMs.schemas.validator.OutputValidator` for default validation
#:         - :mod:`~insideLLMs.schemas.registry` for schema lookups
#:         - Output serialization to embed version metadata in files
#:         - Migration utilities to determine upgrade paths
#:
#:     Type:
#:         str
#:
#:     Value:
#:         "1.0.1"
#:
#:     Format:
#:         MAJOR.MINOR.PATCH where each component is a non-negative integer.
#:
#:     Examples:
#:         Basic access::
#:
#:             >>> from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
#:             >>> DEFAULT_SCHEMA_VERSION
#:             '1.0.1'
#:
#:         Parsing the version components::
#:
#:             >>> major, minor, patch = DEFAULT_SCHEMA_VERSION.split('.')
#:             >>> int(major), int(minor), int(patch)
#:             (1, 0, 1)
#:
#:         Using with validation::
#:
#:             >>> from insideLLMs.schemas.validator import OutputValidator
#:             >>> validator = OutputValidator()
#:             >>> # Uses DEFAULT_SCHEMA_VERSION internally when none specified
#:             >>> result = validator.validate("ProbeResult", {"input": "x", "status": "ok"})
#:
#:         Embedding in output records::
#:
#:             >>> record = {"schema_version": DEFAULT_SCHEMA_VERSION, "data": {...}}
#:
#:     See Also:
#:         - :func:`insideLLMs.schemas.registry.semver_tuple` for version parsing
#:         - :func:`insideLLMs.schemas.registry.get_schema` for version-aware lookups
#:
#:     Notes:
#:         - Always use this constant rather than hardcoding version strings
#:         - The constant is immutable at runtime; changes require code updates
#:         - For testing different versions, use the explicit version parameter
#:           in validation functions rather than modifying this constant
#:
#:     Version History:
#:         - 1.0.0: Initial stable release with core schemas (ProbeResult, RunManifest)
#:         - 1.0.1: Added ``run_completed`` field to RunManifest for completion tracking
#:
#:     Warning:
#:         Changing this value without corresponding schema registry updates will
#:         cause :exc:`~insideLLMs.schemas.exceptions.SchemaVersionError` during
#:         validation operations.
DEFAULT_SCHEMA_VERSION: str = "1.0.1"
