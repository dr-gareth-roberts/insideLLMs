"""Schema validation exceptions for insideLLMs output validation.

This module defines custom exception classes for schema validation errors in the
insideLLMs output validation system. These exceptions provide structured error
information when output data fails to conform to expected schemas, enabling
precise error handling and debugging in validation pipelines.

Module Overview
---------------
The ``exceptions`` module is a core component of the insideLLMs schema validation
subsystem. It provides specialized exception types that carry rich metadata about
validation failures, making it easier to diagnose issues, implement retry logic,
and build robust data processing pipelines.

The exception hierarchy is designed to:

- **Provide clear, actionable error messages**: Each exception formats its
  message to include the schema name, version, and specific validation errors.
- **Include structured data about validation failures**: Exceptions store
  validation errors as a list, enabling programmatic inspection and filtering.
- **Integrate naturally with Python's exception handling**: By inheriting from
  :class:`ValueError`, these exceptions work seamlessly with existing error
  handling patterns and can be caught by generic exception handlers.
- **Support schema versioning**: Every exception includes version information,
  critical for debugging issues in systems processing data from multiple
  schema versions.

Exception Classes
-----------------
OutputValidationError
    The primary exception raised when output validation fails in strict mode.
    Contains schema name, version, and a list of validation error messages.

Integration with Validation System
----------------------------------
This module is tightly integrated with:

- :class:`~insideLLMs.schemas.validator.OutputValidator`: Raises
  ``OutputValidationError`` when validation fails in strict mode.
- :class:`~insideLLMs.schemas.registry.SchemaRegistry`: Provides the schema
  metadata (name, version) included in exceptions.

Examples
--------
Catching validation errors from the OutputValidator:

    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> from insideLLMs.schemas.validator import OutputValidator
    >>> validator = OutputValidator()
    >>> try:
    ...     validator.validate("ProbeResult", {"invalid": "data"}, mode="strict")
    ... except OutputValidationError as e:
    ...     print(f"Schema: {e.schema_name}")
    ...     print(f"Version: {e.schema_version}")
    ...     print(f"Error count: {len(e.errors)}")
    Schema: ProbeResult
    Version: 1.0.0
    Error count: 1

Creating a custom validation error for testing or custom validators:

    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> error = OutputValidationError(
    ...     schema_name="ResultRecord",
    ...     schema_version="1.0.1",
    ...     errors=["Missing required field: run_id", "Invalid status value"]
    ... )
    >>> str(error)
    'Output validation failed for ResultRecord@1.0.1: Missing required field: run_id; Invalid status value'

Using in batch validation pipelines with error collection:

    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> def validate_batch(records, validator, schema_name="ProbeResult"):
    ...     '''Validate a batch of records, collecting errors separately.'''
    ...     valid_records = []
    ...     invalid_records = []
    ...     for idx, record in enumerate(records):
    ...         try:
    ...             validated = validator.validate(schema_name, record, mode="strict")
    ...             valid_records.append(validated)
    ...         except OutputValidationError as e:
    ...             invalid_records.append({
    ...                 "index": idx,
    ...                 "record": record,
    ...                 "schema": e.schema_name,
    ...                 "version": e.schema_version,
    ...                 "errors": e.errors
    ...             })
    ...     return valid_records, invalid_records

Checking exception attributes and type hierarchy:

    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> error = OutputValidationError(
    ...     schema_name="RunManifest",
    ...     schema_version="1.0.0",
    ...     errors=["Invalid datetime format"]
    ... )
    >>> isinstance(error, ValueError)
    True
    >>> isinstance(error, Exception)
    True
    >>> error.schema_name
    'RunManifest'
    >>> error.schema_version
    '1.0.0'

Filtering errors by type in exception handlers:

    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> def process_with_fallback(data, validator):
    ...     '''Try strict validation, fall back to warn mode on failure.'''
    ...     try:
    ...         return validator.validate("ProbeResult", data, mode="strict")
    ...     except OutputValidationError as e:
    ...         # Log the validation errors for debugging
    ...         for err in e.errors:
    ...             print(f"Validation warning: {err}")
    ...         # Re-validate in warn mode to get partial data
    ...         return validator.validate("ProbeResult", data, mode="warn")

Re-raising exceptions with additional context:

    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> def validate_with_source_context(data, validator, source_file):
    ...     '''Add source file context to validation errors.'''
    ...     try:
    ...         return validator.validate("ResultRecord", data, mode="strict")
    ...     except OutputValidationError as e:
    ...         # Add source context to error messages
    ...         contextual_errors = [f"[{source_file}] {err}" for err in e.errors]
    ...         raise OutputValidationError(
    ...             schema_name=e.schema_name,
    ...             schema_version=e.schema_version,
    ...             errors=contextual_errors
    ...         ) from e

Using exception data for structured logging:

    >>> import json
    >>> from insideLLMs.schemas.exceptions import OutputValidationError
    >>> error = OutputValidationError(
    ...     schema_name="HarnessRecord",
    ...     schema_version="1.0.1",
    ...     errors=["status: invalid enum value", "timing: negative duration"]
    ... )
    >>> log_entry = {
    ...     "event": "validation_failed",
    ...     "schema": error.schema_name,
    ...     "version": error.schema_version,
    ...     "error_count": len(error.errors),
    ...     "errors": error.errors
    ... }
    >>> print(json.dumps(log_entry, indent=2))  # doctest: +NORMALIZE_WHITESPACE
    {
      "event": "validation_failed",
      "schema": "HarnessRecord",
      "version": "1.0.1",
      "error_count": 2,
      "errors": [
        "status: invalid enum value",
        "timing: negative duration"
      ]
    }

Notes
-----
- Exceptions in this module are implemented as dataclasses, which provides
  automatic ``__init__``, ``__repr__``, and ``__eq__`` methods while allowing
  custom ``__str__`` formatting.
- The :class:`OutputValidationError` inherits from :class:`ValueError` rather
  than a custom base exception to ensure compatibility with generic exception
  handlers that catch standard Python exceptions.
- Error messages in the ``errors`` list are human-readable strings derived
  from Pydantic validation errors. The exact format may vary between Pydantic
  versions (v1 vs v2).
- When using "warn" mode in :class:`~insideLLMs.schemas.validator.OutputValidator`,
  no exception is raised; instead, a :class:`RuntimeWarning` is issued and the
  original data is returned.

Warnings
--------
- Do not rely on the exact string format of error messages for programmatic
  parsing, as these may change between versions. Use the structured
  ``errors`` list for programmatic access.
- The ``schema_version`` attribute contains the normalized version string
  (e.g., "1.0.0" not "1.0"), which may differ from the version originally
  passed to the validator.

See Also
--------
insideLLMs.schemas.validator.OutputValidator : The validator class that raises
    these exceptions.
insideLLMs.schemas.registry.SchemaRegistry : Registry for looking up schema
    models by name and version.
insideLLMs.schemas.constants.DEFAULT_SCHEMA_VERSION : The default schema
    version used when none is specified.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputValidationError(ValueError):
    """Exception raised when output validation fails in strict mode.

    This exception is raised by :class:`~insideLLMs.schemas.validator.OutputValidator`
    when data fails to validate against a schema and the validation mode is set to
    "strict". It inherits from :class:`ValueError` to integrate naturally with
    standard Python error handling patterns.

    The exception is implemented as a dataclass to provide structured access to
    validation failure details while maintaining compatibility with standard
    exception handling. As a dataclass, it automatically provides ``__init__``,
    ``__repr__``, and ``__eq__`` methods, while the custom ``__str__`` method
    formats a human-readable error message.

    Parameters
    ----------
    schema_name : str
        The canonical name of the schema that validation was attempted against.
        This should match one of the schema names defined in
        :class:`~insideLLMs.schemas.registry.SchemaRegistry`, such as:

        - ``"ProbeResult"`` - Per-item probe execution results
        - ``"ResultRecord"`` - JSONL record format for batch results
        - ``"RunManifest"`` - Manifest file for run directories
        - ``"HarnessRecord"`` - Per-line harness execution records
        - ``"HarnessSummary"`` - Summary of harness execution
        - ``"BenchmarkSummary"`` - Benchmark run output summary
        - ``"ComparisonReport"`` - Comparison between runs
        - ``"DiffReport"`` - Diff output between baselines

    schema_version : str
        The semantic version string of the schema (e.g., "1.0.0", "1.0.1").
        This is the *normalized* version string (always in "MAJOR.MINOR.PATCH"
        format), which helps identify exactly which version of the schema
        contract was expected. Common versions include:

        - ``"1.0.0"`` - Initial schema version
        - ``"1.0.1"`` - Updated schema with additional fields

    errors : list[str]
        A list of human-readable error messages describing what validation
        rules were violated. Each string typically describes a specific field
        error or constraint violation. Common error patterns include:

        - ``"Field required: <field_name>"`` - Missing required field
        - ``"<field>: value is not a valid enumeration member"`` - Invalid enum
        - ``"<field>: invalid datetime format"`` - Malformed datetime string
        - ``"<field>: str type expected"`` - Type mismatch

    Attributes
    ----------
    schema_name : str
        The name of the schema that validation was attempted against.
        Examples include "ProbeResult", "ResultRecord", "RunManifest", etc.
        This attribute is read-only after construction (as the class is a
        dataclass, not frozen, but should be treated as immutable).

    schema_version : str
        The normalized semantic version string of the schema (e.g., "1.0.0",
        "1.0.1"). Note that this is always in full semver format, even if
        a short version like "1.0" was originally passed to the validator.

    errors : list[str]
        A list of human-readable error messages describing validation failures.
        Each string describes a specific field error or constraint violation.
        The list is guaranteed to contain at least one error when the exception
        is raised by :class:`~insideLLMs.schemas.validator.OutputValidator`.

    args : tuple
        The standard exception ``args`` tuple. For ``OutputValidationError``,
        this is typically empty as the dataclass fields store the error details.
        The ``__str__`` method formats these fields into the exception message.

    Examples
    --------
    Basic exception handling - catching and inspecting validation errors:

        >>> try:
        ...     raise OutputValidationError(
        ...         schema_name="ProbeResult",
        ...         schema_version="1.0.0",
        ...         errors=["Field 'status' is required"]
        ...     )
        ... except OutputValidationError as e:
        ...     print(f"Failed to validate {e.schema_name}")
        ...     print(f"Schema version: {e.schema_version}")
        ...     print(f"First error: {e.errors[0]}")
        Failed to validate ProbeResult
        Schema version: 1.0.0
        First error: Field 'status' is required

    Accessing structured error information for detailed diagnostics:

        >>> error = OutputValidationError(
        ...     schema_name="HarnessRecord",
        ...     schema_version="1.0.1",
        ...     errors=[
        ...         "status: value is not a valid enumeration member",
        ...         "started_at: invalid datetime format",
        ...         "model: field required"
        ...     ]
        ... )
        >>> len(error.errors)
        3
        >>> "started_at" in error.errors[1]
        True
        >>> any("required" in err for err in error.errors)
        True

    Filtering errors by pattern for targeted handling:

        >>> error = OutputValidationError(
        ...     schema_name="ResultRecord",
        ...     schema_version="1.0.0",
        ...     errors=[
        ...         "run_id: field required",
        ...         "status: invalid enum value",
        ...         "timing.duration: negative value not allowed"
        ...     ]
        ... )
        >>> required_field_errors = [e for e in error.errors if "required" in e]
        >>> required_field_errors
        ['run_id: field required']
        >>> type_errors = [e for e in error.errors if "invalid" in e or "not allowed" in e]
        >>> len(type_errors)
        2

    Re-raising with additional context for debugging:

        >>> try:
        ...     raise OutputValidationError(
        ...         schema_name="BenchmarkSummary",
        ...         schema_version="1.0.0",
        ...         errors=["Invalid probe field"]
        ...     )
        ... except OutputValidationError as original:
        ...     enhanced_errors = original.errors + [
        ...         "Context: batch processing failed at index 42",
        ...         "Source: benchmark_run_2024.jsonl"
        ...     ]
        ...     raise OutputValidationError(
        ...         schema_name=original.schema_name,
        ...         schema_version=original.schema_version,
        ...         errors=enhanced_errors
        ...     ) from original  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        OutputValidationError: ...

    Using in conditional error handling based on error content:

        >>> error = OutputValidationError(
        ...     schema_name="DiffReport",
        ...     schema_version="1.0.0",
        ...     errors=["Invalid baseline path: /missing/file.json"]
        ... )
        >>> if "baseline" in str(error):
        ...     print("Check your baseline configuration")
        Check your baseline configuration
        >>> if any("path" in e.lower() for e in error.errors):
        ...     print("File path issue detected")
        File path issue detected

    Comparing exceptions for equality (dataclass behavior):

        >>> error1 = OutputValidationError(
        ...     schema_name="ProbeResult",
        ...     schema_version="1.0.0",
        ...     errors=["Missing field: input"]
        ... )
        >>> error2 = OutputValidationError(
        ...     schema_name="ProbeResult",
        ...     schema_version="1.0.0",
        ...     errors=["Missing field: input"]
        ... )
        >>> error1 == error2
        True
        >>> error1.schema_name == error2.schema_name
        True

    Using with Python's exception chaining for root cause analysis:

        >>> def validate_nested(outer_data, inner_key, validator):
        ...     '''Validate nested data with context preservation.'''
        ...     inner_data = outer_data.get(inner_key, {})
        ...     try:
        ...         return validator.validate("ProbeResult", inner_data, mode="strict")
        ...     except OutputValidationError as e:
        ...         raise OutputValidationError(
        ...             schema_name=e.schema_name,
        ...             schema_version=e.schema_version,
        ...             errors=[f"In '{inner_key}': {err}" for err in e.errors]
        ...         ) from e

    Integrating with logging frameworks:

        >>> import logging
        >>> error = OutputValidationError(
        ...     schema_name="RunManifest",
        ...     schema_version="1.0.1",
        ...     errors=["config: missing required field"]
        ... )
        >>> # In production, you would log like this:
        >>> # logger.error(
        >>> #     "Validation failed for %s@%s: %d errors",
        >>> #     error.schema_name, error.schema_version, len(error.errors),
        >>> #     extra={"errors": error.errors}
        >>> # )
        >>> f"Errors: {len(error.errors)}"
        'Errors: 1'

    Creating test fixtures for validation error handling:

        >>> def make_validation_error(schema="ProbeResult", version="1.0.0", *errors):
        ...     '''Factory function for creating test validation errors.'''
        ...     return OutputValidationError(
        ...         schema_name=schema,
        ...         schema_version=version,
        ...         errors=list(errors) if errors else ["test error"]
        ...     )
        >>> test_error = make_validation_error("ResultRecord", "1.0.1", "err1", "err2")
        >>> test_error.schema_name
        'ResultRecord'
        >>> len(test_error.errors)
        2

    Notes
    -----
    - The exception is a dataclass, so it supports all standard dataclass
      operations including ``asdict()``, ``astuple()``, and field access.
    - The ``errors`` list should be treated as immutable after construction
      to maintain exception integrity. Create a new exception if you need
      to modify errors.
    - When catching this exception alongside other ``ValueError`` subclasses,
      catch ``OutputValidationError`` first for specific handling.
    - The exception's ``__str__`` method joins errors with semicolons for
      a compact single-line representation.

    Warnings
    --------
    - Do not mutate the ``errors`` list after construction, as this could
      lead to confusing behavior if the exception is logged or re-raised.
    - The error message format may vary between Pydantic v1 and v2. Do not
      parse error messages programmatically; instead use the structured
      ``errors`` list.
    - Although this is a dataclass, treat it as immutable for thread safety.

    See Also
    --------
    insideLLMs.schemas.validator.OutputValidator : The validator that raises
        this exception.
    insideLLMs.schemas.registry.SchemaRegistry : Registry for schema lookup.
    ValueError : The base exception class.
    """

    schema_name: str
    schema_version: str
    errors: list[str]

    def __str__(self) -> str:  # pragma: no cover
        """Return a human-readable string representation of the validation error.

        Formats the error message to include the schema name, version, and all
        validation errors joined by semicolons. This method is automatically
        called when the exception is converted to a string, printed, or logged.

        The format is designed to be:

        - **Compact**: All information on a single line for log parsing
        - **Informative**: Includes schema identifier and all error details
        - **Consistent**: Same format regardless of error count

        The output format is::

            Output validation failed for {schema_name}@{schema_version}: {error1}; {error2}; ...

        Returns
        -------
        str
            A formatted string describing the validation failure. The string
            contains:

            - A fixed prefix "Output validation failed for "
            - The schema identifier in "name@version" format
            - A colon separator
            - All error messages joined by "; " (semicolon + space)

            The returned string is suitable for logging, display to users,
            and inclusion in exception chains.

        Examples
        --------
        Single error - basic formatting:

            >>> error = OutputValidationError(
            ...     schema_name="ProbeResult",
            ...     schema_version="1.0.0",
            ...     errors=["Missing field: input"]
            ... )
            >>> str(error)
            'Output validation failed for ProbeResult@1.0.0: Missing field: input'
            >>> print(error)
            Output validation failed for ProbeResult@1.0.0: Missing field: input

        Multiple errors - semicolon-joined:

            >>> error = OutputValidationError(
            ...     schema_name="ResultRecord",
            ...     schema_version="1.0.1",
            ...     errors=["Error 1", "Error 2", "Error 3"]
            ... )
            >>> str(error)
            'Output validation failed for ResultRecord@1.0.1: Error 1; Error 2; Error 3'

        Using in formatted strings:

            >>> error = OutputValidationError(
            ...     schema_name="RunManifest",
            ...     schema_version="1.0.0",
            ...     errors=["config: required", "timestamp: invalid format"]
            ... )
            >>> f"Caught exception: {error}"
            'Caught exception: Output validation failed for RunManifest@1.0.0: config: required; timestamp: invalid format'

        Parsing the schema identifier from the string:

            >>> error = OutputValidationError(
            ...     schema_name="HarnessRecord",
            ...     schema_version="1.0.1",
            ...     errors=["test error"]
            ... )
            >>> message = str(error)
            >>> # Extract schema@version from message
            >>> schema_part = message.split(": ")[0].split("for ")[1]
            >>> schema_part
            'HarnessRecord@1.0.1'

        Using repr() vs str() - different outputs:

            >>> error = OutputValidationError(
            ...     schema_name="ProbeResult",
            ...     schema_version="1.0.0",
            ...     errors=["field missing"]
            ... )
            >>> str(error)  # Human-readable message
            'Output validation failed for ProbeResult@1.0.0: field missing'
            >>> "OutputValidationError" in repr(error)  # Dataclass repr
            True

        Empty errors list (edge case):

            >>> error = OutputValidationError(
            ...     schema_name="ProbeResult",
            ...     schema_version="1.0.0",
            ...     errors=[]
            ... )
            >>> str(error)
            'Output validation failed for ProbeResult@1.0.0: '

        Long error messages - all included:

            >>> error = OutputValidationError(
            ...     schema_name="BenchmarkSummary",
            ...     schema_version="1.0.1",
            ...     errors=[
            ...         "Field 'probe_results' validation failed: expected list of ProbeResult objects",
            ...         "Field 'timing.total_duration' must be a positive float",
            ...         "Field 'metadata.version' does not match pattern '^[0-9]+\\.[0-9]+\\.[0-9]+$'"
            ...     ]
            ... )
            >>> len(str(error)) > 200
            True
            >>> str(error).count("; ")
            2

        Notes
        -----
        - This method is marked with ``# pragma: no cover`` because it's
          typically exercised through exception handling rather than direct
          unit tests.
        - The semicolon separator was chosen to avoid conflicts with common
          error message punctuation (commas, periods).
        - For structured logging, prefer accessing ``schema_name``,
          ``schema_version``, and ``errors`` directly rather than parsing
          this string.

        See Also
        --------
        __repr__ : Returns the dataclass representation (auto-generated).
        """
        return (
            f"Output validation failed for {self.schema_name}@{self.schema_version}: "
            + "; ".join(self.errors)
        )
