"""Compatibility shim for insideLLMs.trace.trace_config.

This module re-exports all public symbols from :mod:`insideLLMs.trace.trace_config`
to maintain backwards compatibility with code that imports directly from
``insideLLMs.trace_config``.

Overview
--------
The trace configuration system provides a strongly-typed configuration surface
for trace recording, fingerprinting, and contract validation. Configuration is
typically loaded from YAML files and compiled to internal validator formats.

This shim exists because the trace_config module was reorganized into the
``insideLLMs.trace`` subpackage. Existing code that imports from the old
location (``insideLLMs.trace_config``) will continue to work via this shim.

Key Components
--------------
The following classes and functions are re-exported from this module:

**Configuration Classes:**

- :class:`TraceConfig` : Top-level configuration dataclass
- :class:`TraceStoreConfig` : Storage behavior configuration
- :class:`TraceContractsConfig` : Contract validation settings
- :class:`FingerprintConfig` : Fingerprinting configuration
- :class:`NormaliserConfig` : Payload normalisation settings
- :class:`TraceRedactConfig` : Legacy redaction configuration (deprecated)

**Contract Configuration Classes:**

- :class:`GenerateBoundariesConfig` : Generate event boundary validation
- :class:`StreamBoundariesConfig` : Stream event boundary validation
- :class:`ToolResultsConfig` : Tool call/result pairing validation
- :class:`ToolPayloadsConfig` : Tool payload schema validation
- :class:`ToolPayloadSchemaConfig` : Individual tool schema definition
- :class:`ToolOrderConfig` : Tool ordering constraint validation

**Enums:**

- :class:`OnViolationMode` : Behavior when contracts are violated
- :class:`StoreMode` : How much trace data to persist
- :class:`NormaliserKind` : Type of normaliser (builtin or import)

**Functions:**

- :func:`load_trace_config` : Load configuration from YAML dictionary
- :func:`validate_with_config` : Run validators using config toggles
- :func:`make_structural_v1_normaliser` : Factory for default normaliser

**Classes:**

- :class:`TracePayloadNormaliser` : Event-kind-aware payload normaliser

**Type Aliases:**

- ``TracePayloadNormaliserFunc`` : Callable type for normaliser functions

Examples
--------
Loading configuration from a dictionary (typically parsed from YAML):

    >>> from insideLLMs.trace_config import load_trace_config
    >>> config = load_trace_config({
    ...     "version": 1,
    ...     "enabled": True,
    ...     "store": {"mode": "full"},
    ...     "contracts": {
    ...         "enabled": True,
    ...         "fail_fast": False,
    ...     },
    ... })
    >>> config.enabled
    True
    >>> config.store.mode
    <StoreMode.FULL: 'full'>

Using the preferred import path (recommended for new code):

    >>> from insideLLMs.trace.trace_config import (
    ...     TraceConfig,
    ...     load_trace_config,
    ...     validate_with_config,
    ... )

Creating a custom normaliser for API-specific noise:

    >>> from insideLLMs.trace_config import TracePayloadNormaliser
    >>> normaliser = TracePayloadNormaliser(
    ...     drop_keys=["request_id", "trace_id", "x-amzn-requestid"],
    ...     drop_key_regex=[r"^x-.*", r".*_at$"],
    ...     hash_paths=["response.body", "result.data"],
    ...     hash_strings_over=256,
    ... )
    >>> payload = {"request_id": "abc123", "data": "important"}
    >>> normalised = normaliser.normalise(payload, kind="api_response")
    >>> "request_id" in normalised
    False

Compiling configuration to validator inputs:

    >>> from insideLLMs.trace_config import load_trace_config
    >>> config = load_trace_config({
    ...     "contracts": {
    ...         "tool_payloads": {
    ...             "enabled": True,
    ...             "tools": {
    ...                 "search": {
    ...                     "args_schema": {
    ...                         "type": "object",
    ...                         "properties": {"query": {"type": "string"}},
    ...                         "required": ["query"]
    ...                     }
    ...                 }
    ...             }
    ...         }
    ...     }
    ... })
    >>> compiled = config.to_contracts()
    >>> "search" in compiled["tool_schemas"]
    True

Running validation with configuration:

    >>> from insideLLMs.trace_config import load_trace_config, validate_with_config
    >>> config = load_trace_config({
    ...     "contracts": {
    ...         "stream_boundaries": {"enabled": False},
    ...         "tool_results": {"enabled": True},
    ...     }
    ... })
    >>> events = [...]  # Your trace events
    >>> violations = validate_with_config(events, config)

Configuration for CI environments with compact storage:

    >>> config = load_trace_config({
    ...     "store": {
    ...         "mode": "compact",
    ...         "max_events": 1000,
    ...     },
    ...     "contracts": {
    ...         "fail_fast": True,
    ...     },
    ...     "on_violation": {
    ...         "mode": "fail_probe",
    ...     },
    ... })
    >>> config.store.mode
    <StoreMode.COMPACT: 'compact'>
    >>> config.on_violation.mode
    <OnViolationMode.FAIL_PROBE: 'fail_probe'>

Notes
-----
**Migration Guidance:**

New code should import directly from ``insideLLMs.trace.trace_config``:

    from insideLLMs.trace.trace_config import TraceConfig, load_trace_config

Existing code using this module will continue to work indefinitely.

**Thread Safety:**

The configuration classes are dataclasses and are not inherently thread-safe.
Create separate config instances for concurrent use, or treat them as immutable
after construction.

**Version Compatibility:**

The ``version`` field in TraceConfig indicates the schema version. When the
configuration semantics change (not just new optional fields), the version
is bumped to allow for migration logic.

See Also
--------
insideLLMs.trace.trace_config : The canonical module location (preferred import)
insideLLMs.trace.trace_contracts : Contract validation functions
insideLLMs.trace.tracing : Trace recording utilities

References
----------
.. [1] JSON Schema Draft-07 specification for tool payload validation:
       https://json-schema.org/draft-07/json-schema-release-notes.html
.. [2] RFC 6901 JSON Pointer (used for legacy redaction):
       https://tools.ietf.org/html/rfc6901
"""

from insideLLMs.trace.trace_config import *  # noqa: F401,F403
