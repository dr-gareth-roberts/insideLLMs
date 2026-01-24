"""Trace configuration for deterministic CI enforcement.

This module provides the public configuration surface for trace recording
and contract validation. Configuration is loaded from YAML and compiled
to the internal validator formats.

The module supports several key functionalities:

1. **Configuration Loading**: Parse YAML configuration dictionaries into
   strongly-typed dataclass objects via :func:`load_trace_config`.

2. **Payload Normalization**: Transform trace payloads before fingerprinting
   to strip noise (timestamps, request IDs) and create stable hashes via
   :class:`TracePayloadNormaliser`.

3. **Contract Compilation**: Convert configuration to validator inputs
   suitable for the trace_contracts module via :meth:`TraceConfig.to_contracts`.

4. **Validation Helper**: Run contract validators using config toggles via
   :func:`validate_with_config`.

Examples
--------
Basic configuration loading and validation:

    >>> from insideLLMs.trace.trace_config import load_trace_config, validate_with_config
    >>> config = load_trace_config({
    ...     "version": 1,
    ...     "enabled": True,
    ...     "store": {"mode": "full"},
    ...     "contracts": {"enabled": True, "fail_fast": False},
    ... })
    >>> config.enabled
    True
    >>> config.store.mode
    <StoreMode.FULL: 'full'>

Compiling configuration to validator inputs:

    >>> compiled = config.to_contracts()
    >>> compiled["toggles"]["generate_boundaries"]
    True
    >>> compiled["fail_fast"]
    False

Using a custom normaliser configuration:

    >>> config = load_trace_config({
    ...     "fingerprint": {
    ...         "normaliser": {
    ...             "config": {
    ...                 "drop_keys": ["timestamp", "request_id", "session_id"],
    ...                 "hash_strings_over": 256,
    ...             }
    ...         }
    ...     }
    ... })
    >>> config.fingerprint.normaliser.hash_strings_over
    256

Configuring tool payload validation:

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
    >>> "search" in config.contracts.tool_payloads.tools
    True

See Also
--------
insideLLMs.trace.trace_contracts : Contract validation functions.
insideLLMs.trace.tracing : Trace recording utilities.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

# Type alias for event-kind-aware normaliser function
TracePayloadNormaliserFunc = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class OnViolationMode(str, Enum):
    """Mode determining behavior when trace contracts are violated.

    This enum controls how the system responds when a trace contract
    violation is detected during execution.

    Attributes
    ----------
    RECORD : str
        Write violations to the trace output but continue execution.
        Useful for monitoring and debugging without disrupting runs.
    FAIL_PROBE : str
        Mark the current example/probe as failed and continue to the next.
        Good for CI where you want to test all examples but flag failures.
    FAIL_RUN : str
        Stop the entire harness immediately upon first violation.
        Suitable for fail-fast debugging scenarios.

    Examples
    --------
    Setting violation mode in configuration:

        >>> config = load_trace_config({
        ...     "on_violation": {"mode": "fail_probe"}
        ... })
        >>> config.on_violation.mode == OnViolationMode.FAIL_PROBE
        True

    Using in conditional logic:

        >>> mode = OnViolationMode.RECORD
        >>> if mode == OnViolationMode.RECORD:
        ...     print("Violations will be logged but execution continues")
        Violations will be logged but execution continues
    """

    RECORD = "record"  # Write violations, continue
    FAIL_PROBE = "fail_probe"  # Mark example as error
    FAIL_RUN = "fail_run"  # Stop harness early


class StoreMode(str, Enum):
    """Storage mode controlling how much trace data to persist.

    Controls the verbosity and size of stored trace data, allowing
    trade-offs between storage size and debugging capability.

    Attributes
    ----------
    NONE : str
        No storage. Traces are computed but not persisted.
        Useful for production where only real-time monitoring is needed.
    COMPACT : str
        Store only fingerprint hash and any violations.
        Good balance between storage efficiency and drift detection.
    FULL : str
        Store complete events including payloads, plus fingerprint.
        Best for development and debugging.
    OFF : str
        Alias for NONE (backwards compatibility).
    FINGERPRINT : str
        Alias for COMPACT (backwards compatibility).

    Examples
    --------
    Configuring storage mode:

        >>> config = load_trace_config({"store": {"mode": "compact"}})
        >>> config.store.mode == StoreMode.COMPACT
        True

    Using backwards-compatible mode names:

        >>> config = load_trace_config({"store": {"mode": "fingerprint"}})
        >>> config.store.mode == StoreMode.COMPACT
        True

    Checking if full storage is enabled:

        >>> config = load_trace_config({"store": {"mode": "full"}})
        >>> config.store.mode == StoreMode.FULL
        True
    """

    NONE = "none"  # No storage (renamed from OFF for clarity)
    COMPACT = "compact"  # Hash + violations only (renamed from FINGERPRINT)
    FULL = "full"  # Full events + fingerprint
    # Keep old names as aliases for backwards compatibility
    OFF = "none"
    FINGERPRINT = "compact"


class NormaliserKind(str, Enum):
    """Type of normaliser to use for payload transformation.

    Determines the source of the normalisation logic used to transform
    payloads before fingerprinting.

    Attributes
    ----------
    BUILTIN : str
        Use a library-provided normaliser. Available built-ins include
        'structural_v1' which handles common noise fields.
    IMPORT : str
        Import a user-defined normaliser function from a Python module.
        Specified via import_path in NormaliserConfig.

    Examples
    --------
    Using the default builtin normaliser:

        >>> config = load_trace_config({
        ...     "fingerprint": {"normaliser": {"kind": "builtin", "name": "structural_v1"}}
        ... })
        >>> config.fingerprint.normaliser.kind == NormaliserKind.BUILTIN
        True

    Configuring a custom imported normaliser:

        >>> config = load_trace_config({
        ...     "fingerprint": {
        ...         "normaliser": {
        ...             "kind": "import",
        ...             "import": "mypackage.normalisers:custom_normaliser"
        ...         }
        ...     }
        ... })
        >>> config.fingerprint.normaliser.kind == NormaliserKind.IMPORT
        True
    """

    BUILTIN = "builtin"  # Library-provided normaliser
    IMPORT = "import"  # User-imported function


# =============================================================================
# Normaliser Configuration
# =============================================================================


@dataclass
class NormaliserConfig:
    """Configuration for the payload normaliser.

    The normaliser transforms payloads before fingerprinting to strip
    noise (timestamps, request IDs, etc.) and create stable hashes.
    This ensures that trace fingerprints remain consistent across runs
    even when non-deterministic fields vary.

    Attributes
    ----------
    kind : NormaliserKind
        Type of normaliser: "builtin" uses library-provided logic,
        "import" loads a user-defined function.
    name : str
        Builtin normaliser name when kind=builtin. Default is "structural_v1".
    import_path : Optional[str]
        Python import path "pkg.module:callable" when kind=import.
    drop_keys : list[str]
        Exact key names to drop from payloads before fingerprinting.
    drop_key_regex : list[str]
        Regex patterns for keys to drop. Matched against all keys recursively.
    hash_paths : list[str]
        Dotpaths to hash instead of storing raw values. Useful for large
        payloads that should contribute to fingerprint but not be stored.
    hash_strings_over : int
        Strings longer than this are hashed instead of stored verbatim.

    Examples
    --------
    Default configuration:

        >>> cfg = NormaliserConfig()
        >>> cfg.kind == NormaliserKind.BUILTIN
        True
        >>> "timestamp" in cfg.drop_keys
        True

    Custom drop keys for API-specific noise:

        >>> cfg = NormaliserConfig(
        ...     drop_keys=["request_id", "trace_id", "span_id"],
        ...     drop_key_regex=[r"^x-.*", r".*_at$"]
        ... )
        >>> len(cfg.drop_key_regex)
        2

    Configuring hash paths for large response bodies:

        >>> cfg = NormaliserConfig(
        ...     hash_paths=["result", "raw", "response.body"],
        ...     hash_strings_over=256
        ... )
        >>> "response.body" in cfg.hash_paths
        True

    See Also
    --------
    TracePayloadNormaliser : The normaliser implementation.
    make_structural_v1_normaliser : Factory for the default normaliser.
    """

    kind: NormaliserKind = NormaliserKind.BUILTIN
    name: str = "structural_v1"
    import_path: Optional[str] = None
    drop_keys: list[str] = field(
        default_factory=lambda: ["request_id", "response_id", "created", "timestamp", "latency_ms"]
    )
    drop_key_regex: list[str] = field(default_factory=lambda: [])
    hash_paths: list[str] = field(default_factory=lambda: ["result", "raw"])
    hash_strings_over: int = 512


@dataclass
class FingerprintConfig:
    """Configuration for trace fingerprinting.

    Trace fingerprints are deterministic hashes computed over normalised
    event payloads. They enable detection of behavioral drift between
    runs without storing full trace data.

    Attributes
    ----------
    enabled : bool
        Whether to compute fingerprints. When False, no fingerprint
        is generated and drift detection is disabled.
    algorithm : str
        Hash algorithm for fingerprinting. Currently only "sha256" is
        supported, providing 256-bit collision-resistant hashes.
    normaliser : NormaliserConfig
        Configuration for payload normalisation before hashing.

    Examples
    --------
    Default fingerprinting configuration:

        >>> cfg = FingerprintConfig()
        >>> cfg.enabled
        True
        >>> cfg.algorithm
        'sha256'

    Disabling fingerprinting:

        >>> cfg = FingerprintConfig(enabled=False)
        >>> cfg.enabled
        False

    Custom normaliser configuration:

        >>> cfg = FingerprintConfig(
        ...     normaliser=NormaliserConfig(
        ...         drop_keys=["timestamp"],
        ...         hash_strings_over=128
        ...     )
        ... )
        >>> cfg.normaliser.hash_strings_over
        128

    See Also
    --------
    tracing.trace_fingerprint : Function that computes fingerprints.
    """

    enabled: bool = True
    algorithm: str = "sha256"
    normaliser: NormaliserConfig = field(default_factory=NormaliserConfig)


# =============================================================================
# Store Configuration (backwards compatible)
# =============================================================================


@dataclass
class TraceRedactConfig:
    """Configuration for payload redaction (legacy, use normaliser instead).

    This configuration is deprecated in favor of NormaliserConfig.
    It provides backwards compatibility for existing configurations
    that use JSON pointer-based redaction.

    Attributes
    ----------
    enabled : bool
        Whether legacy redaction is enabled.
    json_pointers : list[str]
        JSON pointers (RFC 6901) specifying paths to redact.
        Example: "/response/content" redacts data["response"]["content"].
    replacement : str
        String to replace redacted values with.

    Examples
    --------
    Legacy redaction (deprecated):

        >>> cfg = TraceRedactConfig(
        ...     enabled=True,
        ...     json_pointers=["/api_key", "/response/raw"],
        ...     replacement="[REDACTED]"
        ... )
        >>> cfg.enabled
        True

    Note
    ----
    Prefer using NormaliserConfig.drop_keys or NormaliserConfig.hash_paths
    for new configurations.
    """

    enabled: bool = False
    json_pointers: list[str] = field(default_factory=list)
    replacement: str = "<redacted>"


@dataclass
class TraceStoreConfig:
    """Configuration for trace storage behavior.

    Controls how trace data is persisted, including limits on storage
    size and whether payloads are included.

    Attributes
    ----------
    mode : StoreMode
        Storage mode controlling verbosity. Options are:
        - StoreMode.NONE: No storage
        - StoreMode.COMPACT: Fingerprint + violations only
        - StoreMode.FULL: Complete events with payloads
    max_events : Optional[int]
        Maximum number of events to store. None means unlimited.
        Oldest events are dropped when limit is exceeded.
    max_event_payload_bytes : Optional[int]
        Maximum size per event payload in bytes. Larger payloads
        are truncated with a marker indicating truncation.
    include_payloads : bool
        Whether to include event payloads in storage. When False,
        only event metadata (kind, seq) is stored.
    redact : TraceRedactConfig
        Legacy redaction config. Prefer using fingerprint.normaliser.

    Examples
    --------
    Full storage with default settings:

        >>> cfg = TraceStoreConfig()
        >>> cfg.mode == StoreMode.FULL
        True
        >>> cfg.include_payloads
        True

    Compact storage for CI environments:

        >>> cfg = TraceStoreConfig(
        ...     mode=StoreMode.COMPACT,
        ...     max_events=1000
        ... )
        >>> cfg.max_events
        1000

    Limited payload storage:

        >>> cfg = TraceStoreConfig(
        ...     mode=StoreMode.FULL,
        ...     max_event_payload_bytes=4096,
        ...     max_events=500
        ... )
        >>> cfg.max_event_payload_bytes
        4096

    No storage (metrics only):

        >>> cfg = TraceStoreConfig(mode=StoreMode.NONE)
        >>> cfg.mode == StoreMode.NONE
        True
    """

    mode: StoreMode = StoreMode.FULL
    max_events: Optional[int] = None
    max_event_payload_bytes: Optional[int] = None
    include_payloads: bool = True
    redact: TraceRedactConfig = field(default_factory=TraceRedactConfig)


# =============================================================================
# Contracts Configuration
# =============================================================================


@dataclass
class GenerateBoundariesConfig:
    """Configuration for generate event boundary validation.

    Controls validation of generate_start/generate_end event pairs
    to ensure proper nesting and matching.

    Attributes
    ----------
    enabled : bool
        Whether to validate generate boundaries.
    start_kind : str
        Event kind name for generation start events.
    end_kind : str
        Event kind name for generation end events.

    Examples
    --------
    Default configuration:

        >>> cfg = GenerateBoundariesConfig()
        >>> cfg.enabled
        True
        >>> cfg.start_kind
        'generate_start'

    Custom event kinds for different providers:

        >>> cfg = GenerateBoundariesConfig(
        ...     start_kind="completion_start",
        ...     end_kind="completion_end"
        ... )
        >>> cfg.start_kind
        'completion_start'

    Disabling generate boundary validation:

        >>> cfg = GenerateBoundariesConfig(enabled=False)
        >>> cfg.enabled
        False
    """

    enabled: bool = True
    start_kind: str = "generate_start"
    end_kind: str = "generate_end"


@dataclass
class StreamBoundariesConfig:
    """Configuration for stream event boundary validation.

    Controls validation of streaming sequences including start/chunk/end
    ordering, chunk index continuity, and proper stream termination.

    Attributes
    ----------
    enabled : bool
        Whether to validate stream boundaries.
    start_kind : str
        Event kind name for stream start events.
    chunk_kind : str
        Event kind name for stream chunk events.
    end_kind : str
        Event kind name for stream end events.
    stream_id_key : str
        Payload key containing the stream identifier for multi-stream support.
    chunk_index_key : str
        Payload key containing the chunk sequence number.
    first_chunk_index : int
        Expected index of the first chunk (typically 0).
    require_end : bool
        Whether every stream_start must have a matching stream_end.
    require_monotonic_chunks : bool
        Whether chunk indices must be strictly increasing.

    Examples
    --------
    Default configuration:

        >>> cfg = StreamBoundariesConfig()
        >>> cfg.enabled
        True
        >>> cfg.require_end
        True

    Relaxed validation for incomplete streams:

        >>> cfg = StreamBoundariesConfig(
        ...     require_end=False,
        ...     require_monotonic_chunks=False
        ... )
        >>> cfg.require_end
        False

    Custom event and payload key names:

        >>> cfg = StreamBoundariesConfig(
        ...     start_kind="sse_open",
        ...     chunk_kind="sse_message",
        ...     end_kind="sse_close",
        ...     chunk_index_key="message_index"
        ... )
        >>> cfg.chunk_kind
        'sse_message'

    One-indexed chunk sequences:

        >>> cfg = StreamBoundariesConfig(first_chunk_index=1)
        >>> cfg.first_chunk_index
        1
    """

    enabled: bool = True
    start_kind: str = "stream_start"
    chunk_kind: str = "stream_chunk"
    end_kind: str = "stream_end"
    stream_id_key: str = "stream_id"
    chunk_index_key: str = "chunk_index"
    first_chunk_index: int = 0
    require_end: bool = True
    require_monotonic_chunks: bool = True


@dataclass
class ToolResultsConfig:
    """Configuration for tool call/result pairing validation.

    Controls validation that ensures every tool call has a corresponding
    result and results don't appear without prior calls.

    Attributes
    ----------
    enabled : bool
        Whether to validate tool call/result pairing.
    call_start_kind : str
        Event kind name for tool call start events.
    call_result_kind : str
        Event kind name for tool result events.
    call_id_key : str
        Payload key containing the tool call identifier for matching.
    require_exactly_one_result : bool
        Whether each tool call must have exactly one result (not zero,
        not multiple).

    Examples
    --------
    Default configuration:

        >>> cfg = ToolResultsConfig()
        >>> cfg.enabled
        True
        >>> cfg.require_exactly_one_result
        True

    Custom event kind names:

        >>> cfg = ToolResultsConfig(
        ...     call_start_kind="function_invoke",
        ...     call_result_kind="function_return"
        ... )
        >>> cfg.call_start_kind
        'function_invoke'

    Allow calls without results (fire-and-forget):

        >>> cfg = ToolResultsConfig(require_exactly_one_result=False)
        >>> cfg.require_exactly_one_result
        False
    """

    enabled: bool = True
    call_start_kind: str = "tool_call_start"
    call_result_kind: str = "tool_result"
    call_id_key: str = "call_id"
    require_exactly_one_result: bool = True


@dataclass
class ToolPayloadSchemaConfig:
    """JSON Schema for validating a tool's arguments.

    Defines the expected structure of arguments for a specific tool,
    enabling type checking and required field validation.

    Attributes
    ----------
    args_schema : dict[str, Any]
        JSON Schema (draft-07 compatible) defining the tool's arguments.
        Supports "type", "properties", "required", and other standard
        JSON Schema keywords.
    required : bool
        Whether this tool must be called with valid arguments. When True,
        invalid arguments generate violations.

    Examples
    --------
    Simple schema with required string argument:

        >>> cfg = ToolPayloadSchemaConfig(
        ...     args_schema={
        ...         "type": "object",
        ...         "properties": {"query": {"type": "string"}},
        ...         "required": ["query"]
        ...     }
        ... )
        >>> "query" in cfg.args_schema["required"]
        True

    Schema with multiple typed arguments:

        >>> cfg = ToolPayloadSchemaConfig(
        ...     args_schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "limit": {"type": "integer"},
        ...             "filters": {"type": "object"},
        ...             "include_metadata": {"type": "boolean"}
        ...         },
        ...         "required": ["limit"]
        ...     }
        ... )
        >>> cfg.args_schema["properties"]["limit"]["type"]
        'integer'

    Optional schema (warnings only):

        >>> cfg = ToolPayloadSchemaConfig(
        ...     args_schema={"type": "object"},
        ...     required=False
        ... )
        >>> cfg.required
        False
    """

    args_schema: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class ToolPayloadsConfig:
    """Configuration for validating tool call payloads against schemas.

    Enables schema-based validation of tool arguments to catch malformed
    or incorrectly typed tool calls.

    Attributes
    ----------
    enabled : bool
        Whether to validate tool payloads.
    tool_key : str
        Payload key containing the tool name.
    args_key : str
        Payload key containing the tool arguments.
    tools : dict[str, ToolPayloadSchemaConfig]
        Mapping of tool names to their validation schemas.

    Examples
    --------
    Default configuration (no schemas):

        >>> cfg = ToolPayloadsConfig()
        >>> cfg.enabled
        True
        >>> len(cfg.tools)
        0

    Configuring schemas for multiple tools:

        >>> cfg = ToolPayloadsConfig(
        ...     tools={
        ...         "search": ToolPayloadSchemaConfig(
        ...             args_schema={
        ...                 "type": "object",
        ...                 "properties": {"query": {"type": "string"}},
        ...                 "required": ["query"]
        ...             }
        ...         ),
        ...         "calculate": ToolPayloadSchemaConfig(
        ...             args_schema={
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "expression": {"type": "string"}
        ...                 },
        ...                 "required": ["expression"]
        ...             }
        ...         )
        ...     }
        ... )
        >>> "search" in cfg.tools
        True
        >>> "calculate" in cfg.tools
        True

    Custom payload key names:

        >>> cfg = ToolPayloadsConfig(
        ...     tool_key="function_name",
        ...     args_key="parameters"
        ... )
        >>> cfg.tool_key
        'function_name'
    """

    enabled: bool = True
    tool_key: str = "tool"
    args_key: str = "args"
    tools: dict[str, ToolPayloadSchemaConfig] = field(default_factory=dict)


@dataclass
class ToolOrderConfig:
    """Configuration for validating tool call ordering constraints.

    Enables enforcement of tool execution order, ensuring certain tools
    are called before/after others or that specific sequences are avoided.

    Attributes
    ----------
    enabled : bool
        Whether to validate tool ordering.
    allow_unknown_tools : bool
        Whether to allow tools not mentioned in ordering rules.
        When False, unknown tools generate violations.
    must_follow : dict[str, list[str]]
        Mapping of tool names to tools that must be called before them.
        Example: {"process": ["fetch"]} means "fetch" must precede "process".
    must_precede : dict[str, list[str]]
        Mapping of tool names to tools that must be called after them.
        Example: {"init": ["run"]} means "init" must precede "run".
    forbidden_sequences : list[list[str]]
        List of tool sequences that are not allowed.
        Example: [["delete", "delete"]] forbids consecutive deletes.

    Examples
    --------
    Default configuration (no ordering rules):

        >>> cfg = ToolOrderConfig()
        >>> cfg.enabled
        True
        >>> len(cfg.must_follow)
        0

    Requiring tool ordering:

        >>> cfg = ToolOrderConfig(
        ...     must_follow={"process_data": ["load_data"]},
        ...     must_precede={"init_session": ["run_query"]}
        ... )
        >>> cfg.must_follow["process_data"]
        ['load_data']

    Forbidding dangerous sequences:

        >>> cfg = ToolOrderConfig(
        ...     forbidden_sequences=[
        ...         ["delete_all", "delete_all"],  # No double delete
        ...         ["write", "write", "write"]    # Max 2 consecutive writes
        ...     ]
        ... )
        >>> len(cfg.forbidden_sequences)
        2

    Strict mode (reject unknown tools):

        >>> cfg = ToolOrderConfig(
        ...     allow_unknown_tools=False,
        ...     must_follow={"save": ["validate"]}
        ... )
        >>> cfg.allow_unknown_tools
        False
    """

    enabled: bool = True
    allow_unknown_tools: bool = False
    must_follow: dict[str, list[str]] = field(default_factory=dict)
    must_precede: dict[str, list[str]] = field(default_factory=dict)
    forbidden_sequences: list[list[str]] = field(default_factory=list)


@dataclass
class TraceContractsConfig:
    """Configuration for all trace contract validators.

    Master configuration controlling which validators run and their
    individual settings. Each validator can be independently enabled
    or disabled.

    Attributes
    ----------
    enabled : bool
        Master switch for all contract validation. When False, no
        validators run regardless of their individual settings.
    fail_fast : bool
        Stop validation on first violation. Faster but provides less
        information. Useful for quick CI checks.
    generate_boundaries : GenerateBoundariesConfig
        Configuration for generate_start/generate_end validation.
    stream_boundaries : StreamBoundariesConfig
        Configuration for stream event sequence validation.
    tool_results : ToolResultsConfig
        Configuration for tool call/result pairing validation.
    tool_payloads : ToolPayloadsConfig
        Configuration for tool argument schema validation.
    tool_order : ToolOrderConfig
        Configuration for tool execution order validation.

    Examples
    --------
    Default configuration (all validators enabled):

        >>> cfg = TraceContractsConfig()
        >>> cfg.enabled
        True
        >>> cfg.generate_boundaries.enabled
        True

    Disable all contract validation:

        >>> cfg = TraceContractsConfig(enabled=False)
        >>> cfg.enabled
        False

    Fail-fast mode for CI:

        >>> cfg = TraceContractsConfig(fail_fast=True)
        >>> cfg.fail_fast
        True

    Selective validation (only stream and tool results):

        >>> cfg = TraceContractsConfig(
        ...     generate_boundaries=GenerateBoundariesConfig(enabled=False),
        ...     tool_payloads=ToolPayloadsConfig(enabled=False),
        ...     tool_order=ToolOrderConfig(enabled=False)
        ... )
        >>> cfg.stream_boundaries.enabled
        True
        >>> cfg.generate_boundaries.enabled
        False

    See Also
    --------
    validate_with_config : Run validators using this configuration.
    trace_contracts : Individual validator implementations.
    """

    enabled: bool = True
    fail_fast: bool = False
    generate_boundaries: GenerateBoundariesConfig = field(default_factory=GenerateBoundariesConfig)
    stream_boundaries: StreamBoundariesConfig = field(default_factory=StreamBoundariesConfig)
    tool_results: ToolResultsConfig = field(default_factory=ToolResultsConfig)
    tool_payloads: ToolPayloadsConfig = field(default_factory=ToolPayloadsConfig)
    tool_order: ToolOrderConfig = field(default_factory=ToolOrderConfig)


@dataclass
class OnViolationConfig:
    """Configuration for handling contract violations.

    Determines system behavior when trace contract violations are detected.

    Attributes
    ----------
    mode : OnViolationMode
        How to handle violations:
        - RECORD: Log violations and continue
        - FAIL_PROBE: Mark current example as failed, continue to next
        - FAIL_RUN: Stop the entire run immediately

    Examples
    --------
    Default (record and continue):

        >>> cfg = OnViolationConfig()
        >>> cfg.mode == OnViolationMode.RECORD
        True

    Fail on first violation:

        >>> cfg = OnViolationConfig(mode=OnViolationMode.FAIL_RUN)
        >>> cfg.mode == OnViolationMode.FAIL_RUN
        True

    Per-probe failure for test suites:

        >>> cfg = OnViolationConfig(mode=OnViolationMode.FAIL_PROBE)
        >>> cfg.mode == OnViolationMode.FAIL_PROBE
        True
    """

    mode: OnViolationMode = OnViolationMode.RECORD


# =============================================================================
# Top-Level TraceConfig
# =============================================================================


@dataclass
class TraceConfig:
    """Top-level trace configuration for recording and validation.

    This is the primary public interface for configuring trace behavior.
    Load from YAML dictionaries with :func:`load_trace_config`, then
    compile to validator inputs with :meth:`to_contracts`.

    The configuration supports:

    - **Storage control**: What and how much trace data to persist
    - **Fingerprinting**: Deterministic hashing for drift detection
    - **Contract validation**: Structural and semantic trace validation
    - **Violation handling**: How to respond to detected issues

    Attributes
    ----------
    version : int
        Configuration schema version. Bump when semantics change to
        ensure backwards compatibility detection.
    enabled : bool
        Master switch for all tracing functionality. When False,
        no tracing occurs regardless of other settings.
    store : TraceStoreConfig
        Configuration for trace data persistence.
    fingerprint : FingerprintConfig
        Configuration for deterministic fingerprint computation.
    contracts : TraceContractsConfig
        Configuration for structural and semantic validation.
    on_violation : OnViolationConfig
        Configuration for violation handling behavior.

    Examples
    --------
    Default configuration:

        >>> cfg = TraceConfig()
        >>> cfg.enabled
        True
        >>> cfg.version
        1

    Creating from YAML dictionary:

        >>> from insideLLMs.trace.trace_config import load_trace_config
        >>> cfg = load_trace_config({
        ...     "version": 1,
        ...     "enabled": True,
        ...     "store": {"mode": "full"},
        ...     "contracts": {"fail_fast": True}
        ... })
        >>> cfg.contracts.fail_fast
        True

    Compiling to validator inputs:

        >>> compiled = cfg.to_contracts()
        >>> "toggles" in compiled
        True
        >>> "tool_schemas" in compiled
        True

    Complete CI configuration example:

        >>> cfg = TraceConfig(
        ...     store=TraceStoreConfig(mode=StoreMode.COMPACT, max_events=1000),
        ...     contracts=TraceContractsConfig(fail_fast=True),
        ...     on_violation=OnViolationConfig(mode=OnViolationMode.FAIL_PROBE)
        ... )
        >>> cfg.store.mode == StoreMode.COMPACT
        True
        >>> cfg.on_violation.mode == OnViolationMode.FAIL_PROBE
        True

    See Also
    --------
    load_trace_config : Create TraceConfig from YAML dictionary.
    validate_with_config : Run validators using TraceConfig.
    """

    version: int = 1
    enabled: bool = True
    store: TraceStoreConfig = field(default_factory=TraceStoreConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    contracts: TraceContractsConfig = field(default_factory=TraceContractsConfig)
    on_violation: OnViolationConfig = field(default_factory=OnViolationConfig)

    def to_contracts(self) -> dict[str, Any]:
        """Compile configuration to validator inputs.

        Transforms the high-level TraceConfig into the data structures
        required by the trace_contracts validation functions.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - **tool_schemas** (dict[str, ToolSchema]): Compiled tool schemas
              for use with validate_tool_payloads().
            - **tool_order_rules** (Optional[ToolOrderRule]): Compiled ordering
              rules for use with validate_tool_order(), or None if no rules.
            - **toggles** (dict[str, bool]): Boolean flags for each validator
              indicating whether it should run.
            - **fail_fast** (bool): Whether to stop on first violation.

        Examples
        --------
        Compile default configuration:

            >>> cfg = TraceConfig()
            >>> compiled = cfg.to_contracts()
            >>> compiled["toggles"]["generate_boundaries"]
            True
            >>> compiled["fail_fast"]
            False

        Compile with tool schemas:

            >>> cfg = load_trace_config({
            ...     "contracts": {
            ...         "tool_payloads": {
            ...             "tools": {
            ...                 "search": {
            ...                     "args_schema": {
            ...                         "properties": {"q": {"type": "string"}},
            ...                         "required": ["q"]
            ...                     }
            ...                 }
            ...             }
            ...         }
            ...     }
            ... })
            >>> compiled = cfg.to_contracts()
            >>> "search" in compiled["tool_schemas"]
            True

        Disabled contracts return empty:

            >>> cfg = TraceConfig(contracts=TraceContractsConfig(enabled=False))
            >>> compiled = cfg.to_contracts()
            >>> compiled["toggles"]["generate_boundaries"]
            False
            >>> len(compiled["tool_schemas"])
            0
        """
        from insideLLMs.trace_contracts import ToolOrderRule, ToolSchema

        # If contracts disabled, return empty
        if not self.contracts.enabled:
            return {
                "tool_schemas": {},
                "tool_order_rules": None,
                "toggles": {
                    "generate_boundaries": False,
                    "stream_boundaries": False,
                    "tool_results": False,
                    "tool_payloads": False,
                    "tool_order": False,
                },
                "fail_fast": False,
            }

        # Compile tool schemas from JSON Schema to ToolSchema
        tool_schemas: dict[str, "ToolSchema"] = {}
        for tool_name, schema_cfg in self.contracts.tool_payloads.tools.items():
            args_schema = schema_cfg.args_schema
            required_args = args_schema.get("required", [])
            properties = args_schema.get("properties", {})

            # Map JSON Schema types to Python types (for isinstance checks).
            arg_types: dict[str, type | tuple[type, ...]] = {}
            for arg_name, prop in properties.items():
                json_type = prop.get("type")
                if json_type == "string":
                    arg_types[arg_name] = str
                elif json_type == "integer":
                    arg_types[arg_name] = int
                elif json_type == "number":
                    # JSON Schema "number" includes integers as well.
                    arg_types[arg_name] = (int, float)
                elif json_type == "boolean":
                    arg_types[arg_name] = bool
                elif json_type == "array":
                    arg_types[arg_name] = list
                elif json_type == "object":
                    arg_types[arg_name] = dict

            # Determine optional args (in properties but not required)
            optional_args = [name for name in properties if name not in required_args]

            tool_schemas[tool_name] = ToolSchema(
                name=tool_name,
                required_args=required_args,
                arg_types=arg_types,
                optional_args=optional_args,
            )

        # Compile tool order rules
        tool_order_cfg = self.contracts.tool_order
        tool_order_rules: Optional["ToolOrderRule"] = None
        if tool_order_cfg.enabled and (
            tool_order_cfg.must_follow
            or tool_order_cfg.must_precede
            or tool_order_cfg.forbidden_sequences
        ):
            tool_order_rules = ToolOrderRule(
                name="trace_config_rules",
                must_precede=tool_order_cfg.must_precede,
                must_follow=tool_order_cfg.must_follow,
                forbidden_sequences=tool_order_cfg.forbidden_sequences,
            )

        # Build toggles
        toggles = {
            "generate_boundaries": self.contracts.generate_boundaries.enabled,
            "stream_boundaries": self.contracts.stream_boundaries.enabled,
            "tool_results": self.contracts.tool_results.enabled,
            "tool_payloads": self.contracts.tool_payloads.enabled,
            "tool_order": self.contracts.tool_order.enabled,
        }

        return {
            "tool_schemas": tool_schemas,
            "tool_order_rules": tool_order_rules,
            "toggles": toggles,
            "fail_fast": self.contracts.fail_fast,
        }


# =============================================================================
# TracePayloadNormaliser (Event-Kind-Aware)
# =============================================================================


def _stable_hash(obj: Any) -> dict[str, Any]:
    """Create a stable hash summary of an object.

    Computes a deterministic SHA-256 hash of any JSON-serializable object.
    The hash is computed over canonical JSON (sorted keys, minimal whitespace)
    to ensure consistency across runs.

    Parameters
    ----------
    obj : Any
        Any JSON-serializable object (dict, list, str, int, float, bool, None).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "sha256": Hexadecimal hash string (64 characters)
        - "len": Original serialized length in characters

    Examples
    --------
    Hash a simple string:

        >>> result = _stable_hash("hello world")
        >>> result["sha256"][:16]  # First 16 chars of hash
        'b5ce89....'  # (actual hash varies)
        >>> result["len"]
        13

    Hash a complex object:

        >>> result = _stable_hash({"key": "value", "nested": [1, 2, 3]})
        >>> "sha256" in result
        True
        >>> result["len"] > 0
        True

    Stability across key ordering:

        >>> h1 = _stable_hash({"a": 1, "b": 2})
        >>> h2 = _stable_hash({"b": 2, "a": 1})
        >>> h1["sha256"] == h2["sha256"]
        True
    """
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return {"sha256": h, "len": len(data)}


class TracePayloadNormaliser:
    """Event-kind-aware normaliser for trace payloads.

    Transforms payloads before fingerprinting to strip noise and create
    stable, deterministic hashes. This ensures trace fingerprints remain
    consistent across runs even when non-deterministic fields vary.

    The normaliser supports several transformation strategies:

    - **Key dropping**: Remove specific keys by exact name or regex pattern
    - **Path hashing**: Replace values at specific dotpaths with their hashes
    - **String hashing**: Auto-hash strings exceeding a length threshold
    - **Event-kind awareness**: Special handling for specific event types
      (e.g., stream chunks have text content summarized to length only)

    Parameters
    ----------
    drop_keys : Optional[list[str]]
        Exact key names to remove from payloads recursively.
    drop_key_regex : Optional[list[str]]
        Regex patterns for keys to remove. Applied to all keys recursively.
    hash_paths : Optional[list[str]]
        Dotpaths where values should be replaced with their hash.
        Example: "result" matches top-level "result" key.
    hash_strings_over : int
        Strings longer than this are replaced with their hash.
    redact_enabled : bool
        Enable legacy JSON pointer-based redaction.
    json_pointers : Optional[list[str]]
        Legacy JSON pointers for redaction (RFC 6901).
    replacement : str
        Legacy replacement string for redacted values.

    Examples
    --------
    Basic normalisation with key dropping:

        >>> normaliser = TracePayloadNormaliser(
        ...     drop_keys=["timestamp", "request_id"],
        ...     hash_strings_over=256
        ... )
        >>> payload = {"data": "test", "timestamp": "2024-01-01T00:00:00Z"}
        >>> result = normaliser.normalise(payload)
        >>> "timestamp" in result
        False
        >>> result["data"]
        'test'

    Hashing large payloads at specific paths:

        >>> normaliser = TracePayloadNormaliser(
        ...     hash_paths=["response.body"],
        ...     hash_strings_over=50
        ... )
        >>> payload = {"response": {"body": "a" * 100}}
        >>> # The body would be hashed since it's at a hash_path

    Using regex patterns to drop headers:

        >>> normaliser = TracePayloadNormaliser(
        ...     drop_key_regex=[r"^x-.*", r".*_id$"]
        ... )
        >>> payload = {"x-request-id": "abc", "user_id": "123", "data": "test"}
        >>> result = normaliser.normalise(payload)
        >>> "x-request-id" in result
        False
        >>> "user_id" in result
        False
        >>> result["data"]
        'test'

    Stream chunk special handling:

        >>> normaliser = TracePayloadNormaliser()
        >>> payload = {"text": "Hello " * 100, "chunk_index": 0}
        >>> result = normaliser.normalise(payload, kind="stream_chunk")
        >>> "text" in result
        False
        >>> "text_len" in result
        True

    See Also
    --------
    TracePayloadNormaliser.from_config : Create from TraceConfig.
    make_structural_v1_normaliser : Factory for default normaliser.
    """

    def __init__(
        self,
        drop_keys: Optional[list[str]] = None,
        drop_key_regex: Optional[list[str]] = None,
        hash_paths: Optional[list[str]] = None,
        hash_strings_over: int = 512,
        # Legacy parameters for backwards compatibility
        redact_enabled: bool = False,
        json_pointers: Optional[list[str]] = None,
        replacement: str = "<redacted>",
    ):
        """Initialize the normaliser with transformation rules.

        Parameters
        ----------
        drop_keys : Optional[list[str]]
            Exact key names to remove from payloads. Keys are matched
            recursively at all levels of nested structures.
        drop_key_regex : Optional[list[str]]
            Regex patterns for keys to remove. Patterns are compiled
            once and matched against all keys during normalisation.
        hash_paths : Optional[list[str]]
            Dotpaths where values should be replaced with their hash.
            Example: "result" matches the top-level "result" key,
            "response.body" matches nested paths.
        hash_strings_over : int
            Strings longer than this threshold are automatically
            replaced with their hash. Default is 512 characters.
        redact_enabled : bool
            Enable legacy JSON pointer-based redaction. Deprecated.
        json_pointers : Optional[list[str]]
            Legacy JSON pointers (RFC 6901) for redaction. Deprecated.
        replacement : str
            Replacement string for legacy redaction. Deprecated.

        Examples
        --------
        Create with default settings:

            >>> normaliser = TracePayloadNormaliser()
            >>> normaliser._hash_strings_over
            512

        Create with custom drop keys:

            >>> normaliser = TracePayloadNormaliser(
            ...     drop_keys=["timestamp", "latency"],
            ...     hash_strings_over=100
            ... )
            >>> "timestamp" in normaliser._drop_keys
            True
        """
        self._drop_keys = set(drop_keys or [])
        self._drop_key_regex = [re.compile(r) for r in (drop_key_regex or [])]
        self._hash_paths = set(hash_paths or [])
        self._hash_strings_over = hash_strings_over

        # Legacy support
        self._redact_enabled = redact_enabled
        self._json_pointers = json_pointers or []
        self._replacement = replacement

    @classmethod
    def from_config(cls, config: "TraceConfig") -> "TracePayloadNormaliser":
        """Create a normaliser from a TraceConfig instance.

        Factory method that extracts normaliser settings from a complete
        trace configuration, including both modern normaliser config and
        legacy redaction settings.

        Parameters
        ----------
        config : TraceConfig
            The trace configuration containing normaliser settings.

        Returns
        -------
        TracePayloadNormaliser
            A normaliser instance configured according to the config.

        Examples
        --------
        Create from default config:

            >>> config = TraceConfig()
            >>> normaliser = TracePayloadNormaliser.from_config(config)
            >>> "timestamp" in normaliser._drop_keys
            True

        Create from custom config:

            >>> config = load_trace_config({
            ...     "fingerprint": {
            ...         "normaliser": {
            ...             "config": {
            ...                 "drop_keys": ["custom_field"],
            ...                 "hash_strings_over": 100
            ...             }
            ...         }
            ...     }
            ... })
            >>> normaliser = TracePayloadNormaliser.from_config(config)
            >>> normaliser._hash_strings_over
            100
        """
        norm_cfg = config.fingerprint.normaliser
        redact_cfg = config.store.redact

        return cls(
            drop_keys=norm_cfg.drop_keys,
            drop_key_regex=norm_cfg.drop_key_regex,
            hash_paths=norm_cfg.hash_paths,
            hash_strings_over=norm_cfg.hash_strings_over,
            # Legacy support
            redact_enabled=redact_cfg.enabled,
            json_pointers=redact_cfg.json_pointers,
            replacement=redact_cfg.replacement,
        )

    def _should_drop_key(self, key: str) -> bool:
        """Check if a key should be dropped from the payload.

        Parameters
        ----------
        key : str
            The key name to check.

        Returns
        -------
        bool
            True if the key should be dropped, False otherwise.

        Notes
        -----
        A key is dropped if:
        1. It matches an exact name in drop_keys, OR
        2. It matches any regex pattern in drop_key_regex
        """
        if key in self._drop_keys:
            return True
        return any(r.search(key) for r in self._drop_key_regex)

    def _walk(self, kind: str, obj: Any, path: str = "") -> Any:
        """Recursively walk and transform an object.

        Traverses the object tree, applying transformations:
        - Hash values at configured paths
        - Hash long strings
        - Drop keys matching drop rules
        - Recurse into nested structures

        Parameters
        ----------
        kind : str
            Event kind for context-specific handling (e.g., "stream_chunk").
        obj : Any
            The object to transform. Can be dict, list, or scalar.
        path : str
            Current dotpath within the payload for path-based hashing.

        Returns
        -------
        Any
            Transformed object with normalisation rules applied.

        Notes
        -----
        The transformation order is:
        1. Check if current path should be hashed (hash_paths)
        2. Check if value is a long string (hash_strings_over)
        3. Recurse into containers (dict, list)
        4. Drop keys matching drop rules
        """
        # Hash selected paths
        if path in self._hash_paths:
            return _stable_hash(obj)

        # Hash long strings
        if isinstance(obj, str) and len(obj) > self._hash_strings_over:
            return _stable_hash(obj)

        # Recurse into lists
        if isinstance(obj, list):
            return [self._walk(kind, v, path=path) for v in obj]

        # Recurse into dicts, dropping keys as configured
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if self._should_drop_key(k):
                    continue
                child_path = f"{path}.{k}" if path else k
                out[k] = self._walk(kind, v, path=child_path)
            return out

        return obj

    def _apply_legacy_redaction(self, obj: dict[str, Any], pointer: str) -> None:
        """Apply legacy JSON pointer redaction to an object.

        Navigates to the location specified by a JSON pointer (RFC 6901)
        and replaces the value with the configured replacement string.

        Parameters
        ----------
        obj : dict[str, Any]
            The object to modify in place.
        pointer : str
            JSON pointer string (e.g., "/response/content").

        Notes
        -----
        This is a deprecated feature. Prefer using drop_keys or hash_paths.
        The method handles JSON pointer escaping (~0 for ~, ~1 for /).
        """
        if not pointer.startswith("/"):
            return

        parts = pointer[1:].split("/")
        if not parts:
            return

        current: Any = obj
        for part in parts[:-1]:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return
                except ValueError:
                    return
            else:
                return

        final_key = parts[-1].replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and final_key in current:
            current[final_key] = self._replacement
        elif isinstance(current, list):
            try:
                idx = int(final_key)
                if 0 <= idx < len(current):
                    current[idx] = self._replacement
            except ValueError:
                pass

    def normalise(
        self,
        payload: dict[str, Any],
        kind: Optional[str] = None,
    ) -> dict[str, Any]:
        """Normalise a payload with dropping, hashing, and canonicalisation.

        Main entry point for payload normalisation. Applies all configured
        transformations and returns a deterministic, canonical form suitable
        for fingerprinting.

        Parameters
        ----------
        payload : dict[str, Any]
            The payload to normalise. A deep copy is made internally.
        kind : Optional[str]
            Event kind for context-specific handling. Some event kinds
            (e.g., "stream_chunk") receive special treatment.

        Returns
        -------
        dict[str, Any]
            Normalised payload with:
            - Configured keys dropped
            - Long strings hashed
            - Selected paths hashed
            - Keys sorted for canonical form
            - Event-kind-specific transformations applied

        Examples
        --------
        Basic normalisation:

            >>> normaliser = TracePayloadNormaliser(drop_keys=["timestamp"])
            >>> payload = {"data": "test", "timestamp": "2024-01-01"}
            >>> result = normaliser.normalise(payload)
            >>> "timestamp" in result
            False

        With event kind context:

            >>> normaliser = TracePayloadNormaliser()
            >>> payload = {"text": "Hello world", "index": 0}
            >>> result = normaliser.normalise(payload, kind="stream_chunk")
            >>> # stream_chunk events have text replaced with text_len

        Long string hashing:

            >>> normaliser = TracePayloadNormaliser(hash_strings_over=10)
            >>> payload = {"message": "This is a very long message"}
            >>> result = normaliser.normalise(payload)
            >>> "sha256" in str(result.get("message", {}))
            True
        """
        result = copy.deepcopy(payload)
        event_kind = kind or ""

        # Special handling for stream chunks
        if event_kind == "stream_chunk":
            # Keep structural info, summarise text content
            text = result.get("text") or result.get("chunk")
            if isinstance(text, str):
                result["text_len"] = len(text)
                result.pop("text", None)
                result.pop("chunk", None)

        # Apply legacy redaction if enabled
        if self._redact_enabled:
            for pointer in self._json_pointers:
                self._apply_legacy_redaction(result, pointer)

        # Apply new normalisation
        result = self._walk(event_kind, result)

        # Canonicalise via JSON round-trip with sorted keys
        canonical = json.dumps(result, sort_keys=True, ensure_ascii=False)
        return json.loads(canonical)

    def __call__(
        self,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Callable interface for use as TracePayloadNormaliserFunc.

        Allows the normaliser to be used as a function, matching the
        TracePayloadNormaliserFunc type signature.

        Parameters
        ----------
        kind : str
            Event kind for context-specific handling.
        payload : Mapping[str, Any]
            Event payload to normalise.

        Returns
        -------
        Mapping[str, Any]
            Normalised payload.

        Examples
        --------
        Using as a callable:

            >>> normaliser = TracePayloadNormaliser(drop_keys=["ts"])
            >>> result = normaliser("generate_end", {"response": "Hi", "ts": 123})
            >>> "ts" in result
            False

        Assigning to a function type:

            >>> from typing import Callable, Mapping, Any
            >>> func: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] = normaliser
            >>> result = func("custom", {"key": "value"})
        """
        return self.normalise(dict(payload), kind=kind)


def make_structural_v1_normaliser(
    *,
    drop_keys: Optional[list[str]] = None,
    drop_key_regex: Optional[list[str]] = None,
    hash_paths: Optional[list[str]] = None,
    hash_strings_over: int = 512,
) -> TracePayloadNormaliser:
    """Factory for the structural_v1 builtin normaliser.

    Creates a normaliser optimized for behavioral drift detection in LLM
    traces. The default configuration strips common noise fields that vary
    between runs but don't indicate behavioral changes.

    The structural_v1 normaliser:

    - **Drops noise keys**: Removes timestamps, request IDs, and similar
      fields that change between runs but don't affect behavior.
    - **Hashes large blobs**: Replaces tool results and raw provider payloads
      with their hashes to reduce storage while preserving change detection.
    - **Summarises streams**: Converts stream chunk text to length-only
      summaries, focusing on structure rather than content.

    Parameters
    ----------
    drop_keys : Optional[list[str]]
        Keys to drop from payloads. Defaults to common noise fields:
        ["request_id", "response_id", "created", "timestamp",
        "latency_ms", "x-request-id"].
    drop_key_regex : Optional[list[str]]
        Regex patterns for keys to drop. Defaults to
        [r"^x-.*", r".*_request_id$"] for HTTP headers and ID suffixes.
    hash_paths : Optional[list[str]]
        Dotpaths to hash instead of storing raw. Defaults to
        ["result", "raw"] for tool results and provider responses.
    hash_strings_over : int
        Hash strings longer than this. Default is 512 characters.

    Returns
    -------
    TracePayloadNormaliser
        A configured normaliser instance.

    Examples
    --------
    Create with default settings:

        >>> normaliser = make_structural_v1_normaliser()
        >>> "timestamp" in normaliser._drop_keys
        True

    Customize drop keys:

        >>> normaliser = make_structural_v1_normaliser(
        ...     drop_keys=["timestamp", "trace_id", "span_id"]
        ... )
        >>> "trace_id" in normaliser._drop_keys
        True

    Adjust string hashing threshold:

        >>> normaliser = make_structural_v1_normaliser(hash_strings_over=256)
        >>> normaliser._hash_strings_over
        256

    Add custom hash paths:

        >>> normaliser = make_structural_v1_normaliser(
        ...     hash_paths=["result", "raw", "response.content"]
        ... )
        >>> "response.content" in normaliser._hash_paths
        True

    See Also
    --------
    TracePayloadNormaliser : The normaliser class.
    NormaliserConfig : Configuration dataclass for normalisers.
    """
    return TracePayloadNormaliser(
        drop_keys=drop_keys
        or [
            "request_id",
            "response_id",
            "created",
            "timestamp",
            "latency_ms",
            "x-request-id",
        ],
        drop_key_regex=drop_key_regex or ["^x-.*", ".*_request_id$"],
        hash_paths=hash_paths or ["result", "raw"],
        hash_strings_over=hash_strings_over,
    )


# =============================================================================
# Config Loading
# =============================================================================


def load_trace_config(yaml_dict: dict[str, Any]) -> TraceConfig:
    """Load TraceConfig from a YAML-parsed dictionary.

    Parses a dictionary (typically from YAML) into a fully-typed TraceConfig
    instance. Missing keys are filled with sensible defaults, and backwards
    compatibility is maintained for older configuration formats.

    Parameters
    ----------
    yaml_dict : dict[str, Any]
        Dictionary parsed from YAML configuration. Expected structure::

            {
                "version": 1,
                "enabled": True,
                "store": {"mode": "full", "max_events": 1000},
                "fingerprint": {"enabled": True, "normaliser": {...}},
                "contracts": {"enabled": True, "fail_fast": False},
                "on_violation": {"mode": "record"}
            }

    Returns
    -------
    TraceConfig
        Fully populated TraceConfig with values from dict and defaults
        for any missing keys.

    Examples
    --------
    Load minimal configuration:

        >>> config = load_trace_config({"enabled": True})
        >>> config.enabled
        True
        >>> config.version
        1

    Load with storage settings:

        >>> config = load_trace_config({
        ...     "store": {"mode": "compact", "max_events": 500}
        ... })
        >>> config.store.mode == StoreMode.COMPACT
        True
        >>> config.store.max_events
        500

    Load with contract configuration:

        >>> config = load_trace_config({
        ...     "contracts": {
        ...         "fail_fast": True,
        ...         "stream_boundaries": {"enabled": False}
        ...     }
        ... })
        >>> config.contracts.fail_fast
        True
        >>> config.contracts.stream_boundaries.enabled
        False

    Handle empty/None input:

        >>> config = load_trace_config({})
        >>> config.enabled
        True
        >>> config = load_trace_config(None)  # doctest: +SKIP
        >>> # Returns default TraceConfig

    Backwards compatibility with old mode names:

        >>> config = load_trace_config({"store": {"mode": "fingerprint"}})
        >>> config.store.mode == StoreMode.COMPACT
        True

    See Also
    --------
    TraceConfig : The configuration dataclass.
    validate_with_config : Run validators using loaded config.
    """
    if not yaml_dict:
        return TraceConfig()

    # Parse store config
    store_dict = yaml_dict.get("store", {})
    redact_dict = store_dict.get("redact", {})
    redact = TraceRedactConfig(
        enabled=redact_dict.get("enabled", False),
        json_pointers=redact_dict.get("json_pointers", []),
        replacement=redact_dict.get("replacement", "<redacted>"),
    )

    # Handle mode with backwards compatibility
    mode_str = store_dict.get("mode", "full")
    # Map old names to new
    mode_map = {"off": "none", "fingerprint": "compact"}
    mode_str = mode_map.get(mode_str, mode_str)

    store = TraceStoreConfig(
        mode=StoreMode(mode_str),
        max_events=store_dict.get("max_events"),
        max_event_payload_bytes=store_dict.get("max_event_payload_bytes"),
        include_payloads=store_dict.get("include_payloads", True),
        redact=redact,
    )

    # Parse fingerprint config
    fp_dict = yaml_dict.get("fingerprint", {})
    norm_dict = fp_dict.get("normaliser", {})

    normaliser = NormaliserConfig(
        kind=NormaliserKind(norm_dict.get("kind", "builtin")),
        name=norm_dict.get("name", "structural_v1"),
        import_path=norm_dict.get("import"),
        drop_keys=norm_dict.get("config", {}).get(
            "drop_keys", ["request_id", "response_id", "created", "timestamp", "latency_ms"]
        ),
        drop_key_regex=norm_dict.get("config", {}).get("drop_key_regex", []),
        hash_paths=norm_dict.get("config", {}).get("hash_paths", ["result", "raw"]),
        hash_strings_over=norm_dict.get("config", {}).get("hash_strings_over", 512),
    )

    fingerprint = FingerprintConfig(
        enabled=fp_dict.get("enabled", True),
        algorithm=fp_dict.get("algorithm", "sha256"),
        normaliser=normaliser,
    )

    # Parse contracts config
    contracts_dict = yaml_dict.get("contracts", {})
    contracts = _parse_contracts_config(contracts_dict)

    # Parse on_violation config
    on_violation_dict = yaml_dict.get("on_violation", {})
    mode_str = on_violation_dict.get("mode", "record")
    on_violation = OnViolationConfig(mode=OnViolationMode(mode_str))

    return TraceConfig(
        version=yaml_dict.get("version", 1),
        enabled=yaml_dict.get("enabled", True),
        store=store,
        fingerprint=fingerprint,
        contracts=contracts,
        on_violation=on_violation,
    )


def _parse_contracts_config(contracts_dict: dict[str, Any]) -> TraceContractsConfig:
    """Parse the contracts section of configuration.

    Internal helper that transforms the contracts dictionary into a
    TraceContractsConfig instance with all nested configuration objects.

    Parameters
    ----------
    contracts_dict : dict[str, Any]
        The "contracts" section of the configuration dictionary.

    Returns
    -------
    TraceContractsConfig
        Fully populated contracts configuration.
    """
    if not contracts_dict:
        return TraceContractsConfig()

    # Generate boundaries
    gen_dict = contracts_dict.get("generate_boundaries", {})
    generate_boundaries = GenerateBoundariesConfig(
        enabled=gen_dict.get("enabled", True),
        start_kind=gen_dict.get("start_kind", "generate_start"),
        end_kind=gen_dict.get("end_kind", "generate_end"),
    )

    # Stream boundaries
    stream_dict = contracts_dict.get("stream_boundaries", {})
    stream_boundaries = StreamBoundariesConfig(
        enabled=stream_dict.get("enabled", True),
        start_kind=stream_dict.get("start_kind", "stream_start"),
        chunk_kind=stream_dict.get("chunk_kind", "stream_chunk"),
        end_kind=stream_dict.get("end_kind", "stream_end"),
        stream_id_key=stream_dict.get("stream_id_key", "stream_id"),
        chunk_index_key=stream_dict.get("chunk_index_key", "chunk_index"),
        first_chunk_index=stream_dict.get("first_chunk_index", 0),
        require_end=stream_dict.get("require_end", True),
        require_monotonic_chunks=stream_dict.get("require_monotonic_chunks", True),
    )

    # Tool results
    results_dict = contracts_dict.get("tool_results", {})
    tool_results = ToolResultsConfig(
        enabled=results_dict.get("enabled", True),
        call_start_kind=results_dict.get("call_start_kind", "tool_call_start"),
        call_result_kind=results_dict.get("call_result_kind", "tool_result"),
        call_id_key=results_dict.get("call_id_key", "call_id"),
        require_exactly_one_result=results_dict.get("require_exactly_one_result", True),
    )

    # Tool payloads
    payloads_dict = contracts_dict.get("tool_payloads", {})
    tools_dict = payloads_dict.get("tools", {})
    tools = {
        name: ToolPayloadSchemaConfig(
            args_schema=spec.get("args_schema", {}),
            required=spec.get("required", True),
        )
        for name, spec in tools_dict.items()
    }
    tool_payloads = ToolPayloadsConfig(
        enabled=payloads_dict.get("enabled", True),
        tool_key=payloads_dict.get("tool_key", "tool"),
        args_key=payloads_dict.get("args_key", "args"),
        tools=tools,
    )

    # Tool order
    order_dict = contracts_dict.get("tool_order", {})
    tool_order = ToolOrderConfig(
        enabled=order_dict.get("enabled", True),
        allow_unknown_tools=order_dict.get("allow_unknown_tools", False),
        must_follow=order_dict.get("must_follow", {}),
        must_precede=order_dict.get("must_precede", {}),
        forbidden_sequences=order_dict.get("forbidden_sequences", []),
    )

    return TraceContractsConfig(
        enabled=contracts_dict.get("enabled", True),
        fail_fast=contracts_dict.get("fail_fast", False),
        generate_boundaries=generate_boundaries,
        stream_boundaries=stream_boundaries,
        tool_results=tool_results,
        tool_payloads=tool_payloads,
        tool_order=tool_order,
    )


# =============================================================================
# Validation Helper
# =============================================================================


def validate_with_config(
    events: list[Any],
    config: TraceConfig,
) -> list[Any]:
    """Run trace contract validators using configuration toggles.

    Convenience function that compiles a TraceConfig to validator inputs
    and runs only the enabled validators. Handles fail-fast mode and
    returns violations sorted by event sequence.

    Parameters
    ----------
    events : list[Any]
        List of trace events. Can be TraceEvent objects or dictionaries
        with the same structure (seq, kind, payload).
    config : TraceConfig
        The trace configuration controlling which validators run and
        their settings.

    Returns
    -------
    list[Violation]
        List of Violation objects from all enabled validators, sorted
        by event sequence number. Empty list if tracing or contracts
        are disabled.

    Examples
    --------
    Validate with default configuration:

        >>> from insideLLMs.trace.tracing import TraceRecorder
        >>> recorder = TraceRecorder()
        >>> recorder.record("generate_start", {"prompt": "Hello"})
        >>> recorder.record("generate_end", {"response": "Hi"})
        >>> config = TraceConfig()
        >>> violations = validate_with_config(recorder.events, config)
        >>> len(violations)
        0

    Detect missing stream_end:

        >>> recorder = TraceRecorder()
        >>> recorder.record("stream_start", {"prompt": "Hello"})
        >>> recorder.record("stream_chunk", {"chunk": "Hi", "chunk_index": 0})
        >>> # Missing stream_end!
        >>> config = TraceConfig()
        >>> violations = validate_with_config(recorder.events, config)
        >>> any(v.code == "STREAM_NO_END" for v in violations)
        True

    Disable specific validators:

        >>> config = load_trace_config({
        ...     "contracts": {
        ...         "stream_boundaries": {"enabled": False},
        ...         "tool_order": {"enabled": False}
        ...     }
        ... })
        >>> violations = validate_with_config(recorder.events, config)
        >>> # Stream violations won't be checked

    Fail-fast mode:

        >>> config = load_trace_config({
        ...     "contracts": {"fail_fast": True}
        ... })
        >>> violations = validate_with_config(recorder.events, config)
        >>> # Stops after first violation

    See Also
    --------
    TraceConfig.to_contracts : Compile config to validator inputs.
    trace_contracts.validate_all : Run all validators directly.
    """
    from insideLLMs.trace_contracts import (
        Violation,
        validate_generate_boundaries,
        validate_stream_boundaries,
        validate_tool_order,
        validate_tool_payloads,
        validate_tool_results,
    )

    if not config.enabled or not config.contracts.enabled:
        return []

    compiled = config.to_contracts()
    toggles = compiled["toggles"]
    tool_schemas = compiled["tool_schemas"]
    tool_order_rules = compiled["tool_order_rules"]
    fail_fast = compiled["fail_fast"]

    all_violations: list[Violation] = []

    def _check_fail_fast() -> bool:
        """Return True if we should stop validation."""
        return fail_fast and len(all_violations) > 0

    if toggles.get("generate_boundaries", True):
        all_violations.extend(validate_generate_boundaries(events))
        if _check_fail_fast():
            return all_violations

    if toggles.get("stream_boundaries", True):
        all_violations.extend(validate_stream_boundaries(events))
        if _check_fail_fast():
            return all_violations

    if toggles.get("tool_results", True):
        all_violations.extend(validate_tool_results(events))
        if _check_fail_fast():
            return all_violations

    if toggles.get("tool_payloads", True) and tool_schemas:
        all_violations.extend(validate_tool_payloads(events, tool_schemas))
        if _check_fail_fast():
            return all_violations

    if toggles.get("tool_order", True) and tool_order_rules:
        all_violations.extend(validate_tool_order(events, tool_order_rules))

    # Sort by event sequence for stable output
    all_violations.sort(key=lambda v: v.event_seq)
    return all_violations
