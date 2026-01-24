"""Trace configuration, contracts, and serialization helpers.

This package provides a comprehensive tracing system for deterministic execution
recording and validation of LLM interactions. The system is designed for CI/CD
pipelines, enabling reproducible trace comparison, behavioral drift detection,
and contract-based validation.

Overview
--------
The trace package consists of three main modules:

1. **trace_config** - Configuration and payload normalization
2. **trace_contracts** - Pure validation functions for trace integrity
3. **tracing** - Event recording and fingerprinting utilities

Design Philosophy
-----------------
Traditional logging uses wall-clock timestamps, which makes traces
non-deterministic across runs (network latency, system load variations, etc.).
This tracing system uses logical sequence numbers instead, ensuring that
identical model behavior produces identical traces.

This enables powerful use cases:

- **Regression detection**: Compare fingerprints to detect behavioral drift
- **Reproducibility**: Replay traces for debugging without timing issues
- **Contract validation**: Verify event ordering and payload schemas
- **CI integration**: Deterministic traces work reliably in CI pipelines

Modules
-------
trace_config
    Configuration loading from YAML, payload normalization for stable
    fingerprinting, and compilation to validator inputs.

trace_contracts
    Pure, deterministic validation functions for trace events including:
    - Stream boundary validation (start/chunk/end ordering)
    - Generate boundary validation (start/end pairing)
    - Tool call/result pairing validation
    - Tool payload schema validation
    - Tool execution order validation

tracing
    Core event recording with ``TraceRecorder`` and fingerprinting with
    ``trace_fingerprint``. Thread-safe and deterministic.

Classes
-------
Configuration Classes
~~~~~~~~~~~~~~~~~~~~~
TraceConfig
    Top-level configuration for all tracing behavior. Load from YAML with
    ``load_trace_config()``.

FingerprintConfig
    Configuration for deterministic fingerprint computation including
    algorithm selection and normalizer settings.

NormaliserConfig
    Configuration for payload transformation before fingerprinting,
    including key dropping and string hashing thresholds.

TracePayloadNormaliser
    Event-kind-aware normalizer that transforms payloads to strip noise
    (timestamps, request IDs) and create stable hashes.

Event Classes
~~~~~~~~~~~~~
TraceEvent
    Immutable record of a single trace event with sequence number, kind,
    and payload.

TraceEventKind
    Enum of standard event types: GENERATE_START, GENERATE_END, STREAM_START,
    STREAM_CHUNK, STREAM_END, TOOL_CALL_START, TOOL_RESULT, ERROR, etc.

TraceRecorder
    Thread-safe collector for recording ordered trace events during execution.

Validation Classes
~~~~~~~~~~~~~~~~~~
Violation
    A trace contract violation with code, event sequence, detail message,
    and optional context.

ViolationCode
    Enum of standard violation codes: STREAM_NO_START, STREAM_NO_END,
    TOOL_NO_RESULT, GENERATE_NESTED, etc.

ToolSchema
    Schema definition for validating tool call arguments including
    required arguments and type constraints.

ToolOrderRule
    Rules for tool ordering constraints including must_precede, must_follow,
    and forbidden_sequences.

Enums
~~~~~
OnViolationMode
    Mode for handling violations: RECORD (log and continue), FAIL_PROBE
    (mark example failed), FAIL_RUN (stop immediately).

StoreMode
    Storage verbosity: NONE (no storage), COMPACT (fingerprint + violations),
    FULL (complete events).

NormaliserKind
    Normalizer source: BUILTIN (library-provided) or IMPORT (user-defined).

Functions
---------
Configuration Functions
~~~~~~~~~~~~~~~~~~~~~~~
load_trace_config(yaml_dict)
    Parse a YAML dictionary into a ``TraceConfig`` object with all defaults
    filled in. Handles backwards compatibility for legacy configuration keys.

make_structural_v1_normaliser(**kwargs)
    Factory for the default builtin normalizer. Drops common noise fields
    (timestamps, request IDs) and hashes large blobs.

validate_with_config(events, config)
    Convenience function to run all enabled validators using a ``TraceConfig``.
    Returns a list of ``Violation`` objects.

Validation Functions
~~~~~~~~~~~~~~~~~~~~
validate_all(events, tool_schemas=None, tool_order_rules=None)
    Run all standard validations and return combined violations.

validate_generate_boundaries(events)
    Validate that generate_start/generate_end events are properly paired
    and not nested.

validate_stream_boundaries(events)
    Validate streaming sequences: start/chunk/end ordering, chunk index
    continuity, and proper termination.

validate_tool_order(events, ruleset)
    Validate tool execution order against must_precede, must_follow, and
    forbidden_sequences constraints.

validate_tool_payloads(events, tool_schemas)
    Validate tool call arguments against JSON Schema definitions including
    required arguments and type checking.

validate_tool_results(events)
    Validate that every tool_call_start has a matching tool_result and
    no orphan results exist.

Tracing Functions
~~~~~~~~~~~~~~~~~
trace_fingerprint(events)
    Compute a stable SHA-256 fingerprint for a list of trace events.
    Returns a string in format "sha256:<hex>".

trace_to_custom_field(**kwargs)
    Format trace data for the insideLLMs.custom.trace@1 schema with
    stable, human-readable key ordering.

Examples
--------
Basic Trace Recording
~~~~~~~~~~~~~~~~~~~~~
Record events during model execution:

    >>> from insideLLMs.trace import TraceRecorder, TraceEventKind
    >>> recorder = TraceRecorder(run_id="run_001", example_id="ex_001")
    >>> recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hello"})
    TraceEvent(seq=0, kind='generate_start', ...)
    >>> recorder.record(TraceEventKind.GENERATE_END, {"response": "Hi there!"})
    TraceEvent(seq=1, kind='generate_end', ...)
    >>> len(recorder.events)
    2

Using Convenience Methods
~~~~~~~~~~~~~~~~~~~~~~~~~
Record common event patterns with less boilerplate:

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("What is 2+2?", temperature=0.7)
    TraceEvent(seq=0, kind='generate_start', ...)
    >>> recorder.record_generate_end("The answer is 4.", usage={"tokens": 15})
    TraceEvent(seq=1, kind='generate_end', ...)

Streaming Traces
~~~~~~~~~~~~~~~~
Track streaming responses with chunk indexing:

    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_start("Explain Python")
    TraceEvent(seq=0, kind='stream_start', ...)
    >>> for i, chunk in enumerate(["Python ", "is ", "great!"]):
    ...     recorder.record_stream_chunk(chunk, i)
    ...
    TraceEvent(seq=1, kind='stream_chunk', ...)
    TraceEvent(seq=2, kind='stream_chunk', ...)
    TraceEvent(seq=3, kind='stream_chunk', ...)
    >>> recorder.record_stream_end("Python is great!", chunk_count=3)
    TraceEvent(seq=4, kind='stream_end', ...)

Tool Call Tracing
~~~~~~~~~~~~~~~~~
Record tool invocations and results:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("calculator", {"expression": "2+2"}, tool_call_id="tc_1")
    TraceEvent(seq=0, kind='tool_call_start', ...)
    >>> recorder.record_tool_result("calculator", {"answer": 4}, tool_call_id="tc_1")
    TraceEvent(seq=1, kind='tool_result', ...)
    >>> recorder.get_tool_sequence()
    ['calculator']

Fingerprinting for Drift Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute stable fingerprints to detect behavioral changes:

    >>> from insideLLMs.trace import trace_fingerprint
    >>> events = recorder.events
    >>> fp = trace_fingerprint(events)
    >>> fp.startswith("sha256:")
    True
    >>> # Same events always produce the same fingerprint
    >>> trace_fingerprint(events) == fp
    True

Configuration Loading
~~~~~~~~~~~~~~~~~~~~~
Load configuration from YAML dictionaries:

    >>> from insideLLMs.trace import load_trace_config, validate_with_config
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

Custom Normalizer Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configure payload normalization for stable fingerprints:

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

Tool Payload Schema Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define and validate tool argument schemas:

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

Contract Validation
~~~~~~~~~~~~~~~~~~~
Validate traces against contracts:

    >>> from insideLLMs.trace import (
    ...     validate_stream_boundaries,
    ...     validate_tool_results,
    ...     Violation,
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record("stream_start", {"prompt": "Hello"})
    TraceEvent(seq=0, kind='stream_start', ...)
    >>> recorder.record("stream_chunk", {"chunk": "Hi", "chunk_index": 0})
    TraceEvent(seq=1, kind='stream_chunk', ...)
    >>> # Missing stream_end - will be detected as violation
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> len(violations)
    1
    >>> violations[0].code
    'STREAM_NO_END'

Using validate_with_config
~~~~~~~~~~~~~~~~~~~~~~~~~~
Run all enabled validators with a single call:

    >>> config = load_trace_config({
    ...     "contracts": {
    ...         "stream_boundaries": {"enabled": True},
    ...         "generate_boundaries": {"enabled": False},
    ...     }
    ... })
    >>> violations = validate_with_config(recorder.events, config)

Tool Order Validation
~~~~~~~~~~~~~~~~~~~~~
Enforce tool execution order constraints:

    >>> from insideLLMs.trace import ToolOrderRule, validate_tool_order
    >>> rules = ToolOrderRule(
    ...     name="data_pipeline",
    ...     must_follow={"process": ["fetch"]},  # process must come after fetch
    ...     forbidden_sequences=[["delete", "delete"]],  # no double deletes
    ... )
    >>> violations = validate_tool_order(recorder.events, rules)

Creating a Custom Normalizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build normalisers programmatically:

    >>> from insideLLMs.trace import (
    ...     TracePayloadNormaliser,
    ...     make_structural_v1_normaliser,
    ... )
    >>> normaliser = make_structural_v1_normaliser(
    ...     drop_keys=["timestamp", "request_id", "trace_id"],
    ...     hash_strings_over=128,
    ... )
    >>> payload = {"timestamp": "2024-01-01", "data": "important"}
    >>> normalized = normaliser.normalise(payload)
    >>> "timestamp" in normalized
    False

Complete CI Pipeline Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full example of trace recording and validation for CI:

    >>> from insideLLMs.trace import (
    ...     TraceRecorder,
    ...     TraceConfig,
    ...     load_trace_config,
    ...     validate_with_config,
    ...     trace_fingerprint,
    ...     OnViolationMode,
    ...     StoreMode,
    ... )
    >>>
    >>> # Load CI configuration
    >>> config = load_trace_config({
    ...     "version": 1,
    ...     "enabled": True,
    ...     "store": {"mode": "compact", "max_events": 1000},
    ...     "contracts": {"enabled": True, "fail_fast": True},
    ...     "on_violation": {"mode": "fail_probe"},
    ... })
    >>>
    >>> # Record execution trace
    >>> recorder = TraceRecorder(run_id="ci_run_123")
    >>> recorder.record_generate_start("Test prompt")
    TraceEvent(seq=0, kind='generate_start', ...)
    >>> recorder.record_generate_end("Test response")
    TraceEvent(seq=1, kind='generate_end', ...)
    >>>
    >>> # Validate contracts
    >>> violations = validate_with_config(recorder.events, config)
    >>> assert len(violations) == 0, f"Contract violations: {violations}"
    >>>
    >>> # Compute fingerprint for regression detection
    >>> fingerprint = trace_fingerprint(recorder.events)
    >>> # Compare with baseline fingerprint from previous runs

Notes
-----
- All validation functions are pure and deterministic
- Sequence numbers are 0-indexed and monotonically increasing
- Thread safety is achieved via ``threading.Lock`` on all state mutations
- Events are immutable after creation
- Fingerprints use SHA-256 for collision resistance

See Also
--------
insideLLMs.trace.trace_config : Full configuration documentation.
insideLLMs.trace.trace_contracts : Detailed validation function docs.
insideLLMs.trace.tracing : Event recording internals.
"""

from insideLLMs.trace.trace_config import (  # noqa: F401
    FingerprintConfig,
    NormaliserConfig,
    NormaliserKind,
    OnViolationMode,
    StoreMode,
    TraceConfig,
    TracePayloadNormaliser,
    load_trace_config,
    make_structural_v1_normaliser,
    validate_with_config,
)
from insideLLMs.trace.trace_contracts import (  # noqa: F401
    ToolOrderRule,
    ToolSchema,
    Violation,
    ViolationCode,
    validate_all,
    validate_generate_boundaries,
    validate_stream_boundaries,
    validate_tool_order,
    validate_tool_payloads,
    validate_tool_results,
)
from insideLLMs.trace.tracing import (  # noqa: F401
    TraceEvent,
    TraceEventKind,
    TraceRecorder,
    trace_fingerprint,
    trace_to_custom_field,
)

__all__ = [
    "FingerprintConfig",
    "NormaliserConfig",
    "NormaliserKind",
    "OnViolationMode",
    "StoreMode",
    "TraceConfig",
    "TracePayloadNormaliser",
    "load_trace_config",
    "make_structural_v1_normaliser",
    "validate_with_config",
    "ToolOrderRule",
    "ToolSchema",
    "Violation",
    "ViolationCode",
    "validate_all",
    "validate_generate_boundaries",
    "validate_stream_boundaries",
    "validate_tool_order",
    "validate_tool_payloads",
    "validate_tool_results",
    "TraceEvent",
    "TraceEventKind",
    "TraceRecorder",
    "trace_fingerprint",
    "trace_to_custom_field",
]
