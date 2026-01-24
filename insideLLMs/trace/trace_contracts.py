"""Trace contract validation for deterministic CI enforcement.

This module provides pure, deterministic validation functions for trace events
generated during LLM model execution. It enforces structural contracts on event
sequences, ensuring that streaming, tool calling, and generation events follow
proper lifecycle patterns.

Overview
--------
The trace contract system validates event sequences against predefined rules:

- **Stream Contracts**: Ensure `stream_start` -> `stream_chunk`* -> `stream_end`
- **Generate Contracts**: Ensure `generate_start` -> `generate_end` pairing
- **Tool Contracts**: Validate tool call/result pairing and argument schemas
- **Order Contracts**: Enforce tool invocation ordering constraints

All validation functions in this module are:

- **Pure**: No I/O operations, no side effects, no global state mutation
- **Deterministic**: Same input always produces exactly the same output
- **Stable Ordering**: Violations are always sorted by event sequence number

This makes the module ideal for use in CI pipelines where reproducibility
is critical.

Key Classes
-----------
ViolationCode : Enum
    Standard codes for categorizing contract violations.
Violation : dataclass
    Represents a single contract violation with context.
ToolSchema : dataclass
    Schema definition for validating tool call arguments.
ToolOrderRule : dataclass
    Rules for enforcing tool invocation ordering.

Key Functions
-------------
validate_stream_boundaries
    Validate streaming event lifecycle.
validate_generate_boundaries
    Validate generation event lifecycle.
validate_tool_payloads
    Validate tool call arguments against schemas.
validate_tool_order
    Validate tool invocation ordering.
validate_tool_results
    Validate tool call/result pairing.
validate_all
    Run all standard validations in one call.

Examples
--------
Basic stream validation:

>>> from insideLLMs.trace.tracing import TraceRecorder
>>> from insideLLMs.trace.trace_contracts import validate_stream_boundaries
>>>
>>> recorder = TraceRecorder()
>>> recorder.record("stream_start", {"prompt": "Hello"})
>>> recorder.record("stream_chunk", {"chunk": "Hi", "chunk_index": 0})
>>> # Missing stream_end - this is a contract violation!
>>>
>>> violations = validate_stream_boundaries(recorder.events)
>>> len(violations)
1
>>> violations[0].code
'STREAM_NO_END'

Validating tool argument schemas:

>>> from insideLLMs.trace.trace_contracts import (
...     validate_tool_payloads,
...     ToolSchema,
... )
>>>
>>> # Define expected schema for a search tool
>>> schemas = {
...     "web_search": ToolSchema(
...         name="web_search",
...         required_args=["query"],
...         arg_types={"query": str, "max_results": int},
...     )
... }
>>>
>>> recorder = TraceRecorder()
>>> recorder.record_tool_call("web_search", {"query": 123})  # Wrong type!
>>>
>>> violations = validate_tool_payloads(recorder.events, schemas)
>>> len(violations)
1
>>> violations[0].code
'TOOL_INVALID_ARG_TYPE'

Running all validations:

>>> from insideLLMs.trace.trace_contracts import validate_all, ToolSchema
>>>
>>> recorder = TraceRecorder()
>>> recorder.record_generate_start("What is 2+2?")
>>> recorder.record_tool_call("calculator", {"expression": "2+2"})
>>> recorder.record_tool_result("calculator", "4")
>>> recorder.record_generate_end("The answer is 4.")
>>>
>>> violations = validate_all(recorder.events)
>>> len(violations)  # All contracts satisfied
0

CI Integration Example
----------------------
A typical CI usage pattern validates trace files and fails on violations:

>>> import json
>>> from insideLLMs.trace.trace_contracts import validate_all, Violation
>>>
>>> def check_trace_file(path: str) -> bool:
...     '''Load a trace file and check for contract violations.'''
...     with open(path) as f:
...         events = json.load(f)
...     violations = validate_all(events)
...     if violations:
...         for v in violations:
...             print(f"[{v.code}] seq={v.event_seq}: {v.detail}")
...         return False
...     return True

Notes
-----
- Event sequences can be provided as either `TraceEvent` objects or
  dictionaries. The module normalizes inputs internally.
- All violations include an `event_seq` field indicating which event
  triggered the violation. For lifecycle violations (missing end events),
  this points to the start event.
- The `context` field in `Violation` provides debugging information
  specific to each violation type.
- Custom violation codes can be used by passing string values directly,
  but using `ViolationCode` enum values is recommended for consistency.

See Also
--------
insideLLMs.trace.tracing : Core tracing infrastructure
insideLLMs.trace.trace_config : Configuration for trace recording

References
----------
.. [1] OpenTelemetry Semantic Conventions for LLM traces
   https://opentelemetry.io/docs/specs/semconv/gen-ai/

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from insideLLMs.tracing import TraceEvent, TraceEventKind


class ViolationCode(str, Enum):
    """Standard violation codes for trace contract validation.

    This enum defines all standard violation codes used by the trace
    contract validation system. Each code uniquely identifies a specific
    type of contract violation, enabling automated categorization and
    handling in CI pipelines.

    The codes are organized by category:

    - ``STREAM_*``: Violations related to streaming event lifecycle
    - ``TOOL_*``: Violations related to tool/function calling
    - ``GENERATE_*``: Violations related to generation lifecycle
    - ``INVALID_*`` / ``MISSING_*``: General payload violations

    Parameters
    ----------
    value : str
        The string value of the violation code (inherited from str).

    Attributes
    ----------
    STREAM_NO_START : str
        A ``stream_end`` event occurred without a preceding ``stream_start``.
    STREAM_NO_END : str
        A ``stream_start`` event was not followed by ``stream_end``.
    STREAM_NESTED : str
        A ``stream_start`` occurred while another stream was active.
    STREAM_CHUNK_BEFORE_START : str
        A ``stream_chunk`` occurred before any ``stream_start``.
    STREAM_CHUNK_AFTER_END : str
        A ``stream_chunk`` occurred after ``stream_end``.
    STREAM_SEQ_DISCONTINUITY : str
        Stream chunks have gaps in their sequence numbers.
    STREAM_CHUNK_INDEX_MISMATCH : str
        A ``stream_chunk`` has an unexpected ``chunk_index`` value.
    TOOL_INVALID_ARGUMENTS : str
        Tool call arguments are malformed or invalid.
    TOOL_MISSING_REQUIRED_ARG : str
        A required argument is missing from a tool call.
    TOOL_INVALID_ARG_TYPE : str
        A tool argument has the wrong type.
    TOOL_ORDER_VIOLATION : str
        Tool calls violated ordering constraints.
    TOOL_NO_RESULT : str
        A ``tool_call_start`` was not followed by ``tool_result``.
    TOOL_RESULT_BEFORE_CALL : str
        A ``tool_result`` occurred without a preceding ``tool_call_start``.
    GENERATE_NO_START : str
        A ``generate_end`` occurred without a preceding ``generate_start``.
    GENERATE_NO_END : str
        A ``generate_start`` was not followed by ``generate_end``.
    GENERATE_NESTED : str
        A ``generate_start`` occurred while another generation was active.
    INVALID_PAYLOAD : str
        Event payload is malformed or contains invalid data.
    MISSING_REQUIRED_FIELD : str
        A required field is missing from the event payload.
    CUSTOM : str
        A custom violation code (for user-defined contracts).

    Examples
    --------
    Using violation codes in validation:

    >>> violation = Violation(
    ...     code=ViolationCode.STREAM_NO_END.value,
    ...     event_seq=0,
    ...     detail="stream_start without matching stream_end",
    ... )
    >>> violation.code
    'STREAM_NO_END'

    Checking violation categories:

    >>> code = ViolationCode.TOOL_MISSING_REQUIRED_ARG
    >>> code.value.startswith("TOOL_")
    True
    >>> code.name
    'TOOL_MISSING_REQUIRED_ARG'

    Iterating over all codes:

    >>> stream_codes = [c for c in ViolationCode if c.value.startswith("STREAM_")]
    >>> len(stream_codes)
    7

    See Also
    --------
    Violation : The dataclass that uses these codes
    validate_stream_boundaries : Produces STREAM_* violations
    validate_tool_payloads : Produces TOOL_* violations
    validate_generate_boundaries : Produces GENERATE_* violations
    """

    # Stream violations
    STREAM_NO_START = "STREAM_NO_START"
    STREAM_NO_END = "STREAM_NO_END"
    STREAM_NESTED = "STREAM_NESTED"
    STREAM_CHUNK_BEFORE_START = "STREAM_CHUNK_BEFORE_START"
    STREAM_CHUNK_AFTER_END = "STREAM_CHUNK_AFTER_END"
    STREAM_SEQ_DISCONTINUITY = "STREAM_SEQ_DISCONTINUITY"
    STREAM_CHUNK_INDEX_MISMATCH = "STREAM_CHUNK_INDEX_MISMATCH"

    # Tool violations
    TOOL_INVALID_ARGUMENTS = "TOOL_INVALID_ARGUMENTS"
    TOOL_MISSING_REQUIRED_ARG = "TOOL_MISSING_REQUIRED_ARG"
    TOOL_INVALID_ARG_TYPE = "TOOL_INVALID_ARG_TYPE"
    TOOL_ORDER_VIOLATION = "TOOL_ORDER_VIOLATION"
    TOOL_NO_RESULT = "TOOL_NO_RESULT"
    TOOL_RESULT_BEFORE_CALL = "TOOL_RESULT_BEFORE_CALL"

    # Generate violations
    GENERATE_NO_START = "GENERATE_NO_START"
    GENERATE_NO_END = "GENERATE_NO_END"
    GENERATE_NESTED = "GENERATE_NESTED"

    # General violations
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    CUSTOM = "CUSTOM"


@dataclass
class Violation:
    """Represents a single trace contract violation.

    A Violation captures detailed information about a contract violation
    detected during trace validation. Each violation includes the specific
    code identifying the violation type, the sequence number of the
    offending event, and a human-readable description.

    Violations are designed to be serializable to JSON for storage and
    reporting in CI pipelines. The `to_dict` and `from_dict` methods
    provide round-trip serialization support.

    Parameters
    ----------
    code : str
        The violation code identifying the type of violation.
        Should be a value from `ViolationCode` or a custom string.
    event_seq : int
        The sequence number of the event that caused the violation.
        For lifecycle violations (e.g., missing end event), this is
        the sequence number of the start event.
    detail : str
        Human-readable description of the violation.
    event_kind : str, optional
        The kind of event that caused the violation (e.g., "stream_start").
        Default is None.
    context : dict[str, Any], optional
        Additional debugging context specific to the violation type.
        Default is an empty dict.

    Attributes
    ----------
    code : str
        The violation code.
    event_seq : int
        Sequence number of the violating event.
    detail : str
        Human-readable description.
    event_kind : str or None
        The event kind that caused the violation.
    context : dict[str, Any]
        Additional debugging context.

    Examples
    --------
    Creating a basic violation:

    >>> violation = Violation(
    ...     code=ViolationCode.STREAM_NO_END.value,
    ...     event_seq=0,
    ...     detail="stream_start without matching stream_end",
    ...     event_kind="stream_start",
    ... )
    >>> violation.code
    'STREAM_NO_END'
    >>> violation.event_seq
    0

    Creating a violation with context:

    >>> violation = Violation(
    ...     code=ViolationCode.TOOL_MISSING_REQUIRED_ARG.value,
    ...     event_seq=5,
    ...     detail="Tool 'search' missing required argument: query",
    ...     event_kind="tool_call_start",
    ...     context={"tool_name": "search", "missing_arg": "query"},
    ... )
    >>> violation.context["missing_arg"]
    'query'

    Serialization round-trip:

    >>> original = Violation(
    ...     code="CUSTOM_ERROR",
    ...     event_seq=10,
    ...     detail="Custom validation failed",
    ...     context={"custom_field": "value"},
    ... )
    >>> data = original.to_dict()
    >>> restored = Violation.from_dict(data)
    >>> restored.code == original.code
    True
    >>> restored.context == original.context
    True

    Using violations in CI assertions:

    >>> def assert_no_violations(violations: list[Violation]) -> None:
    ...     '''Fail if any violations exist.'''
    ...     if violations:
    ...         msgs = [f"[{v.code}] seq={v.event_seq}: {v.detail}"
    ...                 for v in violations]
    ...         raise AssertionError("\\n".join(msgs))
    >>>
    >>> assert_no_violations([])  # Passes
    >>> # assert_no_violations([violation])  # Would raise

    See Also
    --------
    ViolationCode : Standard violation codes
    violations_to_custom_field : Convert violations for storage
    """

    code: str
    event_seq: int
    detail: str
    event_kind: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the violation to a dictionary for serialization.

        Creates a dictionary representation suitable for JSON serialization.
        Only includes `event_kind` and `context` if they have non-empty values.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: "code", "event_seq", "detail", and
            optionally "event_kind" and "context".

        Examples
        --------
        Basic serialization:

        >>> v = Violation(
        ...     code="STREAM_NO_END",
        ...     event_seq=0,
        ...     detail="Missing stream_end",
        ... )
        >>> d = v.to_dict()
        >>> d["code"]
        'STREAM_NO_END'
        >>> "context" in d
        False

        Serialization with all fields:

        >>> v = Violation(
        ...     code="TOOL_INVALID_ARG_TYPE",
        ...     event_seq=5,
        ...     detail="Expected str, got int",
        ...     event_kind="tool_call_start",
        ...     context={"arg": "query", "expected": "str"},
        ... )
        >>> d = v.to_dict()
        >>> d["event_kind"]
        'tool_call_start'
        >>> d["context"]["expected"]
        'str'

        JSON serialization:

        >>> import json
        >>> v = Violation(code="TEST", event_seq=0, detail="Test")
        >>> json_str = json.dumps(v.to_dict())
        >>> isinstance(json_str, str)
        True
        """
        d = {
            "code": self.code,
            "event_seq": self.event_seq,
            "detail": self.detail,
        }
        if self.event_kind:
            d["event_kind"] = self.event_kind
        if self.context:
            d["context"] = self.context
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Violation":
        """Create a Violation from a dictionary.

        Reconstructs a Violation instance from its dictionary representation,
        as produced by `to_dict()`.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing violation data. Required keys are
            "code", "event_seq", and "detail". Optional keys are
            "event_kind" and "context".

        Returns
        -------
        Violation
            A new Violation instance with the given data.

        Raises
        ------
        KeyError
            If required keys ("code", "event_seq", "detail") are missing.

        Examples
        --------
        Basic deserialization:

        >>> data = {
        ...     "code": "STREAM_NO_END",
        ...     "event_seq": 0,
        ...     "detail": "Missing stream_end",
        ... }
        >>> v = Violation.from_dict(data)
        >>> v.code
        'STREAM_NO_END'
        >>> v.event_kind is None
        True

        Full deserialization:

        >>> data = {
        ...     "code": "TOOL_INVALID_ARG_TYPE",
        ...     "event_seq": 5,
        ...     "detail": "Expected str",
        ...     "event_kind": "tool_call_start",
        ...     "context": {"arg": "query"},
        ... }
        >>> v = Violation.from_dict(data)
        >>> v.event_kind
        'tool_call_start'
        >>> v.context["arg"]
        'query'

        Round-trip test:

        >>> original = Violation(
        ...     code="TEST",
        ...     event_seq=42,
        ...     detail="Test violation",
        ...     event_kind="custom",
        ...     context={"key": "value"},
        ... )
        >>> restored = Violation.from_dict(original.to_dict())
        >>> original == restored
        True
        """
        return cls(
            code=data["code"],
            event_seq=data["event_seq"],
            detail=data["detail"],
            event_kind=data.get("event_kind"),
            context=data.get("context", {}),
        )


@dataclass
class ToolSchema:
    """Schema definition for validating tool call arguments.

    Defines the expected structure of arguments for a specific tool,
    including required arguments, optional arguments, and type constraints.
    Used by `validate_tool_payloads` to check that tool calls conform to
    their expected interface.

    Parameters
    ----------
    name : str
        The name of the tool this schema applies to.
    required_args : list[str], optional
        List of argument names that must be present. Default is [].
    arg_types : dict[str, type], optional
        Mapping of argument names to their expected Python types.
        Default is {}.
    optional_args : list[str], optional
        List of argument names that may be present. Default is [].

    Attributes
    ----------
    name : str
        Tool name for identification.
    required_args : list[str]
        Names of required arguments.
    arg_types : dict[str, type]
        Expected types for arguments (both required and optional).
    optional_args : list[str]
        Names of optional arguments.

    Examples
    --------
    Basic schema with required arguments:

    >>> schema = ToolSchema(
    ...     name="web_search",
    ...     required_args=["query"],
    ... )
    >>> schema.name
    'web_search'
    >>> "query" in schema.required_args
    True

    Schema with type constraints:

    >>> schema = ToolSchema(
    ...     name="calculator",
    ...     required_args=["expression"],
    ...     arg_types={
    ...         "expression": str,
    ...         "precision": int,
    ...     },
    ...     optional_args=["precision"],
    ... )
    >>> schema.arg_types["expression"]
    <class 'str'>

    Using schemas for validation:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> from insideLLMs.trace.trace_contracts import validate_tool_payloads
    >>>
    >>> schemas = {
    ...     "fetch_url": ToolSchema(
    ...         name="fetch_url",
    ...         required_args=["url"],
    ...         arg_types={"url": str, "timeout": int},
    ...         optional_args=["timeout"],
    ...     )
    ... }
    >>>
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("fetch_url", {"url": "https://example.com"})
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)
    0

    Detecting missing required arguments:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("fetch_url", {"timeout": 30})  # Missing 'url'
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)
    1
    >>> violations[0].code
    'TOOL_MISSING_REQUIRED_ARG'

    Detecting type mismatches:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("fetch_url", {"url": 12345})  # url should be str
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> violations[0].code
    'TOOL_INVALID_ARG_TYPE'

    See Also
    --------
    validate_tool_payloads : Uses ToolSchema for validation
    ToolOrderRule : For ordering constraints instead of argument validation
    """

    name: str
    required_args: list[str] = field(default_factory=list)
    arg_types: dict[str, type] = field(default_factory=dict)
    optional_args: list[str] = field(default_factory=list)


@dataclass
class ToolOrderRule:
    """Defines ordering constraints for tool invocations.

    Specifies rules that constrain the order in which tools can be called.
    This is useful for enforcing workflows where certain tools must be
    called before or after others, or where certain sequences are forbidden.

    Parameters
    ----------
    name : str
        A descriptive name for this ruleset.
    must_precede : dict[str, list[str]], optional
        Maps tool names to lists of tools they must come before.
        If tool A must precede tool B, A must appear before every
        occurrence of B in the trace. Default is {}.
    must_follow : dict[str, list[str]], optional
        Maps tool names to lists of tools that must come before them.
        If tool B must follow tool A, at least one A must appear
        before B. Default is {}.
    forbidden_sequences : list[list[str]], optional
        Lists of tool name sequences that are not allowed.
        Each sequence is an exact pattern that must not appear.
        Default is [].

    Attributes
    ----------
    name : str
        Rule name for identification in violation messages.
    must_precede : dict[str, list[str]]
        Tool precedence requirements.
    must_follow : dict[str, list[str]]
        Tool succession requirements.
    forbidden_sequences : list[list[str]]
        Forbidden tool call patterns.

    Examples
    --------
    Requiring authentication before API calls:

    >>> rule = ToolOrderRule(
    ...     name="auth_before_api",
    ...     must_precede={"authenticate": ["api_call", "fetch_data"]},
    ... )
    >>> rule.name
    'auth_before_api'

    Requiring initialization:

    >>> rule = ToolOrderRule(
    ...     name="init_required",
    ...     must_follow={"process_data": ["initialize"]},
    ... )
    >>> # process_data must have initialize called before it

    Forbidding dangerous sequences:

    >>> rule = ToolOrderRule(
    ...     name="safety_rules",
    ...     forbidden_sequences=[
    ...         ["delete_file", "delete_file"],  # No double deletes
    ...         ["drop_table", "drop_database"],  # Dangerous combo
    ...     ],
    ... )

    Using rules for validation:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> from insideLLMs.trace.trace_contracts import validate_tool_order
    >>>
    >>> rule = ToolOrderRule(
    ...     name="auth_first",
    ...     must_follow={"query_db": ["authenticate"]},
    ... )
    >>>
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("query_db", {"sql": "SELECT *"})
    <TraceEvent ...>
    >>> # Missing authenticate before query_db!
    >>>
    >>> violations = validate_tool_order(recorder.events, rule)
    >>> len(violations)
    1
    >>> violations[0].code
    'TOOL_ORDER_VIOLATION'

    Complex workflow rules:

    >>> workflow = ToolOrderRule(
    ...     name="data_pipeline",
    ...     must_precede={
    ...         "validate": ["transform", "load"],
    ...         "transform": ["load"],
    ...     },
    ...     must_follow={
    ...         "cleanup": ["load"],  # cleanup after load
    ...     },
    ...     forbidden_sequences=[
    ...         ["load", "validate"],  # Can't validate after loading
    ...     ],
    ... )

    See Also
    --------
    validate_tool_order : Uses ToolOrderRule for validation
    ToolSchema : For argument validation instead of ordering
    """

    name: str
    must_precede: dict[str, list[str]] = field(default_factory=dict)
    must_follow: dict[str, list[str]] = field(default_factory=dict)
    forbidden_sequences: list[list[str]] = field(default_factory=list)


# --- Pure Validation Functions ---


def validate_stream_boundaries(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[Violation]:
    """Validate streaming event boundaries and chunk sequencing.

    Checks that streaming events follow the expected lifecycle pattern:
    ``stream_start`` -> ``stream_chunk``* -> ``stream_end``. Also validates
    that chunk indices are sequential starting from 0.

    This function performs the following validations:

    1. **Start/End Pairing**: Every ``stream_start`` must have a matching
       ``stream_end``, and vice versa.
    2. **No Nesting**: A new ``stream_start`` cannot occur while a stream
       is already active (no nested streams).
    3. **Chunk Ordering**: ``stream_chunk`` events must occur between
       ``stream_start`` and ``stream_end``.
    4. **Chunk Index Continuity**: The ``chunk_index`` field in chunk
       payloads must be sequential (0, 1, 2, ...).

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of trace events to validate. Can be either TraceEvent
        objects or dictionaries with the same structure.

    Returns
    -------
    list[Violation]
        List of violations found, sorted by event sequence number.
        Empty list if no violations detected.

    Examples
    --------
    Valid streaming sequence:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_start("Hello")
    <TraceEvent ...>
    >>> recorder.record_stream_chunk("Hi", chunk_index=0)
    <TraceEvent ...>
    >>> recorder.record_stream_chunk(" there", chunk_index=1)
    <TraceEvent ...>
    >>> recorder.record_stream_end(full_response="Hi there", chunk_count=2)
    <TraceEvent ...>
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> len(violations)
    0

    Missing stream_end:

    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_start("Test")
    <TraceEvent ...>
    >>> recorder.record_stream_chunk("Response", chunk_index=0)
    <TraceEvent ...>
    >>> # Forgot stream_end!
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> len(violations)
    1
    >>> violations[0].code
    'STREAM_NO_END'
    >>> violations[0].event_seq  # Points to the stream_start
    0

    Chunk before start:

    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_chunk("Orphan chunk", chunk_index=0)
    <TraceEvent ...>
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> violations[0].code
    'STREAM_CHUNK_BEFORE_START'

    Non-sequential chunk indices:

    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_start("Test")
    <TraceEvent ...>
    >>> recorder.record_stream_chunk("First", chunk_index=0)
    <TraceEvent ...>
    >>> recorder.record_stream_chunk("Third", chunk_index=2)  # Skipped 1!
    <TraceEvent ...>
    >>> recorder.record_stream_end()
    <TraceEvent ...>
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> len(violations)
    1
    >>> violations[0].code
    'STREAM_CHUNK_INDEX_MISMATCH'
    >>> violations[0].context["expected"]
    1
    >>> violations[0].context["actual"]
    2

    Nested streams:

    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_start("First stream")
    <TraceEvent ...>
    >>> recorder.record_stream_start("Nested stream")  # Invalid!
    <TraceEvent ...>
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> any(v.code == 'STREAM_NESTED' for v in violations)
    True

    Working with dict events:

    >>> events = [
    ...     {"seq": 0, "kind": "stream_start", "payload": {"prompt": "Hi"}},
    ...     {"seq": 1, "kind": "stream_chunk", "payload": {"chunk": "Hello", "chunk_index": 0}},
    ...     {"seq": 2, "kind": "stream_end", "payload": {}},
    ... ]
    >>> violations = validate_stream_boundaries(events)
    >>> len(violations)
    0

    See Also
    --------
    validate_generate_boundaries : Similar validation for generate events
    validate_all : Runs all validators including this one
    ViolationCode : Contains all STREAM_* violation codes
    """
    violations: list[Violation] = []

    # Normalize to TraceEvent if needed
    normalized = _normalize_events(events)

    # Track state
    stream_active = False
    stream_start_seq: Optional[int] = None
    last_chunk_index = -1

    for event in normalized:
        kind = event.kind
        seq = event.seq

        if kind == TraceEventKind.STREAM_START.value:
            if stream_active:
                violations.append(
                    Violation(
                        code=ViolationCode.STREAM_NESTED.value,
                        event_seq=seq,
                        detail="Nested stream_start without prior stream_end",
                        event_kind=kind,
                        context={"prior_start_seq": stream_start_seq},
                    )
                )
            stream_active = True
            stream_start_seq = seq
            last_chunk_index = -1

        elif kind == TraceEventKind.STREAM_CHUNK.value:
            if not stream_active:
                violations.append(
                    Violation(
                        code=ViolationCode.STREAM_CHUNK_BEFORE_START.value,
                        event_seq=seq,
                        detail="stream_chunk without prior stream_start",
                        event_kind=kind,
                    )
                )
            else:
                # Check chunk index continuity
                chunk_index = event.payload.get("chunk_index", -1)
                expected_index = last_chunk_index + 1
                if chunk_index != expected_index:
                    violations.append(
                        Violation(
                            code=ViolationCode.STREAM_CHUNK_INDEX_MISMATCH.value,
                            event_seq=seq,
                            detail=f"Expected chunk_index {expected_index}, got {chunk_index}",
                            event_kind=kind,
                            context={
                                "expected": expected_index,
                                "actual": chunk_index,
                            },
                        )
                    )
                last_chunk_index = chunk_index

        elif kind == TraceEventKind.STREAM_END.value:
            if not stream_active:
                violations.append(
                    Violation(
                        code=ViolationCode.STREAM_NO_START.value,
                        event_seq=seq,
                        detail="stream_end without prior stream_start",
                        event_kind=kind,
                    )
                )
            stream_active = False
            stream_start_seq = None

    # Check for unclosed stream
    if stream_active and stream_start_seq is not None:
        violations.append(
            Violation(
                code=ViolationCode.STREAM_NO_END.value,
                event_seq=stream_start_seq,
                detail="stream_start without matching stream_end",
                event_kind=TraceEventKind.STREAM_START.value,
            )
        )

    # Sort by event sequence for stable ordering
    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_tool_payloads(
    events: list[TraceEvent] | list[dict[str, Any]],
    tool_schemas: dict[str, ToolSchema],
) -> list[Violation]:
    """Validate tool call payloads against defined schemas.

    Checks that tool call events contain valid arguments according to
    the provided schemas. This includes verifying required arguments
    are present and that argument types match expectations.

    This function performs the following validations:

    1. **Required Arguments**: All arguments listed in ``required_args``
       must be present in the tool call payload.
    2. **Type Checking**: Arguments listed in ``arg_types`` must have
       values matching the specified Python types.
    3. **Tool Name**: The ``tool_name`` field must be present in the
       tool call payload.

    Tools without a corresponding schema are skipped (not validated).

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of trace events to validate. Only ``tool_call_start``
        events are checked.
    tool_schemas : dict[str, ToolSchema]
        Mapping of tool names to their schemas. Tools not in this
        dict are not validated.

    Returns
    -------
    list[Violation]
        List of violations found, sorted by event sequence number.
        Empty list if no violations detected.

    Examples
    --------
    Valid tool calls:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> schemas = {
    ...     "search": ToolSchema(
    ...         name="search",
    ...         required_args=["query"],
    ...         arg_types={"query": str, "limit": int},
    ...     )
    ... }
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("search", {"query": "python", "limit": 10})
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)
    0

    Missing required argument:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("search", {"limit": 10})  # Missing 'query'
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)
    1
    >>> violations[0].code
    'TOOL_MISSING_REQUIRED_ARG'
    >>> violations[0].context["missing_arg"]
    'query'

    Wrong argument type:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("search", {"query": 123, "limit": "ten"})
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)
    2
    >>> codes = {v.code for v in violations}
    >>> 'TOOL_INVALID_ARG_TYPE' in codes
    True

    Multiple schemas:

    >>> schemas = {
    ...     "read_file": ToolSchema(
    ...         name="read_file",
    ...         required_args=["path"],
    ...         arg_types={"path": str},
    ...     ),
    ...     "write_file": ToolSchema(
    ...         name="write_file",
    ...         required_args=["path", "content"],
    ...         arg_types={"path": str, "content": str},
    ...     ),
    ... }
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("read_file", {"path": "/tmp/test.txt"})
    <TraceEvent ...>
    >>> recorder.record_tool_call("write_file", {"path": "/tmp/out.txt"})  # Missing content!
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)
    1
    >>> violations[0].context["tool_name"]
    'write_file'

    Unknown tools are skipped:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("unknown_tool", {"anything": "goes"})
    <TraceEvent ...>
    >>> violations = validate_tool_payloads(recorder.events, schemas)
    >>> len(violations)  # No schema for unknown_tool, so no validation
    0

    Missing tool_name in payload:

    >>> events = [{"seq": 0, "kind": "tool_call_start", "payload": {}}]
    >>> violations = validate_tool_payloads(events, schemas)
    >>> violations[0].code
    'INVALID_PAYLOAD'

    See Also
    --------
    ToolSchema : Schema definition for tools
    validate_tool_order : Validates tool ordering instead of arguments
    validate_tool_results : Validates tool call/result pairing
    """
    violations: list[Violation] = []
    normalized = _normalize_events(events)

    for event in normalized:
        if event.kind != TraceEventKind.TOOL_CALL_START.value:
            continue

        tool_name = event.payload.get("tool_name")
        arguments = event.payload.get("arguments", {})

        if not tool_name:
            violations.append(
                Violation(
                    code=ViolationCode.INVALID_PAYLOAD.value,
                    event_seq=event.seq,
                    detail="tool_call_start missing tool_name",
                    event_kind=event.kind,
                )
            )
            continue

        # Check if we have a schema for this tool
        schema = tool_schemas.get(tool_name)
        if not schema:
            # No schema to validate against
            continue

        # Check required arguments
        for req_arg in schema.required_args:
            if req_arg not in arguments:
                violations.append(
                    Violation(
                        code=ViolationCode.TOOL_MISSING_REQUIRED_ARG.value,
                        event_seq=event.seq,
                        detail=f"Tool '{tool_name}' missing required argument: {req_arg}",
                        event_kind=event.kind,
                        context={"tool_name": tool_name, "missing_arg": req_arg},
                    )
                )

        # Check argument types
        for arg_name, expected_type in schema.arg_types.items():
            if arg_name in arguments:
                actual_value = arguments[arg_name]
                if not isinstance(actual_value, expected_type):
                    violations.append(
                        Violation(
                            code=ViolationCode.TOOL_INVALID_ARG_TYPE.value,
                            event_seq=event.seq,
                            detail=f"Tool '{tool_name}' argument '{arg_name}': "
                            f"expected {expected_type.__name__}, "
                            f"got {type(actual_value).__name__}",
                            event_kind=event.kind,
                            context={
                                "tool_name": tool_name,
                                "arg_name": arg_name,
                                "expected_type": expected_type.__name__,
                                "actual_type": type(actual_value).__name__,
                            },
                        )
                    )

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_tool_order(
    events: list[TraceEvent] | list[dict[str, Any]],
    ruleset: ToolOrderRule,
) -> list[Violation]:
    """Validate tool call ordering against defined constraints.

    Checks that tool calls occur in an order consistent with the provided
    rules. This enables enforcement of workflows where certain tools must
    be called before or after others.

    This function performs the following validations:

    1. **must_precede**: If tool A is in ``must_precede`` with value [B, C],
       then every occurrence of B or C must have A appear before it.
    2. **must_follow**: If tool B is in ``must_follow`` with value [A],
       then at least one A must appear somewhere before B.
    3. **forbidden_sequences**: If [A, B, C] is a forbidden sequence,
       then this exact sequence of consecutive tool calls is not allowed.

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of trace events to validate. Only ``tool_call_start``
        events are considered for ordering.
    ruleset : ToolOrderRule
        The ordering rules to enforce.

    Returns
    -------
    list[Violation]
        List of violations found, sorted by event sequence number.
        Empty list if no violations detected.

    Examples
    --------
    must_precede validation:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> rule = ToolOrderRule(
    ...     name="auth_first",
    ...     must_precede={"auth": ["api_call"]},
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("api_call", {})  # auth should come first!
    <TraceEvent ...>
    >>> recorder.record_tool_call("auth", {})
    <TraceEvent ...>
    >>> violations = validate_tool_order(recorder.events, rule)
    >>> len(violations)
    1
    >>> violations[0].detail
    "'auth' must precede 'api_call' but appeared at position 1 vs 0"

    must_follow validation:

    >>> rule = ToolOrderRule(
    ...     name="init_required",
    ...     must_follow={"process": ["init"]},
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("process", {})  # Missing init!
    <TraceEvent ...>
    >>> violations = validate_tool_order(recorder.events, rule)
    >>> len(violations)
    1
    >>> "'process' must follow 'init'" in violations[0].detail
    True

    Correct ordering:

    >>> rule = ToolOrderRule(
    ...     name="workflow",
    ...     must_follow={"cleanup": ["process"]},
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("process", {})
    <TraceEvent ...>
    >>> recorder.record_tool_call("cleanup", {})
    <TraceEvent ...>
    >>> violations = validate_tool_order(recorder.events, rule)
    >>> len(violations)
    0

    Forbidden sequences:

    >>> rule = ToolOrderRule(
    ...     name="no_double_delete",
    ...     forbidden_sequences=[["delete", "delete"]],
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("delete", {"id": 1})
    <TraceEvent ...>
    >>> recorder.record_tool_call("delete", {"id": 2})
    <TraceEvent ...>
    >>> violations = validate_tool_order(recorder.events, rule)
    >>> len(violations)
    1
    >>> "Forbidden tool sequence" in violations[0].detail
    True

    Complex workflow:

    >>> rule = ToolOrderRule(
    ...     name="pipeline",
    ...     must_precede={"extract": ["transform"], "transform": ["load"]},
    ...     must_follow={"load": ["transform"]},
    ...     forbidden_sequences=[["load", "extract"]],
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("extract", {})
    <TraceEvent ...>
    >>> recorder.record_tool_call("transform", {})
    <TraceEvent ...>
    >>> recorder.record_tool_call("load", {})
    <TraceEvent ...>
    >>> violations = validate_tool_order(recorder.events, rule)
    >>> len(violations)
    0

    See Also
    --------
    ToolOrderRule : Rule definition for tool ordering
    validate_tool_payloads : Validates tool arguments instead of ordering
    validate_tool_results : Validates tool call/result pairing
    """
    violations: list[Violation] = []
    normalized = _normalize_events(events)

    # Extract tool sequence
    tool_calls: list[tuple[int, str]] = []
    for event in normalized:
        if event.kind == TraceEventKind.TOOL_CALL_START.value:
            tool_name = event.payload.get("tool_name", "unknown")
            tool_calls.append((event.seq, tool_name))

    tool_sequence = [name for _, name in tool_calls]
    # Check must_precede constraints
    for tool, must_come_after in ruleset.must_precede.items():
        tool_indices = [i for i, t in enumerate(tool_sequence) if t == tool]
        for after_tool in must_come_after:
            after_indices = [i for i, t in enumerate(tool_sequence) if t == after_tool]
            for ti in tool_indices:
                for ai in after_indices:
                    if ti >= ai:
                        # tool should come before after_tool, but it doesn't
                        seq = tool_calls[ti][0]
                        violations.append(
                            Violation(
                                code=ViolationCode.TOOL_ORDER_VIOLATION.value,
                                event_seq=seq,
                                detail=f"'{tool}' must precede '{after_tool}' but "
                                f"appeared at position {ti} vs {ai}",
                                event_kind=TraceEventKind.TOOL_CALL_START.value,
                                context={
                                    "rule_name": ruleset.name,
                                    "tool": tool,
                                    "must_precede": after_tool,
                                    "tool_position": ti,
                                    "after_position": ai,
                                },
                            )
                        )

    # Check must_follow constraints
    for tool, must_come_before in ruleset.must_follow.items():
        tool_indices = [i for i, t in enumerate(tool_sequence) if t == tool]
        for before_tool in must_come_before:
            before_indices = [i for i, t in enumerate(tool_sequence) if t == before_tool]
            for ti in tool_indices:
                # Check if any before_tool comes before this tool
                has_required_predecessor = any(bi < ti for bi in before_indices)
                if not has_required_predecessor:
                    seq = tool_calls[ti][0]
                    violations.append(
                        Violation(
                            code=ViolationCode.TOOL_ORDER_VIOLATION.value,
                            event_seq=seq,
                            detail=f"'{tool}' must follow '{before_tool}'",
                            event_kind=TraceEventKind.TOOL_CALL_START.value,
                            context={
                                "rule_name": ruleset.name,
                                "tool": tool,
                                "must_follow": before_tool,
                            },
                        )
                    )

    # Check forbidden sequences
    for forbidden in ruleset.forbidden_sequences:
        if len(forbidden) < 2:
            continue
        # Look for this sequence in tool_sequence
        for i in range(len(tool_sequence) - len(forbidden) + 1):
            window = tool_sequence[i : i + len(forbidden)]
            if window == forbidden:
                seq = tool_calls[i][0]
                violations.append(
                    Violation(
                        code=ViolationCode.TOOL_ORDER_VIOLATION.value,
                        event_seq=seq,
                        detail=f"Forbidden tool sequence: {' -> '.join(forbidden)}",
                        event_kind=TraceEventKind.TOOL_CALL_START.value,
                        context={
                            "rule_name": ruleset.name,
                            "forbidden_sequence": forbidden,
                            "position": i,
                        },
                    )
                )

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_generate_boundaries(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[Violation]:
    """Validate generation event boundaries.

    Checks that generation events follow the expected lifecycle pattern:
    ``generate_start`` -> ``generate_end``. Error events can terminate
    a generation sequence without requiring ``generate_end``.

    This function performs the following validations:

    1. **Start/End Pairing**: Every ``generate_start`` must have a matching
       ``generate_end`` (unless terminated by an error).
    2. **No Nesting**: A new ``generate_start`` cannot occur while a
       generation is already active.
    3. **Orphan Ends**: A ``generate_end`` must have a preceding
       ``generate_start``.

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of trace events to validate. Can be either TraceEvent
        objects or dictionaries with the same structure.

    Returns
    -------
    list[Violation]
        List of violations found, sorted by event sequence number.
        Empty list if no violations detected.

    Examples
    --------
    Valid generation sequence:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("What is 2+2?")
    <TraceEvent ...>
    >>> recorder.record_generate_end("The answer is 4.")
    <TraceEvent ...>
    >>> violations = validate_generate_boundaries(recorder.events)
    >>> len(violations)
    0

    Missing generate_end:

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Hello")
    <TraceEvent ...>
    >>> # Forgot generate_end!
    >>> violations = validate_generate_boundaries(recorder.events)
    >>> len(violations)
    1
    >>> violations[0].code
    'GENERATE_NO_END'

    Error terminates generation:

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Complex query")
    <TraceEvent ...>
    >>> recorder.record_error("API rate limit exceeded")
    <TraceEvent ...>
    >>> # Error properly terminates, no generate_end needed
    >>> violations = validate_generate_boundaries(recorder.events)
    >>> len(violations)
    0

    Nested generation (invalid):

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("First question")
    <TraceEvent ...>
    >>> recorder.record_generate_start("Nested question")  # Invalid!
    <TraceEvent ...>
    >>> violations = validate_generate_boundaries(recorder.events)
    >>> any(v.code == 'GENERATE_NESTED' for v in violations)
    True

    Orphan generate_end:

    >>> recorder = TraceRecorder()
    >>> recorder.record("generate_end", {"response": "Orphan"})
    <TraceEvent ...>
    >>> violations = validate_generate_boundaries(recorder.events)
    >>> violations[0].code
    'GENERATE_NO_START'

    Multiple valid generations:

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Q1")
    <TraceEvent ...>
    >>> recorder.record_generate_end("A1")
    <TraceEvent ...>
    >>> recorder.record_generate_start("Q2")
    <TraceEvent ...>
    >>> recorder.record_generate_end("A2")
    <TraceEvent ...>
    >>> violations = validate_generate_boundaries(recorder.events)
    >>> len(violations)
    0

    Working with dict events:

    >>> events = [
    ...     {"seq": 0, "kind": "generate_start", "payload": {"prompt": "Hi"}},
    ...     {"seq": 1, "kind": "generate_end", "payload": {"response": "Hello"}},
    ... ]
    >>> violations = validate_generate_boundaries(events)
    >>> len(violations)
    0

    See Also
    --------
    validate_stream_boundaries : Similar validation for streaming events
    validate_all : Runs all validators including this one
    ViolationCode : Contains all GENERATE_* violation codes
    """
    violations: list[Violation] = []
    normalized = _normalize_events(events)

    generate_active = False
    generate_start_seq: Optional[int] = None

    for event in normalized:
        kind = event.kind
        seq = event.seq

        if kind == TraceEventKind.GENERATE_START.value:
            if generate_active:
                violations.append(
                    Violation(
                        code=ViolationCode.GENERATE_NESTED.value,
                        event_seq=seq,
                        detail="Nested generate_start without prior generate_end",
                        event_kind=kind,
                        context={"prior_start_seq": generate_start_seq},
                    )
                )
            generate_active = True
            generate_start_seq = seq

        elif kind == TraceEventKind.GENERATE_END.value:
            if not generate_active:
                violations.append(
                    Violation(
                        code=ViolationCode.GENERATE_NO_START.value,
                        event_seq=seq,
                        detail="generate_end without prior generate_start",
                        event_kind=kind,
                    )
                )
            generate_active = False
            generate_start_seq = None

        elif kind == TraceEventKind.ERROR.value:
            # Errors can terminate a generate sequence
            if generate_active:
                generate_active = False
                generate_start_seq = None

    # Check for unclosed generate
    if generate_active and generate_start_seq is not None:
        violations.append(
            Violation(
                code=ViolationCode.GENERATE_NO_END.value,
                event_seq=generate_start_seq,
                detail="generate_start without matching generate_end",
                event_kind=TraceEventKind.GENERATE_START.value,
            )
        )

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_tool_results(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[Violation]:
    """Validate that tool calls have corresponding results.

    Checks that every ``tool_call_start`` event is followed by a
    corresponding ``tool_result`` event, and that ``tool_result``
    events don't appear without a preceding call.

    Tool calls and results are matched by ``tool_call_id`` if present,
    otherwise by ``tool_name``. This allows for multiple concurrent
    calls to the same tool when IDs are used.

    This function performs the following validations:

    1. **Call/Result Pairing**: Every ``tool_call_start`` must have a
       matching ``tool_result``.
    2. **No Orphan Results**: A ``tool_result`` must have a preceding
       ``tool_call_start`` with matching ID or name.

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of trace events to validate. Considers ``tool_call_start``
        and ``tool_result`` events.

    Returns
    -------
    list[Violation]
        List of violations found, sorted by event sequence number.
        Empty list if no violations detected.

    Examples
    --------
    Valid tool call/result:

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("search", {"query": "python"})
    <TraceEvent ...>
    >>> recorder.record_tool_result("search", ["result1", "result2"])
    <TraceEvent ...>
    >>> violations = validate_tool_results(recorder.events)
    >>> len(violations)
    0

    Missing tool result:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("api_call", {"endpoint": "/users"})
    <TraceEvent ...>
    >>> # Forgot to record the result!
    >>> violations = validate_tool_results(recorder.events)
    >>> len(violations)
    1
    >>> violations[0].code
    'TOOL_NO_RESULT'

    Orphan tool result:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_result("mystery_tool", "some result")
    <TraceEvent ...>
    >>> violations = validate_tool_results(recorder.events)
    >>> violations[0].code
    'TOOL_RESULT_BEFORE_CALL'

    Multiple calls with IDs:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("fetch", {"url": "a.com"}, tool_call_id="call_1")
    <TraceEvent ...>
    >>> recorder.record_tool_call("fetch", {"url": "b.com"}, tool_call_id="call_2")
    <TraceEvent ...>
    >>> recorder.record_tool_result("fetch", "data_b", tool_call_id="call_2")
    <TraceEvent ...>
    >>> recorder.record_tool_result("fetch", "data_a", tool_call_id="call_1")
    <TraceEvent ...>
    >>> violations = validate_tool_results(recorder.events)
    >>> len(violations)  # Order doesn't matter with IDs
    0

    Multiple calls same tool (FIFO matching without IDs):

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("calc", {"expr": "1+1"})
    <TraceEvent ...>
    >>> recorder.record_tool_call("calc", {"expr": "2+2"})
    <TraceEvent ...>
    >>> recorder.record_tool_result("calc", 2)
    <TraceEvent ...>
    >>> recorder.record_tool_result("calc", 4)
    <TraceEvent ...>
    >>> violations = validate_tool_results(recorder.events)
    >>> len(violations)
    0

    Partial results (some calls missing results):

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("tool_a", {})
    <TraceEvent ...>
    >>> recorder.record_tool_call("tool_b", {})
    <TraceEvent ...>
    >>> recorder.record_tool_result("tool_a", "done")
    <TraceEvent ...>
    >>> # tool_b never got a result
    >>> violations = validate_tool_results(recorder.events)
    >>> len(violations)
    1
    >>> "tool_b" in violations[0].detail
    True

    See Also
    --------
    validate_tool_payloads : Validates tool arguments
    validate_tool_order : Validates tool ordering
    ViolationCode : Contains TOOL_NO_RESULT and TOOL_RESULT_BEFORE_CALL
    """
    violations: list[Violation] = []
    normalized = _normalize_events(events)

    # Track pending tool calls by tool_call_id or tool_name.
    # When tool_call_id is missing, multiple calls to the same tool can be in-flight.
    pending_calls: dict[str, list[int]] = {}  # key -> [start_seq...]

    for event in normalized:
        kind = event.kind
        seq = event.seq

        if kind == TraceEventKind.TOOL_CALL_START.value:
            tool_call_id = event.payload.get("tool_call_id")
            tool_name = event.payload.get("tool_name", "unknown")
            key = tool_call_id or tool_name
            pending_calls.setdefault(key, []).append(seq)

        elif kind == TraceEventKind.TOOL_RESULT.value:
            tool_call_id = event.payload.get("tool_call_id")
            tool_name = event.payload.get("tool_name", "unknown")
            key = tool_call_id or tool_name

            if key not in pending_calls or not pending_calls[key]:
                violations.append(
                    Violation(
                        code=ViolationCode.TOOL_RESULT_BEFORE_CALL.value,
                        event_seq=seq,
                        detail=f"tool_result for '{tool_name}' without prior tool_call_start",
                        event_kind=kind,
                        context={"tool_name": tool_name},
                    )
                )
            else:
                pending_calls[key].pop(0)
                if not pending_calls[key]:
                    del pending_calls[key]

    # Check for calls without results
    for key, starts in pending_calls.items():
        for start_seq in starts:
            violations.append(
                Violation(
                    code=ViolationCode.TOOL_NO_RESULT.value,
                    event_seq=start_seq,
                    detail=f"tool_call_start for '{key}' without tool_result",
                    event_kind=TraceEventKind.TOOL_CALL_START.value,
                    context={"tool_key": key},
                )
            )

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_all(
    events: list[TraceEvent] | list[dict[str, Any]],
    tool_schemas: Optional[dict[str, ToolSchema]] = None,
    tool_order_rules: Optional[ToolOrderRule] = None,
) -> list[Violation]:
    """Run all standard trace validations in one call.

    Convenience function that executes all built-in validators and
    combines their results. This is the recommended entry point for
    comprehensive trace validation.

    The following validators are always run:

    - ``validate_generate_boundaries``: Generation lifecycle
    - ``validate_stream_boundaries``: Streaming lifecycle
    - ``validate_tool_results``: Tool call/result pairing

    The following validators are run if configuration is provided:

    - ``validate_tool_payloads``: If ``tool_schemas`` is provided
    - ``validate_tool_order``: If ``tool_order_rules`` is provided

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of trace events to validate.
    tool_schemas : dict[str, ToolSchema], optional
        Tool schemas for payload validation. If None, payload
        validation is skipped.
    tool_order_rules : ToolOrderRule, optional
        Ordering rules for tool validation. If None, order
        validation is skipped.

    Returns
    -------
    list[Violation]
        Combined list of all violations from all validators,
        sorted by event sequence number.

    Examples
    --------
    Basic validation (no schemas):

    >>> from insideLLMs.trace.tracing import TraceRecorder
    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Hello")
    <TraceEvent ...>
    >>> recorder.record_generate_end("Hi there")
    <TraceEvent ...>
    >>> violations = validate_all(recorder.events)
    >>> len(violations)
    0

    Full validation with schemas:

    >>> schemas = {
    ...     "search": ToolSchema(
    ...         name="search",
    ...         required_args=["query"],
    ...         arg_types={"query": str},
    ...     )
    ... }
    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Find info about Python")
    <TraceEvent ...>
    >>> recorder.record_tool_call("search", {"query": "Python programming"})
    <TraceEvent ...>
    >>> recorder.record_tool_result("search", ["docs.python.org"])
    <TraceEvent ...>
    >>> recorder.record_generate_end("Python is a programming language.")
    <TraceEvent ...>
    >>> violations = validate_all(recorder.events, tool_schemas=schemas)
    >>> len(violations)
    0

    Validation with ordering rules:

    >>> order_rules = ToolOrderRule(
    ...     name="auth_required",
    ...     must_follow={"api_call": ["authenticate"]},
    ... )
    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Get user data")
    <TraceEvent ...>
    >>> recorder.record_tool_call("api_call", {})  # Missing auth!
    <TraceEvent ...>
    >>> recorder.record_tool_result("api_call", "unauthorized")
    <TraceEvent ...>
    >>> recorder.record_generate_end("Access denied")
    <TraceEvent ...>
    >>> violations = validate_all(
    ...     recorder.events,
    ...     tool_order_rules=order_rules,
    ... )
    >>> any(v.code == 'TOOL_ORDER_VIOLATION' for v in violations)
    True

    Collecting multiple violation types:

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Complex operation")
    <TraceEvent ...>
    >>> recorder.record_stream_start("Streaming")
    <TraceEvent ...>
    >>> # Missing generate_end, stream_end
    >>> violations = validate_all(recorder.events)
    >>> codes = {v.code for v in violations}
    >>> 'GENERATE_NO_END' in codes
    True
    >>> 'STREAM_NO_END' in codes
    True

    CI integration pattern:

    >>> def check_trace(events, schemas=None, rules=None):
    ...     '''Validate trace and return exit code.'''
    ...     violations = validate_all(events, schemas, rules)
    ...     for v in violations:
    ...         print(f"ERROR [{v.code}] seq={v.event_seq}: {v.detail}")
    ...     return 1 if violations else 0
    >>>
    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("Test")
    <TraceEvent ...>
    >>> recorder.record_generate_end("Done")
    <TraceEvent ...>
    >>> exit_code = check_trace(recorder.events)
    >>> exit_code
    0

    See Also
    --------
    validate_generate_boundaries : Generation lifecycle validation
    validate_stream_boundaries : Streaming lifecycle validation
    validate_tool_results : Tool call/result pairing validation
    validate_tool_payloads : Tool argument validation
    validate_tool_order : Tool ordering validation
    """
    all_violations: list[Violation] = []

    # Run structural validations
    all_violations.extend(validate_generate_boundaries(events))
    all_violations.extend(validate_stream_boundaries(events))
    all_violations.extend(validate_tool_results(events))

    # Run schema validations if provided
    if tool_schemas:
        all_violations.extend(validate_tool_payloads(events, tool_schemas))

    # Run ordering validations if provided
    if tool_order_rules:
        all_violations.extend(validate_tool_order(events, tool_order_rules))

    # Sort all violations by sequence
    all_violations.sort(key=lambda v: v.event_seq)
    return all_violations


def violations_to_custom_field(violations: list[Violation]) -> list[dict[str, Any]]:
    """Convert violations to the format for ResultRecord.custom storage.

    Transforms a list of Violation objects into a list of dictionaries
    suitable for storage in the ``custom`` field of evaluation result
    records. This format is compatible with the insideLLMs result
    storage schema.

    Parameters
    ----------
    violations : list[Violation]
        List of Violation objects to convert.

    Returns
    -------
    list[dict[str, Any]]
        List of violation dictionaries, each containing keys:
        "code", "event_seq", "detail", and optionally "event_kind"
        and "context".

    Examples
    --------
    Basic conversion:

    >>> violations = [
    ...     Violation(
    ...         code="STREAM_NO_END",
    ...         event_seq=0,
    ...         detail="Missing stream_end",
    ...     ),
    ...     Violation(
    ...         code="TOOL_NO_RESULT",
    ...         event_seq=5,
    ...         detail="Missing tool_result",
    ...         context={"tool_key": "search"},
    ...     ),
    ... ]
    >>> dicts = violations_to_custom_field(violations)
    >>> len(dicts)
    2
    >>> dicts[0]["code"]
    'STREAM_NO_END'
    >>> dicts[1]["context"]["tool_key"]
    'search'

    Empty list:

    >>> violations_to_custom_field([])
    []

    Integration with result storage:

    >>> # Typical usage in evaluation pipeline:
    >>> # result = ResultRecord(
    >>> #     example_id="ex_001",
    >>> #     output="model response",
    >>> #     custom={
    >>> #         "trace_violations": violations_to_custom_field(violations),
    >>> #     },
    >>> # )

    JSON serialization:

    >>> import json
    >>> violations = [
    ...     Violation(code="TEST", event_seq=0, detail="Test violation")
    ... ]
    >>> json_str = json.dumps(violations_to_custom_field(violations))
    >>> isinstance(json_str, str)
    True

    See Also
    --------
    Violation.to_dict : Underlying conversion method
    validate_all : Produces violations for conversion
    """
    return [v.to_dict() for v in violations]


# --- Helper Functions ---


def _normalize_events(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[TraceEvent]:
    """Normalize events to TraceEvent objects, sorted by sequence.

    Internal helper function that converts a list of events (which may
    be either TraceEvent objects or dictionaries) into a sorted list
    of TraceEvent objects. This ensures consistent processing across
    all validation functions.

    Parameters
    ----------
    events : list[TraceEvent] | list[dict[str, Any]]
        List of events to normalize. Can be TraceEvent objects or
        dictionaries with keys "seq", "kind", and "payload".

    Returns
    -------
    list[TraceEvent]
        List of TraceEvent objects sorted by sequence number.
        Empty list if input is empty.

    Examples
    --------
    Normalizing dict events:

    >>> events = [
    ...     {"seq": 1, "kind": "generate_end", "payload": {}},
    ...     {"seq": 0, "kind": "generate_start", "payload": {}},
    ... ]
    >>> normalized = _normalize_events(events)
    >>> [e.seq for e in normalized]
    [0, 1]

    Normalizing TraceEvent objects:

    >>> from insideLLMs.trace.tracing import TraceEvent
    >>> events = [
    ...     TraceEvent(seq=2, kind="end", payload={}),
    ...     TraceEvent(seq=0, kind="start", payload={}),
    ...     TraceEvent(seq=1, kind="middle", payload={}),
    ... ]
    >>> normalized = _normalize_events(events)
    >>> [e.seq for e in normalized]
    [0, 1, 2]

    Empty input:

    >>> _normalize_events([])
    []

    Notes
    -----
    This function creates a new list and does not modify the input.
    The sorting is stable for events with the same sequence number.
    """
    if not events:
        return []

    if isinstance(events[0], TraceEvent):
        normalized = list(events)
    else:
        normalized = [TraceEvent.from_dict(e) for e in events]

    # Sort by sequence for consistent processing
    normalized.sort(key=lambda e: e.seq)
    return normalized
