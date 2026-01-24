"""Compatibility shim for insideLLMs.trace.trace_contracts.

This module provides backward-compatible access to trace contract validation
functionality. All classes and functions are re-exported from the canonical
location at :mod:`insideLLMs.trace.trace_contracts`.

Overview
--------
Trace contracts define structural and semantic rules that trace event sequences
must satisfy. This module provides pure, deterministic validation functions that
can be used in CI pipelines to enforce trace correctness.

The validation system supports:

- **Stream boundary validation**: Ensures stream_start/stream_chunk/stream_end
  events follow proper sequencing with correct chunk indices.
- **Tool payload validation**: Validates tool call arguments against schemas,
  checking required fields and type constraints.
- **Tool ordering validation**: Enforces ordering constraints between tool calls
  (must_precede, must_follow, forbidden_sequences).
- **Generate boundary validation**: Ensures generate_start/generate_end events
  are properly paired without illegal nesting.
- **Tool result validation**: Verifies each tool_call_start has a corresponding
  tool_result.

Key Features
------------
- **Pure functions**: No I/O, no side effects, no global state mutation.
- **Deterministic**: Same input always produces the same output.
- **Stable ordering**: Violations are always sorted by event sequence number.
- **CI-friendly**: Designed for automated testing and continuous integration.

Exported Classes
----------------
ViolationCode : Enum
    Standard violation codes (STREAM_NO_START, TOOL_MISSING_REQUIRED_ARG, etc.).
Violation : dataclass
    Represents a single trace contract violation with code, details, and context.
ToolSchema : dataclass
    Schema definition for validating tool call arguments.
ToolOrderRule : dataclass
    Ordering constraints between tool calls.

Exported Functions
------------------
validate_stream_boundaries(events) -> list[Violation]
    Check stream event sequencing and chunk indices.
validate_tool_payloads(events, tool_schemas) -> list[Violation]
    Validate tool arguments against schemas.
validate_tool_order(events, ruleset) -> list[Violation]
    Check tool call ordering constraints.
validate_generate_boundaries(events) -> list[Violation]
    Check generate event pairing.
validate_tool_results(events) -> list[Violation]
    Check tool_call_start/tool_result pairing.
validate_all(events, tool_schemas=None, tool_order_rules=None) -> list[Violation]
    Run all validations and return combined violations.
violations_to_custom_field(violations) -> list[dict]
    Convert violations to serializable format.

Examples
--------
Basic stream validation:

>>> from insideLLMs.trace_contracts import validate_stream_boundaries, Violation
>>> from insideLLMs.tracing import TraceEvent
>>>
>>> # Create events with a missing stream_end
>>> events = [
...     TraceEvent(seq=0, kind="stream_start", payload={"prompt": "Hello"}),
...     TraceEvent(seq=1, kind="stream_chunk", payload={"chunk": "Hi", "chunk_index": 0}),
...     TraceEvent(seq=2, kind="stream_chunk", payload={"chunk": "!", "chunk_index": 1}),
...     # Missing stream_end!
... ]
>>> violations = validate_stream_boundaries(events)
>>> len(violations)
1
>>> violations[0].code
'STREAM_NO_END'

Tool payload validation with schemas:

>>> from insideLLMs.trace_contracts import validate_tool_payloads, ToolSchema
>>> from insideLLMs.tracing import TraceEvent
>>>
>>> # Define schema for a search tool
>>> schemas = {
...     "search": ToolSchema(
...         name="search",
...         required_args=["query"],
...         arg_types={"query": str, "limit": int},
...     )
... }
>>>
>>> # Tool call missing required argument
>>> events = [
...     TraceEvent(seq=0, kind="tool_call_start", payload={
...         "tool_name": "search",
...         "arguments": {"limit": 10}  # Missing 'query'!
...     }),
... ]
>>> violations = validate_tool_payloads(events, schemas)
>>> len(violations)
1
>>> violations[0].code
'TOOL_MISSING_REQUIRED_ARG'

Tool ordering validation:

>>> from insideLLMs.trace_contracts import validate_tool_order, ToolOrderRule
>>> from insideLLMs.tracing import TraceEvent
>>>
>>> # Define ordering rule: 'auth' must precede 'api_call'
>>> rule = ToolOrderRule(
...     name="auth_first",
...     must_precede={"auth": ["api_call"]},
... )
>>>
>>> # Violating sequence: api_call before auth
>>> events = [
...     TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "api_call"}),
...     TraceEvent(seq=1, kind="tool_call_start", payload={"tool_name": "auth"}),
... ]
>>> violations = validate_tool_order(events, rule)
>>> len(violations)
1
>>> violations[0].code
'TOOL_ORDER_VIOLATION'

Running all validations:

>>> from insideLLMs.trace_contracts import validate_all, ToolSchema
>>> from insideLLMs.tracing import TraceEvent
>>>
>>> schemas = {"fetch": ToolSchema(name="fetch", required_args=["url"])}
>>> events = [
...     TraceEvent(seq=0, kind="generate_start", payload={}),
...     TraceEvent(seq=1, kind="tool_call_start", payload={
...         "tool_name": "fetch",
...         "arguments": {}  # Missing 'url'
...     }),
...     TraceEvent(seq=2, kind="generate_end", payload={}),
... ]
>>> violations = validate_all(events, tool_schemas=schemas)
>>> len(violations)
2  # Missing url + missing tool_result

CI Integration Example
----------------------
Typical usage in a pytest test:

>>> def test_trace_contracts(recorded_events):
...     '''Verify trace events satisfy all contracts.'''
...     from insideLLMs.trace_contracts import validate_all, ToolSchema
...
...     schemas = {
...         "search": ToolSchema(name="search", required_args=["query"]),
...         "fetch": ToolSchema(name="fetch", required_args=["url"]),
...     }
...
...     violations = validate_all(recorded_events, tool_schemas=schemas)
...
...     if violations:
...         details = "\\n".join(f"  [{v.code}] seq={v.event_seq}: {v.detail}"
...                             for v in violations)
...         raise AssertionError(f"Trace contract violations:\\n{details}")

Notes
-----
This is a compatibility shim. For new code, prefer importing directly from
:mod:`insideLLMs.trace.trace_contracts`.

The validation functions accept either ``list[TraceEvent]`` or
``list[dict[str, Any]]`` for flexibility when working with serialized traces.

See Also
--------
insideLLMs.trace.trace_contracts : Canonical module location.
insideLLMs.tracing : TraceEvent and TraceRecorder classes.
insideLLMs.trace.tracing : Canonical location for tracing utilities.

Migration Guide
---------------
If you are importing from this module:

    >>> from insideLLMs.trace_contracts import validate_all

Consider migrating to the canonical import:

    >>> from insideLLMs.trace.trace_contracts import validate_all

Both imports work identically; the canonical import is preferred for clarity.
"""

from insideLLMs.trace.trace_contracts import *  # noqa: F401,F403
