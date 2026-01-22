"""Trace contract validation for deterministic CI enforcement.

This module provides pure, deterministic validation functions for trace events.
All functions are:
- Pure (no I/O, no side effects)
- Deterministic (same input always produces same output)
- Stable ordering (violations sorted by event sequence)

Example:
    >>> from insideLLMs.tracing import TraceRecorder
    >>> from insideLLMs.trace_contracts import validate_stream_boundaries
    >>>
    >>> recorder = TraceRecorder()
    >>> recorder.record("stream_start", {"prompt": "Hello"})
    >>> recorder.record("stream_chunk", {"chunk": "Hi", "chunk_index": 0})
    >>> # Missing stream_end!
    >>>
    >>> violations = validate_stream_boundaries(recorder.events)
    >>> len(violations)  # Should find the missing stream_end
    1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.tracing import TraceEvent, TraceEventKind


class ViolationCode(str, Enum):
    """Standard violation codes for trace contracts."""

    # Stream violations
    STREAM_NO_START = "STREAM_NO_START"
    STREAM_NO_END = "STREAM_NO_END"
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
    """A trace contract violation.

    Attributes:
        code: The violation code (from ViolationCode or custom string).
        event_seq: Sequence number of the violating event.
        detail: Human-readable description of the violation.
        event_kind: The kind of event that caused the violation.
        context: Additional context for debugging.
    """

    code: str
    event_seq: int
    detail: str
    event_kind: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
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
        """Create from dictionary."""
        return cls(
            code=data["code"],
            event_seq=data["event_seq"],
            detail=data["detail"],
            event_kind=data.get("event_kind"),
            context=data.get("context", {}),
        )


@dataclass
class ToolSchema:
    """Schema definition for a tool.

    Attributes:
        name: Tool name.
        required_args: List of required argument names.
        arg_types: Optional mapping of argument names to expected types.
        optional_args: List of optional argument names.
    """

    name: str
    required_args: list[str] = field(default_factory=list)
    arg_types: dict[str, type] = field(default_factory=dict)
    optional_args: list[str] = field(default_factory=list)


@dataclass
class ToolOrderRule:
    """A rule for tool ordering constraints.

    Attributes:
        name: Rule name for identification.
        must_precede: Dict mapping tool names to tools that must come after.
        must_follow: Dict mapping tool names to tools that must come before.
        forbidden_sequences: List of forbidden tool sequences.
    """

    name: str
    must_precede: dict[str, list[str]] = field(default_factory=dict)
    must_follow: dict[str, list[str]] = field(default_factory=dict)
    forbidden_sequences: list[list[str]] = field(default_factory=list)


# --- Pure Validation Functions ---


def validate_stream_boundaries(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[Violation]:
    """Validate streaming event boundaries.

    Checks:
    - stream_start must precede stream_chunk and stream_end
    - stream_end must follow stream_start
    - chunk_index should be sequential (0, 1, 2, ...)
    - No chunks after stream_end

    Args:
        events: List of trace events.

    Returns:
        List of violations, sorted by event sequence.
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
                violations.append(Violation(
                    code=ViolationCode.GENERATE_NESTED.value,
                    event_seq=seq,
                    detail="Nested stream_start without prior stream_end",
                    event_kind=kind,
                    context={"prior_start_seq": stream_start_seq},
                ))
            stream_active = True
            stream_start_seq = seq
            last_chunk_index = -1

        elif kind == TraceEventKind.STREAM_CHUNK.value:
            if not stream_active:
                violations.append(Violation(
                    code=ViolationCode.STREAM_CHUNK_BEFORE_START.value,
                    event_seq=seq,
                    detail="stream_chunk without prior stream_start",
                    event_kind=kind,
                ))
            else:
                # Check chunk index continuity
                chunk_index = event.payload.get("chunk_index", -1)
                expected_index = last_chunk_index + 1
                if chunk_index != expected_index:
                    violations.append(Violation(
                        code=ViolationCode.STREAM_CHUNK_INDEX_MISMATCH.value,
                        event_seq=seq,
                        detail=f"Expected chunk_index {expected_index}, got {chunk_index}",
                        event_kind=kind,
                        context={
                            "expected": expected_index,
                            "actual": chunk_index,
                        },
                    ))
                last_chunk_index = chunk_index

        elif kind == TraceEventKind.STREAM_END.value:
            if not stream_active:
                violations.append(Violation(
                    code=ViolationCode.STREAM_NO_START.value,
                    event_seq=seq,
                    detail="stream_end without prior stream_start",
                    event_kind=kind,
                ))
            stream_active = False
            stream_start_seq = None

    # Check for unclosed stream
    if stream_active and stream_start_seq is not None:
        violations.append(Violation(
            code=ViolationCode.STREAM_NO_END.value,
            event_seq=stream_start_seq,
            detail="stream_start without matching stream_end",
            event_kind=TraceEventKind.STREAM_START.value,
        ))

    # Sort by event sequence for stable ordering
    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_tool_payloads(
    events: list[TraceEvent] | list[dict[str, Any]],
    tool_schemas: dict[str, ToolSchema],
) -> list[Violation]:
    """Validate tool call payloads against schemas.

    Checks:
    - Required arguments are present
    - Argument types match if specified
    - Unknown arguments are flagged (optional)

    Args:
        events: List of trace events.
        tool_schemas: Dict mapping tool names to their schemas.

    Returns:
        List of violations, sorted by event sequence.
    """
    violations: list[Violation] = []
    normalized = _normalize_events(events)

    for event in normalized:
        if event.kind != TraceEventKind.TOOL_CALL_START.value:
            continue

        tool_name = event.payload.get("tool_name")
        arguments = event.payload.get("arguments", {})

        if not tool_name:
            violations.append(Violation(
                code=ViolationCode.INVALID_PAYLOAD.value,
                event_seq=event.seq,
                detail="tool_call_start missing tool_name",
                event_kind=event.kind,
            ))
            continue

        # Check if we have a schema for this tool
        schema = tool_schemas.get(tool_name)
        if not schema:
            # No schema to validate against
            continue

        # Check required arguments
        for req_arg in schema.required_args:
            if req_arg not in arguments:
                violations.append(Violation(
                    code=ViolationCode.TOOL_MISSING_REQUIRED_ARG.value,
                    event_seq=event.seq,
                    detail=f"Tool '{tool_name}' missing required argument: {req_arg}",
                    event_kind=event.kind,
                    context={"tool_name": tool_name, "missing_arg": req_arg},
                ))

        # Check argument types
        for arg_name, expected_type in schema.arg_types.items():
            if arg_name in arguments:
                actual_value = arguments[arg_name]
                if not isinstance(actual_value, expected_type):
                    violations.append(Violation(
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
                    ))

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_tool_order(
    events: list[TraceEvent] | list[dict[str, Any]],
    ruleset: ToolOrderRule,
) -> list[Violation]:
    """Validate tool call ordering against constraints.

    Checks:
    - must_precede: Certain tools must come before others
    - must_follow: Certain tools must come after others
    - forbidden_sequences: Certain tool sequences are not allowed

    Args:
        events: List of trace events.
        ruleset: The ordering rules to enforce.

    Returns:
        List of violations, sorted by event sequence.
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
    tool_positions = {seq: i for i, (seq, _) in enumerate(tool_calls)}

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
                        violations.append(Violation(
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
                        ))

    # Check must_follow constraints
    for tool, must_come_before in ruleset.must_follow.items():
        tool_indices = [i for i, t in enumerate(tool_sequence) if t == tool]
        for before_tool in must_come_before:
            before_indices = [i for i, t in enumerate(tool_sequence) if t == before_tool]
            for ti in tool_indices:
                # Check if any before_tool comes before this tool
                has_required_predecessor = any(bi < ti for bi in before_indices)
                if not has_required_predecessor and before_indices:
                    seq = tool_calls[ti][0]
                    violations.append(Violation(
                        code=ViolationCode.TOOL_ORDER_VIOLATION.value,
                        event_seq=seq,
                        detail=f"'{tool}' must follow '{before_tool}'",
                        event_kind=TraceEventKind.TOOL_CALL_START.value,
                        context={
                            "rule_name": ruleset.name,
                            "tool": tool,
                            "must_follow": before_tool,
                        },
                    ))

    # Check forbidden sequences
    for forbidden in ruleset.forbidden_sequences:
        if len(forbidden) < 2:
            continue
        # Look for this sequence in tool_sequence
        for i in range(len(tool_sequence) - len(forbidden) + 1):
            window = tool_sequence[i:i + len(forbidden)]
            if window == forbidden:
                seq = tool_calls[i][0]
                violations.append(Violation(
                    code=ViolationCode.TOOL_ORDER_VIOLATION.value,
                    event_seq=seq,
                    detail=f"Forbidden tool sequence: {' -> '.join(forbidden)}",
                    event_kind=TraceEventKind.TOOL_CALL_START.value,
                    context={
                        "rule_name": ruleset.name,
                        "forbidden_sequence": forbidden,
                        "position": i,
                    },
                ))

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_generate_boundaries(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[Violation]:
    """Validate generate event boundaries.

    Checks:
    - generate_start must have matching generate_end
    - No nested generate calls (unless explicitly allowed)
    - Error events properly terminate sequences

    Args:
        events: List of trace events.

    Returns:
        List of violations, sorted by event sequence.
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
                violations.append(Violation(
                    code=ViolationCode.GENERATE_NESTED.value,
                    event_seq=seq,
                    detail="Nested generate_start without prior generate_end",
                    event_kind=kind,
                    context={"prior_start_seq": generate_start_seq},
                ))
            generate_active = True
            generate_start_seq = seq

        elif kind == TraceEventKind.GENERATE_END.value:
            if not generate_active:
                violations.append(Violation(
                    code=ViolationCode.GENERATE_NO_START.value,
                    event_seq=seq,
                    detail="generate_end without prior generate_start",
                    event_kind=kind,
                ))
            generate_active = False
            generate_start_seq = None

        elif kind == TraceEventKind.ERROR.value:
            # Errors can terminate a generate sequence
            if generate_active:
                generate_active = False
                generate_start_seq = None

    # Check for unclosed generate
    if generate_active and generate_start_seq is not None:
        violations.append(Violation(
            code=ViolationCode.GENERATE_NO_END.value,
            event_seq=generate_start_seq,
            detail="generate_start without matching generate_end",
            event_kind=TraceEventKind.GENERATE_START.value,
        ))

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_tool_results(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[Violation]:
    """Validate that tool calls have corresponding results.

    Checks:
    - Each tool_call_start should have a tool_result
    - tool_result should not appear before tool_call_start

    Args:
        events: List of trace events.

    Returns:
        List of violations, sorted by event sequence.
    """
    violations: list[Violation] = []
    normalized = _normalize_events(events)

    # Track pending tool calls by tool_call_id or tool_name
    pending_calls: dict[str, int] = {}  # key -> start_seq

    for event in normalized:
        kind = event.kind
        seq = event.seq

        if kind == TraceEventKind.TOOL_CALL_START.value:
            tool_call_id = event.payload.get("tool_call_id")
            tool_name = event.payload.get("tool_name", "unknown")
            key = tool_call_id or tool_name
            pending_calls[key] = seq

        elif kind == TraceEventKind.TOOL_RESULT.value:
            tool_call_id = event.payload.get("tool_call_id")
            tool_name = event.payload.get("tool_name", "unknown")
            key = tool_call_id or tool_name

            if key not in pending_calls:
                violations.append(Violation(
                    code=ViolationCode.TOOL_RESULT_BEFORE_CALL.value,
                    event_seq=seq,
                    detail=f"tool_result for '{tool_name}' without prior tool_call_start",
                    event_kind=kind,
                    context={"tool_name": tool_name},
                ))
            else:
                del pending_calls[key]

    # Check for calls without results
    for key, start_seq in pending_calls.items():
        violations.append(Violation(
            code=ViolationCode.TOOL_NO_RESULT.value,
            event_seq=start_seq,
            detail=f"tool_call_start for '{key}' without tool_result",
            event_kind=TraceEventKind.TOOL_CALL_START.value,
            context={"tool_key": key},
        ))

    violations.sort(key=lambda v: v.event_seq)
    return violations


def validate_all(
    events: list[TraceEvent] | list[dict[str, Any]],
    tool_schemas: Optional[dict[str, ToolSchema]] = None,
    tool_order_rules: Optional[ToolOrderRule] = None,
) -> list[Violation]:
    """Run all standard validations.

    Args:
        events: List of trace events.
        tool_schemas: Optional tool schemas for payload validation.
        tool_order_rules: Optional tool ordering rules.

    Returns:
        Combined list of all violations, sorted by event sequence.
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
    """Convert violations to the format for ResultRecord.custom.

    Args:
        violations: List of Violation objects.

    Returns:
        List of violation dictionaries.
    """
    return [v.to_dict() for v in violations]


# --- Helper Functions ---


def _normalize_events(
    events: list[TraceEvent] | list[dict[str, Any]],
) -> list[TraceEvent]:
    """Normalize events to TraceEvent objects, sorted by sequence."""
    if not events:
        return []

    if isinstance(events[0], TraceEvent):
        normalized = list(events)
    else:
        normalized = [TraceEvent.from_dict(e) for e in events]

    # Sort by sequence for consistent processing
    normalized.sort(key=lambda e: e.seq)
    return normalized
