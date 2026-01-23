"""Trace recording for deterministic execution tracing.

This module provides utilities for recording ordered trace events during
model execution. Events are designed to be deterministic (no wall-clock time)
using logical sequencing.

Example:
    >>> recorder = TraceRecorder(run_id="abc123", example_id="ex_001")
    >>> recorder.record("generate_start", {"prompt": "Hello"})
    >>> recorder.record("generate_end", {"response": "Hi there!"})
    >>> fingerprint = trace_fingerprint(recorder.events)
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class TraceEventKind(str, Enum):
    """Standard trace event kinds."""

    # Generation events
    GENERATE_START = "generate_start"
    GENERATE_END = "generate_end"

    # Chat events
    CHAT_START = "chat_start"
    CHAT_END = "chat_end"

    # Streaming events
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"

    # Tool/function calling events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"

    # Error events
    ERROR = "error"
    RETRY = "retry"

    # Custom events
    CUSTOM = "custom"


@dataclass
class TraceEvent:
    """A single trace event with deterministic ordering.

    Attributes:
        seq: Sequence number for ordering (0-indexed).
        kind: Type of event (from TraceEventKind or custom string).
        payload: Event-specific data.
        run_id: Optional run ID for context.
        example_id: Optional example ID for context.
    """

    seq: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    example_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "seq": self.seq,
            "kind": self.kind,
            "payload": self.payload,
        }
        if self.run_id is not None:
            d["run_id"] = self.run_id
        if self.example_id is not None:
            d["example_id"] = self.example_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        """Create from dictionary."""
        return cls(
            seq=data["seq"],
            kind=data["kind"],
            payload=data.get("payload", {}),
            run_id=data.get("run_id"),
            example_id=data.get("example_id"),
        )


class TraceRecorder:
    """Records ordered trace events for a single execution.

    Thread-safe event recording with deterministic sequencing.
    Does not use wall-clock time.

    Attributes:
        run_id: The run ID for context.
        example_id: The example ID for context.
        events: List of recorded events.

    Example:
        >>> recorder = TraceRecorder(run_id="run_001", example_id="ex_001")
        >>> recorder.record("generate_start", {"prompt": "Hello"})
        >>> recorder.record("generate_end", {"response": "World"})
        >>> print(f"Recorded {len(recorder.events)} events")
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ):
        """Initialize the trace recorder.

        Args:
            run_id: Optional run ID for all events.
            example_id: Optional example ID for all events.
        """
        self._run_id = run_id
        self._example_id = example_id
        self._lock = threading.Lock()
        self._events: list[TraceEvent] = []
        self._seq_counter = 0
        self._tool_step_counter = 0  # Track tool invocation sequence

    @property
    def run_id(self) -> Optional[str]:
        """Get the run ID."""
        return self._run_id

    @property
    def example_id(self) -> Optional[str]:
        """Get the example ID."""
        return self._example_id

    @property
    def events(self) -> list[TraceEvent]:
        """Get all recorded events (read-only copy)."""
        with self._lock:
            return list(self._events)

    def record(
        self,
        kind: str | TraceEventKind,
        payload: Optional[dict[str, Any]] = None,
    ) -> TraceEvent:
        """Record a new trace event.

        Args:
            kind: The event kind (string or TraceEventKind enum).
            payload: Optional event-specific data.

        Returns:
            The recorded TraceEvent.
        """
        if isinstance(kind, TraceEventKind):
            kind = kind.value

        with self._lock:
            event = TraceEvent(
                seq=self._seq_counter,
                kind=kind,
                payload=payload or {},
                run_id=self._run_id,
                example_id=self._example_id,
            )
            self._events.append(event)
            self._seq_counter += 1
            return event

    def record_generate_start(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> TraceEvent:
        """Record the start of a generate call.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.

        Returns:
            The recorded event.
        """
        payload = {"prompt": prompt}
        if kwargs:
            payload["params"] = kwargs
        return self.record(TraceEventKind.GENERATE_START, payload)

    def record_generate_end(
        self,
        response: str,
        usage: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """Record the end of a generate call.

        Args:
            response: The generated response.
            usage: Optional token usage information.
            **kwargs: Additional metadata.

        Returns:
            The recorded event.
        """
        payload: dict[str, Any] = {"response": response}
        if usage:
            payload["usage"] = usage
        if kwargs:
            payload.update(kwargs)
        return self.record(TraceEventKind.GENERATE_END, payload)

    def record_stream_start(self, prompt: str, **kwargs: Any) -> TraceEvent:
        """Record the start of streaming.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters.

        Returns:
            The recorded event.
        """
        payload = {"prompt": prompt}
        if kwargs:
            payload["params"] = kwargs
        return self.record(TraceEventKind.STREAM_START, payload)

    def record_stream_chunk(
        self,
        chunk: str,
        chunk_index: int,
        **kwargs: Any,
    ) -> TraceEvent:
        """Record a streaming chunk.

        Args:
            chunk: The chunk content.
            chunk_index: Index of this chunk (0-based).
            **kwargs: Additional metadata.

        Returns:
            The recorded event.
        """
        payload: dict[str, Any] = {
            "chunk": chunk,
            "chunk_index": chunk_index,
        }
        if kwargs:
            payload.update(kwargs)
        return self.record(TraceEventKind.STREAM_CHUNK, payload)

    def record_stream_end(
        self,
        full_response: Optional[str] = None,
        chunk_count: Optional[int] = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """Record the end of streaming.

        Args:
            full_response: Optional full accumulated response.
            chunk_count: Optional total number of chunks.
            **kwargs: Additional metadata.

        Returns:
            The recorded event.
        """
        payload: dict[str, Any] = {}
        if full_response is not None:
            payload["full_response"] = full_response
        if chunk_count is not None:
            payload["chunk_count"] = chunk_count
        if kwargs:
            payload.update(kwargs)
        return self.record(TraceEventKind.STREAM_END, payload)

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> TraceEvent:
        """Record a tool/function call.

        Args:
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.
            tool_call_id: Optional unique ID for the call.
            step: Optional explicit step number. If not provided, auto-incremented.

        Returns:
            The recorded event.
        """
        # Auto-increment step if not provided
        if step is None:
            with self._lock:
                step = self._tool_step_counter
                self._tool_step_counter += 1

        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "step": step,
        }
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        return self.record(TraceEventKind.TOOL_CALL_START, payload)

    def record_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: Optional[str] = None,
        error: Optional[str] = None,
        ok: Optional[bool] = None,
    ) -> TraceEvent:
        """Record a tool/function result.

        Args:
            tool_name: Name of the tool.
            result: The tool's return value.
            tool_call_id: Optional unique ID for the call.
            error: Optional error message if the tool failed.
            ok: Optional success flag. If not provided, inferred from error.

        Returns:
            The recorded event.
        """
        # Infer ok from error if not explicitly provided
        if ok is None:
            ok = error is None

        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "result": result,
            "ok": ok,
        }
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        if error:
            payload["error"] = error
        return self.record(TraceEventKind.TOOL_RESULT, payload)

    def record_error(
        self,
        error: str,
        error_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """Record an error event.

        Args:
            error: The error message.
            error_type: Optional error type/class name.
            **kwargs: Additional context.

        Returns:
            The recorded event.
        """
        payload: dict[str, Any] = {"error": error}
        if error_type:
            payload["error_type"] = error_type
        if kwargs:
            payload.update(kwargs)
        return self.record(TraceEventKind.ERROR, payload)

    def clear(self) -> None:
        """Clear all recorded events and reset counters."""
        with self._lock:
            self._events.clear()
            self._seq_counter = 0
            self._tool_step_counter = 0

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all events to a list of dictionaries."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    def get_tool_sequence(self) -> list[str]:
        """Extract the sequence of tool calls from the trace.

        Returns:
            List of tool names in order of invocation.
        """
        with self._lock:
            return [
                e.payload.get("tool_name", "unknown")
                for e in self._events
                if e.kind == TraceEventKind.TOOL_CALL_START.value
            ]


def trace_fingerprint(events: list[TraceEvent] | list[dict[str, Any]]) -> str:
    """Compute a stable fingerprint for a trace.

    The fingerprint is a SHA-256 hash of the canonical JSON serialization
    of the trace events. This allows detecting trace drift between runs.

    Args:
        events: List of TraceEvent objects or dictionaries.

    Returns:
        A fingerprint string in the format "sha256:<hex>".

    Example:
        >>> events = [{"seq": 0, "kind": "generate_start", "payload": {}}]
        >>> fp = trace_fingerprint(events)
        >>> fp.startswith("sha256:")
        True
    """
    # Normalize to list of dicts
    if events and isinstance(events[0], TraceEvent):
        event_dicts = [e.to_dict() for e in events]
    else:
        event_dicts = list(events)

    # Sort events by sequence to ensure deterministic ordering
    sorted_events = sorted(event_dicts, key=lambda e: e.get("seq", 0))

    # Canonical JSON serialization (sorted keys, no extra whitespace)
    canonical = json.dumps(sorted_events, sort_keys=True, separators=(",", ":"))

    # Compute hash
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


TRACE_BUNDLE_V1_TOP_ORDER = [
    "schema_version",
    "mode",
    "counts",
    "fingerprint",
    "normaliser",
    "contracts",
    "violations",
    "events_view",
    "events",
    "truncation",
    "derived",
]

TRACE_COUNTS_ORDER = ["events_total", "events_stored", "by_kind"]
TRACE_FP_ORDER = ["enabled", "alg", "value", "basis"]
TRACE_NORM_ORDER = ["kind", "name", "import", "config_hash"]
TRACE_CONTRACTS_ORDER = ["enabled", "fail_fast", "violations_total", "violations_stored", "by_code"]

TRACE_EVENT_ORDER = ["seq", "kind", "payload"]
TRACE_VIOLATION_ORDER = ["code", "message", "event_seq", "path", "meta"]

TRUNC_EVENTS_ORDER = ["applied", "policy", "max_events", "dropped", "dropped_by_kind"]
TRUNC_PAYLOADS_ORDER = ["applied", "max_bytes", "omitted_fields"]
TRUNC_VIOLS_ORDER = ["applied", "max_violations", "dropped"]
TRUNCATION_ORDER = ["events", "payloads", "violations"]

DERIVED_TOOL_CALLS_ORDER = ["count", "sequence", "by_tool"]
DERIVED_ORDER = ["tool_calls"]


def _ordered_map(keys_in_order: Iterable[str], data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ordered keys, then remaining keys sorted."""
    out: dict[str, Any] = {}
    seen = set()

    for k in keys_in_order:
        if k in data:
            out[k] = data[k]
            seen.add(k)

    for k in sorted(data.keys()):
        if k not in seen:
            out[k] = data[k]

    return out


def _sorted_int_map(m: Mapping[str, int]) -> dict[str, int]:
    return {k: int(m[k]) for k in sorted(m.keys())}


def trace_to_custom_field(
    *,
    schema_version: str,
    mode: str,
    counts: dict[str, Any],
    fingerprint: dict[str, Any],
    normaliser: dict[str, Any],
    contracts: dict[str, Any],
    violations: list[dict[str, Any]],
    events_view: Optional[str],
    events: Optional[list[dict[str, Any]]],
    truncation: dict[str, Any],
    derived: dict[str, Any],
) -> dict[str, Any]:
    """Emit insideLLMs.custom.trace@1 with a stable, human-oriented key order."""
    counts = dict(counts)
    counts["by_kind"] = _sorted_int_map(counts.get("by_kind", {}))

    contracts = dict(contracts)
    contracts["by_code"] = _sorted_int_map(contracts.get("by_code", {}))

    derived = dict(derived)
    tool_calls = dict(derived.get("tool_calls") or {})
    tool_calls["by_tool"] = _sorted_int_map(tool_calls.get("by_tool", {}))
    derived["tool_calls"] = _ordered_map(DERIVED_TOOL_CALLS_ORDER, tool_calls)
    derived = _ordered_map(DERIVED_ORDER, derived)

    truncation = dict(truncation)
    trunc_events = dict(truncation.get("events") or {})
    trunc_payloads = dict(truncation.get("payloads") or {})
    trunc_viols = dict(truncation.get("violations") or {})

    trunc_events["dropped_by_kind"] = _sorted_int_map(trunc_events.get("dropped_by_kind", {}))
    truncation["events"] = _ordered_map(TRUNC_EVENTS_ORDER, trunc_events)
    truncation["payloads"] = _ordered_map(TRUNC_PAYLOADS_ORDER, trunc_payloads)
    truncation["violations"] = _ordered_map(TRUNC_VIOLS_ORDER, trunc_viols)
    truncation = _ordered_map(TRUNCATION_ORDER, truncation)

    if events is not None:
        events = [_ordered_map(TRACE_EVENT_ORDER, e) for e in events]
    violations = [_ordered_map(TRACE_VIOLATION_ORDER, v) for v in violations]

    out = {
        "schema_version": schema_version,
        "mode": mode,
        "counts": _ordered_map(TRACE_COUNTS_ORDER, counts),
        "fingerprint": _ordered_map(TRACE_FP_ORDER, fingerprint),
        "normaliser": _ordered_map(TRACE_NORM_ORDER, normaliser),
        "contracts": _ordered_map(TRACE_CONTRACTS_ORDER, contracts),
        "violations": violations,
        "truncation": truncation,
        "derived": derived,
    }

    if events_view is not None:
        out["events_view"] = events_view
    if events is not None:
        out["events"] = events

    return _ordered_map(TRACE_BUNDLE_V1_TOP_ORDER, out)
