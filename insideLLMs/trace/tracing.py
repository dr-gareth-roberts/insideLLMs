"""Trace recording for deterministic execution tracing.

This module provides utilities for recording ordered trace events during
model execution. Events are designed to be deterministic (no wall-clock time)
using logical sequencing, making traces reproducible across runs and suitable
for CI/CD pipelines.

Overview
--------
The tracing system is built around three core concepts:

1. **TraceEvent**: Immutable records of individual events with sequence numbers
2. **TraceRecorder**: Thread-safe collector that manages event recording
3. **trace_fingerprint**: Stable hash function for trace comparison

The module also provides:

- **TraceEventKind**: Enum of standard event types for common operations
- **trace_to_custom_field**: Formatter for the insideLLMs.custom.trace@1 schema
- **Key ordering constants**: For stable, human-readable JSON output

Design Philosophy
-----------------
Traditional logging uses wall-clock timestamps, which makes traces
non-deterministic across runs (network latency, system load, etc. cause
variation). This tracing system uses logical sequence numbers instead,
ensuring that identical model behavior produces identical traces.

This enables powerful use cases:

- **Regression detection**: Compare fingerprints to detect behavioral drift
- **Reproducibility**: Replay traces for debugging without timing issues
- **Contract validation**: Verify event ordering and payload schemas
- **CI integration**: Deterministic traces work reliably in CI pipelines

Architecture
------------
The tracing module is part of a larger trace subsystem::

    insideLLMs.trace/
    ├── tracing.py        # This module: event recording and fingerprinting
    ├── trace_config.py   # Configuration and payload normalisation
    └── trace_contracts.py # Contract validation for traces

Events flow through the system as follows::

    User Code → TraceRecorder.record() → TraceEvent → fingerprint/validate

Examples
--------
Basic generation tracing:

    >>> from insideLLMs.trace.tracing import TraceRecorder, TraceEventKind
    >>> recorder = TraceRecorder(run_id="abc123", example_id="ex_001")
    >>> recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hello"})
    TraceEvent(seq=0, kind='generate_start', payload={'prompt': 'Hello'}, ...)
    >>> recorder.record(TraceEventKind.GENERATE_END, {"response": "Hi there!"})
    TraceEvent(seq=1, kind='generate_end', payload={'response': 'Hi there!'}, ...)
    >>> len(recorder.events)
    2

Using convenience methods for common patterns:

    >>> recorder = TraceRecorder()
    >>> recorder.record_generate_start("What is 2+2?", temperature=0.7)
    TraceEvent(seq=0, kind='generate_start', ...)
    >>> recorder.record_generate_end("2+2 equals 4.", usage={"tokens": 10})
    TraceEvent(seq=1, kind='generate_end', ...)

Streaming trace with chunk tracking:

    >>> recorder = TraceRecorder()
    >>> recorder.record_stream_start("Explain Python")
    TraceEvent(seq=0, kind='stream_start', ...)
    >>> for i, chunk in enumerate(["Python ", "is ", "great!"]):
    ...     recorder.record_stream_chunk(chunk, i)
    TraceEvent(seq=1, kind='stream_chunk', ...)
    TraceEvent(seq=2, kind='stream_chunk', ...)
    TraceEvent(seq=3, kind='stream_chunk', ...)
    >>> recorder.record_stream_end("Python is great!", chunk_count=3)
    TraceEvent(seq=4, kind='stream_end', ...)

Tool calling with results:

    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("calculator", {"expression": "2+2"}, tool_call_id="tc_1")
    TraceEvent(seq=0, kind='tool_call_start', ...)
    >>> recorder.record_tool_result("calculator", {"answer": 4}, tool_call_id="tc_1")
    TraceEvent(seq=1, kind='tool_result', ...)
    >>> recorder.get_tool_sequence()
    ['calculator']

Fingerprinting for drift detection:

    >>> from insideLLMs.trace.tracing import trace_fingerprint
    >>> events = recorder.events
    >>> fp = trace_fingerprint(events)
    >>> fp.startswith("sha256:")
    True
    >>> # Same events always produce same fingerprint
    >>> trace_fingerprint(events) == fp
    True

Error recording:

    >>> recorder = TraceRecorder()
    >>> recorder.record_error("API timeout", error_type="TimeoutError", retries=3)
    TraceEvent(seq=0, kind='error', ...)

Thread-safe concurrent recording:

    >>> import threading
    >>> recorder = TraceRecorder()
    >>> def record_in_thread(n):
    ...     recorder.record("custom", {"thread": n})
    >>> threads = [threading.Thread(target=record_in_thread, args=(i,)) for i in range(10)]
    >>> for t in threads: t.start()
    >>> for t in threads: t.join()
    >>> len(recorder.events)  # All 10 events recorded safely
    10

Serializing for persistence:

    >>> recorder = TraceRecorder(run_id="persist_test")
    >>> recorder.record("custom", {"key": "value"})
    TraceEvent(seq=0, kind='custom', ...)
    >>> data = recorder.to_list()
    >>> import json
    >>> json.dumps(data)  # Ready for storage
    '[{"seq": 0, "kind": "custom", "payload": {"key": "value"}, "run_id": "persist_test"}]'

Notes
-----
- Sequence numbers are 0-indexed and monotonically increasing
- Thread safety is achieved via ``threading.Lock`` on all state mutations
- Events are immutable after creation (dataclass with default frozen=False but no setters)
- The ``clear()`` method resets both sequence and tool step counters
- Tool step counter is independent of event sequence counter

See Also
--------
insideLLMs.trace.trace_config : Configuration for normalization and validation.
insideLLMs.trace.trace_contracts : Contract validators for trace integrity.
insideLLMs.tracing : Backwards-compatible import path.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class TraceEventKind(str, Enum):
    """Standard trace event kinds for common LLM operations.

    This enum provides a standardized vocabulary for trace events, ensuring
    consistency across different model providers and use cases. Each event
    kind represents a specific point in the model interaction lifecycle.

    The enum inherits from both ``str`` and ``Enum``, allowing direct string
    comparison and seamless JSON serialization::

        >>> TraceEventKind.GENERATE_START == "generate_start"
        True
        >>> TraceEventKind.GENERATE_START.value
        'generate_start'

    Parameters
    ----------
    value : str
        The string value of the event kind, used in serialization.

    Attributes
    ----------
    GENERATE_START : str
        Marks the beginning of a text generation request.
        Payload typically includes: prompt, model, parameters.
    GENERATE_END : str
        Marks the completion of a text generation request.
        Payload typically includes: response, usage statistics.
    CHAT_START : str
        Marks the beginning of a chat/conversation request.
        Payload typically includes: messages array, model, parameters.
    CHAT_END : str
        Marks the completion of a chat/conversation request.
        Payload typically includes: assistant message, usage statistics.
    STREAM_START : str
        Marks the beginning of a streaming response.
        Payload typically includes: prompt or messages, stream_id.
    STREAM_CHUNK : str
        Records an individual chunk in a streaming response.
        Payload typically includes: chunk content, chunk_index.
    STREAM_END : str
        Marks the completion of a streaming response.
        Payload typically includes: full_response, chunk_count.
    TOOL_CALL_START : str
        Marks when the model initiates a tool/function call.
        Payload typically includes: tool_name, arguments, tool_call_id.
    TOOL_CALL_END : str
        Marks when a tool/function call execution completes.
        Payload typically includes: tool_name, duration (if tracked).
    TOOL_RESULT : str
        Records the result returned from a tool/function call.
        Payload typically includes: tool_name, result, ok status.
    ERROR : str
        Records an error that occurred during execution.
        Payload typically includes: error message, error_type.
    RETRY : str
        Records a retry attempt after a transient failure.
        Payload typically includes: attempt number, reason, delay.
    CUSTOM : str
        Generic event kind for user-defined events.
        Payload structure is entirely user-defined.

    Examples
    --------
    Using event kinds with TraceRecorder:

        >>> from insideLLMs.trace.tracing import TraceRecorder, TraceEventKind
        >>> recorder = TraceRecorder()
        >>> recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hello"})
        TraceEvent(seq=0, kind='generate_start', ...)

    Checking event kind in validation:

        >>> event = recorder.events[0]
        >>> event.kind == TraceEventKind.GENERATE_START.value
        True

    Using in conditional logic:

        >>> if event.kind == TraceEventKind.GENERATE_START.value:
        ...     print("Generation started")
        Generation started

    String comparison works directly:

        >>> TraceEventKind.ERROR == "error"
        True
        >>> "stream_chunk" in [e.value for e in TraceEventKind]
        True

    Iterating over all event kinds:

        >>> generation_kinds = [k for k in TraceEventKind if "GENERATE" in k.name]
        >>> len(generation_kinds)
        2

    See Also
    --------
    TraceEvent : The event dataclass that uses these kinds.
    TraceRecorder : The recorder that creates events with these kinds.
    """

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

    TraceEvent is an immutable record of something that happened during model
    execution. Events are ordered by sequence number rather than timestamp,
    ensuring deterministic traces across runs.

    Each event captures:

    - **What happened**: The ``kind`` field (e.g., "generate_start", "tool_result")
    - **When in sequence**: The ``seq`` field (monotonically increasing integer)
    - **Event details**: The ``payload`` dict with event-specific data
    - **Context**: Optional ``run_id`` and ``example_id`` for grouping

    Parameters
    ----------
    seq : int
        Sequence number for ordering (0-indexed). Assigned automatically by
        TraceRecorder and guaranteed to be unique within a recording session.
    kind : str
        Type of event. Can be a TraceEventKind value or any custom string.
        Convention is snake_case (e.g., "my_custom_event").
    payload : dict[str, Any], optional
        Event-specific data. Structure varies by event kind. Default is
        an empty dict.
    run_id : str, optional
        Identifier for the overall run/session. Useful for grouping related
        events across multiple examples. Default is None.
    example_id : str, optional
        Identifier for the specific example/test case. Useful for filtering
        events in multi-example runs. Default is None.

    Attributes
    ----------
    seq : int
        The event's position in the trace sequence.
    kind : str
        The event type identifier.
    payload : dict[str, Any]
        Event-specific data dictionary.
    run_id : Optional[str]
        Run identifier for context grouping.
    example_id : Optional[str]
        Example identifier for context grouping.

    Examples
    --------
    Creating events manually (usually done by TraceRecorder):

        >>> event = TraceEvent(
        ...     seq=0,
        ...     kind="generate_start",
        ...     payload={"prompt": "Hello, world!"},
        ...     run_id="run_001",
        ...     example_id="ex_001"
        ... )
        >>> event.seq
        0
        >>> event.kind
        'generate_start'

    Converting to dict for JSON serialization:

        >>> event = TraceEvent(seq=0, kind="custom", payload={"key": "value"})
        >>> d = event.to_dict()
        >>> d["seq"]
        0
        >>> d["kind"]
        'custom'
        >>> d["payload"]
        {'key': 'value'}

    Creating from a dictionary (e.g., loaded from JSON):

        >>> data = {"seq": 1, "kind": "generate_end", "payload": {"response": "Hi!"}}
        >>> event = TraceEvent.from_dict(data)
        >>> event.kind
        'generate_end'
        >>> event.payload["response"]
        'Hi!'

    Optional fields are omitted from dict when None:

        >>> event = TraceEvent(seq=0, kind="test", payload={})
        >>> d = event.to_dict()
        >>> "run_id" in d
        False
        >>> event_with_id = TraceEvent(seq=0, kind="test", run_id="run_1")
        >>> "run_id" in event_with_id.to_dict()
        True

    Round-trip serialization:

        >>> original = TraceEvent(seq=5, kind="custom", payload={"data": [1, 2, 3]})
        >>> restored = TraceEvent.from_dict(original.to_dict())
        >>> original.seq == restored.seq
        True
        >>> original.payload == restored.payload
        True

    See Also
    --------
    TraceEventKind : Standard event kind values.
    TraceRecorder : Creates TraceEvent instances with auto-incrementing seq.
    trace_fingerprint : Computes stable hashes over event sequences.
    """

    seq: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    example_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary for serialization.

        Creates a dictionary representation suitable for JSON serialization.
        Optional fields (run_id, example_id) are only included if they have
        non-None values, keeping the output compact.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``seq``: int - The sequence number
            - ``kind``: str - The event kind
            - ``payload``: dict - The event payload
            - ``run_id``: str - Only present if not None
            - ``example_id``: str - Only present if not None

        Examples
        --------
        Basic conversion:

            >>> event = TraceEvent(seq=0, kind="test", payload={"key": "value"})
            >>> event.to_dict()
            {'seq': 0, 'kind': 'test', 'payload': {'key': 'value'}}

        With optional fields:

            >>> event = TraceEvent(
            ...     seq=1,
            ...     kind="generate_start",
            ...     payload={"prompt": "Hi"},
            ...     run_id="run_001"
            ... )
            >>> d = event.to_dict()
            >>> d["run_id"]
            'run_001'
            >>> "example_id" in d
            False

        JSON serialization:

            >>> import json
            >>> event = TraceEvent(seq=0, kind="custom", payload={"nested": {"a": 1}})
            >>> json.dumps(event.to_dict())
            '{"seq": 0, "kind": "custom", "payload": {"nested": {"a": 1}}}'

        See Also
        --------
        from_dict : Inverse operation to reconstruct TraceEvent from dict.
        """
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
        """Create a TraceEvent from a dictionary.

        Factory method that reconstructs a TraceEvent from its dictionary
        representation. Useful for deserializing events from JSON storage
        or API responses.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing event data. Required keys:

            - ``seq``: int - Sequence number
            - ``kind``: str - Event kind

            Optional keys:

            - ``payload``: dict - Event data (default: {})
            - ``run_id``: str - Run identifier
            - ``example_id``: str - Example identifier

        Returns
        -------
        TraceEvent
            A new TraceEvent instance with the provided data.

        Raises
        ------
        KeyError
            If required keys (``seq``, ``kind``) are missing from data.

        Examples
        --------
        Basic reconstruction:

            >>> data = {"seq": 0, "kind": "generate_start", "payload": {"prompt": "Hello"}}
            >>> event = TraceEvent.from_dict(data)
            >>> event.kind
            'generate_start'
            >>> event.payload["prompt"]
            'Hello'

        With optional fields:

            >>> data = {
            ...     "seq": 1,
            ...     "kind": "tool_result",
            ...     "payload": {"result": 42},
            ...     "run_id": "run_001",
            ...     "example_id": "ex_001"
            ... }
            >>> event = TraceEvent.from_dict(data)
            >>> event.run_id
            'run_001'

        Missing payload defaults to empty dict:

            >>> data = {"seq": 0, "kind": "custom"}
            >>> event = TraceEvent.from_dict(data)
            >>> event.payload
            {}

        Loading from JSON:

            >>> import json
            >>> json_str = '{"seq": 0, "kind": "test", "payload": {"x": 1}}'
            >>> event = TraceEvent.from_dict(json.loads(json_str))
            >>> event.payload["x"]
            1

        See Also
        --------
        to_dict : Inverse operation to convert TraceEvent to dict.
        """
        return cls(
            seq=data["seq"],
            kind=data["kind"],
            payload=data.get("payload", {}),
            run_id=data.get("run_id"),
            example_id=data.get("example_id"),
        )


class TraceRecorder:
    """Records ordered trace events for a single execution.

    TraceRecorder is the primary interface for capturing trace events during
    model execution. It provides thread-safe event recording with deterministic
    sequencing, making traces reproducible across runs.

    Key features:

    - **Thread-safe**: All operations use internal locking for concurrent access
    - **Deterministic**: Events use sequence numbers, not wall-clock time
    - **Convenience methods**: Pre-built methods for common event patterns
    - **Flexible**: Accepts both TraceEventKind enums and custom strings

    Parameters
    ----------
    run_id : str, optional
        Identifier for the overall run/session. Applied to all events
        recorded by this instance. Useful for grouping events in storage.
    example_id : str, optional
        Identifier for the specific example/test case. Applied to all
        events recorded by this instance. Useful for filtering in analysis.

    Attributes
    ----------
    run_id : Optional[str]
        The run identifier (read-only property).
    example_id : Optional[str]
        The example identifier (read-only property).
    events : list[TraceEvent]
        All recorded events in sequence order (read-only copy).

    Examples
    --------
    Basic usage with standard event kinds:

        >>> from insideLLMs.trace.tracing import TraceRecorder, TraceEventKind
        >>> recorder = TraceRecorder(run_id="run_001", example_id="ex_001")
        >>> recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hello"})
        TraceEvent(seq=0, kind='generate_start', ...)
        >>> recorder.record(TraceEventKind.GENERATE_END, {"response": "World"})
        TraceEvent(seq=1, kind='generate_end', ...)
        >>> len(recorder.events)
        2

    Using convenience methods for generation:

        >>> recorder = TraceRecorder()
        >>> recorder.record_generate_start("What is Python?", temperature=0.7)
        TraceEvent(seq=0, kind='generate_start', ...)
        >>> recorder.record_generate_end("Python is a programming language.", usage={"tokens": 15})
        TraceEvent(seq=1, kind='generate_end', ...)

    Recording streaming responses:

        >>> recorder = TraceRecorder()
        >>> recorder.record_stream_start("Explain AI")
        TraceEvent(seq=0, kind='stream_start', ...)
        >>> recorder.record_stream_chunk("AI", 0)
        TraceEvent(seq=1, kind='stream_chunk', ...)
        >>> recorder.record_stream_chunk(" is", 1)
        TraceEvent(seq=2, kind='stream_chunk', ...)
        >>> recorder.record_stream_chunk(" amazing!", 2)
        TraceEvent(seq=3, kind='stream_chunk', ...)
        >>> recorder.record_stream_end("AI is amazing!", chunk_count=3)
        TraceEvent(seq=4, kind='stream_end', ...)

    Recording tool calls with results:

        >>> recorder = TraceRecorder()
        >>> recorder.record_tool_call(
        ...     tool_name="calculator",
        ...     arguments={"expression": "2 + 2"},
        ...     tool_call_id="call_001"
        ... )
        TraceEvent(seq=0, kind='tool_call_start', ...)
        >>> recorder.record_tool_result(
        ...     tool_name="calculator",
        ...     result=4,
        ...     tool_call_id="call_001"
        ... )
        TraceEvent(seq=1, kind='tool_result', ...)

    Recording errors:

        >>> recorder = TraceRecorder()
        >>> recorder.record_error(
        ...     "Connection timeout",
        ...     error_type="TimeoutError",
        ...     endpoint="https://api.example.com"
        ... )
        TraceEvent(seq=0, kind='error', ...)

    Custom events with string kinds:

        >>> recorder = TraceRecorder()
        >>> recorder.record("my_custom_event", {"data": "value"})
        TraceEvent(seq=0, kind='my_custom_event', ...)

    Thread-safe concurrent recording:

        >>> import threading
        >>> recorder = TraceRecorder()
        >>> def worker(n):
        ...     recorder.record("worker_event", {"worker_id": n})
        >>> threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        >>> for t in threads: t.start()
        >>> for t in threads: t.join()
        >>> len(recorder.events)
        5

    Serializing for storage:

        >>> recorder = TraceRecorder(run_id="persist_run")
        >>> recorder.record("event_1", {"key": "value1"})
        TraceEvent(seq=0, kind='event_1', ...)
        >>> recorder.record("event_2", {"key": "value2"})
        TraceEvent(seq=1, kind='event_2', ...)
        >>> data = recorder.to_list()
        >>> len(data)
        2
        >>> data[0]["run_id"]
        'persist_run'

    Extracting tool call sequence:

        >>> recorder = TraceRecorder()
        >>> recorder.record_tool_call("search", {"query": "weather"})
        TraceEvent(seq=0, kind='tool_call_start', ...)
        >>> recorder.record_tool_call("format", {"template": "plain"})
        TraceEvent(seq=1, kind='tool_call_start', ...)
        >>> recorder.get_tool_sequence()
        ['search', 'format']

    Clearing and reusing:

        >>> recorder = TraceRecorder()
        >>> recorder.record("test", {})
        TraceEvent(seq=0, kind='test', ...)
        >>> len(recorder.events)
        1
        >>> recorder.clear()
        >>> len(recorder.events)
        0
        >>> recorder.record("new_test", {})
        TraceEvent(seq=0, kind='new_test', ...)

    Notes
    -----
    - The ``events`` property returns a copy to prevent external modification
    - Sequence numbers start at 0 and increment by 1 for each event
    - Tool step counter is separate from event sequence counter
    - The ``clear()`` method resets both sequence and tool step counters

    See Also
    --------
    TraceEvent : The event dataclass created by this recorder.
    TraceEventKind : Standard event kinds for the ``kind`` parameter.
    trace_fingerprint : Compute stable hashes from recorded events.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ):
        """Initialize the trace recorder.

        Creates a new TraceRecorder instance with optional context identifiers
        that will be attached to all recorded events.

        Parameters
        ----------
        run_id : str, optional
            Identifier for the overall run/session. All events recorded by
            this instance will have this run_id. Useful for grouping related
            events when storing traces from multiple runs.
        example_id : str, optional
            Identifier for the specific example/test case. All events will
            have this example_id. Useful for filtering events in multi-example
            evaluation runs.

        Examples
        --------
        Basic initialization:

            >>> recorder = TraceRecorder()
            >>> recorder.run_id is None
            True

        With run context:

            >>> recorder = TraceRecorder(run_id="evaluation_2024_01")
            >>> recorder.run_id
            'evaluation_2024_01'

        With full context:

            >>> recorder = TraceRecorder(run_id="run_001", example_id="example_42")
            >>> recorder.run_id
            'run_001'
            >>> recorder.example_id
            'example_42'

        Context propagates to events:

            >>> recorder = TraceRecorder(run_id="test_run")
            >>> event = recorder.record("test", {})
            >>> event.run_id
            'test_run'
        """
        self._run_id = run_id
        self._example_id = example_id
        self._lock = threading.Lock()
        self._events: list[TraceEvent] = []
        self._seq_counter = 0
        self._tool_step_counter = 0  # Track tool invocation sequence

    @property
    def run_id(self) -> Optional[str]:
        """Get the run ID for this recorder.

        Returns
        -------
        Optional[str]
            The run identifier, or None if not set.

        Examples
        --------
            >>> recorder = TraceRecorder(run_id="run_001")
            >>> recorder.run_id
            'run_001'
            >>> recorder = TraceRecorder()
            >>> recorder.run_id is None
            True
        """
        return self._run_id

    @property
    def example_id(self) -> Optional[str]:
        """Get the example ID for this recorder.

        Returns
        -------
        Optional[str]
            The example identifier, or None if not set.

        Examples
        --------
            >>> recorder = TraceRecorder(example_id="ex_001")
            >>> recorder.example_id
            'ex_001'
            >>> recorder = TraceRecorder()
            >>> recorder.example_id is None
            True
        """
        return self._example_id

    @property
    def events(self) -> list[TraceEvent]:
        """Get all recorded events as a read-only copy.

        Returns a copy of the internal event list to prevent external
        modification. The copy is made under lock for thread safety.

        Returns
        -------
        list[TraceEvent]
            List of all recorded events in sequence order.

        Examples
        --------
            >>> recorder = TraceRecorder()
            >>> recorder.record("event_1", {"a": 1})
            TraceEvent(seq=0, kind='event_1', ...)
            >>> recorder.record("event_2", {"b": 2})
            TraceEvent(seq=1, kind='event_2', ...)
            >>> events = recorder.events
            >>> len(events)
            2
            >>> events[0].kind
            'event_1'

        Modifications to returned list don't affect recorder:

            >>> events = recorder.events
            >>> events.clear()
            >>> len(recorder.events)  # Original unchanged
            2

        Thread-safe access:

            >>> import threading
            >>> def read_events():
            ...     return len(recorder.events)
            >>> # Safe to call from multiple threads
        """
        with self._lock:
            return list(self._events)

    def record(
        self,
        kind: str | TraceEventKind,
        payload: Optional[dict[str, Any]] = None,
    ) -> TraceEvent:
        """Record a new trace event.

        Creates a new TraceEvent with the next sequence number and appends it
        to the internal event list. This is the core recording method; all
        convenience methods (record_generate_start, etc.) delegate to this.

        Parameters
        ----------
        kind : str or TraceEventKind
            The event kind. Can be a TraceEventKind enum member or any string.
            If a TraceEventKind is provided, its string value is used.
        payload : dict[str, Any], optional
            Event-specific data to include in the event. If None, an empty
            dict is used. Default is None.

        Returns
        -------
        TraceEvent
            The newly created and recorded event.

        Examples
        --------
        Using TraceEventKind enum:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hi"})
            >>> event.kind
            'generate_start'
            >>> event.seq
            0

        Using string kind:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record("my_custom_kind", {"data": 123})
            >>> event.kind
            'my_custom_kind'

        Recording without payload:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record("marker")
            >>> event.payload
            {}

        Sequence numbers auto-increment:

            >>> recorder = TraceRecorder()
            >>> e1 = recorder.record("first", {})
            >>> e2 = recorder.record("second", {})
            >>> e1.seq, e2.seq
            (0, 1)

        Context is applied automatically:

            >>> recorder = TraceRecorder(run_id="run_1", example_id="ex_1")
            >>> event = recorder.record("test", {})
            >>> event.run_id
            'run_1'
            >>> event.example_id
            'ex_1'

        See Also
        --------
        record_generate_start : Convenience method for generate start events.
        record_tool_call : Convenience method for tool call events.
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
        """Record the start of a text generation call.

        Convenience method that creates a GENERATE_START event with the
        prompt and optional generation parameters.

        Parameters
        ----------
        prompt : str
            The input prompt sent to the model.
        **kwargs : Any
            Additional generation parameters (e.g., temperature, max_tokens).
            These are stored under a "params" key in the payload.

        Returns
        -------
        TraceEvent
            The recorded GENERATE_START event.

        Examples
        --------
        Basic generation start:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_generate_start("What is Python?")
            >>> event.kind
            'generate_start'
            >>> event.payload["prompt"]
            'What is Python?'

        With generation parameters:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_generate_start(
            ...     "Explain AI",
            ...     temperature=0.7,
            ...     max_tokens=100
            ... )
            >>> event.payload["params"]["temperature"]
            0.7

        In a complete generation trace:

            >>> recorder = TraceRecorder()
            >>> recorder.record_generate_start("Hello")
            TraceEvent(seq=0, kind='generate_start', ...)
            >>> recorder.record_generate_end("Hi there!")
            TraceEvent(seq=1, kind='generate_end', ...)

        See Also
        --------
        record_generate_end : Record the end of a generation.
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
        """Record the end of a text generation call.

        Convenience method that creates a GENERATE_END event with the
        generated response and optional usage statistics.

        Parameters
        ----------
        response : str
            The generated text response from the model.
        usage : dict[str, Any], optional
            Token usage information. Typically includes keys like
            "prompt_tokens", "completion_tokens", "total_tokens".
        **kwargs : Any
            Additional metadata to include in the payload (e.g., model name,
            finish_reason). Merged directly into the payload dict.

        Returns
        -------
        TraceEvent
            The recorded GENERATE_END event.

        Examples
        --------
        Basic generation end:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_generate_end("Python is a programming language.")
            >>> event.kind
            'generate_end'
            >>> event.payload["response"]
            'Python is a programming language.'

        With usage statistics:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_generate_end(
            ...     "Hello!",
            ...     usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            ... )
            >>> event.payload["usage"]["total_tokens"]
            7

        With additional metadata:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_generate_end(
            ...     "Result",
            ...     model="gpt-4",
            ...     finish_reason="stop"
            ... )
            >>> event.payload["finish_reason"]
            'stop'

        See Also
        --------
        record_generate_start : Record the start of a generation.
        """
        payload: dict[str, Any] = {"response": response}
        if usage:
            payload["usage"] = usage
        if kwargs:
            payload.update(kwargs)
        return self.record(TraceEventKind.GENERATE_END, payload)

    def record_stream_start(self, prompt: str, **kwargs: Any) -> TraceEvent:
        """Record the start of a streaming response.

        Convenience method that creates a STREAM_START event. Use this
        before recording stream chunks.

        Parameters
        ----------
        prompt : str
            The input prompt sent to the model.
        **kwargs : Any
            Additional streaming parameters (e.g., stream_id, model).
            Stored under a "params" key in the payload.

        Returns
        -------
        TraceEvent
            The recorded STREAM_START event.

        Examples
        --------
        Basic stream start:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_stream_start("Explain quantum computing")
            >>> event.kind
            'stream_start'
            >>> event.payload["prompt"]
            'Explain quantum computing'

        With stream identifier:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_stream_start(
            ...     "Tell me a story",
            ...     stream_id="stream_001"
            ... )
            >>> event.payload["params"]["stream_id"]
            'stream_001'

        Complete streaming trace:

            >>> recorder = TraceRecorder()
            >>> recorder.record_stream_start("Count to 3")
            TraceEvent(seq=0, kind='stream_start', ...)
            >>> recorder.record_stream_chunk("1", 0)
            TraceEvent(seq=1, kind='stream_chunk', ...)
            >>> recorder.record_stream_chunk(" 2", 1)
            TraceEvent(seq=2, kind='stream_chunk', ...)
            >>> recorder.record_stream_chunk(" 3", 2)
            TraceEvent(seq=3, kind='stream_chunk', ...)
            >>> recorder.record_stream_end("1 2 3", chunk_count=3)
            TraceEvent(seq=4, kind='stream_end', ...)

        See Also
        --------
        record_stream_chunk : Record an individual stream chunk.
        record_stream_end : Record the end of streaming.
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
        """Record an individual streaming chunk.

        Convenience method that creates a STREAM_CHUNK event for each
        piece of content received during streaming.

        Parameters
        ----------
        chunk : str
            The text content of this chunk.
        chunk_index : int
            The 0-based index of this chunk in the stream sequence.
            Used for ordering and validation.
        **kwargs : Any
            Additional metadata (e.g., stream_id, finish_reason for final chunk).
            Merged directly into the payload.

        Returns
        -------
        TraceEvent
            The recorded STREAM_CHUNK event.

        Examples
        --------
        Recording individual chunks:

            >>> recorder = TraceRecorder()
            >>> e1 = recorder.record_stream_chunk("Hello", 0)
            >>> e2 = recorder.record_stream_chunk(" world", 1)
            >>> e1.payload["chunk_index"]
            0
            >>> e2.payload["chunk"]
            ' world'

        With additional metadata:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_stream_chunk(
            ...     "final chunk",
            ...     5,
            ...     is_final=True
            ... )
            >>> event.payload["is_final"]
            True

        In a loop processing stream:

            >>> recorder = TraceRecorder()
            >>> chunks = ["Python", " is", " great!"]
            >>> for i, chunk in enumerate(chunks):
            ...     recorder.record_stream_chunk(chunk, i)
            TraceEvent(seq=0, kind='stream_chunk', ...)
            TraceEvent(seq=1, kind='stream_chunk', ...)
            TraceEvent(seq=2, kind='stream_chunk', ...)

        See Also
        --------
        record_stream_start : Record the start of streaming.
        record_stream_end : Record the end of streaming.
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
        """Record the end of a streaming response.

        Convenience method that creates a STREAM_END event. Should be
        called after all stream chunks have been recorded.

        Parameters
        ----------
        full_response : str, optional
            The complete accumulated response from all chunks. Useful
            for validation and debugging.
        chunk_count : int, optional
            Total number of chunks received. Useful for validation.
        **kwargs : Any
            Additional metadata (e.g., usage statistics, finish_reason).
            Merged directly into the payload.

        Returns
        -------
        TraceEvent
            The recorded STREAM_END event.

        Examples
        --------
        Basic stream end:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_stream_end()
            >>> event.kind
            'stream_end'

        With accumulated response:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_stream_end(
            ...     full_response="Python is great!",
            ...     chunk_count=3
            ... )
            >>> event.payload["full_response"]
            'Python is great!'
            >>> event.payload["chunk_count"]
            3

        With usage statistics:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_stream_end(
            ...     full_response="Done",
            ...     chunk_count=1,
            ...     total_tokens=10
            ... )
            >>> event.payload["total_tokens"]
            10

        See Also
        --------
        record_stream_start : Record the start of streaming.
        record_stream_chunk : Record individual chunks.
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
        """Record a tool/function call initiated by the model.

        Convenience method that creates a TOOL_CALL_START event. The
        recorder maintains a separate step counter for tool calls.

        Parameters
        ----------
        tool_name : str
            Name of the tool/function being called.
        arguments : dict[str, Any]
            Arguments passed to the tool. Structure depends on the tool.
        tool_call_id : str, optional
            Unique identifier for this tool call. Useful for matching
            calls with their results.
        step : int, optional
            Explicit step number for this tool call. If not provided,
            the recorder's internal tool step counter is used and
            auto-incremented.

        Returns
        -------
        TraceEvent
            The recorded TOOL_CALL_START event.

        Examples
        --------
        Basic tool call:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_call(
            ...     tool_name="search",
            ...     arguments={"query": "weather today"}
            ... )
            >>> event.kind
            'tool_call_start'
            >>> event.payload["tool_name"]
            'search'
            >>> event.payload["arguments"]["query"]
            'weather today'

        With tool call ID for matching:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_call(
            ...     tool_name="calculator",
            ...     arguments={"expression": "2 + 2"},
            ...     tool_call_id="call_abc123"
            ... )
            >>> event.payload["tool_call_id"]
            'call_abc123'

        Auto-incrementing step numbers:

            >>> recorder = TraceRecorder()
            >>> e1 = recorder.record_tool_call("tool_a", {"arg": 1})
            >>> e2 = recorder.record_tool_call("tool_b", {"arg": 2})
            >>> e1.payload["step"], e2.payload["step"]
            (0, 1)

        Explicit step numbers:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_call(
            ...     "my_tool",
            ...     {"x": 1},
            ...     step=5
            ... )
            >>> event.payload["step"]
            5

        Complete tool call/result sequence:

            >>> recorder = TraceRecorder()
            >>> recorder.record_tool_call("calc", {"x": 1}, tool_call_id="tc1")
            TraceEvent(seq=0, kind='tool_call_start', ...)
            >>> recorder.record_tool_result("calc", 42, tool_call_id="tc1")
            TraceEvent(seq=1, kind='tool_result', ...)

        See Also
        --------
        record_tool_result : Record the result of a tool call.
        get_tool_sequence : Extract the sequence of tool names.
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
        """Record the result of a tool/function call.

        Convenience method that creates a TOOL_RESULT event. Should be
        called after the tool has executed.

        Parameters
        ----------
        tool_name : str
            Name of the tool that produced this result.
        result : Any
            The return value from the tool. Can be any JSON-serializable
            type (string, dict, list, number, etc.).
        tool_call_id : str, optional
            Unique identifier matching the corresponding tool call.
        error : str, optional
            Error message if the tool call failed. When provided and
            ok is not explicitly set, ok will be set to False.
        ok : bool, optional
            Explicit success/failure flag. If not provided, inferred
            from the presence of error (ok=True if no error).

        Returns
        -------
        TraceEvent
            The recorded TOOL_RESULT event.

        Examples
        --------
        Successful tool result:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_result(
            ...     tool_name="calculator",
            ...     result=4
            ... )
            >>> event.kind
            'tool_result'
            >>> event.payload["result"]
            4
            >>> event.payload["ok"]
            True

        With matching tool call ID:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_result(
            ...     tool_name="search",
            ...     result={"items": ["a", "b", "c"]},
            ...     tool_call_id="call_123"
            ... )
            >>> event.payload["tool_call_id"]
            'call_123'

        Recording a tool error:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_result(
            ...     tool_name="api_call",
            ...     result=None,
            ...     error="Connection timeout"
            ... )
            >>> event.payload["ok"]
            False
            >>> event.payload["error"]
            'Connection timeout'

        Explicit ok flag:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_tool_result(
            ...     tool_name="validator",
            ...     result={"warnings": ["minor issue"]},
            ...     ok=True  # Still ok despite warnings
            ... )
            >>> event.payload["ok"]
            True

        See Also
        --------
        record_tool_call : Record the initiation of a tool call.
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

        Convenience method that creates an ERROR event for exceptions
        or failures that occur during execution.

        Parameters
        ----------
        error : str
            The error message describing what went wrong.
        error_type : str, optional
            The type/class name of the error (e.g., "TimeoutError",
            "ValueError", "APIError").
        **kwargs : Any
            Additional context about the error (e.g., endpoint, retries,
            stack_trace). Merged directly into the payload.

        Returns
        -------
        TraceEvent
            The recorded ERROR event.

        Examples
        --------
        Basic error recording:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_error("Connection failed")
            >>> event.kind
            'error'
            >>> event.payload["error"]
            'Connection failed'

        With error type:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_error(
            ...     "Request timed out after 30s",
            ...     error_type="TimeoutError"
            ... )
            >>> event.payload["error_type"]
            'TimeoutError'

        With additional context:

            >>> recorder = TraceRecorder()
            >>> event = recorder.record_error(
            ...     "API rate limit exceeded",
            ...     error_type="RateLimitError",
            ...     endpoint="/v1/completions",
            ...     retry_after=60,
            ...     attempts=3
            ... )
            >>> event.payload["endpoint"]
            '/v1/completions'
            >>> event.payload["retry_after"]
            60

        From exception handling:

            >>> recorder = TraceRecorder()
            >>> try:
            ...     raise ValueError("Invalid input")
            ... except Exception as e:
            ...     recorder.record_error(str(e), error_type=type(e).__name__)
            TraceEvent(seq=0, kind='error', ...)

        See Also
        --------
        TraceEventKind.RETRY : For recording retry attempts.
        """
        payload: dict[str, Any] = {"error": error}
        if error_type:
            payload["error_type"] = error_type
        if kwargs:
            payload.update(kwargs)
        return self.record(TraceEventKind.ERROR, payload)

    def clear(self) -> None:
        """Clear all recorded events and reset counters.

        Removes all events from the recorder and resets both the sequence
        counter and tool step counter to 0. Useful for reusing a recorder
        instance across multiple test cases.

        Examples
        --------
        Basic clearing:

            >>> recorder = TraceRecorder()
            >>> recorder.record("event_1", {})
            TraceEvent(seq=0, kind='event_1', ...)
            >>> recorder.record("event_2", {})
            TraceEvent(seq=1, kind='event_2', ...)
            >>> len(recorder.events)
            2
            >>> recorder.clear()
            >>> len(recorder.events)
            0

        Sequence counter resets:

            >>> recorder = TraceRecorder()
            >>> recorder.record("first", {})
            TraceEvent(seq=0, kind='first', ...)
            >>> recorder.clear()
            >>> event = recorder.record("after_clear", {})
            >>> event.seq  # Back to 0
            0

        Tool step counter also resets:

            >>> recorder = TraceRecorder()
            >>> recorder.record_tool_call("tool_a", {})
            TraceEvent(seq=0, kind='tool_call_start', ...)
            >>> recorder.clear()
            >>> event = recorder.record_tool_call("tool_b", {})
            >>> event.payload["step"]  # Back to 0
            0

        Thread-safe clearing:

            >>> import threading
            >>> recorder = TraceRecorder()
            >>> # Safe to call from any thread
        """
        with self._lock:
            self._events.clear()
            self._seq_counter = 0
            self._tool_step_counter = 0

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all events to a list of dictionaries.

        Creates a list of dictionary representations of all recorded events,
        suitable for JSON serialization and storage.

        Returns
        -------
        list[dict[str, Any]]
            List of event dictionaries, each containing seq, kind, payload,
            and optionally run_id and example_id.

        Examples
        --------
        Basic conversion:

            >>> recorder = TraceRecorder()
            >>> recorder.record("event_1", {"key": "value"})
            TraceEvent(seq=0, kind='event_1', ...)
            >>> data = recorder.to_list()
            >>> data[0]["kind"]
            'event_1'
            >>> data[0]["payload"]["key"]
            'value'

        Ready for JSON serialization:

            >>> import json
            >>> recorder = TraceRecorder(run_id="test")
            >>> recorder.record("test", {"x": 1})
            TraceEvent(seq=0, kind='test', ...)
            >>> json_str = json.dumps(recorder.to_list())
            >>> "test" in json_str
            True

        Preserves event order:

            >>> recorder = TraceRecorder()
            >>> recorder.record("first", {})
            TraceEvent(seq=0, kind='first', ...)
            >>> recorder.record("second", {})
            TraceEvent(seq=1, kind='second', ...)
            >>> data = recorder.to_list()
            >>> [d["kind"] for d in data]
            ['first', 'second']

        See Also
        --------
        TraceEvent.to_dict : Conversion method for individual events.
        trace_fingerprint : Compute hash from event list.
        """
        with self._lock:
            return [e.to_dict() for e in self._events]

    def get_tool_sequence(self) -> list[str]:
        """Extract the sequence of tool calls from the trace.

        Returns a list of tool names in the order they were called,
        filtering to only TOOL_CALL_START events.

        Returns
        -------
        list[str]
            List of tool names in order of invocation. Returns "unknown"
            for any tool call missing the tool_name in its payload.

        Examples
        --------
        Basic tool sequence extraction:

            >>> recorder = TraceRecorder()
            >>> recorder.record_tool_call("search", {"query": "test"})
            TraceEvent(seq=0, kind='tool_call_start', ...)
            >>> recorder.record_tool_call("format", {"style": "json"})
            TraceEvent(seq=1, kind='tool_call_start', ...)
            >>> recorder.get_tool_sequence()
            ['search', 'format']

        Ignores other event types:

            >>> recorder = TraceRecorder()
            >>> recorder.record_generate_start("prompt")
            TraceEvent(seq=0, kind='generate_start', ...)
            >>> recorder.record_tool_call("calculator", {"x": 1})
            TraceEvent(seq=1, kind='tool_call_start', ...)
            >>> recorder.record_generate_end("response")
            TraceEvent(seq=2, kind='generate_end', ...)
            >>> recorder.get_tool_sequence()
            ['calculator']

        Empty when no tool calls:

            >>> recorder = TraceRecorder()
            >>> recorder.record("custom", {})
            TraceEvent(seq=0, kind='custom', ...)
            >>> recorder.get_tool_sequence()
            []

        Useful for validation:

            >>> recorder = TraceRecorder()
            >>> recorder.record_tool_call("auth", {})
            TraceEvent(seq=0, kind='tool_call_start', ...)
            >>> recorder.record_tool_call("fetch", {})
            TraceEvent(seq=1, kind='tool_call_start', ...)
            >>> sequence = recorder.get_tool_sequence()
            >>> "auth" in sequence and sequence.index("auth") < sequence.index("fetch")
            True

        See Also
        --------
        record_tool_call : Record tool calls.
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
    of the trace events. This allows detecting behavioral drift between runs
    by comparing fingerprints rather than full event data.

    The fingerprinting process:

    1. Convert TraceEvent objects to dictionaries (if needed)
    2. Sort events by sequence number for deterministic ordering
    3. Serialize to canonical JSON (sorted keys, minimal whitespace)
    4. Compute SHA-256 hash of the UTF-8 encoded JSON

    Parameters
    ----------
    events : list[TraceEvent] or list[dict[str, Any]]
        The trace events to fingerprint. Can be either TraceEvent objects
        or their dictionary representations.

    Returns
    -------
    str
        A fingerprint string in the format "sha256:<64-character-hex>".
        The prefix identifies the algorithm used.

    Examples
    --------
    Basic fingerprinting from recorder:

        >>> recorder = TraceRecorder()
        >>> recorder.record("generate_start", {"prompt": "test"})
        TraceEvent(seq=0, kind='generate_start', ...)
        >>> recorder.record("generate_end", {"response": "result"})
        TraceEvent(seq=1, kind='generate_end', ...)
        >>> fp = trace_fingerprint(recorder.events)
        >>> fp.startswith("sha256:")
        True
        >>> len(fp)  # "sha256:" (7) + 64 hex chars
        71

    Fingerprinting from dictionaries:

        >>> events = [
        ...     {"seq": 0, "kind": "generate_start", "payload": {"prompt": "Hello"}},
        ...     {"seq": 1, "kind": "generate_end", "payload": {"response": "Hi"}}
        ... ]
        >>> fp = trace_fingerprint(events)
        >>> fp.startswith("sha256:")
        True

    Same events produce same fingerprint:

        >>> recorder1 = TraceRecorder()
        >>> recorder1.record("test", {"data": 123})
        TraceEvent(seq=0, kind='test', ...)
        >>> recorder2 = TraceRecorder()
        >>> recorder2.record("test", {"data": 123})
        TraceEvent(seq=0, kind='test', ...)
        >>> trace_fingerprint(recorder1.events) == trace_fingerprint(recorder2.events)
        True

    Different events produce different fingerprints:

        >>> recorder1 = TraceRecorder()
        >>> recorder1.record("test", {"data": 123})
        TraceEvent(seq=0, kind='test', ...)
        >>> recorder2 = TraceRecorder()
        >>> recorder2.record("test", {"data": 456})
        TraceEvent(seq=0, kind='test', ...)
        >>> trace_fingerprint(recorder1.events) == trace_fingerprint(recorder2.events)
        False

    Empty event list:

        >>> fp = trace_fingerprint([])
        >>> fp.startswith("sha256:")
        True

    Drift detection workflow:

        >>> # Record baseline
        >>> baseline = TraceRecorder()
        >>> baseline.record("step_1", {"value": "a"})
        TraceEvent(seq=0, kind='step_1', ...)
        >>> baseline_fp = trace_fingerprint(baseline.events)
        >>>
        >>> # Later run
        >>> current = TraceRecorder()
        >>> current.record("step_1", {"value": "a"})
        TraceEvent(seq=0, kind='step_1', ...)
        >>> current_fp = trace_fingerprint(current.events)
        >>>
        >>> # Check for drift
        >>> if baseline_fp != current_fp:
        ...     print("Behavioral drift detected!")
        ... else:
        ...     print("Behavior unchanged")
        Behavior unchanged

    Notes
    -----
    - The fingerprint is order-independent due to sorting by seq
    - run_id and example_id are included in the hash if present
    - Use payload normalisation (see trace_config) before fingerprinting
      to exclude non-deterministic fields like timestamps

    See Also
    --------
    TraceRecorder : Records events for fingerprinting.
    insideLLMs.trace.trace_config.TracePayloadNormaliser : Normalise payloads before fingerprinting.
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


# =============================================================================
# Key Ordering Constants for Stable JSON Output
# =============================================================================
#
# These constants define the preferred key ordering for trace bundle output.
# Using consistent ordering makes JSON output human-readable and diff-friendly.

TRACE_BUNDLE_V1_TOP_ORDER: list[str] = [
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
"""Top-level key ordering for trace bundle v1 schema.

Defines the preferred order of keys in the top-level trace bundle object.
Keys not in this list are appended in sorted order.
"""

TRACE_COUNTS_ORDER: list[str] = ["events_total", "events_stored", "by_kind"]
"""Key ordering for the counts section of trace bundles."""

TRACE_FP_ORDER: list[str] = ["enabled", "alg", "value", "basis"]
"""Key ordering for the fingerprint section of trace bundles."""

TRACE_NORM_ORDER: list[str] = ["kind", "name", "import", "config_hash"]
"""Key ordering for the normaliser section of trace bundles."""

TRACE_CONTRACTS_ORDER: list[str] = [
    "enabled",
    "fail_fast",
    "violations_total",
    "violations_stored",
    "by_code",
]
"""Key ordering for the contracts section of trace bundles."""

TRACE_EVENT_ORDER: list[str] = ["seq", "kind", "payload"]
"""Key ordering for individual event objects in trace bundles."""

TRACE_VIOLATION_ORDER: list[str] = ["code", "message", "event_seq", "path", "meta"]
"""Key ordering for violation objects in trace bundles."""

TRUNC_EVENTS_ORDER: list[str] = ["applied", "policy", "max_events", "dropped", "dropped_by_kind"]
"""Key ordering for event truncation settings."""

TRUNC_PAYLOADS_ORDER: list[str] = ["applied", "max_bytes", "omitted_fields"]
"""Key ordering for payload truncation settings."""

TRUNC_VIOLS_ORDER: list[str] = ["applied", "max_violations", "dropped"]
"""Key ordering for violation truncation settings."""

TRUNCATION_ORDER: list[str] = ["events", "payloads", "violations"]
"""Key ordering for the truncation section of trace bundles."""

DERIVED_TOOL_CALLS_ORDER: list[str] = ["count", "sequence", "by_tool"]
"""Key ordering for derived tool call statistics."""

DERIVED_ORDER: list[str] = ["tool_calls"]
"""Key ordering for the derived section of trace bundles."""


def _ordered_map(keys_in_order: Iterable[str], data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ordered keys, then remaining keys sorted.

    Creates a dictionary with keys in a specified preferred order, followed
    by any remaining keys in alphabetically sorted order. This ensures
    consistent, human-readable JSON output.

    Parameters
    ----------
    keys_in_order : Iterable[str]
        Preferred key ordering. Keys in data matching these come first,
        in this order.
    data : Mapping[str, Any]
        The source data to reorder.

    Returns
    -------
    dict[str, Any]
        A new dictionary with keys in the preferred order.

    Examples
    --------
    Basic key ordering:

        >>> data = {"c": 3, "a": 1, "b": 2}
        >>> _ordered_map(["a", "b"], data)
        {'a': 1, 'b': 2, 'c': 3}

    Missing preferred keys are skipped:

        >>> data = {"b": 2, "c": 3}
        >>> _ordered_map(["a", "b"], data)
        {'b': 2, 'c': 3}

    Extra keys are sorted alphabetically:

        >>> data = {"z": 26, "a": 1, "m": 13}
        >>> _ordered_map(["a"], data)
        {'a': 1, 'm': 13, 'z': 26}
    """
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
    """Create a dict with alphabetically sorted keys from an int mapping.

    Parameters
    ----------
    m : Mapping[str, int]
        Source mapping with string keys and integer values.

    Returns
    -------
    dict[str, int]
        New dictionary with keys in sorted order.

    Examples
    --------
        >>> _sorted_int_map({"z": 3, "a": 1, "m": 2})
        {'a': 1, 'm': 2, 'z': 3}
    """
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
    """Emit trace data in insideLLMs.custom.trace@1 format with stable key ordering.

    Creates a trace bundle dictionary conforming to the insideLLMs custom trace
    schema version 1. The output has consistent, human-oriented key ordering
    for readability and diff-friendliness.

    This function is typically used when storing trace data in evaluation
    results or exporting traces for analysis.

    Parameters
    ----------
    schema_version : str
        Schema version identifier (e.g., "1.0").
    mode : str
        Storage mode used (e.g., "full", "compact", "none").
    counts : dict[str, Any]
        Event count statistics. Expected keys: events_total, events_stored,
        by_kind (mapping of event kind to count).
    fingerprint : dict[str, Any]
        Fingerprint configuration and value. Expected keys: enabled, alg,
        value, basis.
    normaliser : dict[str, Any]
        Normaliser configuration. Expected keys: kind, name, import,
        config_hash.
    contracts : dict[str, Any]
        Contract validation results. Expected keys: enabled, fail_fast,
        violations_total, violations_stored, by_code.
    violations : list[dict[str, Any]]
        List of contract violation records. Each should have: code, message,
        event_seq, path, meta.
    events_view : str, optional
        Description of event storage view (e.g., "full", "truncated").
        Omitted from output if None.
    events : list[dict[str, Any]], optional
        The actual trace events. Omitted from output if None (compact mode).
    truncation : dict[str, Any]
        Truncation settings and statistics with sub-dicts for events,
        payloads, and violations.
    derived : dict[str, Any]
        Derived statistics with tool_calls sub-dict containing count,
        sequence, and by_tool.

    Returns
    -------
    dict[str, Any]
        Complete trace bundle with consistent key ordering suitable for
        JSON serialization.

    Examples
    --------
    Minimal trace bundle:

        >>> bundle = trace_to_custom_field(
        ...     schema_version="1.0",
        ...     mode="full",
        ...     counts={"events_total": 2, "events_stored": 2, "by_kind": {"generate_start": 1, "generate_end": 1}},
        ...     fingerprint={"enabled": True, "alg": "sha256", "value": "sha256:abc...", "basis": "normalized"},
        ...     normaliser={"kind": "builtin", "name": "structural_v1"},
        ...     contracts={"enabled": True, "fail_fast": False, "violations_total": 0, "violations_stored": 0, "by_code": {}},
        ...     violations=[],
        ...     events_view="full",
        ...     events=[{"seq": 0, "kind": "generate_start", "payload": {}}],
        ...     truncation={"events": {"applied": False}, "payloads": {"applied": False}, "violations": {"applied": False}},
        ...     derived={"tool_calls": {"count": 0, "sequence": [], "by_tool": {}}}
        ... )
        >>> bundle["schema_version"]
        '1.0'
        >>> bundle["mode"]
        'full'

    Compact mode (no events):

        >>> bundle = trace_to_custom_field(
        ...     schema_version="1.0",
        ...     mode="compact",
        ...     counts={"events_total": 10, "events_stored": 0, "by_kind": {}},
        ...     fingerprint={"enabled": True, "alg": "sha256", "value": "sha256:xyz..."},
        ...     normaliser={"kind": "builtin", "name": "structural_v1"},
        ...     contracts={"enabled": False},
        ...     violations=[],
        ...     events_view=None,
        ...     events=None,  # Not stored in compact mode
        ...     truncation={"events": {}, "payloads": {}, "violations": {}},
        ...     derived={"tool_calls": {"count": 5, "sequence": ["a", "b", "c", "d", "e"], "by_tool": {"a": 2, "b": 3}}}
        ... )
        >>> "events" in bundle
        False
        >>> bundle["derived"]["tool_calls"]["count"]
        5

    Notes
    -----
    - All sub-dictionaries have their keys reordered for consistency
    - by_kind and by_tool maps are alphabetically sorted
    - events_view and events are omitted when None
    - This function does not validate the input data structure

    See Also
    --------
    trace_fingerprint : Compute the fingerprint value.
    TraceRecorder.to_list : Get events in dict form.
    """
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
