"""Compatibility shim for insideLLMs.trace.tracing.

This module provides backwards-compatible access to the trace recording
functionality that has been reorganized into the ``insideLLMs.trace`` subpackage.
All public symbols from ``insideLLMs.trace.tracing`` are re-exported here.

Overview
--------
The insideLLMs tracing system provides deterministic execution tracing for
LLM model interactions. Unlike traditional logging that uses wall-clock
timestamps, this tracing system uses logical sequence numbers to ensure
traces are deterministic and reproducible across runs.

Key features:

- **Deterministic ordering**: Events use sequence numbers instead of timestamps
- **Thread-safe recording**: ``TraceRecorder`` handles concurrent event recording
- **Stable fingerprinting**: ``trace_fingerprint()`` creates reproducible hashes
- **Structured events**: ``TraceEvent`` captures event kind, payload, and context
- **Standard event kinds**: ``TraceEventKind`` provides common event types

Re-exported Classes
-------------------
TraceEventKind : enum.Enum
    Standard trace event kinds (GENERATE_START, GENERATE_END, CHAT_START, etc.).
TraceEvent : dataclass
    A single trace event with deterministic ordering via sequence numbers.
TraceRecorder : class
    Thread-safe recorder for capturing ordered trace events.

Re-exported Functions
---------------------
trace_fingerprint : callable
    Compute a stable SHA-256 fingerprint for a sequence of trace events.
trace_to_custom_field : callable
    Emit trace data in the insideLLMs.custom.trace@1 format with stable key ordering.

Migration Guide
---------------
This module exists for backwards compatibility. New code should import directly
from the ``insideLLMs.trace`` package::

    # Preferred (new code)
    from insideLLMs.trace import TraceRecorder, TraceEvent, trace_fingerprint

    # Still supported (backwards compatible)
    from insideLLMs.tracing import TraceRecorder, TraceEvent, trace_fingerprint

Examples
--------
Basic trace recording for a generate call:

    >>> from insideLLMs.tracing import TraceRecorder, TraceEventKind
    >>> recorder = TraceRecorder(run_id="run_001", example_id="ex_001")
    >>> recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hello, world!"})
    TraceEvent(seq=0, kind='generate_start', ...)
    >>> recorder.record(TraceEventKind.GENERATE_END, {"response": "Hi there!"})
    TraceEvent(seq=1, kind='generate_end', ...)
    >>> len(recorder.events)
    2

Recording streaming responses:

    >>> from insideLLMs.tracing import TraceRecorder
    >>> recorder = TraceRecorder(run_id="stream_run")
    >>> recorder.record_stream_start("What is Python?")
    TraceEvent(seq=0, kind='stream_start', ...)
    >>> recorder.record_stream_chunk("Python", 0)
    TraceEvent(seq=1, kind='stream_chunk', ...)
    >>> recorder.record_stream_chunk(" is a programming language.", 1)
    TraceEvent(seq=2, kind='stream_chunk', ...)
    >>> recorder.record_stream_end("Python is a programming language.", chunk_count=2)
    TraceEvent(seq=3, kind='stream_end', ...)

Recording tool calls and results:

    >>> from insideLLMs.tracing import TraceRecorder
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call(
    ...     tool_name="search",
    ...     arguments={"query": "weather today"},
    ...     tool_call_id="call_001"
    ... )
    TraceEvent(seq=0, kind='tool_call_start', ...)
    >>> recorder.record_tool_result(
    ...     tool_name="search",
    ...     result={"temperature": 72, "conditions": "sunny"},
    ...     tool_call_id="call_001"
    ... )
    TraceEvent(seq=1, kind='tool_result', ...)

Computing trace fingerprints for drift detection:

    >>> from insideLLMs.tracing import TraceRecorder, trace_fingerprint
    >>> recorder = TraceRecorder()
    >>> recorder.record("generate_start", {"prompt": "test"})
    TraceEvent(seq=0, kind='generate_start', ...)
    >>> recorder.record("generate_end", {"response": "result"})
    TraceEvent(seq=1, kind='generate_end', ...)
    >>> fingerprint = trace_fingerprint(recorder.events)
    >>> fingerprint.startswith("sha256:")
    True

Serializing events for persistence:

    >>> from insideLLMs.tracing import TraceRecorder
    >>> recorder = TraceRecorder(run_id="persist_run")
    >>> recorder.record("custom", {"data": "value"})
    TraceEvent(seq=0, kind='custom', ...)
    >>> event_list = recorder.to_list()
    >>> event_list[0]["kind"]
    'custom'
    >>> event_list[0]["run_id"]
    'persist_run'

Extracting tool call sequences:

    >>> from insideLLMs.tracing import TraceRecorder
    >>> recorder = TraceRecorder()
    >>> recorder.record_tool_call("search", {"q": "test"})
    TraceEvent(seq=0, kind='tool_call_start', ...)
    >>> recorder.record_tool_call("calculate", {"expr": "2+2"})
    TraceEvent(seq=1, kind='tool_call_start', ...)
    >>> recorder.get_tool_sequence()
    ['search', 'calculate']

Notes
-----
- All trace operations are thread-safe via internal locking
- Sequence numbers start at 0 and increment monotonically
- Events are immutable once recorded (no modification API)
- The fingerprint algorithm is SHA-256 with canonical JSON serialization
- Tool step counters are separate from event sequence counters

See Also
--------
insideLLMs.trace.tracing : The canonical location of trace recording utilities.
insideLLMs.trace.trace_config : Configuration for trace recording and validation.
insideLLMs.trace.trace_contracts : Contract validation for trace events.
"""

from insideLLMs.trace.tracing import *  # noqa: F401,F403
