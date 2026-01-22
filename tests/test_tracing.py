"""Tests for insideLLMs.tracing module."""

import pytest

from insideLLMs.tracing import (
    TraceEvent,
    TraceEventKind,
    TraceRecorder,
    trace_fingerprint,
    trace_to_custom_field,
)


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_basic_creation(self):
        """Test basic event creation."""
        event = TraceEvent(seq=0, kind="test", payload={"key": "value"})
        assert event.seq == 0
        assert event.kind == "test"
        assert event.payload == {"key": "value"}
        assert event.run_id is None
        assert event.example_id is None

    def test_with_context(self):
        """Test event with run_id and example_id."""
        event = TraceEvent(
            seq=5,
            kind="generate_start",
            payload={"prompt": "Hello"},
            run_id="run_001",
            example_id="ex_001",
        )
        assert event.run_id == "run_001"
        assert event.example_id == "ex_001"

    def test_to_dict(self):
        """Test serialization to dict."""
        event = TraceEvent(
            seq=0,
            kind="test",
            payload={"x": 1},
            run_id="r1",
        )
        d = event.to_dict()
        assert d == {
            "seq": 0,
            "kind": "test",
            "payload": {"x": 1},
            "run_id": "r1",
        }

    def test_to_dict_minimal(self):
        """Test to_dict excludes None values."""
        event = TraceEvent(seq=0, kind="test")
        d = event.to_dict()
        assert "run_id" not in d
        assert "example_id" not in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "seq": 10,
            "kind": "stream_chunk",
            "payload": {"chunk": "Hello"},
            "run_id": "r1",
        }
        event = TraceEvent.from_dict(data)
        assert event.seq == 10
        assert event.kind == "stream_chunk"
        assert event.payload == {"chunk": "Hello"}
        assert event.run_id == "r1"

    def test_roundtrip(self):
        """Test dict roundtrip preserves data."""
        original = TraceEvent(
            seq=5,
            kind="generate_end",
            payload={"response": "Hi!"},
            run_id="run_001",
            example_id="ex_001",
        )
        d = original.to_dict()
        restored = TraceEvent.from_dict(d)
        assert restored.seq == original.seq
        assert restored.kind == original.kind
        assert restored.payload == original.payload
        assert restored.run_id == original.run_id
        assert restored.example_id == original.example_id


class TestTraceRecorder:
    """Tests for TraceRecorder class."""

    def test_basic_recording(self):
        """Test basic event recording."""
        recorder = TraceRecorder()
        event = recorder.record("test", {"data": "value"})

        assert event.seq == 0
        assert event.kind == "test"
        assert len(recorder.events) == 1

    def test_sequence_incrementing(self):
        """Test that sequence numbers increment."""
        recorder = TraceRecorder()
        e1 = recorder.record("first")
        e2 = recorder.record("second")
        e3 = recorder.record("third")

        assert e1.seq == 0
        assert e2.seq == 1
        assert e3.seq == 2

    def test_context_propagation(self):
        """Test run_id and example_id propagate to events."""
        recorder = TraceRecorder(run_id="run_123", example_id="ex_456")
        event = recorder.record("test")

        assert event.run_id == "run_123"
        assert event.example_id == "ex_456"

    def test_record_with_enum_kind(self):
        """Test recording with TraceEventKind enum."""
        recorder = TraceRecorder()
        event = recorder.record(TraceEventKind.GENERATE_START, {"prompt": "Hi"})

        assert event.kind == "generate_start"

    def test_record_generate_start(self):
        """Test convenience method for generate_start."""
        recorder = TraceRecorder()
        event = recorder.record_generate_start("Hello world", temperature=0.7)

        assert event.kind == "generate_start"
        assert event.payload["prompt"] == "Hello world"
        assert event.payload["params"]["temperature"] == 0.7

    def test_record_generate_end(self):
        """Test convenience method for generate_end."""
        recorder = TraceRecorder()
        event = recorder.record_generate_end(
            "Response text",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        assert event.kind == "generate_end"
        assert event.payload["response"] == "Response text"
        assert event.payload["usage"]["prompt_tokens"] == 10

    def test_record_stream_events(self):
        """Test streaming event recording."""
        recorder = TraceRecorder()

        recorder.record_stream_start("Prompt", max_tokens=100)
        recorder.record_stream_chunk("Hello", 0)
        recorder.record_stream_chunk(" world", 1)
        recorder.record_stream_end("Hello world", chunk_count=2)

        assert len(recorder.events) == 4
        assert recorder.events[0].kind == "stream_start"
        assert recorder.events[1].kind == "stream_chunk"
        assert recorder.events[1].payload["chunk_index"] == 0
        assert recorder.events[2].kind == "stream_chunk"
        assert recorder.events[2].payload["chunk_index"] == 1
        assert recorder.events[3].kind == "stream_end"
        assert recorder.events[3].payload["chunk_count"] == 2

    def test_record_tool_call(self):
        """Test tool call recording."""
        recorder = TraceRecorder()
        event = recorder.record_tool_call(
            "search",
            {"query": "weather"},
            tool_call_id="call_001",
        )

        assert event.kind == "tool_call_start"
        assert event.payload["tool_name"] == "search"
        assert event.payload["arguments"]["query"] == "weather"
        assert event.payload["tool_call_id"] == "call_001"

    def test_record_tool_result(self):
        """Test tool result recording."""
        recorder = TraceRecorder()
        event = recorder.record_tool_result(
            "search",
            {"results": ["item1", "item2"]},
            tool_call_id="call_001",
        )

        assert event.kind == "tool_result"
        assert event.payload["tool_name"] == "search"
        assert event.payload["result"]["results"] == ["item1", "item2"]

    def test_record_error(self):
        """Test error event recording."""
        recorder = TraceRecorder()
        event = recorder.record_error(
            "Connection failed",
            error_type="ConnectionError",
            attempt=3,
        )

        assert event.kind == "error"
        assert event.payload["error"] == "Connection failed"
        assert event.payload["error_type"] == "ConnectionError"
        assert event.payload["attempt"] == 3

    def test_clear(self):
        """Test clearing the recorder."""
        recorder = TraceRecorder()
        recorder.record("one")
        recorder.record("two")
        recorder.record("three")

        assert len(recorder.events) == 3

        recorder.clear()

        assert len(recorder.events) == 0
        # Sequence should reset too
        event = recorder.record("after_clear")
        assert event.seq == 0

    def test_to_list(self):
        """Test conversion to list of dicts."""
        recorder = TraceRecorder()
        recorder.record("first", {"a": 1})
        recorder.record("second", {"b": 2})

        events_list = recorder.to_list()

        assert len(events_list) == 2
        assert events_list[0]["kind"] == "first"
        assert events_list[1]["kind"] == "second"

    def test_get_tool_sequence(self):
        """Test extracting tool sequence."""
        recorder = TraceRecorder()
        recorder.record("generate_start")
        recorder.record_tool_call("search", {"q": "test"})
        recorder.record_tool_result("search", {})
        recorder.record_tool_call("retrieve", {"id": "123"})
        recorder.record_tool_result("retrieve", {})
        recorder.record_tool_call("summarize", {"text": "..."})
        recorder.record("generate_end")

        sequence = recorder.get_tool_sequence()
        assert sequence == ["search", "retrieve", "summarize"]

    def test_events_property_returns_copy(self):
        """Test that events property returns a copy."""
        recorder = TraceRecorder()
        recorder.record("test")

        events1 = recorder.events
        events2 = recorder.events

        assert events1 is not events2
        assert events1 == events2


class TestTraceFingerprint:
    """Tests for trace_fingerprint function."""

    def test_basic_fingerprint(self):
        """Test basic fingerprint generation."""
        events = [
            TraceEvent(seq=0, kind="start"),
            TraceEvent(seq=1, kind="end"),
        ]
        fp = trace_fingerprint(events)

        assert fp.startswith("sha256:")
        assert len(fp) == 7 + 64  # "sha256:" + 64 hex chars

    def test_deterministic(self):
        """Test fingerprint is deterministic."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={"prompt": "Hello"}),
            TraceEvent(seq=1, kind="generate_end", payload={"response": "Hi"}),
        ]

        fp1 = trace_fingerprint(events)
        fp2 = trace_fingerprint(events)

        assert fp1 == fp2

    def test_order_independent(self):
        """Test fingerprint is stable regardless of input order (sorted by seq)."""
        events_ordered = [
            TraceEvent(seq=0, kind="first"),
            TraceEvent(seq=1, kind="second"),
        ]
        events_reversed = [
            TraceEvent(seq=1, kind="second"),
            TraceEvent(seq=0, kind="first"),
        ]

        fp1 = trace_fingerprint(events_ordered)
        fp2 = trace_fingerprint(events_reversed)

        assert fp1 == fp2

    def test_different_traces_different_fingerprints(self):
        """Test different traces produce different fingerprints."""
        events1 = [TraceEvent(seq=0, kind="a")]
        events2 = [TraceEvent(seq=0, kind="b")]

        fp1 = trace_fingerprint(events1)
        fp2 = trace_fingerprint(events2)

        assert fp1 != fp2

    def test_accepts_dict_list(self):
        """Test fingerprint accepts list of dicts."""
        events_dicts = [
            {"seq": 0, "kind": "start", "payload": {}},
            {"seq": 1, "kind": "end", "payload": {}},
        ]
        events_objects = [
            TraceEvent(seq=0, kind="start"),
            TraceEvent(seq=1, kind="end"),
        ]

        fp_dicts = trace_fingerprint(events_dicts)
        fp_objects = trace_fingerprint(events_objects)

        assert fp_dicts == fp_objects

    def test_empty_events(self):
        """Test fingerprint of empty events list."""
        fp = trace_fingerprint([])
        assert fp.startswith("sha256:")


class TestTraceToCustomField:
    """Tests for trace_to_custom_field function."""

    def test_basic_conversion(self):
        """Test basic conversion to custom field."""
        recorder = TraceRecorder()
        recorder.record("generate_start", {"prompt": "Hello"})
        recorder.record("generate_end", {"response": "Hi"})

        custom = trace_to_custom_field(recorder)

        assert "trace_fingerprint" in custom
        assert custom["trace_fingerprint"].startswith("sha256:")
        assert "trace_violations" in custom
        assert custom["trace_violations"] == []
        assert "tool_sequence" in custom
        assert custom["tool_sequence"] == []
        assert custom["trace_event_count"] == 2

    def test_with_tool_calls(self):
        """Test conversion with tool calls."""
        recorder = TraceRecorder()
        recorder.record_tool_call("search", {"q": "test"})
        recorder.record_tool_result("search", {})
        recorder.record_tool_call("summarize", {})

        custom = trace_to_custom_field(recorder)

        assert custom["tool_sequence"] == ["search", "summarize"]
        assert custom["trace_event_count"] == 3


class TestTraceEventKind:
    """Tests for TraceEventKind enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert TraceEventKind.GENERATE_START.value == "generate_start"
        assert TraceEventKind.GENERATE_END.value == "generate_end"
        assert TraceEventKind.STREAM_START.value == "stream_start"
        assert TraceEventKind.STREAM_CHUNK.value == "stream_chunk"
        assert TraceEventKind.STREAM_END.value == "stream_end"
        assert TraceEventKind.TOOL_CALL_START.value == "tool_call_start"
        assert TraceEventKind.TOOL_RESULT.value == "tool_result"
        assert TraceEventKind.ERROR.value == "error"

    def test_string_equality(self):
        """Test enum values compare equal to strings."""
        assert TraceEventKind.GENERATE_START == "generate_start"
        assert TraceEventKind.ERROR == "error"


class TestToolStepAndOkFields:
    """Tests for step/ok fields in tool payloads."""

    def test_tool_call_includes_step(self):
        """Test tool_call_start includes step field."""
        recorder = TraceRecorder()
        event = recorder.record_tool_call("search", {"query": "test"})

        assert "step" in event.payload
        assert event.payload["step"] == 0

    def test_tool_call_step_auto_increments(self):
        """Test step auto-increments across tool calls."""
        recorder = TraceRecorder()
        e1 = recorder.record_tool_call("search", {})
        recorder.record_tool_result("search", {})
        e2 = recorder.record_tool_call("summarize", {})
        recorder.record_tool_result("summarize", {})
        e3 = recorder.record_tool_call("email", {})

        assert e1.payload["step"] == 0
        assert e2.payload["step"] == 1
        assert e3.payload["step"] == 2

    def test_tool_call_explicit_step(self):
        """Test explicit step overrides auto-increment."""
        recorder = TraceRecorder()
        event = recorder.record_tool_call("search", {}, step=42)

        assert event.payload["step"] == 42

    def test_tool_result_includes_ok(self):
        """Test tool_result includes ok field."""
        recorder = TraceRecorder()
        event = recorder.record_tool_result("search", {"data": "result"})

        assert "ok" in event.payload
        assert event.payload["ok"] is True

    def test_tool_result_ok_false_on_error(self):
        """Test ok=False when error is present."""
        recorder = TraceRecorder()
        event = recorder.record_tool_result(
            "search",
            None,
            error="API rate limit exceeded",
        )

        assert event.payload["ok"] is False
        assert event.payload["error"] == "API rate limit exceeded"

    def test_tool_result_explicit_ok(self):
        """Test explicit ok overrides inference."""
        recorder = TraceRecorder()
        # Force ok=False even without error
        event = recorder.record_tool_result("search", {"data": "partial"}, ok=False)

        assert event.payload["ok"] is False

    def test_clear_resets_tool_step_counter(self):
        """Test clear resets the tool step counter."""
        recorder = TraceRecorder()
        recorder.record_tool_call("search", {})
        recorder.record_tool_call("summarize", {})

        recorder.clear()

        event = recorder.record_tool_call("new_search", {})
        assert event.payload["step"] == 0

    def test_step_ok_workflow(self):
        """Test complete workflow with step/ok fields."""
        recorder = TraceRecorder()

        # Simulate a multi-tool workflow
        recorder.record_generate_start("Find weather in NYC")

        # Tool 1: search (success)
        t1 = recorder.record_tool_call("search", {"query": "NYC weather"})
        r1 = recorder.record_tool_result("search", {"temp": 72})

        # Tool 2: format (failure)
        t2 = recorder.record_tool_call("format", {"data": {"temp": 72}})
        r2 = recorder.record_tool_result("format", None, error="Invalid template")

        # Tool 3: summarize (success)
        t3 = recorder.record_tool_call("summarize", {"text": "fallback"})
        r3 = recorder.record_tool_result("summarize", {"summary": "NYC: 72F"})

        recorder.record_generate_end("The weather in NYC is 72F")

        # Verify step progression
        assert t1.payload["step"] == 0
        assert t2.payload["step"] == 1
        assert t3.payload["step"] == 2

        # Verify ok status
        assert r1.payload["ok"] is True
        assert r2.payload["ok"] is False
        assert r3.payload["ok"] is True
