"""Tests for insideLLMs.trace_contracts module."""

import pytest

from insideLLMs.trace_contracts import (
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
    violations_to_custom_field,
)
from insideLLMs.tracing import TraceEvent, TraceEventKind


class TestViolation:
    """Tests for Violation dataclass."""

    def test_basic_creation(self):
        """Test basic violation creation."""
        v = Violation(
            code="TEST_ERROR",
            event_seq=5,
            detail="Something went wrong",
        )
        assert v.code == "TEST_ERROR"
        assert v.event_seq == 5
        assert v.detail == "Something went wrong"
        assert v.event_kind is None
        assert v.context == {}

    def test_with_context(self):
        """Test violation with context."""
        v = Violation(
            code=ViolationCode.STREAM_NO_END.value,
            event_seq=0,
            detail="Missing stream_end",
            event_kind="stream_start",
            context={"start_seq": 0},
        )
        assert v.context == {"start_seq": 0}

    def test_to_dict(self):
        """Test serialization to dict."""
        v = Violation(
            code="ERROR",
            event_seq=10,
            detail="Test error",
            event_kind="test",
            context={"key": "value"},
        )
        d = v.to_dict()
        assert d["code"] == "ERROR"
        assert d["event_seq"] == 10
        assert d["detail"] == "Test error"
        assert d["event_kind"] == "test"
        assert d["context"] == {"key": "value"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "code": "TEST",
            "event_seq": 5,
            "detail": "Detail",
            "event_kind": "kind",
            "context": {"a": 1},
        }
        v = Violation.from_dict(data)
        assert v.code == "TEST"
        assert v.event_seq == 5
        assert v.detail == "Detail"
        assert v.event_kind == "kind"
        assert v.context == {"a": 1}


class TestValidateStreamBoundaries:
    """Tests for validate_stream_boundaries function."""

    def test_valid_stream(self):
        """Test valid streaming sequence."""
        events = [
            TraceEvent(seq=0, kind="stream_start", payload={"prompt": "Hi"}),
            TraceEvent(seq=1, kind="stream_chunk", payload={"chunk": "H", "chunk_index": 0}),
            TraceEvent(seq=2, kind="stream_chunk", payload={"chunk": "i", "chunk_index": 1}),
            TraceEvent(seq=3, kind="stream_end", payload={"chunk_count": 2}),
        ]
        violations = validate_stream_boundaries(events)
        assert len(violations) == 0

    def test_missing_stream_start(self):
        """Test chunk without stream_start."""
        events = [
            TraceEvent(seq=0, kind="stream_chunk", payload={"chunk": "x", "chunk_index": 0}),
        ]
        violations = validate_stream_boundaries(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.STREAM_CHUNK_BEFORE_START.value

    def test_missing_stream_end(self):
        """Test stream_start without stream_end."""
        events = [
            TraceEvent(seq=0, kind="stream_start", payload={"prompt": "Hi"}),
            TraceEvent(seq=1, kind="stream_chunk", payload={"chunk": "x", "chunk_index": 0}),
        ]
        violations = validate_stream_boundaries(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.STREAM_NO_END.value

    def test_stream_end_without_start(self):
        """Test stream_end without prior stream_start."""
        events = [
            TraceEvent(seq=0, kind="stream_end", payload={}),
        ]
        violations = validate_stream_boundaries(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.STREAM_NO_START.value

    def test_chunk_index_mismatch(self):
        """Test non-sequential chunk indices."""
        events = [
            TraceEvent(seq=0, kind="stream_start", payload={}),
            TraceEvent(seq=1, kind="stream_chunk", payload={"chunk": "a", "chunk_index": 0}),
            TraceEvent(
                seq=2, kind="stream_chunk", payload={"chunk": "b", "chunk_index": 2}
            ),  # Should be 1
            TraceEvent(seq=3, kind="stream_end", payload={}),
        ]
        violations = validate_stream_boundaries(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.STREAM_CHUNK_INDEX_MISMATCH.value

    def test_accepts_dict_list(self):
        """Test validation accepts list of dicts."""
        events = [
            {"seq": 0, "kind": "stream_start", "payload": {}},
            {"seq": 1, "kind": "stream_chunk", "payload": {"chunk_index": 0}},
            {"seq": 2, "kind": "stream_end", "payload": {}},
        ]
        violations = validate_stream_boundaries(events)
        assert len(violations) == 0


class TestValidateToolPayloads:
    """Tests for validate_tool_payloads function."""

    def test_valid_tool_call(self):
        """Test valid tool call payload."""
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={
                    "tool_name": "search",
                    "arguments": {"query": "test", "limit": 10},
                },
            ),
        ]
        schemas = {
            "search": ToolSchema(
                name="search",
                required_args=["query"],
                arg_types={"query": str, "limit": int},
            ),
        }
        violations = validate_tool_payloads(events, schemas)
        assert len(violations) == 0

    def test_missing_required_arg(self):
        """Test missing required argument."""
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={"tool_name": "search", "arguments": {}},
            ),
        ]
        schemas = {
            "search": ToolSchema(name="search", required_args=["query"]),
        }
        violations = validate_tool_payloads(events, schemas)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.TOOL_MISSING_REQUIRED_ARG.value

    def test_wrong_arg_type(self):
        """Test wrong argument type."""
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={
                    "tool_name": "search",
                    "arguments": {"query": 123},  # Should be str
                },
            ),
        ]
        schemas = {
            "search": ToolSchema(
                name="search",
                required_args=["query"],
                arg_types={"query": str},
            ),
        }
        violations = validate_tool_payloads(events, schemas)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.TOOL_INVALID_ARG_TYPE.value

    def test_no_schema_for_tool(self):
        """Test tool call without matching schema (no violation)."""
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={"tool_name": "unknown", "arguments": {}},
            ),
        ]
        schemas = {}  # No schema for "unknown"
        violations = validate_tool_payloads(events, schemas)
        assert len(violations) == 0

    def test_missing_tool_name(self):
        """Test tool call without tool_name."""
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={"arguments": {}},  # Missing tool_name
            ),
        ]
        violations = validate_tool_payloads(events, {})
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.INVALID_PAYLOAD.value

    def test_non_dict_arguments_is_invalid_payload(self):
        """Tool arguments must be a dict (avoid TypeError crashes)."""
        schemas = {"search": ToolSchema(name="search", required_args=["query"])}
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={"tool_name": "search", "arguments": None},
            ),
            TraceEvent(
                seq=1,
                kind="tool_call_start",
                payload={"tool_name": "search", "arguments": "not-a-dict"},
            ),
        ]
        violations = validate_tool_payloads(events, schemas)
        assert len(violations) == 2
        assert all(v.code == ViolationCode.INVALID_PAYLOAD.value for v in violations)


class TestValidateToolOrder:
    """Tests for validate_tool_order function."""

    def test_valid_order(self):
        """Test valid tool ordering."""
        events = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "search"}),
            TraceEvent(seq=1, kind="tool_call_start", payload={"tool_name": "retrieve"}),
            TraceEvent(seq=2, kind="tool_call_start", payload={"tool_name": "summarize"}),
        ]
        ruleset = ToolOrderRule(
            name="test_rules",
            must_precede={"search": ["summarize"]},  # search must come before summarize
        )
        violations = validate_tool_order(events, ruleset)
        assert len(violations) == 0

    def test_must_precede_allows_interleaving(self):
        """must_precede means each 'after' needs a prior 'before' (A B A B is OK)."""
        events = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "search"}),
            TraceEvent(seq=1, kind="tool_call_start", payload={"tool_name": "summarize"}),
            TraceEvent(seq=2, kind="tool_call_start", payload={"tool_name": "search"}),
            TraceEvent(seq=3, kind="tool_call_start", payload={"tool_name": "summarize"}),
        ]
        ruleset = ToolOrderRule(name="test_rules", must_precede={"search": ["summarize"]})
        violations = validate_tool_order(events, ruleset)
        assert len(violations) == 0

    def test_violated_must_precede(self):
        """Test violated must_precede constraint."""
        events = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "summarize"}),
            TraceEvent(seq=1, kind="tool_call_start", payload={"tool_name": "search"}),
        ]
        ruleset = ToolOrderRule(
            name="test_rules",
            must_precede={"search": ["summarize"]},  # search should come before summarize
        )
        violations = validate_tool_order(events, ruleset)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.TOOL_ORDER_VIOLATION.value

    def test_must_follow(self):
        """Test must_follow constraint."""
        ruleset = ToolOrderRule(
            name="test_rules",
            must_follow={"summarize": ["search"]},  # summarize must follow search
        )
        # No search before summarize, but also no search at all, so this might pass
        # Actually, must_follow means there should be a before_tool before this tool
        events_with_both = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "summarize"}),
            TraceEvent(seq=1, kind="tool_call_start", payload={"tool_name": "search"}),
        ]
        violations = validate_tool_order(events_with_both, ruleset)
        # summarize at 0, search at 1 - summarize should follow search but it doesn't
        assert len(violations) == 1

    def test_forbidden_sequence(self):
        """Test forbidden sequence detection."""
        events = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "delete"}),
            TraceEvent(seq=1, kind="tool_call_start", payload={"tool_name": "commit"}),
        ]
        ruleset = ToolOrderRule(
            name="test_rules",
            forbidden_sequences=[["delete", "commit"]],
        )
        violations = validate_tool_order(events, ruleset)
        assert len(violations) == 1
        assert "Forbidden tool sequence" in violations[0].detail


class TestValidateGenerateBoundaries:
    """Tests for validate_generate_boundaries function."""

    def test_valid_generate(self):
        """Test valid generate sequence."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={"prompt": "Hi"}),
            TraceEvent(seq=1, kind="generate_end", payload={"response": "Hello"}),
        ]
        violations = validate_generate_boundaries(events)
        assert len(violations) == 0

    def test_missing_generate_end(self):
        """Test generate_start without generate_end."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
        ]
        violations = validate_generate_boundaries(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.GENERATE_NO_END.value

    def test_nested_generate(self):
        """Test nested generate calls."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=1, kind="generate_start", payload={}),  # Nested!
            TraceEvent(seq=2, kind="generate_end", payload={}),
        ]
        violations = validate_generate_boundaries(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.GENERATE_NESTED.value

    def test_error_terminates_sequence(self):
        """Test error event properly terminates generate sequence."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=1, kind="error", payload={"error": "Failed"}),
        ]
        violations = validate_generate_boundaries(events)
        # Error terminates the sequence, so no violation
        assert len(violations) == 0


class TestValidateToolResults:
    """Tests for validate_tool_results function."""

    def test_valid_tool_call_with_result(self):
        """Test tool call properly paired with result."""
        events = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "search"}),
            TraceEvent(seq=1, kind="tool_result", payload={"tool_name": "search", "result": {}}),
        ]
        violations = validate_tool_results(events)
        assert len(violations) == 0

    def test_tool_call_without_result(self):
        """Test tool call without corresponding result."""
        events = [
            TraceEvent(seq=0, kind="tool_call_start", payload={"tool_name": "search"}),
        ]
        violations = validate_tool_results(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.TOOL_NO_RESULT.value

    def test_tool_result_without_call(self):
        """Test tool result without prior call."""
        events = [
            TraceEvent(seq=0, kind="tool_result", payload={"tool_name": "search", "result": {}}),
        ]
        violations = validate_tool_results(events)
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.TOOL_RESULT_BEFORE_CALL.value

    def test_tool_call_id_matching(self):
        """Test tool_call_id matching for pairing."""
        events = [
            TraceEvent(
                seq=0,
                kind="tool_call_start",
                payload={"tool_name": "a", "tool_call_id": "call_1"},
            ),
            TraceEvent(
                seq=1,
                kind="tool_call_start",
                payload={"tool_name": "b", "tool_call_id": "call_2"},
            ),
            TraceEvent(
                seq=2,
                kind="tool_result",
                payload={"tool_name": "b", "tool_call_id": "call_2", "result": {}},
            ),
            TraceEvent(
                seq=3,
                kind="tool_result",
                payload={"tool_name": "a", "tool_call_id": "call_1", "result": {}},
            ),
        ]
        violations = validate_tool_results(events)
        assert len(violations) == 0


class TestValidateAll:
    """Tests for validate_all function."""

    def test_empty_events(self):
        """Test validation of empty events."""
        violations = validate_all([])
        assert len(violations) == 0

    def test_combines_all_validations(self):
        """Test that all validations run."""
        events = [
            # Unclosed generate
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # Unclosed stream
            TraceEvent(seq=1, kind="stream_start", payload={}),
            # Tool without result
            TraceEvent(seq=2, kind="tool_call_start", payload={"tool_name": "test"}),
        ]
        violations = validate_all(events)
        # Should have violations from multiple validators
        assert len(violations) >= 2

    def test_with_tool_schemas(self):
        """Test with tool schemas provided."""
        events = [
            TraceEvent(
                seq=0,
                kind="generate_start",
                payload={},
            ),
            TraceEvent(
                seq=1,
                kind="tool_call_start",
                payload={"tool_name": "search", "arguments": {}},
            ),
            TraceEvent(
                seq=2,
                kind="tool_result",
                payload={"tool_name": "search", "result": {}},
            ),
            TraceEvent(
                seq=3,
                kind="generate_end",
                payload={},
            ),
        ]
        schemas = {"search": ToolSchema(name="search", required_args=["query"])}
        violations = validate_all(events, tool_schemas=schemas)
        # Should have one violation for missing required arg
        missing_arg_violations = [
            v for v in violations if v.code == ViolationCode.TOOL_MISSING_REQUIRED_ARG.value
        ]
        assert len(missing_arg_violations) == 1

    def test_sorted_by_sequence(self):
        """Test violations are sorted by event sequence."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=10, kind="stream_start", payload={}),
            TraceEvent(seq=5, kind="tool_call_start", payload={"tool_name": "x"}),
        ]
        violations = validate_all(events)
        # Check sorting
        seqs = [v.event_seq for v in violations]
        assert seqs == sorted(seqs)


class TestViolationsToCustomField:
    """Tests for violations_to_custom_field function."""

    def test_basic_conversion(self):
        """Test basic conversion to custom field format."""
        violations = [
            Violation(code="ERROR_1", event_seq=0, detail="First error"),
            Violation(code="ERROR_2", event_seq=5, detail="Second error"),
        ]
        result = violations_to_custom_field(violations)
        assert len(result) == 2
        assert result[0]["code"] == "ERROR_1"
        assert result[1]["code"] == "ERROR_2"
        assert result[0]["message"] == "First error"
        assert isinstance(result[0]["meta"], dict)

    def test_empty_list(self):
        """Test conversion of empty list."""
        result = violations_to_custom_field([])
        assert result == []
