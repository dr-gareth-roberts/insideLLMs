"""Tests for the prompt debugging and trace visualization module."""

import json
import time

import pytest

from insideLLMs.contrib.debugging import (
    DebugBreakpoint,
    DebugIssue,
    DebugLevel,
    DebugSession,
    ExecutionTrace,
    IssueCategory,
    IssueSeverity,
    # Classes
    PromptDebugger,
    PromptFlowAnalyzer,
    PromptInspector,
    TraceComparator,
    # Dataclasses
    TraceEvent,
    # Enums
    TraceEventType,
    TraceExporter,
    TraceVisualizer,
    VariableSnapshot,
    analyze_trace,
    compare_traces,
    # Functions
    create_debugger,
    create_visualizer,
    export_trace,
    inspect_prompt,
    quick_debug,
    render_trace,
)

# =============================================================================
# TraceEventType Enum Tests
# =============================================================================


class TestTraceEventType:
    """Tests for TraceEventType enum."""

    def test_all_event_types_exist(self):
        """Test all expected event types exist."""
        assert TraceEventType.STEP_START.value == "step_start"
        assert TraceEventType.STEP_END.value == "step_end"
        assert TraceEventType.PROMPT_SENT.value == "prompt_sent"
        assert TraceEventType.RESPONSE_RECEIVED.value == "response_received"
        assert TraceEventType.VARIABLE_SET.value == "variable_set"
        assert TraceEventType.ERROR_OCCURRED.value == "error_occurred"

    def test_event_type_values(self):
        """Test event type values."""
        assert TraceEventType.WARNING.value == "warning"
        assert TraceEventType.RETRY.value == "retry"
        assert TraceEventType.BRANCH_TAKEN.value == "branch_taken"
        assert TraceEventType.LOOP_ITERATION.value == "loop_iteration"


class TestDebugLevel:
    """Tests for DebugLevel enum."""

    def test_all_levels_exist(self):
        """Test all debug levels exist."""
        assert DebugLevel.MINIMAL.value == "minimal"
        assert DebugLevel.BASIC.value == "basic"
        assert DebugLevel.DETAILED.value == "detailed"
        assert DebugLevel.VERBOSE.value == "verbose"


class TestIssueCategory:
    """Tests for IssueCategory enum."""

    def test_all_categories_exist(self):
        """Test all issue categories exist."""
        assert IssueCategory.PERFORMANCE.value == "performance"
        assert IssueCategory.LOGIC_ERROR.value == "logic_error"
        assert IssueCategory.DATA_QUALITY.value == "data_quality"
        assert IssueCategory.PROMPT_ISSUE.value == "prompt_issue"
        assert IssueCategory.API_ERROR.value == "api_error"


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_all_severities_exist(self):
        """Test all severity levels exist."""
        assert IssueSeverity.INFO.value == "info"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.ERROR.value == "error"
        assert IssueSeverity.CRITICAL.value == "critical"


# =============================================================================
# TraceEvent Tests
# =============================================================================


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_create_event(self):
        """Test creating a trace event."""
        event = TraceEvent(
            event_type=TraceEventType.STEP_START,
            timestamp=time.time(),
            step_id="step_1",
            step_name="test_step",
        )

        assert event.event_type == TraceEventType.STEP_START
        assert event.step_id == "step_1"
        assert event.step_name == "test_step"
        assert event.event_id.startswith("evt_")

    def test_event_with_data(self):
        """Test event with data."""
        event = TraceEvent(
            event_type=TraceEventType.PROMPT_SENT,
            timestamp=time.time(),
            data={"prompt": "Hello", "model": "gpt-4"},
        )

        assert event.data["prompt"] == "Hello"
        assert event.data["model"] == "gpt-4"

    def test_event_to_dict(self):
        """Test event serialization."""
        event = TraceEvent(
            event_type=TraceEventType.STEP_END,
            timestamp=1234567890.0,
            step_id="step_1",
            step_name="test",
            data={"result": "success"},
            duration_ms=100.5,
        )

        d = event.to_dict()

        assert d["event_type"] == "step_end"
        assert d["timestamp"] == 1234567890.0
        assert d["step_id"] == "step_1"
        assert d["duration_ms"] == 100.5


# =============================================================================
# ExecutionTrace Tests
# =============================================================================


class TestExecutionTrace:
    """Tests for ExecutionTrace dataclass."""

    def test_create_trace(self):
        """Test creating an execution trace."""
        trace = ExecutionTrace(
            trace_id="test_trace",
            start_time=time.time(),
        )

        assert trace.trace_id == "test_trace"
        assert len(trace.events) == 0
        assert trace.end_time is None

    def test_trace_duration(self):
        """Test trace duration calculation."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=1000.0,
            end_time=1001.5,
        )

        assert trace.duration_ms == 1500.0

    def test_trace_duration_none(self):
        """Test trace duration when not ended."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=time.time(),
        )

        assert trace.duration_ms is None

    def test_event_count(self):
        """Test event count property."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=time.time(),
        )
        trace.events.append(
            TraceEvent(
                event_type=TraceEventType.STEP_START,
                timestamp=time.time(),
            )
        )
        trace.events.append(
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=time.time(),
            )
        )

        assert trace.event_count == 2

    def test_get_events_by_type(self):
        """Test filtering events by type."""
        trace = ExecutionTrace(trace_id="test", start_time=time.time())
        trace.events = [
            TraceEvent(event_type=TraceEventType.STEP_START, timestamp=time.time()),
            TraceEvent(event_type=TraceEventType.PROMPT_SENT, timestamp=time.time()),
            TraceEvent(event_type=TraceEventType.STEP_START, timestamp=time.time()),
        ]

        starts = trace.get_events_by_type(TraceEventType.STEP_START)

        assert len(starts) == 2

    def test_get_step_events(self):
        """Test filtering events by step ID."""
        trace = ExecutionTrace(trace_id="test", start_time=time.time())
        trace.events = [
            TraceEvent(event_type=TraceEventType.STEP_START, timestamp=time.time(), step_id="s1"),
            TraceEvent(event_type=TraceEventType.STEP_START, timestamp=time.time(), step_id="s2"),
            TraceEvent(event_type=TraceEventType.STEP_END, timestamp=time.time(), step_id="s1"),
        ]

        s1_events = trace.get_step_events("s1")

        assert len(s1_events) == 2

    def test_trace_to_dict(self):
        """Test trace serialization."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=1000.0,
            end_time=1001.0,
            metadata={"key": "value"},
        )
        trace.events.append(
            TraceEvent(
                event_type=TraceEventType.STEP_START,
                timestamp=1000.5,
            )
        )

        d = trace.to_dict()

        assert d["trace_id"] == "test"
        assert d["duration_ms"] == 1000.0
        assert d["event_count"] == 1
        assert d["metadata"]["key"] == "value"


# =============================================================================
# DebugIssue Tests
# =============================================================================


class TestDebugIssue:
    """Tests for DebugIssue dataclass."""

    def test_create_issue(self):
        """Test creating a debug issue."""
        issue = DebugIssue(
            category=IssueCategory.PERFORMANCE,
            severity=IssueSeverity.WARNING,
            message="Slow step detected",
            step_id="step_1",
        )

        assert issue.category == IssueCategory.PERFORMANCE
        assert issue.severity == IssueSeverity.WARNING
        assert issue.message == "Slow step detected"

    def test_issue_with_suggestion(self):
        """Test issue with suggestion."""
        issue = DebugIssue(
            category=IssueCategory.PROMPT_ISSUE,
            severity=IssueSeverity.INFO,
            message="Long prompt",
            suggestion="Consider compression",
        )

        assert issue.suggestion == "Consider compression"

    def test_issue_to_dict(self):
        """Test issue serialization."""
        issue = DebugIssue(
            category=IssueCategory.API_ERROR,
            severity=IssueSeverity.ERROR,
            message="Rate limit",
        )

        d = issue.to_dict()

        assert d["category"] == "api_error"
        assert d["severity"] == "error"
        assert d["message"] == "Rate limit"


# =============================================================================
# DebugBreakpoint Tests
# =============================================================================


class TestDebugBreakpoint:
    """Tests for DebugBreakpoint dataclass."""

    def test_create_breakpoint(self):
        """Test creating a breakpoint."""
        bp = DebugBreakpoint(
            breakpoint_id="bp_1",
            step_id="step_1",
        )

        assert bp.breakpoint_id == "bp_1"
        assert bp.step_id == "step_1"
        assert bp.enabled is True
        assert bp.hit_count == 0

    def test_breakpoint_with_condition(self):
        """Test breakpoint with condition."""

        def condition(ctx):
            return ctx.get("x", 0) > 10

        bp = DebugBreakpoint(
            breakpoint_id="bp_1",
            condition=condition,
        )

        assert bp.should_break({"x": 15}) is True
        assert bp.should_break({"x": 5}) is False

    def test_breakpoint_disabled(self):
        """Test disabled breakpoint."""
        bp = DebugBreakpoint(
            breakpoint_id="bp_1",
            enabled=False,
        )

        assert bp.should_break({}) is False


# =============================================================================
# VariableSnapshot Tests
# =============================================================================


class TestVariableSnapshot:
    """Tests for VariableSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a variable snapshot."""
        snap = VariableSnapshot(
            name="my_var",
            value=42,
            timestamp=time.time(),
        )

        assert snap.name == "my_var"
        assert snap.value == 42
        assert snap.operation == "set"

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snap = VariableSnapshot(
            name="data",
            value={"key": "value"},
            timestamp=1000.0,
            step_id="step_1",
        )

        d = snap.to_dict()

        assert d["name"] == "data"
        assert d["value_type"] == "dict"
        assert d["step_id"] == "step_1"


# =============================================================================
# PromptDebugger Tests
# =============================================================================


class TestPromptDebugger:
    """Tests for PromptDebugger class."""

    def test_create_debugger(self):
        """Test creating a debugger."""
        debugger = PromptDebugger()

        assert debugger.debug_level == DebugLevel.DETAILED
        assert debugger.auto_detect_issues is True

    def test_debugger_custom_settings(self):
        """Test debugger with custom settings."""
        debugger = PromptDebugger(
            debug_level=DebugLevel.VERBOSE,
            max_events=5000,
            auto_detect_issues=False,
        )

        assert debugger.debug_level == DebugLevel.VERBOSE
        assert debugger.max_events == 5000
        assert debugger.auto_detect_issues is False

    def test_start_and_end_trace(self):
        """Test starting and ending a trace."""
        debugger = PromptDebugger()

        trace = debugger.start_trace({"test": True})

        assert trace.trace_id.startswith("trace_")
        assert trace.metadata["test"] is True

        result = debugger.end_trace(result="success")

        assert result.end_time is not None
        assert result.final_result == "success"

    def test_trace_context_manager(self):
        """Test trace context manager."""
        debugger = PromptDebugger()

        with debugger.trace({"context": "test"}) as trace:
            debugger.log_event(TraceEventType.STEP_START, step_name="test")

        assert trace.end_time is not None
        assert len(trace.events) >= 1

    def test_log_event(self):
        """Test logging events."""
        debugger = PromptDebugger()
        debugger.start_trace()

        event = debugger.log_event(
            TraceEventType.STEP_START,
            step_id="s1",
            step_name="test_step",
            data={"info": "value"},
        )

        assert event.event_type == TraceEventType.STEP_START
        assert event.step_id == "s1"

        debugger.end_trace()

    def test_step_start_and_end(self):
        """Test step logging methods."""
        debugger = PromptDebugger()
        debugger.start_trace()

        debugger.step_start("s1", "step_one", param="value")
        debugger.step_end("s1", "step_one", 100.0, result="done")

        trace = debugger.end_trace()

        assert len(trace.events) >= 2
        starts = trace.get_events_by_type(TraceEventType.STEP_START)
        ends = trace.get_events_by_type(TraceEventType.STEP_END)
        assert len(starts) == 1
        assert len(ends) == 1

    def test_log_prompt(self):
        """Test logging prompts."""
        debugger = PromptDebugger()
        debugger.start_trace()

        event = debugger.log_prompt(
            "What is 2+2?",
            step_id="s1",
            model="gpt-4",
        )

        assert event.event_type == TraceEventType.PROMPT_SENT
        assert event.data["prompt"] == "What is 2+2?"
        assert event.data["model"] == "gpt-4"

        debugger.end_trace()

    def test_log_response(self):
        """Test logging responses."""
        debugger = PromptDebugger()
        debugger.start_trace()

        event = debugger.log_response(
            "The answer is 4",
            step_id="s1",
            tokens_used=10,
            latency_ms=150.0,
        )

        assert event.event_type == TraceEventType.RESPONSE_RECEIVED
        assert event.data["tokens_used"] == 10
        assert event.data["latency_ms"] == 150.0

        debugger.end_trace()

    def test_log_variable(self):
        """Test logging variables."""
        debugger = PromptDebugger()
        debugger.start_trace()

        debugger.log_variable("x", 42, step_id="s1")
        debugger.log_variable("y", "hello", operation="read")

        history = debugger.get_variable_history()

        assert len(history) == 2
        assert history[0].name == "x"
        assert history[0].value == 42

        debugger.end_trace()

    def test_log_error_string(self):
        """Test logging error as string."""
        debugger = PromptDebugger()
        debugger.start_trace()

        event = debugger.log_error("Something went wrong", step_id="s1")

        assert event.event_type == TraceEventType.ERROR_OCCURRED
        assert "Something went wrong" in event.data["error"]

        debugger.end_trace()

    def test_log_error_exception(self):
        """Test logging error as exception."""
        debugger = PromptDebugger()
        debugger.start_trace()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            event = debugger.log_error(e)

        assert "ValueError" in event.data["error"]
        assert "Test error" in event.data["error"]

        debugger.end_trace()

    def test_log_warning(self):
        """Test logging warnings."""
        debugger = PromptDebugger()
        debugger.start_trace()

        event = debugger.log_warning("This is a warning", step_id="s1")

        assert event.event_type == TraceEventType.WARNING
        assert event.data["message"] == "This is a warning"

        debugger.end_trace()

    def test_add_and_remove_breakpoint(self):
        """Test breakpoint management."""
        debugger = PromptDebugger()

        bp = debugger.add_breakpoint(
            "bp_1",
            step_id="step_1",
        )

        assert bp.breakpoint_id == "bp_1"
        assert "bp_1" in debugger._breakpoints

        removed = debugger.remove_breakpoint("bp_1")

        assert removed is True
        assert "bp_1" not in debugger._breakpoints

    def test_remove_nonexistent_breakpoint(self):
        """Test removing nonexistent breakpoint."""
        debugger = PromptDebugger()

        removed = debugger.remove_breakpoint("nonexistent")

        assert removed is False

    def test_breakpoint_handler(self):
        """Test breakpoint handler callback."""
        debugger = PromptDebugger()
        handler_called = []

        def handler(ctx):
            handler_called.append(ctx)

        debugger.set_break_handler(handler)
        debugger.add_breakpoint("bp_1", step_name="trigger")
        debugger.start_trace()

        debugger.log_event(TraceEventType.STEP_START, step_name="trigger")

        assert len(handler_called) == 1

        debugger.end_trace()

    def test_get_variable_history_filtered(self):
        """Test getting filtered variable history."""
        debugger = PromptDebugger()
        debugger.start_trace()

        debugger.log_variable("x", 1)
        debugger.log_variable("y", 2)
        debugger.log_variable("x", 3)

        x_history = debugger.get_variable_history("x")
        all_history = debugger.get_variable_history()

        assert len(x_history) == 2
        assert len(all_history) == 3

        debugger.end_trace()

    def test_get_issues_filtered(self):
        """Test getting filtered issues."""
        debugger = PromptDebugger()
        debugger.start_trace()

        # Log an error to create an issue
        debugger.log_error("Test error")

        issues = debugger.get_issues()
        error_issues = debugger.get_issues(severity=IssueSeverity.ERROR)

        assert len(issues) >= 1
        assert len(error_issues) >= 1

        debugger.end_trace()

    def test_max_events_limit(self):
        """Test max events limit."""
        debugger = PromptDebugger(max_events=5)
        debugger.start_trace()

        for i in range(10):
            debugger.log_event(TraceEventType.STEP_START, step_name=f"step_{i}")

        trace = debugger.end_trace()

        assert len(trace.events) <= 5

    def test_end_trace_without_start(self):
        """Test ending trace without start raises error."""
        debugger = PromptDebugger()

        with pytest.raises(RuntimeError):
            debugger.end_trace()


# =============================================================================
# TraceVisualizer Tests
# =============================================================================


class TestTraceVisualizer:
    """Tests for TraceVisualizer class."""

    @pytest.fixture
    def sample_trace(self):
        """Create a sample trace for testing."""
        trace = ExecutionTrace(
            trace_id="test_trace",
            start_time=1000.0,
            end_time=1002.0,
        )
        trace.events = [
            TraceEvent(
                event_type=TraceEventType.STEP_START,
                timestamp=1000.0,
                step_id="s1",
                step_name="step_one",
            ),
            TraceEvent(
                event_type=TraceEventType.PROMPT_SENT,
                timestamp=1000.5,
                step_id="s1",
                data={"prompt_length": 100},
            ),
            TraceEvent(
                event_type=TraceEventType.RESPONSE_RECEIVED,
                timestamp=1001.0,
                step_id="s1",
                data={"response_length": 50, "latency_ms": 500},
            ),
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=1001.5,
                step_id="s1",
                step_name="step_one",
                duration_ms=1500.0,
                data={"success": True},
            ),
        ]
        return trace

    def test_create_visualizer(self):
        """Test creating a visualizer."""
        viz = TraceVisualizer()

        assert viz.width == 80

    def test_visualizer_custom_width(self):
        """Test visualizer with custom width."""
        viz = TraceVisualizer(width=120)

        assert viz.width == 120

    def test_render_timeline(self, sample_trace):
        """Test rendering timeline."""
        viz = TraceVisualizer()

        output = viz.render_timeline(sample_trace)

        assert "EXECUTION TRACE" in output
        assert "test_trace" in output
        assert "step_one" in output
        assert "SUCCESS" in output

    def test_render_tree(self, sample_trace):
        """Test rendering tree."""
        viz = TraceVisualizer()

        output = viz.render_tree(sample_trace)

        assert "test_trace" in output
        assert "step_one" in output

    def test_render_summary(self, sample_trace):
        """Test rendering summary."""
        viz = TraceVisualizer()

        output = viz.render_summary(sample_trace)

        assert "Trace Summary" in output
        assert "Total Events" in output
        assert "Duration" in output

    def test_render_flame_chart(self, sample_trace):
        """Test rendering flame chart."""
        viz = TraceVisualizer()

        output = viz.render_flame_chart(sample_trace)

        assert "Flame Chart" in output

    def test_render_flame_chart_incomplete_trace(self):
        """Test flame chart with incomplete trace."""
        viz = TraceVisualizer()
        trace = ExecutionTrace(trace_id="test", start_time=time.time())

        output = viz.render_flame_chart(trace)

        assert "Cannot render" in output

    def test_render_timeline_with_error(self):
        """Test timeline with error trace."""
        viz = TraceVisualizer()
        trace = ExecutionTrace(
            trace_id="test",
            start_time=1000.0,
            end_time=1001.0,
            error="Something failed",
        )
        trace.events = [
            TraceEvent(
                event_type=TraceEventType.ERROR_OCCURRED,
                timestamp=1000.5,
                data={"error": "Test error"},
            ),
        ]

        output = viz.render_timeline(trace)

        assert "ERROR" in output


# =============================================================================
# PromptFlowAnalyzer Tests
# =============================================================================


class TestPromptFlowAnalyzer:
    """Tests for PromptFlowAnalyzer class."""

    def test_create_analyzer(self):
        """Test creating an analyzer."""
        analyzer = PromptFlowAnalyzer()

        assert len(analyzer._patterns) > 0

    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        analyzer = PromptFlowAnalyzer()
        initial_count = len(analyzer._patterns)

        analyzer.add_pattern(
            "custom",
            lambda t: True,
            "Custom suggestion",
        )

        assert len(analyzer._patterns) == initial_count + 1

    def test_analyze_empty_response(self):
        """Test detecting empty response."""
        analyzer = PromptFlowAnalyzer()
        trace = ExecutionTrace(trace_id="test", start_time=time.time())
        trace.events = [
            TraceEvent(
                event_type=TraceEventType.RESPONSE_RECEIVED,
                timestamp=time.time(),
                data={"response_length": 0},
            ),
        ]

        findings = analyzer.analyze(trace)

        assert any(f["pattern"] == "empty_response" for f in findings)

    def test_analyze_high_latency(self):
        """Test detecting high latency."""
        analyzer = PromptFlowAnalyzer()
        trace = ExecutionTrace(trace_id="test", start_time=time.time())
        trace.events = [
            TraceEvent(
                event_type=TraceEventType.RESPONSE_RECEIVED,
                timestamp=time.time(),
                data={"latency_ms": 15000},
            ),
        ]

        findings = analyzer.analyze(trace)

        assert any(f["pattern"] == "high_latency" for f in findings)

    def test_analyze_multiple_errors(self):
        """Test detecting multiple errors."""
        analyzer = PromptFlowAnalyzer()
        trace = ExecutionTrace(trace_id="test", start_time=time.time())
        trace.events = [
            TraceEvent(event_type=TraceEventType.ERROR_OCCURRED, timestamp=time.time()),
            TraceEvent(event_type=TraceEventType.ERROR_OCCURRED, timestamp=time.time()),
            TraceEvent(event_type=TraceEventType.ERROR_OCCURRED, timestamp=time.time()),
        ]

        findings = analyzer.analyze(trace)

        assert any(f["pattern"] == "multiple_errors" for f in findings)

    def test_get_bottlenecks(self):
        """Test identifying bottlenecks."""
        analyzer = PromptFlowAnalyzer()
        trace = ExecutionTrace(trace_id="test", start_time=time.time())
        trace.events = [
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=time.time(),
                step_name="fast",
                duration_ms=100,
            ),
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=time.time(),
                step_name="slow",
                duration_ms=1000,
            ),
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=time.time(),
                step_name="medium",
                duration_ms=200,
            ),
        ]

        bottlenecks = analyzer.get_bottlenecks(trace)

        assert len(bottlenecks) >= 1
        assert bottlenecks[0]["step_name"] == "slow"


# =============================================================================
# DebugSession Tests
# =============================================================================


class TestDebugSession:
    """Tests for DebugSession dataclass."""

    def test_create_session(self):
        """Test creating a debug session."""
        debugger = PromptDebugger()
        session = DebugSession(
            session_id="session_1",
            debugger=debugger,
        )

        assert session.session_id == "session_1"
        assert len(session.traces) == 0
        assert len(session.notes) == 0

    def test_add_trace(self):
        """Test adding trace to session."""
        debugger = PromptDebugger()
        session = DebugSession(session_id="test", debugger=debugger)
        trace = ExecutionTrace(trace_id="trace_1", start_time=time.time())

        session.add_trace(trace)

        assert len(session.traces) == 1

    def test_add_note(self):
        """Test adding note to session."""
        debugger = PromptDebugger()
        session = DebugSession(session_id="test", debugger=debugger)

        session.add_note("Test note")

        assert len(session.notes) == 1
        assert "Test note" in session.notes[0]

    def test_session_to_dict(self):
        """Test session serialization."""
        debugger = PromptDebugger()
        session = DebugSession(session_id="test", debugger=debugger)

        d = session.to_dict()

        assert d["session_id"] == "test"
        assert d["trace_count"] == 0


# =============================================================================
# TraceComparator Tests
# =============================================================================


class TestTraceComparator:
    """Tests for TraceComparator class."""

    def test_compare_traces(self):
        """Test comparing two traces."""
        comparator = TraceComparator()

        trace1 = ExecutionTrace(trace_id="t1", start_time=1000.0, end_time=1001.0)
        trace1.events = [
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=time.time(),
                step_name="step_a",
                duration_ms=100,
            ),
        ]

        trace2 = ExecutionTrace(trace_id="t2", start_time=1000.0, end_time=1002.0)
        trace2.events = [
            TraceEvent(
                event_type=TraceEventType.STEP_END,
                timestamp=time.time(),
                step_name="step_a",
                duration_ms=150,
            ),
        ]

        comparison = comparator.compare(trace1, trace2)

        assert comparison["duration_diff_ms"] == 1000.0
        assert len(comparison["step_comparisons"]) >= 1

    def test_compare_errors(self):
        """Test comparing trace errors."""
        comparator = TraceComparator()

        trace1 = ExecutionTrace(trace_id="t1", start_time=1000.0)
        trace1.events = [
            TraceEvent(
                event_type=TraceEventType.ERROR_OCCURRED,
                timestamp=time.time(),
                data={"error": "Error A"},
            ),
        ]

        trace2 = ExecutionTrace(trace_id="t2", start_time=1000.0)
        trace2.events = [
            TraceEvent(
                event_type=TraceEventType.ERROR_OCCURRED,
                timestamp=time.time(),
                data={"error": "Error B"},
            ),
        ]

        comparison = comparator.compare(trace1, trace2)

        assert "Error B" in comparison["new_errors_in_trace2"]
        assert "Error A" in comparison["resolved_errors"]


# =============================================================================
# TraceExporter Tests
# =============================================================================


class TestTraceExporter:
    """Tests for TraceExporter class."""

    @pytest.fixture
    def sample_trace(self):
        """Create a sample trace."""
        trace = ExecutionTrace(
            trace_id="test_trace",
            start_time=1000.0,
            end_time=1001.0,
        )
        trace.events = [
            TraceEvent(
                event_type=TraceEventType.STEP_START,
                timestamp=1000.0,
                step_name="step_one",
            ),
        ]
        return trace

    def test_to_json(self, sample_trace):
        """Test JSON export."""
        exporter = TraceExporter()

        json_str = exporter.to_json(sample_trace)
        data = json.loads(json_str)

        assert data["trace_id"] == "test_trace"

    def test_to_jsonl(self, sample_trace):
        """Test JSONL export."""
        exporter = TraceExporter()

        jsonl = exporter.to_jsonl(sample_trace)
        lines = jsonl.strip().split("\n")

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "step_start"

    def test_to_markdown(self, sample_trace):
        """Test Markdown export."""
        exporter = TraceExporter()

        md = exporter.to_markdown(sample_trace)

        assert "# Execution Trace" in md
        assert "test_trace" in md
        assert "| Time (ms) |" in md

    def test_from_json(self, sample_trace):
        """Test JSON import."""
        exporter = TraceExporter()

        json_str = exporter.to_json(sample_trace)
        imported = exporter.from_json(json_str)

        assert imported.trace_id == "test_trace"
        assert len(imported.events) == 1

    def test_roundtrip(self, sample_trace):
        """Test export/import roundtrip."""
        exporter = TraceExporter()

        json_str = exporter.to_json(sample_trace)
        imported = exporter.from_json(json_str)

        assert imported.trace_id == sample_trace.trace_id
        assert imported.duration_ms == sample_trace.duration_ms


# =============================================================================
# PromptInspector Tests
# =============================================================================


class TestPromptInspector:
    """Tests for PromptInspector class."""

    def test_create_inspector(self):
        """Test creating an inspector."""
        inspector = PromptInspector()

        assert len(inspector._warning_patterns) > 0

    def test_inspect_basic_prompt(self):
        """Test inspecting basic prompt."""
        inspector = PromptInspector()

        result = inspector.inspect("What is 2+2?")

        assert result["length"] == 12  # "What is 2+2?" is 12 chars
        assert result["word_count"] == 3  # "What", "is", "2+2?"
        assert "warnings" in result

    def test_inspect_long_prompt(self):
        """Test inspecting long prompt."""
        inspector = PromptInspector()
        long_prompt = "Test " * 3000  # 15000 chars

        result = inspector.inspect(long_prompt)

        assert any("breaking into smaller" in s for s in result["suggestions"])

    def test_inspect_unsubstituted_variable(self):
        """Test detecting unsubstituted variables."""
        inspector = PromptInspector()

        result = inspector.inspect("Hello {name}, how are you?")

        assert any("Unsubstituted" in w for w in result["warnings"])

    def test_inspect_please_warning(self):
        """Test warning about 'please'."""
        inspector = PromptInspector()

        result = inspector.inspect("Please answer the question.")

        assert any("please" in w.lower() for w in result["warnings"])

    def test_diff_prompts(self):
        """Test comparing prompts."""
        inspector = PromptInspector()

        diff = inspector.diff_prompts(
            "What is the capital of France?",
            "What is the capital of Germany?",
        )

        assert "length_diff" in diff
        assert "added_words" in diff
        assert "removed_words" in diff
        assert "similarity" in diff


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_debugger_function(self):
        """Test create_debugger function."""
        debugger = create_debugger()

        assert isinstance(debugger, PromptDebugger)

    def test_create_debugger_with_level(self):
        """Test create_debugger with custom level."""
        debugger = create_debugger(level=DebugLevel.VERBOSE)

        assert debugger.debug_level == DebugLevel.VERBOSE

    def test_create_visualizer_function(self):
        """Test create_visualizer function."""
        viz = create_visualizer()

        assert isinstance(viz, TraceVisualizer)

    def test_create_visualizer_custom_width(self):
        """Test create_visualizer with custom width."""
        viz = create_visualizer(width=100)

        assert viz.width == 100

    def test_render_trace_timeline(self):
        """Test render_trace with timeline format."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=1000.0,
            end_time=1001.0,
        )

        output = render_trace(trace, format="timeline")

        assert "EXECUTION TRACE" in output

    def test_render_trace_tree(self):
        """Test render_trace with tree format."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=1000.0,
            end_time=1001.0,
        )

        output = render_trace(trace, format="tree")

        assert "test" in output

    def test_render_trace_summary(self):
        """Test render_trace with summary format."""
        trace = ExecutionTrace(
            trace_id="test",
            start_time=1000.0,
            end_time=1001.0,
        )

        output = render_trace(trace, format="summary")

        assert "Summary" in output

    def test_export_trace_json(self):
        """Test export_trace JSON format."""
        trace = ExecutionTrace(trace_id="test", start_time=1000.0)

        output = export_trace(trace, format="json")
        data = json.loads(output)

        assert data["trace_id"] == "test"

    def test_export_trace_markdown(self):
        """Test export_trace Markdown format."""
        trace = ExecutionTrace(trace_id="test", start_time=1000.0)

        output = export_trace(trace, format="markdown")

        assert "# Execution Trace" in output

    def test_analyze_trace_function(self):
        """Test analyze_trace function."""
        trace = ExecutionTrace(trace_id="test", start_time=time.time())

        result = analyze_trace(trace)

        assert "patterns" in result
        assert "bottlenecks" in result

    def test_inspect_prompt_function(self):
        """Test inspect_prompt function."""
        result = inspect_prompt("Hello world")

        assert "length" in result
        assert "word_count" in result

    def test_compare_traces_function(self):
        """Test compare_traces function."""
        trace1 = ExecutionTrace(trace_id="t1", start_time=1000.0, end_time=1001.0)
        trace2 = ExecutionTrace(trace_id="t2", start_time=1000.0, end_time=1002.0)

        comparison = compare_traces(trace1, trace2)

        assert "duration_diff_ms" in comparison

    def test_quick_debug_function(self):
        """Test quick_debug function."""

        def simple_func(x):
            return x * 2

        result, trace = quick_debug(simple_func, 5)

        assert result == 10
        assert isinstance(trace, ExecutionTrace)

    def test_quick_debug_with_error(self):
        """Test quick_debug with error."""

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            quick_debug(failing_func)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDebuggerIntegration:
    """Integration tests for debugging workflow."""

    def test_full_debug_workflow(self):
        """Test a complete debugging workflow."""
        debugger = create_debugger()

        with debugger.trace({"workflow": "test"}) as trace:
            # Simulate a prompt workflow
            debugger.step_start("s1", "preprocessing")
            debugger.log_variable("input", "test data")
            time.sleep(0.01)
            debugger.step_end("s1", "preprocessing", 10.0)

            debugger.step_start("s2", "llm_call")
            debugger.log_prompt("What is the meaning?", step_id="s2")
            time.sleep(0.01)
            debugger.log_response("42", step_id="s2", tokens_used=5)
            debugger.step_end("s2", "llm_call", 500.0)

            debugger.step_start("s3", "postprocessing")
            debugger.log_variable("output", "42")
            debugger.step_end("s3", "postprocessing", 5.0)

        # Verify trace
        assert trace.event_count > 0
        assert len(trace.get_events_by_type(TraceEventType.STEP_START)) == 3
        assert len(trace.get_events_by_type(TraceEventType.STEP_END)) == 3

        # Render and analyze
        viz = create_visualizer()
        timeline = viz.render_timeline(trace)
        assert "preprocessing" in timeline
        assert "llm_call" in timeline

        analysis = analyze_trace(trace)
        assert "patterns" in analysis

    def test_debug_with_breakpoints(self):
        """Test debugging with breakpoints."""
        debugger = create_debugger()
        breaks = []

        def on_break(ctx):
            breaks.append(ctx["event"].step_name)

        debugger.set_break_handler(on_break)
        debugger.add_breakpoint("bp1", step_name="target_step")

        with debugger.trace():
            debugger.step_start("s1", "other_step")
            debugger.step_end("s1", "other_step", 10.0)

            debugger.step_start("s2", "target_step")
            debugger.step_end("s2", "target_step", 20.0)

        assert "target_step" in breaks

    def test_export_import_roundtrip(self):
        """Test exporting and importing a trace."""
        debugger = create_debugger()

        with debugger.trace() as original_trace:
            debugger.step_start("s1", "test")
            debugger.log_prompt("Hello", step_id="s1")
            debugger.step_end("s1", "test", 100.0)

        # Export
        exporter = TraceExporter()
        json_data = exporter.to_json(original_trace)

        # Import
        imported_trace = exporter.from_json(json_data)

        # Verify
        assert imported_trace.trace_id == original_trace.trace_id
        assert len(imported_trace.events) == len(original_trace.events)
