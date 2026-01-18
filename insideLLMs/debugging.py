"""Prompt debugging and trace visualization module.

This module provides tools for debugging prompt workflows, tracing execution,
visualizing prompt flows, and identifying issues in LLM interactions.

Key Features:
- Execution tracing with detailed step-by-step logging
- Prompt flow visualization (text-based and structured)
- Variable inspection and state tracking
- Error analysis and debugging suggestions
- Performance profiling of prompt chains
- Interactive debugging sessions
- Trace export and replay

Example:
    >>> from insideLLMs.debugging import PromptDebugger, TraceVisualizer
    >>>
    >>> debugger = PromptDebugger()
    >>> with debugger.trace() as trace:
    ...     result = run_prompt_chain(chain, input_data)
    >>>
    >>> visualizer = TraceVisualizer()
    >>> visualizer.render_timeline(trace)
"""

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union


class TraceEventType(Enum):
    """Types of trace events."""

    STEP_START = "step_start"
    STEP_END = "step_end"
    PROMPT_SENT = "prompt_sent"
    RESPONSE_RECEIVED = "response_received"
    VARIABLE_SET = "variable_set"
    VARIABLE_READ = "variable_read"
    CONDITION_EVALUATED = "condition_evaluated"
    ERROR_OCCURRED = "error_occurred"
    WARNING = "warning"
    RETRY = "retry"
    BRANCH_TAKEN = "branch_taken"
    LOOP_ITERATION = "loop_iteration"
    CHECKPOINT = "checkpoint"


class DebugLevel(Enum):
    """Debug verbosity levels."""

    MINIMAL = "minimal"  # Only errors
    BASIC = "basic"  # Steps and errors
    DETAILED = "detailed"  # All events
    VERBOSE = "verbose"  # Everything including internal state


class IssueCategory(Enum):
    """Categories of detected issues."""

    PERFORMANCE = "performance"
    LOGIC_ERROR = "logic_error"
    DATA_QUALITY = "data_quality"
    PROMPT_ISSUE = "prompt_issue"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


class IssueSeverity(Enum):
    """Severity levels for detected issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TraceEvent:
    """A single event in an execution trace."""

    event_type: TraceEventType
    timestamp: float
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    parent_event_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"evt_{time.time_ns()}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "parent_event_id": self.parent_event_id,
        }


@dataclass
class ExecutionTrace:
    """A complete execution trace."""

    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Total trace duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def event_count(self) -> int:
        """Number of events in trace."""
        return len(self.events)

    def get_events_by_type(self, event_type: TraceEventType) -> list[TraceEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_step_events(self, step_id: str) -> list[TraceEvent]:
        """Get all events for a specific step."""
        return [e for e in self.events if e.step_id == step_id]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "event_count": self.event_count,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
            "final_result": str(self.final_result) if self.final_result else None,
            "error": self.error,
        }


@dataclass
class DebugIssue:
    """A detected issue during debugging."""

    category: IssueCategory
    severity: IssueSeverity
    message: str
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    suggestion: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "suggestion": self.suggestion,
            "context": self.context,
            "timestamp": self.timestamp,
        }


@dataclass
class DebugBreakpoint:
    """A breakpoint in a prompt workflow."""

    breakpoint_id: str
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    condition: Optional[Callable[[dict[str, Any]], bool]] = None
    on_variable: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0

    def should_break(self, context: dict[str, Any]) -> bool:
        """Check if breakpoint should trigger."""
        if not self.enabled:
            return False
        if self.condition is not None:
            return self.condition(context)
        return True


@dataclass
class VariableSnapshot:
    """A snapshot of variable state."""

    name: str
    value: Any
    timestamp: float
    step_id: Optional[str] = None
    operation: str = "set"  # set, read, modified

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": str(self.value)[:500],  # Truncate long values
            "value_type": type(self.value).__name__,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "operation": self.operation,
        }


class PromptDebugger:
    """Main debugger for prompt workflows."""

    def __init__(
        self,
        debug_level: DebugLevel = DebugLevel.DETAILED,
        max_events: int = 10000,
        auto_detect_issues: bool = True,
    ):
        """Initialize debugger.

        Args:
            debug_level: Level of debug detail
            max_events: Maximum events to store
            auto_detect_issues: Whether to auto-detect issues
        """
        self.debug_level = debug_level
        self.max_events = max_events
        self.auto_detect_issues = auto_detect_issues

        self._current_trace: Optional[ExecutionTrace] = None
        self._breakpoints: dict[str, DebugBreakpoint] = {}
        self._variable_history: list[VariableSnapshot] = []
        self._detected_issues: list[DebugIssue] = []
        self._step_stack: list[str] = []
        self._break_handler: Optional[Callable[[dict[str, Any]], None]] = None

    def start_trace(self, metadata: Optional[dict[str, Any]] = None) -> ExecutionTrace:
        """Start a new execution trace."""
        trace_id = f"trace_{time.time_ns()}"
        self._current_trace = ExecutionTrace(
            trace_id=trace_id,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._variable_history.clear()
        self._detected_issues.clear()
        self._step_stack.clear()
        return self._current_trace

    def end_trace(
        self,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> ExecutionTrace:
        """End the current trace."""
        if self._current_trace is None:
            raise RuntimeError("No active trace")

        self._current_trace.end_time = time.time()
        self._current_trace.final_result = result
        self._current_trace.error = error

        # Auto-detect issues
        if self.auto_detect_issues:
            self._analyze_trace()

        trace = self._current_trace
        self._current_trace = None
        return trace

    def trace(self, metadata: Optional[dict[str, Any]] = None):
        """Context manager for tracing."""
        return _TraceContext(self, metadata)

    def log_event(
        self,
        event_type: TraceEventType,
        step_id: Optional[str] = None,
        step_name: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> Optional[TraceEvent]:
        """Log a trace event."""
        if self._current_trace is None:
            return None

        if len(self._current_trace.events) >= self.max_events:
            return None

        parent_id = self._step_stack[-1] if self._step_stack else None

        event = TraceEvent(
            event_type=event_type,
            timestamp=time.time(),
            step_id=step_id,
            step_name=step_name,
            data=data or {},
            duration_ms=duration_ms,
            parent_event_id=parent_id,
        )

        self._current_trace.events.append(event)

        # Check breakpoints
        self._check_breakpoints(event)

        return event

    def step_start(self, step_id: str, step_name: str, **kwargs) -> TraceEvent:
        """Log step start."""
        event = self.log_event(
            TraceEventType.STEP_START,
            step_id=step_id,
            step_name=step_name,
            data=kwargs,
        )
        self._step_stack.append(event.event_id if event else step_id)
        return event

    def step_end(
        self,
        step_id: str,
        step_name: str,
        duration_ms: float,
        result: Optional[Any] = None,
        success: bool = True,
    ) -> TraceEvent:
        """Log step end."""
        if self._step_stack:
            self._step_stack.pop()

        return self.log_event(
            TraceEventType.STEP_END,
            step_id=step_id,
            step_name=step_name,
            data={"result": str(result)[:500], "success": success},
            duration_ms=duration_ms,
        )

    def log_prompt(
        self,
        prompt: str,
        step_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> TraceEvent:
        """Log a prompt being sent."""
        return self.log_event(
            TraceEventType.PROMPT_SENT,
            step_id=step_id,
            data={
                "prompt": prompt[:1000],  # Truncate
                "prompt_length": len(prompt),
                "model": model,
            },
        )

    def log_response(
        self,
        response: str,
        step_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log a response received."""
        return self.log_event(
            TraceEventType.RESPONSE_RECEIVED,
            step_id=step_id,
            data={
                "response": response[:1000],  # Truncate
                "response_length": len(response),
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
            },
        )

    def log_variable(
        self,
        name: str,
        value: Any,
        step_id: Optional[str] = None,
        operation: str = "set",
    ) -> TraceEvent:
        """Log a variable operation."""
        snapshot = VariableSnapshot(
            name=name,
            value=value,
            timestamp=time.time(),
            step_id=step_id,
            operation=operation,
        )
        self._variable_history.append(snapshot)

        event_type = (
            TraceEventType.VARIABLE_SET if operation == "set" else TraceEventType.VARIABLE_READ
        )

        return self.log_event(
            event_type,
            step_id=step_id,
            data=snapshot.to_dict(),
        )

    def log_error(
        self,
        error: Union[str, Exception],
        step_id: Optional[str] = None,
        step_name: Optional[str] = None,
    ) -> TraceEvent:
        """Log an error."""
        if isinstance(error, Exception):
            error_str = f"{type(error).__name__}: {str(error)}"
            tb = traceback.format_exc()
        else:
            error_str = error
            tb = None

        issue = DebugIssue(
            category=IssueCategory.LOGIC_ERROR,
            severity=IssueSeverity.ERROR,
            message=error_str,
            step_id=step_id,
            step_name=step_name,
            context={"traceback": tb} if tb else {},
        )
        self._detected_issues.append(issue)

        return self.log_event(
            TraceEventType.ERROR_OCCURRED,
            step_id=step_id,
            step_name=step_name,
            data={"error": error_str, "traceback": tb},
        )

    def log_warning(
        self,
        message: str,
        step_id: Optional[str] = None,
    ) -> TraceEvent:
        """Log a warning."""
        return self.log_event(
            TraceEventType.WARNING,
            step_id=step_id,
            data={"message": message},
        )

    def add_breakpoint(
        self,
        breakpoint_id: str,
        step_id: Optional[str] = None,
        step_name: Optional[str] = None,
        condition: Optional[Callable[[dict[str, Any]], bool]] = None,
        on_variable: Optional[str] = None,
    ) -> DebugBreakpoint:
        """Add a breakpoint."""
        bp = DebugBreakpoint(
            breakpoint_id=breakpoint_id,
            step_id=step_id,
            step_name=step_name,
            condition=condition,
            on_variable=on_variable,
        )
        self._breakpoints[breakpoint_id] = bp
        return bp

    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint."""
        if breakpoint_id in self._breakpoints:
            del self._breakpoints[breakpoint_id]
            return True
        return False

    def set_break_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """Set handler called when breakpoint is hit."""
        self._break_handler = handler

    def get_variable_history(self, name: Optional[str] = None) -> list[VariableSnapshot]:
        """Get variable history, optionally filtered by name."""
        if name is None:
            return list(self._variable_history)
        return [v for v in self._variable_history if v.name == name]

    def get_issues(
        self,
        severity: Optional[IssueSeverity] = None,
        category: Optional[IssueCategory] = None,
    ) -> list[DebugIssue]:
        """Get detected issues with optional filtering."""
        issues = self._detected_issues
        if severity:
            issues = [i for i in issues if i.severity == severity]
        if category:
            issues = [i for i in issues if i.category == category]
        return issues

    def _check_breakpoints(self, event: TraceEvent) -> None:
        """Check if any breakpoint should trigger."""
        context = {
            "event": event,
            "trace": self._current_trace,
            "variables": {v.name: v.value for v in self._variable_history},
        }

        for bp in self._breakpoints.values():
            should_break = False

            # Check step match
            if (
                bp.step_id
                and event.step_id == bp.step_id
                or bp.step_name
                and event.step_name == bp.step_name
            ):
                should_break = bp.should_break(context)
            # Check variable match
            elif bp.on_variable and event.event_type == TraceEventType.VARIABLE_SET:
                if event.data.get("name") == bp.on_variable:
                    should_break = bp.should_break(context)

            if should_break:
                bp.hit_count += 1
                if self._break_handler:
                    self._break_handler(context)

    def _analyze_trace(self) -> None:
        """Analyze trace for issues."""
        if self._current_trace is None:
            return

        trace = self._current_trace

        # Check for slow steps
        for event in trace.get_events_by_type(TraceEventType.STEP_END):
            if event.duration_ms and event.duration_ms > 5000:  # 5 seconds
                self._detected_issues.append(
                    DebugIssue(
                        category=IssueCategory.PERFORMANCE,
                        severity=IssueSeverity.WARNING,
                        message=f"Step '{event.step_name}' took {event.duration_ms:.0f}ms",
                        step_id=event.step_id,
                        step_name=event.step_name,
                        suggestion="Consider optimizing this step or adding caching",
                    )
                )

        # Check for retry events
        retries = trace.get_events_by_type(TraceEventType.RETRY)
        if len(retries) > 3:
            self._detected_issues.append(
                DebugIssue(
                    category=IssueCategory.API_ERROR,
                    severity=IssueSeverity.WARNING,
                    message=f"High retry count: {len(retries)} retries",
                    suggestion="Check API rate limits or network stability",
                )
            )

        # Check for long prompts
        for event in trace.get_events_by_type(TraceEventType.PROMPT_SENT):
            prompt_length = event.data.get("prompt_length", 0)
            if prompt_length > 10000:
                self._detected_issues.append(
                    DebugIssue(
                        category=IssueCategory.PROMPT_ISSUE,
                        severity=IssueSeverity.INFO,
                        message=f"Long prompt ({prompt_length} chars)",
                        step_id=event.step_id,
                        suggestion="Consider using prompt compression",
                    )
                )


class _TraceContext:
    """Context manager for tracing."""

    def __init__(self, debugger: PromptDebugger, metadata: Optional[dict[str, Any]]):
        self.debugger = debugger
        self.metadata = metadata
        self.trace: Optional[ExecutionTrace] = None

    def __enter__(self) -> ExecutionTrace:
        self.trace = self.debugger.start_trace(self.metadata)
        return self.trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        error = str(exc_val) if exc_val else None
        self.debugger.end_trace(error=error)
        return False


class TraceVisualizer:
    """Visualize execution traces."""

    def __init__(self, width: int = 80):
        """Initialize visualizer.

        Args:
            width: Maximum width for text output
        """
        self.width = width

    def render_timeline(self, trace: ExecutionTrace) -> str:
        """Render trace as timeline text."""
        lines = []
        lines.append("=" * self.width)
        lines.append(f"EXECUTION TRACE: {trace.trace_id}")
        lines.append(
            f"Duration: {trace.duration_ms:.2f}ms" if trace.duration_ms else "Duration: N/A"
        )
        lines.append("=" * self.width)

        # Group events by step
        start_time = trace.start_time

        for event in trace.events:
            elapsed = (event.timestamp - start_time) * 1000

            # Format based on event type
            if event.event_type == TraceEventType.STEP_START:
                lines.append(f"\n[{elapsed:8.1f}ms] ▶ START: {event.step_name}")
                if event.data:
                    for k, v in event.data.items():
                        lines.append(f"              │ {k}: {str(v)[:50]}")

            elif event.event_type == TraceEventType.STEP_END:
                status = "✓" if event.data.get("success", True) else "✗"
                lines.append(
                    f"[{elapsed:8.1f}ms] {status} END: {event.step_name} ({event.duration_ms:.1f}ms)"
                )

            elif event.event_type == TraceEventType.PROMPT_SENT:
                prompt_len = event.data.get("prompt_length", 0)
                lines.append(f"[{elapsed:8.1f}ms]   → Prompt sent ({prompt_len} chars)")

            elif event.event_type == TraceEventType.RESPONSE_RECEIVED:
                resp_len = event.data.get("response_length", 0)
                latency = event.data.get("latency_ms")
                latency_str = f"{latency:.0f}ms" if latency is not None else "N/A"
                lines.append(f"[{elapsed:8.1f}ms]   ← Response ({resp_len} chars, {latency_str})")

            elif event.event_type == TraceEventType.ERROR_OCCURRED:
                error = event.data.get("error", "Unknown error")
                lines.append(f"[{elapsed:8.1f}ms]   ✗ ERROR: {error[:60]}")

            elif event.event_type == TraceEventType.WARNING:
                msg = event.data.get("message", "")
                lines.append(f"[{elapsed:8.1f}ms]   ⚠ WARNING: {msg[:60]}")

            elif event.event_type == TraceEventType.VARIABLE_SET:
                name = event.data.get("name", "")
                lines.append(f"[{elapsed:8.1f}ms]   ◆ SET: {name}")

        lines.append("\n" + "=" * self.width)

        if trace.error:
            lines.append(f"RESULT: ERROR - {trace.error}")
        else:
            lines.append("RESULT: SUCCESS")

        lines.append("=" * self.width)

        return "\n".join(lines)

    def render_tree(self, trace: ExecutionTrace) -> str:
        """Render trace as tree structure."""
        lines = []
        lines.append(f"Trace: {trace.trace_id}")

        indent_level = 0
        step_stack = []

        for event in trace.events:
            if event.event_type == TraceEventType.STEP_START:
                indent = "  " * indent_level
                lines.append(f"{indent}├─ {event.step_name}")
                step_stack.append(event.step_id)
                indent_level += 1

            elif event.event_type == TraceEventType.STEP_END:
                indent_level = max(0, indent_level - 1)
                if step_stack:
                    step_stack.pop()

                indent = "  " * indent_level
                status = "✓" if event.data.get("success", True) else "✗"
                lines.append(f"{indent}└─ {status} {event.duration_ms:.1f}ms")

            elif event.event_type == TraceEventType.ERROR_OCCURRED:
                indent = "  " * indent_level
                lines.append(f"{indent}│  ✗ {event.data.get('error', '')[:40]}")

        return "\n".join(lines)

    def render_summary(self, trace: ExecutionTrace) -> str:
        """Render trace summary."""
        lines = []
        lines.append(f"Trace Summary: {trace.trace_id}")
        lines.append("-" * 40)

        # Count events
        event_counts = {}
        for event in trace.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        lines.append(f"Total Events: {trace.event_count}")
        lines.append(
            f"Duration: {trace.duration_ms:.2f}ms" if trace.duration_ms else "Duration: N/A"
        )

        lines.append("\nEvent Breakdown:")
        for event_type, count in sorted(event_counts.items()):
            lines.append(f"  {event_type}: {count}")

        # Step timing
        step_times = []
        for event in trace.get_events_by_type(TraceEventType.STEP_END):
            if event.duration_ms:
                step_times.append((event.step_name, event.duration_ms))

        if step_times:
            lines.append("\nStep Timing:")
            for name, duration in sorted(step_times, key=lambda x: -x[1])[:5]:
                lines.append(f"  {name}: {duration:.1f}ms")

        return "\n".join(lines)

    def render_flame_chart(self, trace: ExecutionTrace) -> str:
        """Render a text-based flame chart."""
        if not trace.duration_ms:
            return "Cannot render: trace incomplete"

        lines = []
        total_width = self.width - 20  # Leave room for labels
        ms_per_char = trace.duration_ms / total_width

        lines.append("Flame Chart")
        lines.append("=" * self.width)

        # Build step timings
        step_starts = {}
        step_ends = {}

        for event in trace.events:
            if event.event_type == TraceEventType.STEP_START:
                step_starts[event.step_id] = event.timestamp
            elif event.event_type == TraceEventType.STEP_END:
                step_ends[event.step_id] = (
                    event.timestamp,
                    event.duration_ms,
                    event.step_name,
                )

        # Render each step
        for step_id, (_end_time, duration, name) in step_ends.items():
            if step_id not in step_starts:
                continue

            start_time = step_starts[step_id]
            start_offset = int((start_time - trace.start_time) * 1000 / ms_per_char)
            width = max(1, int(duration / ms_per_char))

            label = name[:15].ljust(15)
            bar = " " * start_offset + "█" * width
            lines.append(f"{label} │{bar}")

        lines.append("=" * self.width)
        lines.append(f"0ms{' ' * (total_width - 10)}{trace.duration_ms:.0f}ms")

        return "\n".join(lines)


class PromptFlowAnalyzer:
    """Analyze prompt flow patterns and issues."""

    def __init__(self):
        """Initialize analyzer."""
        self._patterns: list[tuple[str, Callable[[ExecutionTrace], bool], str]] = []
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Set up default analysis patterns."""

        # Empty response pattern
        def check_empty_response(trace: ExecutionTrace) -> bool:
            for event in trace.get_events_by_type(TraceEventType.RESPONSE_RECEIVED):
                if event.data.get("response_length", 0) == 0:
                    return True
            return False

        self._patterns.append(
            (
                "empty_response",
                check_empty_response,
                "Empty response received - check prompt clarity",
            )
        )

        # High latency pattern
        def check_high_latency(trace: ExecutionTrace) -> bool:
            for event in trace.get_events_by_type(TraceEventType.RESPONSE_RECEIVED):
                if event.data.get("latency_ms", 0) > 10000:
                    return True
            return False

        self._patterns.append(
            (
                "high_latency",
                check_high_latency,
                "High latency detected - consider timeout handling",
            )
        )

        # Multiple errors pattern
        def check_multiple_errors(trace: ExecutionTrace) -> bool:
            return len(trace.get_events_by_type(TraceEventType.ERROR_OCCURRED)) > 2

        self._patterns.append(
            (
                "multiple_errors",
                check_multiple_errors,
                "Multiple errors in trace - review error handling",
            )
        )

    def add_pattern(
        self,
        name: str,
        check: Callable[[ExecutionTrace], bool],
        suggestion: str,
    ) -> None:
        """Add a custom analysis pattern."""
        self._patterns.append((name, check, suggestion))

    def analyze(self, trace: ExecutionTrace) -> list[dict[str, str]]:
        """Analyze trace for patterns."""
        findings = []

        for name, check, suggestion in self._patterns:
            try:
                if check(trace):
                    findings.append(
                        {
                            "pattern": name,
                            "suggestion": suggestion,
                        }
                    )
            except Exception:
                pass  # Skip failing patterns

        return findings

    def get_bottlenecks(self, trace: ExecutionTrace) -> list[dict[str, Any]]:
        """Identify bottlenecks in the trace."""
        bottlenecks = []

        step_events = trace.get_events_by_type(TraceEventType.STEP_END)

        if not step_events:
            return bottlenecks

        # Calculate timing statistics
        durations = [e.duration_ms for e in step_events if e.duration_ms]
        if not durations:
            return bottlenecks

        avg_duration = sum(durations) / len(durations)

        # Find steps significantly slower than average
        for event in step_events:
            if event.duration_ms and event.duration_ms > avg_duration * 2:
                bottlenecks.append(
                    {
                        "step_name": event.step_name,
                        "step_id": event.step_id,
                        "duration_ms": event.duration_ms,
                        "slowdown_factor": event.duration_ms / avg_duration,
                    }
                )

        return sorted(bottlenecks, key=lambda x: -x["duration_ms"])


@dataclass
class DebugSession:
    """An interactive debug session."""

    session_id: str
    debugger: PromptDebugger
    traces: list[ExecutionTrace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_trace(self, trace: ExecutionTrace) -> None:
        """Add a trace to the session."""
        self.traces.append(trace)

    def add_note(self, note: str) -> None:
        """Add a note to the session."""
        self.notes.append(f"[{datetime.now().isoformat()}] {note}")

    def get_all_issues(self) -> list[DebugIssue]:
        """Get all issues from all traces."""
        return self.debugger.get_issues()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "trace_count": len(self.traces),
            "traces": [t.to_dict() for t in self.traces],
            "notes": self.notes,
        }


class TraceComparator:
    """Compare multiple execution traces."""

    def compare(
        self,
        trace1: ExecutionTrace,
        trace2: ExecutionTrace,
    ) -> dict[str, Any]:
        """Compare two traces."""
        comparison = {
            "duration_diff_ms": None,
            "event_count_diff": trace2.event_count - trace1.event_count,
            "step_comparisons": [],
            "new_errors_in_trace2": [],
            "resolved_errors": [],
        }

        # Duration comparison
        if trace1.duration_ms and trace2.duration_ms:
            comparison["duration_diff_ms"] = trace2.duration_ms - trace1.duration_ms
            comparison["duration_change_percent"] = (
                (trace2.duration_ms - trace1.duration_ms) / trace1.duration_ms * 100
            )

        # Compare steps
        steps1 = {e.step_name: e for e in trace1.get_events_by_type(TraceEventType.STEP_END)}
        steps2 = {e.step_name: e for e in trace2.get_events_by_type(TraceEventType.STEP_END)}

        all_steps = set(steps1.keys()) | set(steps2.keys())
        for step_name in all_steps:
            step_comp = {"step_name": step_name}

            if step_name in steps1 and step_name in steps2:
                d1 = steps1[step_name].duration_ms or 0
                d2 = steps2[step_name].duration_ms or 0
                step_comp["duration_diff_ms"] = d2 - d1
                step_comp["status"] = "present_in_both"
            elif step_name in steps1:
                step_comp["status"] = "removed_in_trace2"
            else:
                step_comp["status"] = "new_in_trace2"

            comparison["step_comparisons"].append(step_comp)

        # Compare errors
        errors1 = {
            e.data.get("error", ""): e
            for e in trace1.get_events_by_type(TraceEventType.ERROR_OCCURRED)
        }
        errors2 = {
            e.data.get("error", ""): e
            for e in trace2.get_events_by_type(TraceEventType.ERROR_OCCURRED)
        }

        comparison["new_errors_in_trace2"] = list(set(errors2.keys()) - set(errors1.keys()))
        comparison["resolved_errors"] = list(set(errors1.keys()) - set(errors2.keys()))

        return comparison


class TraceExporter:
    """Export traces to various formats."""

    def to_json(self, trace: ExecutionTrace) -> str:
        """Export trace to JSON."""
        return json.dumps(trace.to_dict(), indent=2, default=str)

    def to_jsonl(self, trace: ExecutionTrace) -> str:
        """Export trace events as JSON lines."""
        lines = []
        for event in trace.events:
            lines.append(json.dumps(event.to_dict(), default=str))
        return "\n".join(lines)

    def to_markdown(self, trace: ExecutionTrace) -> str:
        """Export trace as markdown report."""
        lines = []
        lines.append(f"# Execution Trace: {trace.trace_id}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- **Start Time**: {datetime.fromtimestamp(trace.start_time).isoformat()}")
        if trace.end_time:
            lines.append(f"- **End Time**: {datetime.fromtimestamp(trace.end_time).isoformat()}")
        if trace.duration_ms:
            lines.append(f"- **Duration**: {trace.duration_ms:.2f}ms")
        lines.append(f"- **Event Count**: {trace.event_count}")
        lines.append(f"- **Result**: {'Success' if not trace.error else f'Error: {trace.error}'}")

        lines.append("")
        lines.append("## Events")
        lines.append("")
        lines.append("| Time (ms) | Type | Step | Details |")
        lines.append("|-----------|------|------|---------|")

        for event in trace.events:
            elapsed = (event.timestamp - trace.start_time) * 1000
            details = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(event.data.items())[:3])
            lines.append(
                f"| {elapsed:.1f} | {event.event_type.value} | {event.step_name or '-'} | {details} |"
            )

        return "\n".join(lines)

    def from_json(self, json_str: str) -> ExecutionTrace:
        """Import trace from JSON."""
        data = json.loads(json_str)

        events = []
        for evt_data in data.get("events", []):
            events.append(
                TraceEvent(
                    event_type=TraceEventType(evt_data["event_type"]),
                    timestamp=evt_data["timestamp"],
                    step_id=evt_data.get("step_id"),
                    step_name=evt_data.get("step_name"),
                    data=evt_data.get("data", {}),
                    duration_ms=evt_data.get("duration_ms"),
                    parent_event_id=evt_data.get("parent_event_id"),
                    event_id=evt_data.get("event_id", f"evt_{time.time_ns()}"),
                )
            )

        return ExecutionTrace(
            trace_id=data["trace_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            events=events,
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )


class PromptInspector:
    """Inspect and analyze prompts."""

    def __init__(self):
        """Initialize inspector."""
        self._warning_patterns = [
            (r"\bplease\b", "Using 'please' may not affect model behavior"),
            (r"\bremember\b", "LLMs don't remember between calls"),
            (r"\{[^}]+\}", "Unsubstituted template variable detected"),
        ]

    def inspect(self, prompt: str) -> dict[str, Any]:
        """Inspect a prompt for potential issues."""
        result = {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "line_count": len(prompt.split("\n")),
            "warnings": [],
            "suggestions": [],
        }

        # Check patterns
        for pattern, message in self._warning_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                result["warnings"].append(message)

        # Check length
        if len(prompt) > 10000:
            result["suggestions"].append("Consider breaking into smaller prompts")

        # Check for structure
        if "\n\n" not in prompt and len(prompt) > 500:
            result["suggestions"].append("Consider adding paragraph breaks")

        # Check for instructions
        instruction_patterns = [
            "you are",
            "you will",
            "your task",
            "please",
            "write",
            "generate",
        ]
        has_instruction = any(p in prompt.lower() for p in instruction_patterns)
        if not has_instruction and len(prompt) > 100:
            result["suggestions"].append("Consider adding clear instructions")

        return result

    def diff_prompts(self, prompt1: str, prompt2: str) -> dict[str, Any]:
        """Compare two prompts."""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        return {
            "length_diff": len(prompt2) - len(prompt1),
            "word_count_diff": len(prompt2.split()) - len(prompt1.split()),
            "added_words": list(words2 - words1)[:20],
            "removed_words": list(words1 - words2)[:20],
            "common_words": len(words1 & words2),
            "similarity": len(words1 & words2) / max(len(words1 | words2), 1),
        }


# Convenience functions
def create_debugger(
    level: DebugLevel = DebugLevel.DETAILED,
    auto_detect: bool = True,
) -> PromptDebugger:
    """Create a new debugger instance."""
    return PromptDebugger(
        debug_level=level,
        auto_detect_issues=auto_detect,
    )


def create_visualizer(width: int = 80) -> TraceVisualizer:
    """Create a new visualizer instance."""
    return TraceVisualizer(width=width)


def render_trace(
    trace: ExecutionTrace,
    format: str = "timeline",
) -> str:
    """Render a trace in the specified format."""
    visualizer = TraceVisualizer()

    if format == "timeline":
        return visualizer.render_timeline(trace)
    elif format == "tree":
        return visualizer.render_tree(trace)
    elif format == "summary":
        return visualizer.render_summary(trace)
    elif format == "flame":
        return visualizer.render_flame_chart(trace)
    else:
        return visualizer.render_timeline(trace)


def export_trace(
    trace: ExecutionTrace,
    format: str = "json",
) -> str:
    """Export a trace in the specified format."""
    exporter = TraceExporter()

    if format == "json":
        return exporter.to_json(trace)
    elif format == "jsonl":
        return exporter.to_jsonl(trace)
    elif format == "markdown":
        return exporter.to_markdown(trace)
    else:
        return exporter.to_json(trace)


def analyze_trace(trace: ExecutionTrace) -> dict[str, Any]:
    """Analyze a trace for issues and patterns."""
    analyzer = PromptFlowAnalyzer()

    return {
        "patterns": analyzer.analyze(trace),
        "bottlenecks": analyzer.get_bottlenecks(trace),
    }


def inspect_prompt(prompt: str) -> dict[str, Any]:
    """Inspect a prompt for issues."""
    inspector = PromptInspector()
    return inspector.inspect(prompt)


def compare_traces(
    trace1: ExecutionTrace,
    trace2: ExecutionTrace,
) -> dict[str, Any]:
    """Compare two execution traces."""
    comparator = TraceComparator()
    return comparator.compare(trace1, trace2)


def quick_debug(
    func: Callable,
    *args,
    **kwargs,
) -> tuple[Any, ExecutionTrace]:
    """Quick debug wrapper for a function."""
    debugger = create_debugger()

    with debugger.trace() as trace:
        try:
            result = func(*args, **kwargs)
            return result, trace
        except Exception as e:
            debugger.log_error(e)
            raise
