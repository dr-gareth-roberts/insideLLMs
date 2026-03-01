"""Prompt debugging and trace visualization module.

This module provides comprehensive tools for debugging prompt workflows, tracing
execution paths, visualizing prompt flows, and identifying issues in LLM
interactions. It is designed to help developers understand, optimize, and
troubleshoot complex prompt chains and agent workflows.

Key Features
------------
- **Execution Tracing**: Detailed step-by-step logging of prompt chain execution
  with timing information, variable snapshots, and event hierarchies.
- **Prompt Flow Visualization**: Multiple visualization formats including
  timeline, tree, flame chart, and summary views.
- **Variable Inspection**: Track variable state changes throughout execution
  with complete history and filtering capabilities.
- **Error Analysis**: Automatic detection and categorization of issues with
  actionable suggestions for resolution.
- **Performance Profiling**: Identify bottlenecks, measure step durations, and
  compare execution traces.
- **Interactive Debugging**: Set breakpoints, inspect state, and step through
  prompt workflows.
- **Trace Export/Import**: Export traces to JSON, JSONL, or Markdown formats
  for sharing and replay.

Architecture
------------
The module is organized around several core concepts:

1. **PromptDebugger**: The main entry point for tracing prompt workflows.
   Manages trace lifecycle, event logging, and breakpoints.

2. **ExecutionTrace**: A complete record of a workflow execution containing
   all events, metadata, and results.

3. **TraceVisualizer**: Renders traces in human-readable formats for
   analysis and debugging.

4. **PromptFlowAnalyzer**: Analyzes traces for patterns, issues, and
   performance bottlenecks.

5. **TraceExporter**: Handles serialization and deserialization of traces.

Examples
--------
Basic tracing of a prompt chain:

>>> from insideLLMs.contrib.debugging import PromptDebugger, TraceVisualizer
>>>
>>> debugger = PromptDebugger()
>>> with debugger.trace(metadata={"workflow": "summarization"}) as trace:
...     debugger.step_start("step1", "Fetch Document")
...     document = fetch_document(url)
...     debugger.step_end("step1", "Fetch Document", duration_ms=150)
...
...     debugger.step_start("step2", "Generate Summary")
...     debugger.log_prompt(summary_prompt, step_id="step2", model="claude-3")
...     summary = call_llm(summary_prompt)
...     debugger.log_response(summary, step_id="step2", latency_ms=1200)
...     debugger.step_end("step2", "Generate Summary", duration_ms=1250)
>>>
>>> visualizer = TraceVisualizer()
>>> print(visualizer.render_timeline(trace))

Using breakpoints for interactive debugging:

>>> debugger = PromptDebugger()
>>> debugger.set_break_handler(lambda ctx: print(f"Break at {ctx['event'].step_name}"))
>>> debugger.add_breakpoint(
...     "bp1",
...     step_name="Generate Summary",
...     condition=lambda ctx: len(ctx['variables'].get('document', '')) > 10000
... )
>>> with debugger.trace() as trace:
...     # Execution will pause at breakpoint when condition is met
...     run_workflow(debugger)

Analyzing traces for performance issues:

>>> from insideLLMs.contrib.debugging import analyze_trace
>>>
>>> analysis = analyze_trace(trace)
>>> for pattern in analysis['patterns']:
...     print(f"Found pattern: {pattern['pattern']}")
...     print(f"  Suggestion: {pattern['suggestion']}")
>>>
>>> for bottleneck in analysis['bottlenecks']:
...     print(f"Bottleneck: {bottleneck['step_name']} ({bottleneck['duration_ms']:.0f}ms)")

Exporting and sharing traces:

>>> from insideLLMs.contrib.debugging import export_trace
>>>
>>> # Export to JSON for programmatic analysis
>>> json_trace = export_trace(trace, format="json")
>>> with open("trace.json", "w") as f:
...     f.write(json_trace)
>>>
>>> # Export to Markdown for documentation
>>> md_report = export_trace(trace, format="markdown")
>>> print(md_report)

Notes
-----
- Traces are limited to a configurable maximum number of events (default 10,000)
  to prevent memory issues in long-running workflows.
- Long prompt and response contents are automatically truncated in event data
  to manage memory usage while preserving debugging utility.
- The module uses nanosecond timestamps internally for event ordering but
  displays millisecond precision in visualizations.
- Breakpoint conditions are evaluated synchronously and should be lightweight
  to avoid affecting workflow performance.

See Also
--------
insideLLMs.tracing : Lower-level tracing primitives
insideLLMs.profiling : Performance profiling utilities
insideLLMs.logging : Logging integration

References
----------
.. [1] OpenTelemetry Tracing Specification
   https://opentelemetry.io/docs/concepts/signals/traces/
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
    """Enumeration of trace event types for categorizing workflow events.

    This enum defines all possible event types that can occur during prompt
    workflow execution. Each event type represents a distinct category of
    action or state change that the debugger can track.

    Attributes
    ----------
    STEP_START : str
        Marks the beginning of a workflow step. Paired with STEP_END.
    STEP_END : str
        Marks the completion of a workflow step. Contains duration and result.
    PROMPT_SENT : str
        Records when a prompt is sent to an LLM. Contains prompt content.
    RESPONSE_RECEIVED : str
        Records when a response is received from an LLM. Contains response data.
    VARIABLE_SET : str
        Records when a variable is created or modified.
    VARIABLE_READ : str
        Records when a variable value is accessed.
    CONDITION_EVALUATED : str
        Records the evaluation of a conditional branch.
    ERROR_OCCURRED : str
        Records an error during execution. Contains error details.
    WARNING : str
        Records a non-fatal warning condition.
    RETRY : str
        Records a retry attempt after a failure.
    BRANCH_TAKEN : str
        Records which branch was taken in a conditional flow.
    LOOP_ITERATION : str
        Records an iteration in a loop construct.
    CHECKPOINT : str
        Records a manual checkpoint for debugging reference.

    Examples
    --------
    Using event types to filter trace events:

    >>> trace = debugger.end_trace()
    >>> step_starts = trace.get_events_by_type(TraceEventType.STEP_START)
    >>> print(f"Workflow had {len(step_starts)} steps")
    Workflow had 5 steps

    >>> errors = trace.get_events_by_type(TraceEventType.ERROR_OCCURRED)
    >>> if errors:
    ...     print(f"Found {len(errors)} errors")
    ...     for error in errors:
    ...         print(f"  - {error.data.get('error')}")

    Checking event type in event handlers:

    >>> def event_handler(event):
    ...     if event.event_type == TraceEventType.PROMPT_SENT:
    ...         print(f"Prompt sent: {event.data['prompt'][:50]}...")
    ...     elif event.event_type == TraceEventType.RESPONSE_RECEIVED:
    ...         print(f"Response received: {event.data['response_length']} chars")

    See Also
    --------
    DebugTraceEvent : Event data structure that uses these types
    ExecutionTrace.get_events_by_type : Filter events by type
    """

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
    """Debug verbosity levels controlling the amount of trace information captured.

    These levels allow developers to balance between comprehensive debugging
    information and performance/memory overhead. Higher verbosity levels capture
    more events but may impact workflow performance.

    Attributes
    ----------
    MINIMAL : str
        Captures only error events. Best for production monitoring with
        minimal overhead. Use when you only need to know if something failed.
    BASIC : str
        Captures step boundaries and errors. Provides workflow structure
        without detailed internal state. Good for basic troubleshooting.
    DETAILED : str
        Captures all event types including prompts, responses, and variables.
        Default level for development debugging. Provides complete picture.
    VERBOSE : str
        Captures everything including internal state snapshots and additional
        context. Use for deep debugging of complex issues.

    Examples
    --------
    Creating debuggers with different verbosity levels:

    >>> # Production monitoring - only catch errors
    >>> prod_debugger = PromptDebugger(debug_level=DebugLevel.MINIMAL)

    >>> # Development debugging - see everything
    >>> dev_debugger = PromptDebugger(debug_level=DebugLevel.DETAILED)

    >>> # Deep investigation of specific issue
    >>> verbose_debugger = PromptDebugger(debug_level=DebugLevel.VERBOSE)

    Adjusting level based on environment:

    >>> import os
    >>> level = DebugLevel.VERBOSE if os.getenv("DEBUG") else DebugLevel.MINIMAL
    >>> debugger = PromptDebugger(debug_level=level)

    See Also
    --------
    PromptDebugger : Uses these levels to control event capture
    """

    MINIMAL = "minimal"  # Only errors
    BASIC = "basic"  # Steps and errors
    DETAILED = "detailed"  # All events
    VERBOSE = "verbose"  # Everything including internal state


class IssueCategory(Enum):
    """Categories for classifying detected issues during debugging.

    Issues detected during trace analysis are categorized to help developers
    quickly identify the nature of problems and apply appropriate fixes.
    Each category suggests different remediation strategies.

    Attributes
    ----------
    PERFORMANCE : str
        Issues related to execution speed, including slow steps, high latency,
        and inefficient processing patterns.
    LOGIC_ERROR : str
        Programming errors, incorrect conditions, or unexpected execution paths
        that produce wrong results.
    DATA_QUALITY : str
        Issues with input or output data, including malformed responses,
        missing fields, or validation failures.
    PROMPT_ISSUE : str
        Problems with prompt construction, including excessive length,
        ambiguous instructions, or template errors.
    API_ERROR : str
        Errors from external API calls, including rate limits, authentication
        failures, and service unavailability.
    TIMEOUT : str
        Operations that exceeded time limits, including LLM calls and
        external service requests.
    RESOURCE_LIMIT : str
        Exceeded resource constraints such as token limits, memory usage,
        or request quotas.

    Examples
    --------
    Filtering issues by category:

    >>> issues = debugger.get_issues(category=IssueCategory.PERFORMANCE)
    >>> for issue in issues:
    ...     print(f"Performance issue in {issue.step_name}: {issue.message}")
    Performance issue in Generate Summary: Step took 8500ms

    Grouping issues by category for reporting:

    >>> from collections import defaultdict
    >>> issues_by_category = defaultdict(list)
    >>> for issue in debugger.get_issues():
    ...     issues_by_category[issue.category].append(issue)
    >>> for category, category_issues in issues_by_category.items():
    ...     print(f"{category.value}: {len(category_issues)} issues")
    performance: 2 issues
    prompt_issue: 1 issues

    See Also
    --------
    DebugIssue : Issue data structure that uses these categories
    IssueSeverity : Severity levels for issues
    """

    PERFORMANCE = "performance"
    LOGIC_ERROR = "logic_error"
    DATA_QUALITY = "data_quality"
    PROMPT_ISSUE = "prompt_issue"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


class IssueSeverity(Enum):
    """Severity levels for classifying the impact of detected issues.

    Severity helps prioritize which issues to address first and determines
    appropriate alerting and escalation behavior. Higher severity issues
    typically require immediate attention.

    Attributes
    ----------
    INFO : str
        Informational findings that may be worth noting but don't require
        action. Examples: optimization opportunities, best practice suggestions.
    WARNING : str
        Potential problems that should be reviewed but don't prevent execution.
        Examples: slow steps, high retry counts, approaching limits.
    ERROR : str
        Actual failures that affected execution but were recoverable or
        isolated. Examples: failed steps, exceptions caught and handled.
    CRITICAL : str
        Severe failures that prevented successful completion or could cause
        data loss. Examples: unhandled exceptions, corrupted state, timeouts.

    Examples
    --------
    Filtering issues by severity:

    >>> # Get only errors and critical issues for alerting
    >>> critical_issues = debugger.get_issues(severity=IssueSeverity.ERROR)
    >>> critical_issues += debugger.get_issues(severity=IssueSeverity.CRITICAL)
    >>> if critical_issues:
    ...     send_alert(critical_issues)

    Severity-based reporting:

    >>> severity_counts = {}
    >>> for issue in debugger.get_issues():
    ...     sev = issue.severity.value
    ...     severity_counts[sev] = severity_counts.get(sev, 0) + 1
    >>> print(severity_counts)
    {'warning': 3, 'error': 1, 'info': 2}

    Conditional logging based on severity:

    >>> for issue in debugger.get_issues():
    ...     if issue.severity == IssueSeverity.CRITICAL:
    ...         logger.critical(issue.message)
    ...     elif issue.severity == IssueSeverity.ERROR:
    ...         logger.error(issue.message)
    ...     elif issue.severity == IssueSeverity.WARNING:
    ...         logger.warning(issue.message)

    See Also
    --------
    DebugIssue : Issue data structure that uses these severities
    IssueCategory : Categories for issues
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DebugTraceEvent:
    """A single event in an execution trace capturing a discrete action or state change.

    DebugTraceEvent represents an atomic unit of information in a trace. Events
    are collected chronologically and can be filtered, grouped, and analyzed to
    understand workflow behavior.

    Parameters
    ----------
    event_type : TraceEventType
        The category of this event (e.g., STEP_START, PROMPT_SENT, ERROR_OCCURRED).
    timestamp : float
        Unix timestamp when the event occurred, from time.time().
    step_id : str, optional
        Unique identifier for the step this event belongs to.
    step_name : str, optional
        Human-readable name of the step for display purposes.
    data : dict[str, Any], optional
        Event-specific payload data. Contents vary by event type.
    duration_ms : float, optional
        Duration in milliseconds, typically set for STEP_END events.
    parent_event_id : str, optional
        ID of the parent event for nested event hierarchies.
    event_id : str, optional
        Unique identifier for this event. Auto-generated if not provided.

    Attributes
    ----------
    event_type : TraceEventType
        The type/category of this trace event.
    timestamp : float
        When the event occurred as Unix timestamp.
    step_id : str or None
        Identifier of the associated workflow step.
    step_name : str or None
        Display name of the associated workflow step.
    data : dict[str, Any]
        Event payload with type-specific information.
    duration_ms : float or None
        Duration for timed events (e.g., STEP_END).
    parent_event_id : str or None
        Reference to parent event for hierarchy.
    event_id : str
        Unique event identifier for correlation.

    Examples
    --------
    Creating events manually (typically done by PromptDebugger):

    >>> event = DebugTraceEvent(
    ...     event_type=TraceEventType.PROMPT_SENT,
    ...     timestamp=time.time(),
    ...     step_id="step_123",
    ...     step_name="Generate Summary",
    ...     data={
    ...         "prompt": "Summarize the following text...",
    ...         "prompt_length": 1500,
    ...         "model": "claude-3-sonnet"
    ...     }
    ... )
    >>> print(f"Event {event.event_id}: {event.event_type.value}")
    Event evt_1234567890: prompt_sent

    Converting events to dictionaries for serialization:

    >>> event_dict = event.to_dict()
    >>> import json
    >>> json_str = json.dumps(event_dict)
    >>> print(json.loads(json_str)['event_type'])
    prompt_sent

    Accessing event data:

    >>> for event in trace.events:
    ...     if event.event_type == TraceEventType.RESPONSE_RECEIVED:
    ...         latency = event.data.get('latency_ms', 'N/A')
    ...         print(f"Step '{event.step_name}' latency: {latency}ms")
    Step 'Generate Summary' latency: 1250ms

    See Also
    --------
    TraceEventType : Available event types
    ExecutionTrace : Collection of events forming a complete trace
    PromptDebugger.log_event : Primary method for creating events
    """

    event_type: TraceEventType
    timestamp: float
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    parent_event_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"evt_{time.time_ns()}")

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary for serialization.

        Creates a JSON-serializable dictionary representation of the event,
        converting enum values to their string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all event fields with serializable values.

        Examples
        --------
        >>> event = DebugTraceEvent(
        ...     event_type=TraceEventType.STEP_START,
        ...     timestamp=1699999999.123,
        ...     step_id="s1",
        ...     step_name="Fetch Data"
        ... )
        >>> d = event.to_dict()
        >>> print(d['event_type'])
        step_start
        >>> print(d['step_name'])
        Fetch Data

        Serializing to JSON:

        >>> import json
        >>> json_str = json.dumps(event.to_dict(), indent=2)
        >>> print(json_str)
        {
          "event_id": "evt_...",
          "event_type": "step_start",
          ...
        }
        """
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
    """A complete execution trace containing all events from a workflow run.

    ExecutionTrace is the primary data structure for capturing and analyzing
    prompt workflow execution. It contains a chronological sequence of events,
    timing information, metadata, and the final result or error.

    Parameters
    ----------
    trace_id : str
        Unique identifier for this trace, typically auto-generated.
    start_time : float
        Unix timestamp when the trace started.
    end_time : float, optional
        Unix timestamp when the trace ended. None if still running.
    events : list[DebugTraceEvent], optional
        List of events captured during execution.
    metadata : dict[str, Any], optional
        Arbitrary metadata about the trace (workflow name, user, etc.).
    final_result : Any, optional
        The final output of the traced workflow.
    error : str, optional
        Error message if the trace ended with an error.

    Attributes
    ----------
    trace_id : str
        Unique trace identifier.
    start_time : float
        Trace start timestamp.
    end_time : float or None
        Trace end timestamp, None if incomplete.
    events : list[DebugTraceEvent]
        All captured events in chronological order.
    metadata : dict[str, Any]
        User-provided trace metadata.
    final_result : Any or None
        Workflow output, if successful.
    error : str or None
        Error message, if failed.

    Examples
    --------
    Accessing trace information after completion:

    >>> debugger = PromptDebugger()
    >>> with debugger.trace({"workflow": "qa_chain"}) as trace:
    ...     # Run workflow steps...
    ...     pass
    >>> print(f"Trace {trace.trace_id}")
    Trace trace_1699999999123456789
    >>> print(f"Duration: {trace.duration_ms:.2f}ms")
    Duration: 2543.75ms
    >>> print(f"Events captured: {trace.event_count}")
    Events captured: 12

    Filtering events by type:

    >>> prompts = trace.get_events_by_type(TraceEventType.PROMPT_SENT)
    >>> responses = trace.get_events_by_type(TraceEventType.RESPONSE_RECEIVED)
    >>> print(f"Made {len(prompts)} LLM calls")
    Made 3 LLM calls

    Getting all events for a specific step:

    >>> step_events = trace.get_step_events("step_summarize")
    >>> for event in step_events:
    ...     print(f"  {event.event_type.value}: {event.data}")

    Checking for errors:

    >>> if trace.error:
    ...     print(f"Trace failed: {trace.error}")
    ...     errors = trace.get_events_by_type(TraceEventType.ERROR_OCCURRED)
    ...     for error in errors:
    ...         print(f"  Error at {error.step_name}: {error.data.get('error')}")
    ... else:
    ...     print(f"Success! Result: {trace.final_result}")

    Exporting trace data:

    >>> trace_dict = trace.to_dict()
    >>> import json
    >>> with open("trace_export.json", "w") as f:
    ...     json.dump(trace_dict, f, indent=2, default=str)

    See Also
    --------
    DebugTraceEvent : Individual events within the trace
    PromptDebugger.trace : Context manager for creating traces
    TraceExporter : Export traces to various formats
    """

    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    events: list[DebugTraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate total trace duration in milliseconds.

        Returns
        -------
        float or None
            Duration in milliseconds, or None if trace hasn't ended.

        Examples
        --------
        >>> trace = ExecutionTrace(
        ...     trace_id="t1",
        ...     start_time=1000.0,
        ...     end_time=1002.5
        ... )
        >>> print(f"{trace.duration_ms:.0f}ms")
        2500ms
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def event_count(self) -> int:
        """Get the number of events in the trace.

        Returns
        -------
        int
            Total count of events captured.

        Examples
        --------
        >>> trace = ExecutionTrace(trace_id="t1", start_time=1000.0)
        >>> trace.events.append(DebugTraceEvent(
        ...     event_type=TraceEventType.STEP_START,
        ...     timestamp=1000.1
        ... ))
        >>> print(trace.event_count)
        1
        """
        return len(self.events)

    def get_events_by_type(self, event_type: TraceEventType) -> list[DebugTraceEvent]:
        """Filter events by their type.

        Args
        ----
        event_type : TraceEventType
            The type of events to retrieve.

        Returns
        -------
        list[DebugTraceEvent]
            All events matching the specified type, in chronological order.

        Examples
        --------
        Get all step start events:

        >>> starts = trace.get_events_by_type(TraceEventType.STEP_START)
        >>> print(f"Workflow has {len(starts)} steps")

        Analyze prompt/response pairs:

        >>> prompts = trace.get_events_by_type(TraceEventType.PROMPT_SENT)
        >>> responses = trace.get_events_by_type(TraceEventType.RESPONSE_RECEIVED)
        >>> for p, r in zip(prompts, responses):
        ...     latency = r.data.get('latency_ms', 0)
        ...     print(f"Call latency: {latency}ms")

        Check for errors:

        >>> errors = trace.get_events_by_type(TraceEventType.ERROR_OCCURRED)
        >>> if errors:
        ...     print(f"Found {len(errors)} errors")
        """
        return [e for e in self.events if e.event_type == event_type]

    def get_step_events(self, step_id: str) -> list[DebugTraceEvent]:
        """Get all events associated with a specific step.

        Args
        ----
        step_id : str
            The unique identifier of the step.

        Returns
        -------
        list[DebugTraceEvent]
            All events with matching step_id, in chronological order.

        Examples
        --------
        Investigate a specific step:

        >>> events = trace.get_step_events("step_generate")
        >>> for event in events:
        ...     print(f"{event.event_type.value}: {event.timestamp}")
        step_start: 1699999999.123
        prompt_sent: 1699999999.125
        response_received: 1700000000.350
        step_end: 1700000000.355

        Calculate step duration from events:

        >>> events = trace.get_step_events("step_1")
        >>> start = next(e for e in events if e.event_type == TraceEventType.STEP_START)
        >>> end = next(e for e in events if e.event_type == TraceEventType.STEP_END)
        >>> print(f"Step took {(end.timestamp - start.timestamp) * 1000:.0f}ms")
        """
        return [e for e in self.events if e.step_id == step_id]

    def to_dict(self) -> dict[str, Any]:
        """Convert the trace to a dictionary for serialization.

        Creates a complete JSON-serializable representation of the trace,
        including all events and metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all trace data.

        Examples
        --------
        Export trace to JSON file:

        >>> trace_data = trace.to_dict()
        >>> import json
        >>> with open("trace.json", "w") as f:
        ...     json.dump(trace_data, f, indent=2, default=str)

        Send trace to monitoring service:

        >>> import requests
        >>> requests.post(
        ...     "https://api.monitoring.com/traces",
        ...     json=trace.to_dict()
        ... )

        Access trace metadata:

        >>> data = trace.to_dict()
        >>> print(f"Trace ID: {data['trace_id']}")
        >>> print(f"Duration: {data['duration_ms']}ms")
        >>> print(f"Total events: {data['event_count']}")
        """
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
    """A detected issue or problem found during debugging analysis.

    DebugIssue captures problems identified during trace analysis, including
    performance issues, errors, and anti-patterns. Each issue includes context
    about where it occurred and suggestions for resolution.

    Parameters
    ----------
    category : IssueCategory
        Classification of the issue type (e.g., PERFORMANCE, LOGIC_ERROR).
    severity : IssueSeverity
        How serious the issue is (INFO, WARNING, ERROR, CRITICAL).
    message : str
        Human-readable description of the issue.
    step_id : str, optional
        ID of the step where the issue occurred.
    step_name : str, optional
        Name of the step where the issue occurred.
    suggestion : str, optional
        Recommended action to resolve the issue.
    context : dict[str, Any], optional
        Additional context data (stack traces, values, etc.).
    timestamp : float, optional
        When the issue was detected. Defaults to current time.

    Attributes
    ----------
    category : IssueCategory
        The type/category of issue.
    severity : IssueSeverity
        The severity level.
    message : str
        Description of what went wrong.
    step_id : str or None
        Associated step identifier.
    step_name : str or None
        Associated step name.
    suggestion : str or None
        How to fix the issue.
    context : dict[str, Any]
        Extra diagnostic information.
    timestamp : float
        Detection timestamp.

    Examples
    --------
    Creating an issue manually:

    >>> issue = DebugIssue(
    ...     category=IssueCategory.PERFORMANCE,
    ...     severity=IssueSeverity.WARNING,
    ...     message="Step 'Generate Summary' took 8500ms",
    ...     step_id="step_2",
    ...     step_name="Generate Summary",
    ...     suggestion="Consider using a faster model or caching results"
    ... )
    >>> print(f"[{issue.severity.value.upper()}] {issue.message}")
    [WARNING] Step 'Generate Summary' took 8500ms

    Processing detected issues:

    >>> issues = debugger.get_issues()
    >>> for issue in issues:
    ...     print(f"{issue.category.value}: {issue.message}")
    ...     if issue.suggestion:
    ...         print(f"  Fix: {issue.suggestion}")
    performance: Step 'Generate Summary' took 8500ms
      Fix: Consider using a faster model or caching results

    Filtering by severity for alerts:

    >>> critical_issues = [
    ...     issue for issue in debugger.get_issues()
    ...     if issue.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
    ... ]
    >>> if critical_issues:
    ...     send_alert(f"Found {len(critical_issues)} critical issues")

    Exporting issues to JSON:

    >>> issues_data = [issue.to_dict() for issue in debugger.get_issues()]
    >>> import json
    >>> print(json.dumps(issues_data, indent=2))

    See Also
    --------
    IssueCategory : Available issue categories
    IssueSeverity : Severity levels
    PromptDebugger.get_issues : Retrieve detected issues
    """

    category: IssueCategory
    severity: IssueSeverity
    message: str
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    suggestion: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert the issue to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary with all issue fields, enums converted to strings.

        Examples
        --------
        >>> issue = DebugIssue(
        ...     category=IssueCategory.API_ERROR,
        ...     severity=IssueSeverity.ERROR,
        ...     message="Rate limit exceeded",
        ...     suggestion="Add exponential backoff"
        ... )
        >>> d = issue.to_dict()
        >>> print(d['category'])
        api_error
        >>> print(d['severity'])
        error

        Export to JSON:

        >>> import json
        >>> json_str = json.dumps(issue.to_dict(), indent=2)
        """
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
    """A breakpoint for pausing workflow execution at specific points.

    Breakpoints allow interactive debugging by pausing execution when certain
    conditions are met. They can trigger on specific steps, variable changes,
    or custom conditions.

    Parameters
    ----------
    breakpoint_id : str
        Unique identifier for this breakpoint.
    step_id : str, optional
        Break when this step ID is encountered.
    step_name : str, optional
        Break when this step name is encountered.
    condition : Callable[[dict[str, Any]], bool], optional
        Custom condition function. Receives context dict with 'event',
        'trace', and 'variables' keys. Returns True to break.
    on_variable : str, optional
        Break when this variable is set or modified.
    enabled : bool, optional
        Whether the breakpoint is active. Defaults to True.
    hit_count : int, optional
        Number of times breakpoint has triggered. Defaults to 0.

    Attributes
    ----------
    breakpoint_id : str
        Unique identifier.
    step_id : str or None
        Step ID to break on.
    step_name : str or None
        Step name to break on.
    condition : Callable or None
        Custom break condition.
    on_variable : str or None
        Variable name to watch.
    enabled : bool
        Whether breakpoint is active.
    hit_count : int
        Number of times triggered.

    Examples
    --------
    Creating a simple step breakpoint:

    >>> bp = DebugBreakpoint(
    ...     breakpoint_id="bp1",
    ...     step_name="Generate Summary"
    ... )
    >>> debugger.add_breakpoint("bp1", step_name="Generate Summary")

    Creating a conditional breakpoint:

    >>> def check_long_document(context):
    ...     doc = context['variables'].get('document', '')
    ...     return len(doc) > 10000
    >>>
    >>> bp = DebugBreakpoint(
    ...     breakpoint_id="bp_long_doc",
    ...     step_name="Process Document",
    ...     condition=check_long_document
    ... )

    Creating a variable watchpoint:

    >>> bp = DebugBreakpoint(
    ...     breakpoint_id="watch_result",
    ...     on_variable="final_result"
    ... )

    Checking if breakpoint should trigger:

    >>> context = {
    ...     'event': current_event,
    ...     'trace': current_trace,
    ...     'variables': {'document': 'short text'}
    ... }
    >>> if bp.should_break(context):
    ...     print(f"Breakpoint {bp.breakpoint_id} triggered")

    Managing breakpoint state:

    >>> bp.enabled = False  # Disable temporarily
    >>> print(f"Breakpoint hit {bp.hit_count} times")

    See Also
    --------
    PromptDebugger.add_breakpoint : Add breakpoints to debugger
    PromptDebugger.set_break_handler : Set callback for breakpoint hits
    """

    breakpoint_id: str
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    condition: Optional[Callable[[dict[str, Any]], bool]] = None
    on_variable: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0

    def should_break(self, context: dict[str, Any]) -> bool:
        """Determine if the breakpoint should trigger based on current context.

        Evaluates the breakpoint's enabled state and optional condition function
        to determine if execution should pause.

        Args
        ----
        context : dict[str, Any]
            Current execution context containing:
            - 'event': The current DebugTraceEvent
            - 'trace': The current ExecutionTrace
            - 'variables': Dict of current variable values

        Returns
        -------
        bool
            True if breakpoint should trigger, False otherwise.

        Examples
        --------
        Basic check without condition:

        >>> bp = DebugBreakpoint(breakpoint_id="bp1", enabled=True)
        >>> bp.should_break({})
        True
        >>> bp.enabled = False
        >>> bp.should_break({})
        False

        Check with custom condition:

        >>> def error_condition(ctx):
        ...     return ctx.get('event', {}).get('data', {}).get('success') == False
        >>> bp = DebugBreakpoint(
        ...     breakpoint_id="bp_error",
        ...     condition=error_condition
        ... )
        >>> bp.should_break({'event': {'data': {'success': False}}})
        True
        >>> bp.should_break({'event': {'data': {'success': True}}})
        False
        """
        if not self.enabled:
            return False
        if self.condition is not None:
            return self.condition(context)
        return True


@dataclass
class VariableSnapshot:
    """A snapshot of a variable's state at a point in time during execution.

    VariableSnapshot records the value and metadata of a variable when it is
    created, read, or modified. This enables tracking variable state changes
    throughout workflow execution for debugging purposes.

    Parameters
    ----------
    name : str
        The name/identifier of the variable.
    value : Any
        The value of the variable at this point in time.
    timestamp : float
        Unix timestamp when the snapshot was taken.
    step_id : str, optional
        ID of the step where this operation occurred.
    operation : str, optional
        Type of operation: "set" (create/update), "read" (access),
        or "modified" (in-place change). Defaults to "set".

    Attributes
    ----------
    name : str
        Variable name.
    value : Any
        Variable value.
    timestamp : float
        When snapshot was taken.
    step_id : str or None
        Associated step ID.
    operation : str
        Operation type ("set", "read", "modified").

    Examples
    --------
    Creating a snapshot when setting a variable:

    >>> snapshot = VariableSnapshot(
    ...     name="document",
    ...     value="The quick brown fox...",
    ...     timestamp=time.time(),
    ...     step_id="step_fetch",
    ...     operation="set"
    ... )
    >>> print(f"Set {snapshot.name} = {snapshot.value[:20]}...")

    Recording a variable read:

    >>> read_snapshot = VariableSnapshot(
    ...     name="config",
    ...     value={"model": "claude-3", "temperature": 0.7},
    ...     timestamp=time.time(),
    ...     operation="read"
    ... )

    Converting to dictionary:

    >>> data = snapshot.to_dict()
    >>> print(f"{data['operation']}: {data['name']} ({data['value_type']})")
    set: document (str)

    Tracking variable history:

    >>> history = debugger.get_variable_history("document")
    >>> for snap in history:
    ...     print(f"{snap.operation} at {snap.timestamp}: {str(snap.value)[:50]}")
    set at 1699999999.1: The quick brown fox jumps over the l...
    modified at 1699999999.5: THE QUICK BROWN FOX JUMPS OVER THE L...

    See Also
    --------
    PromptDebugger.log_variable : Create variable snapshots
    PromptDebugger.get_variable_history : Retrieve variable history
    """

    name: str
    value: Any
    timestamp: float
    step_id: Optional[str] = None
    operation: str = "set"  # set, read, modified

    def to_dict(self) -> dict[str, Any]:
        """Convert the snapshot to a dictionary for serialization.

        The value is truncated to 500 characters to prevent excessive
        memory usage when serializing large values.

        Returns
        -------
        dict[str, Any]
            Dictionary containing snapshot data with truncated value.

        Examples
        --------
        >>> snapshot = VariableSnapshot(
        ...     name="data",
        ...     value={"key": "value", "items": [1, 2, 3]},
        ...     timestamp=1699999999.0
        ... )
        >>> d = snapshot.to_dict()
        >>> print(d['name'])
        data
        >>> print(d['value_type'])
        dict

        Long values are truncated:

        >>> long_snapshot = VariableSnapshot(
        ...     name="text",
        ...     value="x" * 1000,
        ...     timestamp=time.time()
        ... )
        >>> d = long_snapshot.to_dict()
        >>> len(d['value'])  # Truncated to 500 chars
        500
        """
        return {
            "name": self.name,
            "value": str(self.value)[:500],  # Truncate long values
            "value_type": type(self.value).__name__,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "operation": self.operation,
        }


class PromptDebugger:
    """Main debugger for tracing and analyzing prompt workflow execution.

    PromptDebugger is the central component for debugging prompt chains and
    agent workflows. It captures execution traces, logs events, manages
    breakpoints, and detects issues automatically.

    Parameters
    ----------
    debug_level : DebugLevel, optional
        Verbosity level controlling which events are captured.
        Defaults to DebugLevel.DETAILED.
    max_events : int, optional
        Maximum number of events to store in a trace before truncating.
        Helps prevent memory issues in long workflows. Defaults to 10000.
    auto_detect_issues : bool, optional
        Whether to automatically analyze traces for issues when they end.
        Defaults to True.

    Attributes
    ----------
    debug_level : DebugLevel
        Current verbosity level.
    max_events : int
        Maximum events per trace.
    auto_detect_issues : bool
        Whether auto-detection is enabled.

    Examples
    --------
    Basic usage with context manager:

    >>> debugger = PromptDebugger()
    >>> with debugger.trace({"workflow": "summarization"}) as trace:
    ...     debugger.step_start("s1", "Fetch Document")
    ...     doc = fetch_document(url)
    ...     debugger.step_end("s1", "Fetch Document", duration_ms=150)
    ...
    ...     debugger.step_start("s2", "Summarize")
    ...     debugger.log_prompt(prompt, step_id="s2")
    ...     result = call_llm(prompt)
    ...     debugger.log_response(result, step_id="s2", latency_ms=1200)
    ...     debugger.step_end("s2", "Summarize", duration_ms=1250)
    >>> print(f"Trace completed in {trace.duration_ms:.0f}ms")

    Using breakpoints for interactive debugging:

    >>> debugger = PromptDebugger()
    >>> def on_break(ctx):
    ...     print(f"Paused at {ctx['event'].step_name}")
    ...     print(f"Variables: {list(ctx['variables'].keys())}")
    >>> debugger.set_break_handler(on_break)
    >>> debugger.add_breakpoint("bp1", step_name="Process Data")
    >>> with debugger.trace() as trace:
    ...     run_workflow(debugger)

    Tracking variables throughout execution:

    >>> debugger = PromptDebugger()
    >>> with debugger.trace() as trace:
    ...     debugger.log_variable("input", user_query, step_id="s1")
    ...     # ... process ...
    ...     debugger.log_variable("result", final_result, step_id="s3")
    >>> history = debugger.get_variable_history("input")
    >>> print(f"Input was set {len(history)} times")

    Checking for issues after execution:

    >>> debugger = PromptDebugger(auto_detect_issues=True)
    >>> with debugger.trace() as trace:
    ...     run_slow_workflow(debugger)
    >>> issues = debugger.get_issues(severity=IssueSeverity.WARNING)
    >>> for issue in issues:
    ...     print(f"{issue.category.value}: {issue.message}")
    ...     if issue.suggestion:
    ...         print(f"  Suggestion: {issue.suggestion}")

    Production configuration with minimal overhead:

    >>> prod_debugger = PromptDebugger(
    ...     debug_level=DebugLevel.MINIMAL,
    ...     max_events=1000,
    ...     auto_detect_issues=False
    ... )

    Notes
    -----
    - Only one trace can be active at a time per debugger instance.
    - Events are truncated when max_events is reached; use larger values
      for complex workflows.
    - Breakpoint conditions should be lightweight to avoid performance impact.
    - Variable values are stored by reference; modifications after logging
      may affect stored snapshots.

    See Also
    --------
    ExecutionTrace : The trace data structure
    TraceVisualizer : Visualize traces
    PromptFlowAnalyzer : Analyze traces for patterns
    """

    def __init__(
        self,
        debug_level: DebugLevel = DebugLevel.DETAILED,
        max_events: int = 10000,
        auto_detect_issues: bool = True,
    ):
        """Initialize a new PromptDebugger instance.

        Args
        ----
        debug_level : DebugLevel, optional
            Verbosity level for event capture. Higher levels capture more
            detail but have higher overhead. Defaults to DebugLevel.DETAILED.
        max_events : int, optional
            Maximum events to store per trace. When exceeded, new events are
            dropped. Defaults to 10000.
        auto_detect_issues : bool, optional
            If True, automatically analyze traces for issues when end_trace()
            is called. Defaults to True.

        Examples
        --------
        Default configuration for development:

        >>> debugger = PromptDebugger()

        Minimal configuration for production:

        >>> debugger = PromptDebugger(
        ...     debug_level=DebugLevel.MINIMAL,
        ...     auto_detect_issues=False
        ... )

        High-detail configuration for complex debugging:

        >>> debugger = PromptDebugger(
        ...     debug_level=DebugLevel.VERBOSE,
        ...     max_events=50000
        ... )
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
        """Start a new execution trace for recording workflow events.

        Creates a new ExecutionTrace and sets it as the active trace. Clears
        any previous variable history, detected issues, and step stack.

        Args
        ----
        metadata : dict[str, Any], optional
            Arbitrary metadata to attach to the trace. Useful for recording
            workflow name, user ID, request ID, etc.

        Returns
        -------
        ExecutionTrace
            The newly created trace object.

        Raises
        ------
        RuntimeError
            If called while another trace is already active.

        Examples
        --------
        Basic trace start:

        >>> debugger = PromptDebugger()
        >>> trace = debugger.start_trace()
        >>> print(trace.trace_id)
        trace_1699999999123456789

        With metadata:

        >>> trace = debugger.start_trace({
        ...     "workflow": "document_qa",
        ...     "user_id": "user_123",
        ...     "request_id": "req_abc"
        ... })
        >>> print(trace.metadata['workflow'])
        document_qa

        Notes
        -----
        Prefer using the `trace()` context manager instead of manually
        calling start_trace/end_trace, as it ensures proper cleanup.

        See Also
        --------
        end_trace : End the current trace
        trace : Context manager for tracing
        """
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
        """End the current trace and finalize its data.

        Sets the end time, stores the result or error, runs auto-analysis
        if enabled, and returns the completed trace.

        Args
        ----
        result : Any, optional
            The final output of the workflow, if successful.
        error : str, optional
            Error message if the workflow failed.

        Returns
        -------
        ExecutionTrace
            The completed trace with all events and metadata.

        Raises
        ------
        RuntimeError
            If no trace is currently active.

        Examples
        --------
        End trace with result:

        >>> debugger = PromptDebugger()
        >>> trace = debugger.start_trace()
        >>> # ... do work ...
        >>> trace = debugger.end_trace(result="Summary: The document...")
        >>> print(trace.final_result)
        Summary: The document...

        End trace with error:

        >>> trace = debugger.start_trace()
        >>> try:
        ...     # ... do work that fails ...
        ...     raise ValueError("Invalid input")
        ... except Exception as e:
        ...     trace = debugger.end_trace(error=str(e))
        >>> print(trace.error)
        Invalid input

        Check auto-detected issues:

        >>> debugger = PromptDebugger(auto_detect_issues=True)
        >>> trace = debugger.start_trace()
        >>> # ... slow workflow ...
        >>> trace = debugger.end_trace()
        >>> issues = debugger.get_issues()
        >>> print(f"Found {len(issues)} issues")

        See Also
        --------
        start_trace : Start a new trace
        trace : Context manager alternative
        get_issues : Retrieve detected issues
        """
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
        """Create a context manager for automatic trace lifecycle management.

        This is the recommended way to trace workflow execution. The context
        manager automatically starts a trace on entry and ends it on exit,
        properly handling exceptions.

        Args
        ----
        metadata : dict[str, Any], optional
            Arbitrary metadata to attach to the trace.

        Returns
        -------
        _TraceContext
            A context manager that yields the ExecutionTrace.

        Examples
        --------
        Basic tracing:

        >>> debugger = PromptDebugger()
        >>> with debugger.trace() as trace:
        ...     debugger.step_start("s1", "Process")
        ...     result = do_work()
        ...     debugger.step_end("s1", "Process", duration_ms=100)
        >>> print(f"Completed in {trace.duration_ms:.0f}ms")

        With metadata:

        >>> with debugger.trace({"user": "alice", "action": "summarize"}) as trace:
        ...     summary = generate_summary(document)
        >>> print(trace.metadata)
        {'user': 'alice', 'action': 'summarize'}

        Exception handling:

        >>> try:
        ...     with debugger.trace() as trace:
        ...         raise ValueError("Something went wrong")
        ... except ValueError:
        ...     pass
        >>> print(trace.error)
        Something went wrong

        Nested with other context managers:

        >>> with debugger.trace() as trace, open("log.txt", "w") as log:
        ...     log.write(f"Starting trace {trace.trace_id}\\n")
        ...     run_workflow(debugger)

        See Also
        --------
        start_trace : Manual trace start
        end_trace : Manual trace end
        _TraceContext : The context manager implementation
        """
        return _TraceContext(self, metadata)

    def log_event(
        self,
        event_type: TraceEventType,
        step_id: Optional[str] = None,
        step_name: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> Optional[DebugTraceEvent]:
        """Log a generic trace event to the current trace.

        This is the low-level method for recording events. For common event
        types, prefer the specialized methods (step_start, step_end, log_prompt,
        etc.) which provide appropriate defaults and data formatting.

        Args
        ----
        event_type : TraceEventType
            The type of event being logged.
        step_id : str, optional
            Identifier of the associated workflow step.
        step_name : str, optional
            Display name of the associated step.
        data : dict[str, Any], optional
            Event-specific payload data.
        duration_ms : float, optional
            Duration in milliseconds (typically for STEP_END events).

        Returns
        -------
        DebugTraceEvent or None
            The created event, or None if no trace is active or max events
            has been reached.

        Examples
        --------
        Log a custom checkpoint event:

        >>> debugger = PromptDebugger()
        >>> with debugger.trace() as trace:
        ...     event = debugger.log_event(
        ...         TraceEventType.CHECKPOINT,
        ...         step_id="s1",
        ...         step_name="Data Validation",
        ...         data={"records_processed": 1000, "valid": 995}
        ...     )
        >>> print(event.event_type.value)
        checkpoint

        Log a loop iteration:

        >>> for i in range(10):
        ...     debugger.log_event(
        ...         TraceEventType.LOOP_ITERATION,
        ...         data={"iteration": i, "item_id": items[i].id}
        ...     )

        Log a conditional branch:

        >>> debugger.log_event(
        ...     TraceEventType.BRANCH_TAKEN,
        ...     step_id="decision",
        ...     data={"condition": "document_length > 1000", "branch": "long_doc"}
        ... )

        Notes
        -----
        Events are silently dropped if:
        - No trace is currently active
        - The trace has reached max_events limit

        Events automatically get the current parent from the step stack for
        hierarchical event relationships.

        See Also
        --------
        step_start : Log step start events
        step_end : Log step end events
        log_prompt : Log prompt sent events
        log_response : Log response received events
        """
        if self._current_trace is None:
            return None

        if len(self._current_trace.events) >= self.max_events:
            return None

        parent_id = self._step_stack[-1] if self._step_stack else None

        event = DebugTraceEvent(
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

    def step_start(self, step_id: str, step_name: str, **kwargs) -> DebugTraceEvent:
        """Log the beginning of a workflow step.

        Records a STEP_START event and pushes the step onto the internal stack
        for parent-child event relationships. Must be paired with a corresponding
        step_end() call.

        Args
        ----
        step_id : str
            Unique identifier for this step. Used for filtering and correlation.
        step_name : str
            Human-readable name for display in visualizations.
        **kwargs
            Additional data to include in the event (e.g., input parameters).

        Returns
        -------
        DebugTraceEvent
            The created step start event.

        Examples
        --------
        Basic step tracking:

        >>> import time
        >>> debugger = PromptDebugger()
        >>> with debugger.trace() as trace:
        ...     start_time = time.time()
        ...     debugger.step_start("s1", "Fetch Data")
        ...     data = fetch_data(url)
        ...     duration = (time.time() - start_time) * 1000
        ...     debugger.step_end("s1", "Fetch Data", duration_ms=duration)

        With additional data:

        >>> debugger.step_start(
        ...     "summarize",
        ...     "Generate Summary",
        ...     input_length=len(document),
        ...     model="claude-3-sonnet",
        ...     max_tokens=500
        ... )

        Nested steps:

        >>> debugger.step_start("outer", "Process Batch")
        >>> for item in items:
        ...     debugger.step_start(f"inner_{item.id}", "Process Item")
        ...     # ... process item ...
        ...     debugger.step_end(f"inner_{item.id}", "Process Item", duration_ms=50)
        >>> debugger.step_end("outer", "Process Batch", duration_ms=500)

        See Also
        --------
        step_end : Mark step completion
        log_event : Low-level event logging
        """
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
    ) -> DebugTraceEvent:
        """Log the completion of a workflow step.

        Records a STEP_END event with duration and result information, and
        pops the step from the internal stack.

        Args
        ----
        step_id : str
            Identifier matching the corresponding step_start call.
        step_name : str
            Name matching the corresponding step_start call.
        duration_ms : float
            Total step duration in milliseconds.
        result : Any, optional
            The output/result of the step. Truncated to 500 chars in storage.
        success : bool, optional
            Whether the step completed successfully. Defaults to True.

        Returns
        -------
        DebugTraceEvent
            The created step end event.

        Examples
        --------
        Basic step completion:

        >>> import time
        >>> start = time.time()
        >>> debugger.step_start("s1", "Process")
        >>> output = do_processing()
        >>> duration = (time.time() - start) * 1000
        >>> debugger.step_end("s1", "Process", duration_ms=duration, result=output)

        Recording failure:

        >>> try:
        ...     debugger.step_start("risky", "Risky Operation")
        ...     result = risky_operation()
        ...     debugger.step_end("risky", "Risky Operation", duration_ms=100, result=result)
        ... except Exception as e:
        ...     debugger.step_end(
        ...         "risky", "Risky Operation",
        ...         duration_ms=50,
        ...         result=str(e),
        ...         success=False
        ...     )

        Analyzing step performance:

        >>> trace = debugger.end_trace()
        >>> for event in trace.get_events_by_type(TraceEventType.STEP_END):
        ...     if not event.data.get('success'):
        ...         print(f"Failed: {event.step_name}")
        ...     elif event.duration_ms > 1000:
        ...         print(f"Slow: {event.step_name} ({event.duration_ms:.0f}ms)")

        See Also
        --------
        step_start : Mark step beginning
        """
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
    ) -> DebugTraceEvent:
        """Log a prompt being sent to an LLM.

        Records a PROMPT_SENT event with the prompt content (truncated),
        full length, and optionally the target model.

        Args
        ----
        prompt : str
            The prompt text being sent. Truncated to 1000 chars in storage.
        step_id : str, optional
            Identifier of the step making this call.
        model : str, optional
            The LLM model being called (e.g., "claude-3-sonnet").

        Returns
        -------
        DebugTraceEvent
            The created prompt sent event.

        Examples
        --------
        Log a simple prompt:

        >>> debugger.step_start("qa", "Answer Question")
        >>> prompt = f"Answer this question: {user_question}"
        >>> debugger.log_prompt(prompt, step_id="qa", model="claude-3-opus")
        >>> response = call_llm(prompt)
        >>> debugger.log_response(response, step_id="qa")

        Track multiple prompts in a chain:

        >>> for i, prompt in enumerate(prompt_chain):
        ...     debugger.log_prompt(prompt, step_id=f"chain_{i}")
        ...     response = call_llm(prompt)
        ...     debugger.log_response(response, step_id=f"chain_{i}")

        Analyze prompts in completed trace:

        >>> trace = debugger.end_trace()
        >>> for event in trace.get_events_by_type(TraceEventType.PROMPT_SENT):
        ...     print(f"Model: {event.data.get('model')}")
        ...     print(f"Prompt length: {event.data['prompt_length']} chars")

        See Also
        --------
        log_response : Log corresponding response
        """
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
    ) -> DebugTraceEvent:
        """Log a response received from an LLM.

        Records a RESPONSE_RECEIVED event with the response content (truncated),
        full length, token usage, and latency information.

        Args
        ----
        response : str
            The response text received. Truncated to 1000 chars in storage.
        step_id : str, optional
            Identifier of the step that made the call.
        tokens_used : int, optional
            Number of tokens consumed by this call.
        latency_ms : float, optional
            Time in milliseconds from prompt sent to response received.

        Returns
        -------
        DebugTraceEvent
            The created response received event.

        Examples
        --------
        Log response with metrics:

        >>> import time
        >>> start = time.time()
        >>> debugger.log_prompt(prompt, step_id="gen")
        >>> response = call_llm(prompt)
        >>> latency = (time.time() - start) * 1000
        >>> debugger.log_response(
        ...     response,
        ...     step_id="gen",
        ...     tokens_used=150,
        ...     latency_ms=latency
        ... )

        Calculate average latency:

        >>> trace = debugger.end_trace()
        >>> responses = trace.get_events_by_type(TraceEventType.RESPONSE_RECEIVED)
        >>> latencies = [r.data.get('latency_ms', 0) for r in responses]
        >>> avg_latency = sum(latencies) / len(latencies) if latencies else 0
        >>> print(f"Average latency: {avg_latency:.0f}ms")

        Track token usage:

        >>> total_tokens = sum(
        ...     r.data.get('tokens_used', 0)
        ...     for r in trace.get_events_by_type(TraceEventType.RESPONSE_RECEIVED)
        ... )
        >>> print(f"Total tokens used: {total_tokens}")

        See Also
        --------
        log_prompt : Log corresponding prompt
        """
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
    ) -> DebugTraceEvent:
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
    ) -> DebugTraceEvent:
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
    ) -> DebugTraceEvent:
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

    def _check_breakpoints(self, event: DebugTraceEvent) -> None:
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
                status = "OK" if event.data.get("success", True) else "FAIL"
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
                lines.append(f"[{elapsed:8.1f}ms]   ERROR: {error[:60]}")

            elif event.event_type == TraceEventType.WARNING:
                msg = event.data.get("message", "")
                lines.append(f"[{elapsed:8.1f}ms]   WARN: {msg[:60]}")

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
                status = "OK" if event.data.get("success", True) else "FAIL"
                lines.append(f"{indent}└─ {status} {event.duration_ms:.1f}ms")

            elif event.event_type == TraceEventType.ERROR_OCCURRED:
                indent = "  " * indent_level
                lines.append(f"{indent}│  ERROR {event.data.get('error', '')[:40]}")

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
                DebugTraceEvent(
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import TraceEvent. The canonical name is DebugTraceEvent.
TraceEvent = DebugTraceEvent
