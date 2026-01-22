"""Agent probe for testing tool-using LLM agents.

This module provides AgentProbe, a specialized probe for testing LLM agents
that use tools. It integrates with the tracing system for deterministic
execution recording and contract validation.

Example:
    >>> class SearchAgentProbe(AgentProbe):
    ...     def run_agent(self, model, prompt, tools):
    ...         return model.run(prompt, tools=tools)
    ...
    >>> probe = SearchAgentProbe(
    ...     name="search_agent",
    ...     tools=[search_tool, summarize_tool],
    ...     trace_config={"contracts": {"tool_order": {"must_follow": {"summarize": ["search"]}}}},
    ... )
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from insideLLMs.probes.base import Probe
from insideLLMs.trace_config import (
    OnViolationMode,
    TraceConfig,
    TracePayloadNormaliser,
    load_trace_config,
    validate_with_config,
)
from insideLLMs.tracing import TraceEventKind, TraceRecorder
from insideLLMs.types import ProbeCategory, ProbeResult, ProbeScore, ResultStatus


@dataclass
class ToolDefinition:
    """Definition for a tool available to the agent.

    Attributes:
        name: The tool's identifier.
        description: Human-readable description for the model.
        parameters: JSON Schema for the tool's parameters.
        handler: Optional callable that implements the tool.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Optional[Any] = None


@dataclass
class AgentProbeResult:
    """Result from running an agent probe.

    Attributes:
        prompt: The input prompt given to the agent.
        final_response: The agent's final response.
        tool_calls: List of tool calls made during execution.
        trace_events: Full trace of execution events.
        trace_fingerprint: Deterministic hash of the trace.
        violations: Any contract violations detected.
        metadata: Additional probe-specific data.
    """

    prompt: str
    final_response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    trace_fingerprint: Optional[str] = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentProbe(Probe[AgentProbeResult]):
    """Probe for testing tool-using LLM agents with trace integration.

    AgentProbe provides:
    - Tool definition and dispatch
    - Automatic trace recording via TraceRecorder
    - Contract validation via validate_with_config()
    - Configurable violation handling (record, fail_probe, fail_run)

    Subclasses must implement run_agent() which executes the agent
    and records trace events.

    Attributes:
        tools: List of tool definitions available to the agent.
        trace_config: Configuration for tracing and validation.
        normaliser: Payload normaliser for redaction.

    Example:
        >>> class MyAgentProbe(AgentProbe):
        ...     def run_agent(self, model, prompt, tools, recorder):
        ...         recorder.record_generate_start(prompt)
        ...         response = model.run_with_tools(prompt, tools)
        ...         recorder.record_generate_end(response)
        ...         return response
    """

    default_category: ProbeCategory = ProbeCategory.REASONING

    def __init__(
        self,
        name: str,
        tools: Optional[list[ToolDefinition]] = None,
        trace_config: Optional[dict[str, Any] | TraceConfig] = None,
        category: Optional[ProbeCategory] = None,
        description: Optional[str] = None,
    ):
        """Initialize the agent probe.

        Args:
            name: Human-readable name for this probe.
            tools: List of tool definitions available to the agent.
            trace_config: Configuration dict or TraceConfig for tracing.
            category: The category this probe belongs to.
            description: A brief description of what this probe tests.
        """
        super().__init__(name=name, category=category, description=description)

        self.tools = tools or []

        # Load trace config
        if trace_config is None:
            self._trace_config = TraceConfig()
        elif isinstance(trace_config, TraceConfig):
            self._trace_config = trace_config
        else:
            self._trace_config = load_trace_config(trace_config)

        # Create normaliser from config
        self._normaliser = TracePayloadNormaliser.from_config(self._trace_config)

    @property
    def trace_config(self) -> TraceConfig:
        """Get the trace configuration."""
        return self._trace_config

    @abstractmethod
    def run_agent(
        self,
        model: Any,
        prompt: str,
        tools: list[ToolDefinition],
        recorder: TraceRecorder,
    ) -> str:
        """Run the agent and record trace events.

        Subclasses must implement this method to:
        1. Execute the agent with the given model and tools
        2. Record relevant trace events using the recorder
        3. Return the final response

        Args:
            model: The model to test.
            prompt: The input prompt for the agent.
            tools: List of available tools.
            recorder: TraceRecorder for recording events.

        Returns:
            The agent's final response string.
        """
        raise NotImplementedError("Subclasses must implement run_agent")

    def run(
        self,
        model: Any,
        data: Any,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentProbeResult:
        """Run the agent probe on a single input.

        Args:
            model: The model to test.
            data: The input data (prompt string or dict with 'prompt' key).
            run_id: Optional run ID for the trace.
            example_id: Optional example ID for the trace.
            **kwargs: Additional arguments passed to run_agent.

        Returns:
            AgentProbeResult with trace data and any violations.

        Raises:
            ValueError: If violations found and on_violation is fail_probe.
        """
        from insideLLMs.tracing import trace_fingerprint

        # Extract prompt
        if isinstance(data, str):
            prompt = data
        elif isinstance(data, dict):
            prompt = data.get("prompt", str(data))
        else:
            prompt = str(data)

        # Create recorder
        recorder = TraceRecorder(run_id=run_id, example_id=example_id)

        # Run the agent
        final_response = self.run_agent(model, prompt, self.tools, recorder, **kwargs)

        # Get trace events
        events = recorder.events
        event_dicts = [e.to_dict() for e in events]

        # Normalise payloads if configured
        if self._trace_config.store.redact.enabled:
            for event_dict in event_dicts:
                event_dict["payload"] = self._normaliser.normalise(event_dict.get("payload", {}))

        # Compute fingerprint
        fingerprint = trace_fingerprint(events)

        # Validate contracts
        violations = validate_with_config(events, self._trace_config)
        violation_dicts = [v.to_dict() for v in violations]

        # Extract tool calls from trace
        tool_calls = [
            {
                "tool_name": e.payload.get("tool_name"),
                "arguments": e.payload.get("arguments"),
                "seq": e.seq,
            }
            for e in events
            if e.kind == TraceEventKind.TOOL_CALL_START.value
        ]

        result = AgentProbeResult(
            prompt=prompt,
            final_response=final_response,
            tool_calls=tool_calls,
            trace_events=event_dicts,
            trace_fingerprint=fingerprint,
            violations=violation_dicts,
            metadata=kwargs.get("metadata", {}),
        )

        # Handle violations based on config
        if violations and self._trace_config.on_violation.mode == OnViolationMode.FAIL_PROBE:
            raise ValueError(f"Trace contract violations: {[v.detail for v in violations]}")

        return result

    def score(self, results: list[ProbeResult[AgentProbeResult]]) -> ProbeScore:
        """Calculate aggregate scores from agent probe results.

        Includes violation rate in custom metrics.

        Args:
            results: The list of probe results to score.

        Returns:
            A ProbeScore with aggregate metrics.
        """
        base_score = super().score(results)

        # Calculate violation rate
        violation_count = 0
        total_with_output = 0

        for r in results:
            if r.status == ResultStatus.SUCCESS and r.output:
                total_with_output += 1
                if r.output.violations:
                    violation_count += 1

        if total_with_output > 0:
            base_score.custom_metrics["violation_rate"] = violation_count / total_with_output

        return base_score

    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name.

        Args:
            name: The tool name to find.

        Returns:
            The ToolDefinition or None if not found.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def info(self) -> dict[str, Any]:
        """Return probe metadata including tool info."""
        base_info = super().info()
        base_info["tools"] = [{"name": t.name, "description": t.description} for t in self.tools]
        base_info["trace_enabled"] = self._trace_config.enabled
        base_info["contracts_enabled"] = self._trace_config.contracts.enabled
        return base_info
