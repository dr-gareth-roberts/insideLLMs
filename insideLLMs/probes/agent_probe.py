"""Agent probe for testing tool-using LLM agents.

This module provides ``AgentProbe``, a specialized probe for testing LLM agents
that use tools. It integrates with the tracing system for deterministic
execution recording and contract validation. The probe captures complete
execution traces including tool calls, results, and final responses, enabling
reproducible testing and behavioral analysis of agentic systems.

Overview
--------
Agent probes are designed for testing LLMs that interact with external tools
or functions. Unlike simple text-in-text-out probes, agent probes capture
the full execution flow including:

- Tool invocations with arguments
- Tool results and error handling
- Multi-step reasoning chains
- Contract violations and ordering constraints

The tracing system uses logical sequence numbers (not wall-clock time) to
ensure traces are deterministic and reproducible across runs.

Key Components
--------------
ToolDefinition
    Dataclass for defining tools available to the agent, including name,
    description, parameters schema, and optional handler function.

AgentProbeResult
    Dataclass containing the complete result of an agent probe run, including
    the final response, tool calls, trace events, fingerprint, and violations.

AgentProbe
    Abstract base class for agent probes. Subclasses implement ``run_agent()``
    to define how the agent executes with the given model and tools.

Architecture
------------
The typical flow for an agent probe:

1. **Initialization**: Configure tools, trace settings, and contracts
2. **Execution**: Call ``run()`` which creates a ``TraceRecorder`` and
   invokes the subclass's ``run_agent()`` method
3. **Recording**: The implementation records trace events for tool calls,
   results, and model interactions
4. **Validation**: Trace events are validated against configured contracts
5. **Results**: An ``AgentProbeResult`` is returned with the complete trace

Examples
--------
Basic agent probe for a search-and-summarize workflow:

    >>> from insideLLMs.probes.agent_probe import AgentProbe, ToolDefinition
    >>>
    >>> # Define the tools
    >>> search_tool = ToolDefinition(
    ...     name="search",
    ...     description="Search the web for information",
    ...     parameters={"query": {"type": "string"}}
    ... )
    >>> summarize_tool = ToolDefinition(
    ...     name="summarize",
    ...     description="Summarize text content",
    ...     parameters={"text": {"type": "string"}}
    ... )
    >>>
    >>> class SearchAndSummarizeProbe(AgentProbe):
    ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
    ...         # Record the start of generation
    ...         recorder.record_generate_start(prompt)
    ...
    ...         # Simulate agent execution with tool calls
    ...         response = model.run_with_tools(prompt, tools)
    ...
    ...         # Record tool calls from the response
    ...         for call in response.tool_calls:
    ...             recorder.record_tool_call(call.name, call.arguments)
    ...             result = execute_tool(call)
    ...             recorder.record_tool_result(call.name, result)
    ...
    ...         # Record final response
    ...         recorder.record_generate_end(response.final_answer)
    ...         return response.final_answer
    ...
    >>> probe = SearchAndSummarizeProbe(
    ...     name="search_summarize",
    ...     tools=[search_tool, summarize_tool],
    ... )

Agent probe with contract validation:

    >>> probe = SearchAndSummarizeProbe(
    ...     name="search_summarize_validated",
    ...     tools=[search_tool, summarize_tool],
    ...     trace_config={
    ...         "enabled": True,
    ...         "contracts": {
    ...             "enabled": True,
    ...             "tool_order": {
    ...                 "enabled": True,
    ...                 "must_follow": {"summarize": ["search"]}
    ...             },
    ...             "tool_results": {"enabled": True}
    ...         },
    ...         "on_violation": {"mode": "fail_probe"}
    ...     }
    ... )

Running an agent probe and inspecting results:

    >>> result = probe.run(model, "Find information about Python and summarize it")
    >>> print(f"Final response: {result.final_response}")
    Final response: Python is a high-level programming language...
    >>> print(f"Tool calls made: {[tc['tool_name'] for tc in result.tool_calls]}")
    Tool calls made: ['search', 'summarize']
    >>> print(f"Trace fingerprint: {result.trace_fingerprint}")
    Trace fingerprint: sha256:abc123...
    >>> if result.violations:
    ...     print(f"Violations found: {result.violations}")

Batch execution with scoring:

    >>> from insideLLMs.types import ProbeResult, ResultStatus
    >>>
    >>> prompts = [
    ...     "Search for climate change data and summarize",
    ...     "Find Python tutorials and create a summary",
    ...     "Look up machine learning basics and explain",
    ... ]
    >>> results = probe.run_batch(model, prompts)
    >>> score = probe.score(results)
    >>> print(f"Success rate: {score.accuracy:.1%}")
    Success rate: 100.0%
    >>> print(f"Violation rate: {score.custom_metrics.get('violation_rate', 0):.1%}")
    Violation rate: 0.0%

Agent probe with tool handlers:

    >>> def search_handler(query: str) -> dict:
    ...     \"\"\"Execute a web search.\"\"\"
    ...     return {"results": ["result1", "result2"]}
    ...
    >>> search_tool = ToolDefinition(
    ...     name="search",
    ...     description="Search the web",
    ...     parameters={"query": {"type": "string"}},
    ...     handler=search_handler
    ... )
    >>>
    >>> # Access the handler
    >>> tool = probe.get_tool_by_name("search")
    >>> if tool and tool.handler:
    ...     result = tool.handler("Python programming")
    ...     print(result)
    {'results': ['result1', 'result2']}

Testing with trace fingerprinting for drift detection:

    >>> # Run the same prompt twice
    >>> result1 = probe.run(model, "Search for AI news", run_id="run_001")
    >>> result2 = probe.run(model, "Search for AI news", run_id="run_002")
    >>>
    >>> # Compare fingerprints to detect behavioral drift
    >>> if result1.trace_fingerprint == result2.trace_fingerprint:
    ...     print("Deterministic behavior confirmed")
    ... else:
    ...     print("Warning: Different execution traces detected")

Notes
-----
- All trace operations are thread-safe via internal locking in ``TraceRecorder``
- Sequence numbers start at 0 and increment monotonically
- The fingerprint algorithm uses SHA-256 with canonical JSON serialization
- Contract validation is configurable; violations can be recorded, cause
  probe failure, or cause the entire run to fail
- Tool step counters are separate from event sequence counters

Thread Safety
-------------
``AgentProbe`` instances are thread-safe for concurrent ``run()`` calls.
Each call creates its own ``TraceRecorder`` instance with isolated state.
However, the probe's tools list and configuration are shared; do not modify
them after construction if using the probe from multiple threads.

See Also
--------
insideLLMs.probes.base.Probe : Base class for all probes
insideLLMs.tracing.TraceRecorder : Thread-safe trace event recorder
insideLLMs.trace_config.TraceConfig : Configuration for tracing and validation
insideLLMs.trace_config.validate_with_config : Contract validation function

References
----------
.. [1] ReAct: Synergizing Reasoning and Acting in Language Models
       https://arxiv.org/abs/2210.03629
.. [2] Toolformer: Language Models Can Teach Themselves to Use Tools
       https://arxiv.org/abs/2302.04761
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

    This dataclass defines a tool that an LLM agent can invoke during execution.
    Tools are the primary mechanism for agents to interact with external systems,
    retrieve information, or perform actions. The definition includes metadata
    for the LLM (name, description, parameters) and optionally a handler function
    for actual execution.

    Tools are typically passed to language models in a format they can understand
    (often JSON Schema-based), allowing the model to decide when and how to use
    each tool based on the task at hand.

    Parameters
    ----------
    name : str
        The tool's unique identifier. This is the name the LLM will use when
        invoking the tool. Should be descriptive but concise, using snake_case
        or camelCase consistently.
    description : str
        Human-readable description of what the tool does. This is provided to
        the LLM to help it understand when to use the tool. Should clearly
        explain the tool's purpose, inputs, and outputs.
    parameters : dict[str, Any], optional
        JSON Schema defining the tool's parameters. This schema is provided to
        the LLM to structure its tool calls correctly. Default is an empty dict,
        meaning the tool takes no parameters.
    handler : callable, optional
        A callable that implements the tool's functionality. When provided,
        the handler can be invoked directly to execute the tool. The handler
        should accept keyword arguments matching the parameters schema.
        Default is None (tool is for schema definition only).

    Attributes
    ----------
    name : str
        The tool's unique identifier.
    description : str
        Human-readable description for the model.
    parameters : dict[str, Any]
        JSON Schema for the tool's parameters.
    handler : callable or None
        Optional callable that implements the tool.

    Examples
    --------
    Basic tool definition without handler (schema only):

        >>> from insideLLMs.probes.agent_probe import ToolDefinition
        >>>
        >>> search_tool = ToolDefinition(
        ...     name="web_search",
        ...     description="Search the web for current information on any topic",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "query": {
        ...                 "type": "string",
        ...                 "description": "The search query"
        ...             },
        ...             "num_results": {
        ...                 "type": "integer",
        ...                 "description": "Number of results to return",
        ...                 "default": 5
        ...             }
        ...         },
        ...         "required": ["query"]
        ...     }
        ... )
        >>> print(search_tool.name)
        web_search

    Tool with a handler function:

        >>> def calculator_handler(expression: str) -> dict:
        ...     \"\"\"Evaluate a mathematical expression.\"\"\"
        ...     try:
        ...         result = eval(expression, {"__builtins__": {}})
        ...         return {"result": result, "success": True}
        ...     except Exception as e:
        ...         return {"error": str(e), "success": False}
        ...
        >>> calc_tool = ToolDefinition(
        ...     name="calculator",
        ...     description="Evaluate mathematical expressions",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "expression": {"type": "string"}
        ...         },
        ...         "required": ["expression"]
        ...     },
        ...     handler=calculator_handler
        ... )
        >>> result = calc_tool.handler(expression="2 + 2 * 3")
        >>> print(result)
        {'result': 8, 'success': True}

    Tool with complex parameter schema:

        >>> email_tool = ToolDefinition(
        ...     name="send_email",
        ...     description="Send an email to specified recipients",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "to": {
        ...                 "type": "array",
        ...                 "items": {"type": "string", "format": "email"},
        ...                 "description": "List of recipient email addresses"
        ...             },
        ...             "subject": {
        ...                 "type": "string",
        ...                 "maxLength": 200
        ...             },
        ...             "body": {
        ...                 "type": "string",
        ...                 "description": "Email body content"
        ...             },
        ...             "priority": {
        ...                 "type": "string",
        ...                 "enum": ["low", "normal", "high"],
        ...                 "default": "normal"
        ...             }
        ...         },
        ...         "required": ["to", "subject", "body"]
        ...     }
        ... )

    Converting tool definitions for OpenAI function calling:

        >>> def to_openai_function(tool: ToolDefinition) -> dict:
        ...     \"\"\"Convert ToolDefinition to OpenAI function format.\"\"\"
        ...     return {
        ...         "name": tool.name,
        ...         "description": tool.description,
        ...         "parameters": tool.parameters or {"type": "object", "properties": {}}
        ...     }
        ...
        >>> openai_format = to_openai_function(search_tool)
        >>> print(openai_format["name"])
        web_search

    Creating tools from existing APIs:

        >>> import requests
        >>>
        >>> def weather_handler(city: str, units: str = "metric") -> dict:
        ...     \"\"\"Get weather data for a city.\"\"\"
        ...     # In production, this would call an actual API
        ...     return {"city": city, "temp": 22, "units": units}
        ...
        >>> weather_tool = ToolDefinition(
        ...     name="get_weather",
        ...     description="Get current weather for a city",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "city": {"type": "string"},
        ...             "units": {"type": "string", "enum": ["metric", "imperial"]}
        ...         },
        ...         "required": ["city"]
        ...     },
        ...     handler=weather_handler
        ... )

    See Also
    --------
    AgentProbe : The probe class that uses ToolDefinition
    AgentProbe.get_tool_by_name : Retrieve a tool by name from the probe

    Notes
    -----
    The parameters schema follows JSON Schema Draft-07 specification. Common
    schema properties include:

    - ``type``: The data type (string, number, integer, boolean, array, object)
    - ``properties``: For objects, defines each property's schema
    - ``required``: List of required property names
    - ``enum``: Allowed values for the parameter
    - ``default``: Default value if not provided
    - ``description``: Human-readable description of the parameter
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Optional[Any] = None


@dataclass
class AgentProbeResult:
    """Result from running an agent probe.

    This dataclass contains the complete output from a single agent probe
    execution, including the agent's final response, all tool interactions,
    the full execution trace, and any contract violations detected. It provides
    a comprehensive record for analysis, debugging, and regression testing.

    The trace fingerprint enables drift detection by providing a deterministic
    hash of the execution trace. Identical fingerprints across runs indicate
    identical behavior, while differing fingerprints signal behavioral changes.

    Parameters
    ----------
    prompt : str
        The input prompt given to the agent. This is the original task or
        question the agent was asked to complete.
    final_response : str
        The agent's final textual response after completing all tool
        interactions and reasoning steps.
    tool_calls : list[dict[str, Any]], optional
        List of tool calls made during execution. Each dict contains:
        - ``tool_name``: Name of the tool called
        - ``arguments``: Arguments passed to the tool
        - ``seq``: Sequence number in the trace
        Default is an empty list.
    trace_events : list[dict[str, Any]], optional
        Full trace of all execution events in chronological order. Each
        event is a serialized ``TraceEvent`` containing seq, kind, payload,
        and context IDs. Default is an empty list.
    trace_fingerprint : str, optional
        Deterministic SHA-256 hash of the trace events, formatted as
        "sha256:<hex>". Used for detecting behavioral drift between runs.
        None if fingerprinting is disabled.
    violations : list[dict[str, Any]], optional
        Contract violations detected during validation. Each dict contains:
        - ``code``: Violation code (e.g., "TOOL_ORDER_VIOLATION")
        - ``detail``: Human-readable description
        - ``event_seq``: Sequence number of the violating event
        Default is an empty list.
    metadata : dict[str, Any], optional
        Additional probe-specific data such as model configuration, timing
        information, or custom annotations. Default is an empty dict.

    Attributes
    ----------
    prompt : str
        The input prompt given to the agent.
    final_response : str
        The agent's final response.
    tool_calls : list[dict[str, Any]]
        List of tool calls made during execution.
    trace_events : list[dict[str, Any]]
        Full trace of execution events.
    trace_fingerprint : str or None
        Deterministic hash of the trace.
    violations : list[dict[str, Any]]
        Contract violations detected.
    metadata : dict[str, Any]
        Additional probe-specific data.

    Examples
    --------
    Inspecting a successful agent run:

        >>> from insideLLMs.probes.agent_probe import AgentProbeResult
        >>>
        >>> result = AgentProbeResult(
        ...     prompt="Find the weather in Paris and summarize it",
        ...     final_response="The weather in Paris is currently sunny with 22C.",
        ...     tool_calls=[
        ...         {"tool_name": "weather_api", "arguments": {"city": "Paris"}, "seq": 1},
        ...         {"tool_name": "summarize", "arguments": {"text": "..."}, "seq": 3}
        ...     ],
        ...     trace_fingerprint="sha256:abc123def456..."
        ... )
        >>> print(f"Agent made {len(result.tool_calls)} tool calls")
        Agent made 2 tool calls
        >>> print(f"Tools used: {[tc['tool_name'] for tc in result.tool_calls]}")
        Tools used: ['weather_api', 'summarize']

    Analyzing tool call patterns:

        >>> # Count tool usage
        >>> from collections import Counter
        >>> tool_counts = Counter(tc["tool_name"] for tc in result.tool_calls)
        >>> print(f"Tool usage: {dict(tool_counts)}")
        Tool usage: {'weather_api': 1, 'summarize': 1}
        >>>
        >>> # Check tool ordering
        >>> tool_sequence = [tc["tool_name"] for tc in result.tool_calls]
        >>> assert tool_sequence.index("weather_api") < tool_sequence.index("summarize")

    Working with trace events:

        >>> result = AgentProbeResult(
        ...     prompt="Calculate 2+2",
        ...     final_response="4",
        ...     trace_events=[
        ...         {"seq": 0, "kind": "generate_start", "payload": {"prompt": "Calculate 2+2"}},
        ...         {"seq": 1, "kind": "tool_call_start", "payload": {"tool_name": "calculator"}},
        ...         {"seq": 2, "kind": "tool_result", "payload": {"result": 4}},
        ...         {"seq": 3, "kind": "generate_end", "payload": {"response": "4"}}
        ...     ]
        ... )
        >>> # Filter events by kind
        >>> tool_events = [e for e in result.trace_events if "tool" in e["kind"]]
        >>> print(f"Tool-related events: {len(tool_events)}")
        Tool-related events: 2

    Handling violations:

        >>> result = AgentProbeResult(
        ...     prompt="Search and summarize",
        ...     final_response="Summary without search",
        ...     violations=[
        ...         {
        ...             "code": "TOOL_ORDER_VIOLATION",
        ...             "detail": "summarize called before required search",
        ...             "event_seq": 1
        ...         }
        ...     ]
        ... )
        >>> if result.violations:
        ...     print(f"Found {len(result.violations)} violations:")
        ...     for v in result.violations:
        ...         print(f"  - {v['code']}: {v['detail']}")
        Found 1 violations:
          - TOOL_ORDER_VIOLATION: summarize called before required search

    Using fingerprints for drift detection:

        >>> # Store baseline fingerprint
        >>> baseline = result.trace_fingerprint
        >>>
        >>> # Later, compare new runs
        >>> new_result = probe.run(model, same_prompt)
        >>> if new_result.trace_fingerprint != baseline:
        ...     print("WARNING: Agent behavior has changed!")
        ...     # Investigate differences in trace_events

    Adding custom metadata:

        >>> result = AgentProbeResult(
        ...     prompt="Complex task",
        ...     final_response="Done",
        ...     metadata={
        ...         "model_temperature": 0.7,
        ...         "total_tokens": 1500,
        ...         "execution_time_ms": 2500,
        ...         "retry_count": 0,
        ...         "custom_tag": "experiment_v2"
        ...     }
        ... )
        >>> print(f"Execution time: {result.metadata['execution_time_ms']}ms")
        Execution time: 2500ms

    See Also
    --------
    AgentProbe.run : Method that produces AgentProbeResult
    insideLLMs.tracing.TraceEvent : Individual trace event structure
    insideLLMs.tracing.trace_fingerprint : Function to compute trace fingerprints

    Notes
    -----
    The trace_events list contains serialized TraceEvent objects. Each event
    has these standard fields:

    - ``seq``: Monotonically increasing sequence number (0-indexed)
    - ``kind``: Event type (e.g., "generate_start", "tool_call_start")
    - ``payload``: Event-specific data dictionary
    - ``run_id``: Optional run identifier (if provided to probe.run)
    - ``example_id``: Optional example identifier (if provided to probe.run)

    The violations list contains serialized ContractViolation objects from
    the trace_config validation system. Empty list indicates no violations.
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

    AgentProbe is an abstract base class for creating probes that test LLM agents
    capable of using tools. It provides a complete framework for:

    - Tool definition and management
    - Automatic trace recording via ``TraceRecorder``
    - Contract validation via ``validate_with_config()``
    - Configurable violation handling (record, fail_probe, fail_run)
    - Deterministic fingerprinting for behavioral drift detection

    Subclasses must implement the abstract ``run_agent()`` method, which defines
    how the agent executes with the provided model, tools, and trace recorder.
    The base class handles trace management, validation, and result packaging.

    Parameters
    ----------
    name : str
        Human-readable name for this probe. Used for identification in
        reports, logs, and result aggregation.
    tools : list[ToolDefinition], optional
        List of tool definitions available to the agent. These define
        the tools the agent can call during execution. Default is empty.
    trace_config : dict or TraceConfig, optional
        Configuration for tracing and contract validation. Can be a dict
        (parsed via ``load_trace_config``) or a ``TraceConfig`` object.
        Default creates a default TraceConfig.
    category : ProbeCategory, optional
        The category this probe belongs to. Default is ``ProbeCategory.REASONING``.
    description : str, optional
        Brief description of what this probe tests. Defaults to class docstring.

    Attributes
    ----------
    tools : list[ToolDefinition]
        List of tool definitions available to the agent.
    trace_config : TraceConfig
        Configuration for tracing and validation (read-only property).
    default_category : ProbeCategory
        Class-level default category. Set to ``ProbeCategory.REASONING``.

    Examples
    --------
    Minimal agent probe implementation:

        >>> from insideLLMs.probes.agent_probe import AgentProbe, ToolDefinition
        >>> from insideLLMs.tracing import TraceRecorder
        >>>
        >>> class SimpleAgentProbe(AgentProbe):
        ...     \"\"\"A simple agent probe that executes tools in sequence.\"\"\"
        ...
        ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
        ...         # Record the start of agent execution
        ...         recorder.record_generate_start(prompt)
        ...
        ...         # Call the model with available tools
        ...         response = model.generate_with_tools(prompt, tools)
        ...
        ...         # Record any tool calls made by the model
        ...         for tool_call in response.tool_calls:
        ...             recorder.record_tool_call(
        ...                 tool_name=tool_call.name,
        ...                 arguments=tool_call.arguments
        ...             )
        ...             # Execute the tool and record result
        ...             tool = self.get_tool_by_name(tool_call.name)
        ...             if tool and tool.handler:
        ...                 result = tool.handler(**tool_call.arguments)
        ...                 recorder.record_tool_result(tool_call.name, result)
        ...
        ...         # Record the final response
        ...         recorder.record_generate_end(response.content)
        ...         return response.content

    Creating and running an agent probe:

        >>> # Define tools
        >>> search_tool = ToolDefinition(
        ...     name="search",
        ...     description="Search for information",
        ...     parameters={"type": "object", "properties": {"query": {"type": "string"}}}
        ... )
        >>>
        >>> # Create probe
        >>> probe = SimpleAgentProbe(
        ...     name="search_agent_test",
        ...     tools=[search_tool],
        ...     description="Test agent search capabilities"
        ... )
        >>>
        >>> # Run the probe
        >>> result = probe.run(model, "Find information about Python")
        >>> print(f"Response: {result.final_response}")
        >>> print(f"Tools used: {[tc['tool_name'] for tc in result.tool_calls]}")

    Agent probe with contract validation:

        >>> probe = SimpleAgentProbe(
        ...     name="validated_agent",
        ...     tools=[search_tool, summarize_tool],
        ...     trace_config={
        ...         "enabled": True,
        ...         "contracts": {
        ...             "enabled": True,
        ...             "tool_order": {
        ...                 "enabled": True,
        ...                 "must_follow": {
        ...                     "summarize": ["search"]  # summarize requires search first
        ...                 }
        ...             },
        ...             "tool_results": {
        ...                 "enabled": True  # Every tool call must have a result
        ...             }
        ...         },
        ...         "on_violation": {
        ...             "mode": "fail_probe"  # Raise exception on violation
        ...         }
        ...     }
        ... )
        >>>
        >>> try:
        ...     result = probe.run(model, "Search and summarize Python info")
        ... except ValueError as e:
        ...     print(f"Contract violation: {e}")

    Agent probe with payload redaction:

        >>> probe = SimpleAgentProbe(
        ...     name="redacted_agent",
        ...     tools=[search_tool],
        ...     trace_config={
        ...         "store": {
        ...             "redact": {
        ...                 "enabled": True,
        ...                 "keys": ["api_key", "password", "token"],
        ...                 "patterns": ["[A-Za-z0-9]{32,}"]
        ...             }
        ...         }
        ...     }
        ... )

    Multi-tool agent with handler functions:

        >>> def search_handler(query: str) -> dict:
        ...     return {"results": ["Python is a programming language"]}
        ...
        >>> def summarize_handler(text: str) -> dict:
        ...     return {"summary": text[:100]}
        ...
        >>> class MultiToolAgentProbe(AgentProbe):
        ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
        ...         recorder.record_generate_start(prompt)
        ...
        ...         # Simulate multi-step agent execution
        ...         search_result = None
        ...         for tool in tools:
        ...             if tool.name == "search" and tool.handler:
        ...                 recorder.record_tool_call("search", {"query": prompt})
        ...                 search_result = tool.handler(query=prompt)
        ...                 recorder.record_tool_result("search", search_result)
        ...
        ...         if search_result:
        ...             for tool in tools:
        ...                 if tool.name == "summarize" and tool.handler:
        ...                     text = str(search_result.get("results", []))
        ...                     recorder.record_tool_call("summarize", {"text": text})
        ...                     summary = tool.handler(text=text)
        ...                     recorder.record_tool_result("summarize", summary)
        ...                     final = summary.get("summary", "")
        ...                     recorder.record_generate_end(final)
        ...                     return final
        ...
        ...         recorder.record_generate_end("No results")
        ...         return "No results"
        ...
        >>> probe = MultiToolAgentProbe(
        ...     name="multi_tool_agent",
        ...     tools=[
        ...         ToolDefinition("search", "Search", {}, search_handler),
        ...         ToolDefinition("summarize", "Summarize", {}, summarize_handler)
        ...     ]
        ... )

    Getting probe metadata and tool information:

        >>> info = probe.info()
        >>> print(f"Probe: {info['name']}")
        Probe: multi_tool_agent
        >>> print(f"Tools: {[t['name'] for t in info['tools']]}")
        Tools: ['search', 'summarize']
        >>> print(f"Tracing enabled: {info['trace_enabled']}")
        Tracing enabled: True

    Batch execution with violation rate scoring:

        >>> prompts = [
        ...     "Search for AI news",
        ...     "Find Python tutorials",
        ...     "Look up machine learning"
        ... ]
        >>> results = probe.run_batch(model, prompts)
        >>> score = probe.score(results)
        >>> print(f"Success rate: {score.accuracy:.1%}")
        >>> print(f"Violation rate: {score.custom_metrics.get('violation_rate', 0):.1%}")

    Using run IDs for trace correlation:

        >>> result = probe.run(
        ...     model=model,
        ...     data="Complex query",
        ...     run_id="experiment_001",
        ...     example_id="query_42"
        ... )
        >>> # Trace events will include run_id and example_id
        >>> for event in result.trace_events[:1]:
        ...     print(f"Run ID: {event.get('run_id')}")
        Run ID: experiment_001

    See Also
    --------
    Probe : Base class for all probes
    ToolDefinition : Tool definition dataclass
    AgentProbeResult : Result dataclass for agent probes
    insideLLMs.tracing.TraceRecorder : Trace event recorder
    insideLLMs.trace_config.TraceConfig : Trace configuration

    Notes
    -----
    **Thread Safety**: AgentProbe instances are thread-safe for concurrent
    ``run()`` calls. Each call creates its own ``TraceRecorder`` with isolated
    state. However, the tools list and configuration are shared and should not
    be modified after construction.

    **Contract Validation**: When ``on_violation.mode`` is set to ``fail_probe``
    or ``fail_run``, the probe will raise ``ValueError`` if any contract
    violations are detected. In ``record`` mode (default), violations are
    captured in the result but do not cause failures.

    **Fingerprinting**: The trace fingerprint is computed using SHA-256 on
    canonicalized JSON of trace events. This excludes timestamps to ensure
    determinism across runs with the same logical behavior.
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

        Creates a new agent probe instance with the specified configuration,
        tools, and tracing settings. The probe is ready for use immediately
        after construction.

        Parameters
        ----------
        name : str
            Human-readable name for this probe. This should be unique within
            a probe suite and is used for identification in reports, logs,
            and result aggregation. Use descriptive names like "search_agent"
            or "multi_tool_reasoning".
        tools : list[ToolDefinition], optional
            List of tool definitions available to the agent during execution.
            Tools define what external capabilities the agent can invoke.
            Default is an empty list (no tools available).
        trace_config : dict or TraceConfig, optional
            Configuration for tracing and contract validation. Accepts either:

            - ``dict``: Configuration dictionary (parsed via ``load_trace_config``)
            - ``TraceConfig``: Pre-configured TraceConfig object

            Default creates a new ``TraceConfig()`` with default settings.
            See ``TraceConfig`` documentation for available options.
        category : ProbeCategory, optional
            The evaluation category this probe belongs to. If not provided,
            uses the class-level ``default_category`` (``ProbeCategory.REASONING``).
            Categories help organize probes and enable category-based filtering.
        description : str, optional
            Brief description of what this probe tests. If not provided,
            defaults to the class docstring. This is included in probe
            metadata and reports.

        Raises
        ------
        TypeError
            If trace_config is neither a dict, TraceConfig, nor None.

        Examples
        --------
        Basic initialization with tools:

            >>> from insideLLMs.probes.agent_probe import AgentProbe, ToolDefinition
            >>>
            >>> search_tool = ToolDefinition(
            ...     name="search",
            ...     description="Search the web",
            ...     parameters={"type": "object", "properties": {"query": {"type": "string"}}}
            ... )
            >>>
            >>> class MyAgentProbe(AgentProbe):
            ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
            ...         recorder.record_generate_start(prompt)
            ...         response = model.generate(prompt)
            ...         recorder.record_generate_end(response)
            ...         return response
            ...
            >>> probe = MyAgentProbe(
            ...     name="web_search_agent",
            ...     tools=[search_tool],
            ...     description="Tests web search agent behavior"
            ... )
            >>> print(probe.name)
            web_search_agent
            >>> print(len(probe.tools))
            1

        Initialization with trace configuration dict:

            >>> probe = MyAgentProbe(
            ...     name="validated_agent",
            ...     tools=[search_tool],
            ...     trace_config={
            ...         "enabled": True,
            ...         "contracts": {
            ...             "enabled": True,
            ...             "tool_results": {"enabled": True}
            ...         },
            ...         "on_violation": {"mode": "fail_probe"}
            ...     }
            ... )
            >>> print(probe.trace_config.enabled)
            True

        Initialization with TraceConfig object:

            >>> from insideLLMs.trace_config import TraceConfig, OnViolationConfig
            >>>
            >>> config = TraceConfig(
            ...     enabled=True,
            ...     on_violation=OnViolationConfig(mode="record")
            ... )
            >>> probe = MyAgentProbe(
            ...     name="custom_config_agent",
            ...     tools=[],
            ...     trace_config=config
            ... )

        Initialization with explicit category:

            >>> from insideLLMs.types import ProbeCategory
            >>>
            >>> probe = MyAgentProbe(
            ...     name="safety_agent",
            ...     tools=[],
            ...     category=ProbeCategory.SAFETY,
            ...     description="Tests agent safety guardrails"
            ... )
            >>> print(probe.category)
            <ProbeCategory.SAFETY: 'safety'>

        See Also
        --------
        AgentProbe : Full class documentation with usage examples
        TraceConfig : Configuration options for tracing
        load_trace_config : Function to parse configuration dicts
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
        """Get the trace configuration.

        Returns the immutable trace configuration for this probe. The configuration
        controls tracing behavior, contract validation, payload normalization,
        and violation handling.

        Returns
        -------
        TraceConfig
            The trace configuration object for this probe.

        Examples
        --------
        Accessing trace configuration properties:

            >>> probe = MyAgentProbe(name="test", tools=[])
            >>> config = probe.trace_config
            >>> print(f"Tracing enabled: {config.enabled}")
            Tracing enabled: True
            >>> print(f"Contracts enabled: {config.contracts.enabled}")
            Contracts enabled: False

        Checking violation mode:

            >>> from insideLLMs.trace_config import OnViolationMode
            >>>
            >>> probe = MyAgentProbe(
            ...     name="test",
            ...     trace_config={"on_violation": {"mode": "fail_probe"}}
            ... )
            >>> if probe.trace_config.on_violation.mode == OnViolationMode.FAIL_PROBE:
            ...     print("Probe will fail on violations")
            Probe will fail on violations

        See Also
        --------
        TraceConfig : Full configuration class documentation
        """
        return self._trace_config

    @abstractmethod
    def run_agent(
        self,
        model: Any,
        prompt: str,
        tools: list[ToolDefinition],
        recorder: TraceRecorder,
        **kwargs: Any,
    ) -> str:
        """Run the agent and record trace events.

        This is the core abstract method that subclasses must implement. It defines
        how the agent interacts with the model, uses tools, and records trace events.
        The implementation should capture the complete execution flow for analysis
        and contract validation.

        Subclasses must implement this method to:

        1. Execute the agent with the given model and tools
        2. Record relevant trace events using the recorder
        3. Handle tool calls and record their results
        4. Return the final response

        Parameters
        ----------
        model : Any
            The model to test. Should implement a method for generating responses,
            typically with tool/function calling support. The exact interface
            depends on the model implementation (e.g., OpenAI client, Anthropic
            client, or custom wrapper).
        prompt : str
            The input prompt for the agent. This is the task or question the
            agent should complete using available tools.
        tools : list[ToolDefinition]
            List of available tools the agent can invoke. This is the same as
            ``self.tools`` but passed explicitly for clarity and potential
            per-run customization.
        recorder : TraceRecorder
            Thread-safe recorder for capturing trace events. Use recorder methods
            like ``record_generate_start()``, ``record_tool_call()``, etc. to
            capture the execution flow. All events are automatically sequenced.
        **kwargs : Any
            Additional arguments that may be passed from ``run()``. Common kwargs
            include model parameters (temperature, max_tokens) or custom context.

        Returns
        -------
        str
            The agent's final response after completing all tool interactions
            and reasoning. This becomes ``AgentProbeResult.final_response``.

        Raises
        ------
        NotImplementedError
            Always raised if a subclass does not override this method.

        Examples
        --------
        Minimal implementation without tool handling:

            >>> class BasicAgentProbe(AgentProbe):
            ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
            ...         recorder.record_generate_start(prompt)
            ...         response = model.generate(prompt)
            ...         recorder.record_generate_end(response)
            ...         return response

        Implementation with tool execution:

            >>> class ToolAgentProbe(AgentProbe):
            ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
            ...         recorder.record_generate_start(prompt)
            ...
            ...         # Get model response with potential tool calls
            ...         response = model.generate_with_tools(
            ...             prompt,
            ...             tools=[{"name": t.name, "description": t.description} for t in tools]
            ...         )
            ...
            ...         # Handle tool calls
            ...         while response.has_tool_calls:
            ...             for call in response.tool_calls:
            ...                 recorder.record_tool_call(call.name, call.arguments)
            ...
            ...                 # Execute the tool
            ...                 tool = self.get_tool_by_name(call.name)
            ...                 if tool and tool.handler:
            ...                     try:
            ...                         result = tool.handler(**call.arguments)
            ...                         recorder.record_tool_result(call.name, result)
            ...                     except Exception as e:
            ...                         recorder.record_tool_result(
            ...                             call.name, None, error=str(e)
            ...                         )
            ...                 else:
            ...                     recorder.record_tool_result(
            ...                         call.name, None, error="Tool not found"
            ...                     )
            ...
            ...             # Continue conversation with tool results
            ...             response = model.continue_with_results(response.tool_results)
            ...
            ...         recorder.record_generate_end(response.content)
            ...         return response.content

        Implementation with streaming:

            >>> class StreamingAgentProbe(AgentProbe):
            ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
            ...         recorder.record_stream_start(prompt)
            ...
            ...         chunks = []
            ...         for i, chunk in enumerate(model.stream(prompt)):
            ...             recorder.record_stream_chunk(chunk, i)
            ...             chunks.append(chunk)
            ...
            ...         full_response = "".join(chunks)
            ...         recorder.record_stream_end(full_response, chunk_count=len(chunks))
            ...         return full_response

        Implementation with error handling:

            >>> class RobustAgentProbe(AgentProbe):
            ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
            ...         recorder.record_generate_start(prompt)
            ...
            ...         try:
            ...             response = model.generate_with_tools(prompt, tools)
            ...
            ...             for call in response.tool_calls or []:
            ...                 recorder.record_tool_call(call.name, call.arguments)
            ...                 try:
            ...                     result = self._execute_tool(call)
            ...                     recorder.record_tool_result(call.name, result)
            ...                 except TimeoutError as e:
            ...                     recorder.record_error(str(e), "TimeoutError")
            ...                     recorder.record_tool_result(call.name, None, error="Timeout")
            ...                 except Exception as e:
            ...                     recorder.record_error(str(e), type(e).__name__)
            ...                     recorder.record_tool_result(call.name, None, error=str(e))
            ...
            ...             recorder.record_generate_end(response.content)
            ...             return response.content
            ...
            ...         except Exception as e:
            ...             recorder.record_error(str(e), type(e).__name__)
            ...             raise

        See Also
        --------
        run : Public method that calls run_agent and handles result packaging
        TraceRecorder : Full documentation of available recording methods
        TraceRecorder.record_tool_call : Record tool invocations
        TraceRecorder.record_tool_result : Record tool results

        Notes
        -----
        - Always call ``recorder.record_generate_start()`` at the beginning
          and ``recorder.record_generate_end()`` at the end for proper trace
          structure
        - For every ``record_tool_call()``, there should be a corresponding
          ``record_tool_result()`` to satisfy the tool_results contract
        - The recorder is thread-safe, but typically a single execution thread
          is used per run
        - Trace events are ordered by sequence number, not wall-clock time
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

        Executes the agent probe by creating a ``TraceRecorder``, invoking the
        subclass's ``run_agent()`` implementation, validating contracts, and
        packaging the complete result. This is the primary public interface
        for running agent probes.

        The execution flow:

        1. Extract prompt from input data
        2. Create ``TraceRecorder`` with optional run_id and example_id
        3. Call ``run_agent()`` to execute the agent
        4. Optionally redact sensitive data from trace payloads
        5. Compute trace fingerprint
        6. Validate contracts and collect violations
        7. Extract tool calls from trace events
        8. Package and return ``AgentProbeResult``
        9. Optionally raise ``ValueError`` if violations found and configured

        Parameters
        ----------
        model : Any
            The model to test. This is passed directly to ``run_agent()``.
            Should implement whatever interface the subclass expects.
        data : Any
            The input data for the probe. Accepts multiple formats:

            - ``str``: Used directly as the prompt
            - ``dict``: Extracts ``data["prompt"]`` as the prompt
            - Other: Converted to string via ``str(data)``
        run_id : str, optional
            Optional run identifier for trace correlation. Included in all
            trace events and useful for linking events across systems or
            matching runs in logs. Default is None.
        example_id : str, optional
            Optional example identifier for trace correlation. Useful for
            linking results to specific test cases or dataset entries.
            Default is None.
        **kwargs : Any
            Additional arguments passed through to ``run_agent()``. Common
            kwargs include:

            - ``temperature``: Model sampling temperature
            - ``max_tokens``: Maximum response length
            - ``metadata``: Dict stored in result.metadata

        Returns
        -------
        AgentProbeResult
            Complete result containing:

            - ``prompt``: The extracted input prompt
            - ``final_response``: Agent's final response
            - ``tool_calls``: List of tool invocations with arguments
            - ``trace_events``: Full execution trace
            - ``trace_fingerprint``: Deterministic hash for drift detection
            - ``violations``: Contract violations (empty if none)
            - ``metadata``: Custom metadata from kwargs

        Raises
        ------
        ValueError
            If contract violations are detected and ``on_violation.mode``
            is set to ``fail_probe`` or ``fail_run``. The error message
            includes details of all violations.

        Examples
        --------
        Basic execution with string prompt:

            >>> result = probe.run(model, "Search for Python tutorials")
            >>> print(f"Response: {result.final_response}")
            Response: Here are some Python tutorials...
            >>> print(f"Tool calls: {len(result.tool_calls)}")
            Tool calls: 2

        Execution with dict input:

            >>> result = probe.run(
            ...     model,
            ...     {"prompt": "Find weather in Paris", "context": "user query"}
            ... )
            >>> print(result.prompt)
            Find weather in Paris

        Execution with run and example IDs:

            >>> result = probe.run(
            ...     model=model,
            ...     data="Complex multi-step query",
            ...     run_id="experiment_2024_01",
            ...     example_id="test_case_42"
            ... )
            >>> # Check that IDs are in trace events
            >>> result.trace_events[0].get("run_id")
            'experiment_2024_01'

        Execution with custom metadata:

            >>> result = probe.run(
            ...     model,
            ...     "Test prompt",
            ...     metadata={
            ...         "user": "test_user",
            ...         "experiment": "v2",
            ...         "timestamp": "2024-01-01T00:00:00Z"
            ...     }
            ... )
            >>> print(result.metadata["experiment"])
            v2

        Handling violations in record mode (default):

            >>> result = probe.run(model, "Invalid workflow")
            >>> if result.violations:
            ...     print(f"Found {len(result.violations)} violations:")
            ...     for v in result.violations:
            ...         print(f"  - {v['code']}: {v['detail']}")
            Found 1 violations:
              - TOOL_ORDER_VIOLATION: summarize called before search

        Handling violations in fail_probe mode:

            >>> probe = MyAgentProbe(
            ...     name="strict_probe",
            ...     trace_config={"on_violation": {"mode": "fail_probe"}}
            ... )
            >>> try:
            ...     result = probe.run(model, "Invalid workflow")
            ... except ValueError as e:
            ...     print(f"Probe failed: {e}")
            Probe failed: Trace contract violations: ['summarize called before search']

        Using fingerprints for regression testing:

            >>> baseline_result = probe.run(model, "Standard query")
            >>> baseline_fingerprint = baseline_result.trace_fingerprint
            >>>
            >>> # Later, after changes...
            >>> new_result = probe.run(model, "Standard query")
            >>> if new_result.trace_fingerprint != baseline_fingerprint:
            ...     print("Behavior changed! Investigate differences.")

        Inspecting extracted tool calls:

            >>> result = probe.run(model, "Search and summarize")
            >>> for tc in result.tool_calls:
            ...     print(f"Tool: {tc['tool_name']}")
            ...     print(f"  Args: {tc['arguments']}")
            ...     print(f"  Seq: {tc['seq']}")
            Tool: search
              Args: {'query': 'Python tutorials'}
              Seq: 1
            Tool: summarize
              Args: {'text': '...'}
              Seq: 3

        See Also
        --------
        run_agent : Abstract method that subclasses implement
        run_batch : Run probe on multiple inputs
        AgentProbeResult : Result dataclass structure
        TraceRecorder : Trace event recorder

        Notes
        -----
        - The ``run_id`` and ``example_id`` are purely for correlation and do
          not affect execution logic
        - Payload redaction (if configured) is applied after trace recording
        - The fingerprint is computed on the full trace, not the redacted version
        - Tool calls are extracted from ``TOOL_CALL_START`` events only
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
        if violations and self._trace_config.on_violation.mode in {
            OnViolationMode.FAIL_PROBE,
            OnViolationMode.FAIL_RUN,
        }:
            raise ValueError(f"Trace contract violations: {[v.detail for v in violations]}")

        return result

    def score(self, results: list[ProbeResult[AgentProbeResult]]) -> ProbeScore:
        """Calculate aggregate scores from agent probe results.

        Computes standard metrics (accuracy, error rate, mean latency) from the
        parent class and adds an agent-specific ``violation_rate`` metric that
        tracks the proportion of successful runs with contract violations.

        This method is typically called after ``run_batch()`` to analyze overall
        probe performance and agent behavior compliance.

        Parameters
        ----------
        results : list[ProbeResult[AgentProbeResult]]
            The list of probe results to score. Each result should contain
            an ``AgentProbeResult`` in its ``output`` field. Results with
            non-SUCCESS status are excluded from violation rate calculation.

        Returns
        -------
        ProbeScore
            Aggregate score containing:

            - ``accuracy``: Proportion of successful results (SUCCESS / total)
            - ``error_rate``: Proportion of error results (ERROR / total)
            - ``mean_latency_ms``: Average latency of successful results
            - ``custom_metrics["violation_rate"]``: Proportion of successful
              results that had contract violations

        Examples
        --------
        Basic scoring after batch execution:

            >>> results = probe.run_batch(model, prompts)
            >>> score = probe.score(results)
            >>> print(f"Success rate: {score.accuracy:.1%}")
            Success rate: 95.0%
            >>> print(f"Mean latency: {score.mean_latency_ms:.0f}ms")
            Mean latency: 250ms

        Analyzing violation rates:

            >>> score = probe.score(results)
            >>> violation_rate = score.custom_metrics.get("violation_rate", 0.0)
            >>> print(f"Violation rate: {violation_rate:.1%}")
            Violation rate: 5.0%
            >>>
            >>> if violation_rate > 0.1:
            ...     print("WARNING: High violation rate detected!")

        Comparing scores across probes:

            >>> scores = {}
            >>> for probe in probes:
            ...     results = probe.run_batch(model, test_data)
            ...     scores[probe.name] = probe.score(results)
            ...
            >>> # Find probe with lowest violation rate
            >>> best = min(
            ...     scores.items(),
            ...     key=lambda x: x[1].custom_metrics.get("violation_rate", 0)
            ... )
            >>> print(f"Best compliance: {best[0]}")

        Handling empty results:

            >>> score = probe.score([])
            >>> print(score.accuracy)
            None
            >>> print(score.custom_metrics)
            {}

        Filtering results before scoring:

            >>> # Score only successful results
            >>> successful = [r for r in results if r.status == ResultStatus.SUCCESS]
            >>> score = probe.score(successful)
            >>>
            >>> # Score results from a specific time window
            >>> recent = [r for r in results if r.latency_ms and r.latency_ms < 1000]
            >>> fast_score = probe.score(recent)

        See Also
        --------
        Probe.score : Parent implementation with base metrics
        ProbeScore : Score dataclass structure
        run_batch : Method to generate results for scoring

        Notes
        -----
        - The ``violation_rate`` is only computed when there are successful
          results with valid output
        - Violations in ERROR, TIMEOUT, or RATE_LIMITED results are not counted
        - The metric is added to ``custom_metrics`` and can be accessed via
          ``score.custom_metrics["violation_rate"]``
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

        Searches the probe's tool list for a tool with the specified name and
        returns the matching ``ToolDefinition`` if found. This is commonly used
        in ``run_agent()`` implementations to look up tool handlers for execution.

        The search is case-sensitive and matches exact tool names only.

        Parameters
        ----------
        name : str
            The tool name to find. Must match exactly (case-sensitive).

        Returns
        -------
        ToolDefinition or None
            The ``ToolDefinition`` with the matching name, or ``None`` if no
            tool with that name exists in the probe's tool list.

        Examples
        --------
        Looking up a tool by name:

            >>> from insideLLMs.probes.agent_probe import AgentProbe, ToolDefinition
            >>>
            >>> search_tool = ToolDefinition(
            ...     name="search",
            ...     description="Search the web",
            ...     parameters={"query": {"type": "string"}}
            ... )
            >>> probe = MyAgentProbe(name="test", tools=[search_tool])
            >>>
            >>> tool = probe.get_tool_by_name("search")
            >>> print(tool.description)
            Search the web

        Handling missing tools:

            >>> tool = probe.get_tool_by_name("nonexistent")
            >>> print(tool)
            None
            >>>
            >>> if tool is None:
            ...     print("Tool not found!")
            Tool not found!

        Using in run_agent implementation:

            >>> class MyAgentProbe(AgentProbe):
            ...     def run_agent(self, model, prompt, tools, recorder, **kwargs):
            ...         recorder.record_generate_start(prompt)
            ...         response = model.generate_with_tools(prompt)
            ...
            ...         for call in response.tool_calls:
            ...             tool = self.get_tool_by_name(call.name)
            ...             if tool and tool.handler:
            ...                 recorder.record_tool_call(call.name, call.arguments)
            ...                 result = tool.handler(**call.arguments)
            ...                 recorder.record_tool_result(call.name, result)
            ...             else:
            ...                 recorder.record_error(
            ...                     f"Unknown tool: {call.name}",
            ...                     "ToolNotFoundError"
            ...                 )
            ...
            ...         recorder.record_generate_end(response.content)
            ...         return response.content

        Executing a tool handler:

            >>> def calc_handler(expr: str) -> dict:
            ...     return {"result": eval(expr)}
            ...
            >>> calc_tool = ToolDefinition("calc", "Calculator", {}, calc_handler)
            >>> probe = MyAgentProbe(name="test", tools=[calc_tool])
            >>>
            >>> tool = probe.get_tool_by_name("calc")
            >>> if tool and tool.handler:
            ...     result = tool.handler(expr="2 + 2")
            ...     print(result)
            {'result': 4}

        Case sensitivity:

            >>> probe = MyAgentProbe(name="test", tools=[search_tool])
            >>> probe.get_tool_by_name("Search")  # Wrong case
            None
            >>> probe.get_tool_by_name("search")  # Correct case
            ToolDefinition(name='search', ...)

        See Also
        --------
        ToolDefinition : Tool definition dataclass
        tools : The list of tools searched by this method

        Notes
        -----
        - This method performs a linear search through the tools list, which
          is efficient for typical probe configurations with a small number
          of tools
        - For probes with many tools, consider caching in a dict by name
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def info(self) -> dict[str, Any]:
        """Return probe metadata including tool info.

        Extends the parent ``info()`` method to include agent-specific metadata
        such as available tools and tracing configuration. This provides a
        complete snapshot of the probe's configuration for logging, reporting,
        and serialization.

        Returns
        -------
        dict[str, Any]
            A dictionary containing probe metadata:

            - ``name``: Probe name (from parent)
            - ``category``: Probe category value (from parent)
            - ``description``: Probe description (from parent)
            - ``type``: Class name (from parent)
            - ``tools``: List of tool info dicts with name and description
            - ``trace_enabled``: Whether tracing is enabled
            - ``contracts_enabled``: Whether contract validation is enabled

        Examples
        --------
        Getting probe information:

            >>> probe = MyAgentProbe(
            ...     name="search_agent",
            ...     tools=[search_tool, summarize_tool],
            ...     description="Tests search and summarize workflow"
            ... )
            >>> info = probe.info()
            >>> print(info["name"])
            search_agent
            >>> print(info["tools"])
            [{'name': 'search', 'description': 'Search the web'},
             {'name': 'summarize', 'description': 'Summarize text'}]

        Checking configuration status:

            >>> probe = MyAgentProbe(
            ...     name="test",
            ...     trace_config={
            ...         "enabled": True,
            ...         "contracts": {"enabled": True}
            ...     }
            ... )
            >>> info = probe.info()
            >>> print(f"Tracing: {info['trace_enabled']}")
            Tracing: True
            >>> print(f"Contracts: {info['contracts_enabled']}")
            Contracts: True

        Serializing to JSON:

            >>> import json
            >>> info = probe.info()
            >>> json_str = json.dumps(info, indent=2)
            >>> print(json_str)
            {
              "name": "search_agent",
              "category": "reasoning",
              "description": "Tests search and summarize workflow",
              "type": "MyAgentProbe",
              "tools": [...],
              "trace_enabled": true,
              "contracts_enabled": false
            }

        Logging probe information:

            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>>
            >>> for probe in probe_suite:
            ...     info = probe.info()
            ...     logger.info(
            ...         f"Probe: {info['name']} | "
            ...         f"Tools: {len(info['tools'])} | "
            ...         f"Contracts: {info['contracts_enabled']}"
            ...     )

        Building a probe registry:

            >>> registry = {}
            >>> for probe in probes:
            ...     info = probe.info()
            ...     registry[info["name"]] = {
            ...         "tools": [t["name"] for t in info["tools"]],
            ...         "category": info["category"],
            ...         "trace_enabled": info["trace_enabled"]
            ...     }

        See Also
        --------
        Probe.info : Parent implementation with base metadata
        trace_config : Access full trace configuration
        tools : Access full tool definitions
        """
        base_info = super().info()
        base_info["tools"] = [{"name": t.name, "description": t.description} for t in self.tools]
        base_info["trace_enabled"] = self._trace_config.enabled
        base_info["contracts_enabled"] = self._trace_config.contracts.enabled
        return base_info
