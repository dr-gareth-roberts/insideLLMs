"""Agent configuration and result data models."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AgentStatus(Enum):
    """Status of agent execution lifecycle.

    This enum represents the various states an agent can be in during its
    execution lifecycle. Agents transition through these states as they
    process queries, invoke tools, and generate responses.

    Attributes:
        IDLE: Agent is initialized but not currently processing a query.
        THINKING: Agent is reasoning about what to do next.
        ACTING: Agent is executing a tool action.
        OBSERVING: Agent is processing the result of an action.
        FINISHED: Agent has completed successfully with a final answer.
        ERROR: Agent encountered an unrecoverable error.
        MAX_ITERATIONS: Agent stopped due to reaching iteration limit.

    Examples:
        Checking agent result status:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentStatus
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> result = agent.run("Hello")
            >>> if result.status == AgentStatus.FINISHED:
            ...     print(f"Success: {result.answer}")
            ... elif result.status == AgentStatus.ERROR:
            ...     print("Agent encountered an error")

        Using status in conditional logic:

            >>> from insideLLMs.contrib.agents import AgentStatus
            >>>
            >>> def handle_result(result):
            ...     if result.status == AgentStatus.FINISHED:
            ...         return result.answer
            ...     elif result.status == AgentStatus.MAX_ITERATIONS:
            ...         return "Agent timed out - partial result: " + str(result.steps[-1])
            ...     else:
            ...         return f"Unexpected status: {result.status.value}"

        Comparing status values:

            >>> from insideLLMs.contrib.agents import AgentStatus
            >>>
            >>> status = AgentStatus.THINKING
            >>> print(status.value)
            'thinking'
            >>> print(status == AgentStatus.THINKING)
            True

        Iterating over all statuses:

            >>> from insideLLMs.contrib.agents import AgentStatus
            >>>
            >>> for status in AgentStatus:
            ...     print(f"{status.name}: {status.value}")
            IDLE: idle
            THINKING: thinking
            ...
    """

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AgentConfig:
    """Configuration for agent behavior and execution parameters.

    This dataclass holds all configurable parameters that control how an agent
    behaves during execution. It covers execution limits, reasoning format,
    memory management, and error handling strategies.

    Attributes:
        max_iterations: Maximum number of reasoning/action cycles before stopping.
            Prevents infinite loops. Default is 10.
        max_execution_time: Maximum wall-clock time in seconds for agent execution.
            Agent will stop if this limit is exceeded. Default is 300 (5 minutes).
        early_stop_on_finish: If True, agent stops immediately when a final answer
            is found. If False, continues until max_iterations. Default is True.
        verbose: If True, enables detailed logging of agent reasoning steps.
            Useful for debugging. Default is False.
        include_scratchpad: If True, includes previous reasoning steps in the prompt
            to maintain context. Default is True.
        thought_prefix: String prefix used to identify thought lines in agent output.
            Default is "Thought: ".
        action_prefix: String prefix used to identify action lines. Default is "Action: ".
        observation_prefix: String prefix for tool observation results.
            Default is "Observation: ".
        final_answer_prefix: String prefix indicating the final answer.
            Default is "Final Answer: ".
        memory_limit: Maximum number of steps to retain in agent memory.
            Older steps are evicted when limit is reached. Default is 20.
        retry_on_error: If True, agent will retry after tool execution errors.
            Default is True.
        max_retries: Maximum number of retry attempts for failed tool executions.
            Default is 2.

    Examples:
        Creating a default configuration:

            >>> from insideLLMs.contrib.agents import AgentConfig
            >>>
            >>> config = AgentConfig()
            >>> print(config.max_iterations)
            10
            >>> print(config.verbose)
            False

        Creating a custom configuration for debugging:

            >>> from insideLLMs.contrib.agents import AgentConfig, ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> debug_config = AgentConfig(
            ...     max_iterations=5,
            ...     verbose=True,
            ...     max_execution_time=60,
            ...     memory_limit=10
            ... )
            >>> agent = ReActAgent(DummyModel(), config=debug_config)

        Configuration for production with higher limits:

            >>> from insideLLMs.contrib.agents import AgentConfig
            >>>
            >>> prod_config = AgentConfig(
            ...     max_iterations=20,
            ...     max_execution_time=600,
            ...     retry_on_error=True,
            ...     max_retries=3,
            ...     verbose=False
            ... )

        Converting configuration to dictionary:

            >>> from insideLLMs.contrib.agents import AgentConfig
            >>>
            >>> config = AgentConfig(max_iterations=15, verbose=True)
            >>> config_dict = config.to_dict()
            >>> print(config_dict['max_iterations'])
            15
            >>> print(config_dict['verbose'])
            True

    See Also:
        :class:`ReActAgent`: Uses this configuration for execution behavior.
        :class:`AgentMemory`: Uses memory_limit from this configuration.
    """

    # Execution settings
    max_iterations: int = 10
    max_execution_time: int = 300  # seconds
    early_stop_on_finish: bool = True

    # Reasoning settings
    verbose: bool = False
    include_scratchpad: bool = True
    # NOTE: these prefixes only affect scratchpad *rendering* of prior steps.
    # The model-facing prompt template and the response parser use fixed
    # "Thought:"/"Action:"/"Action Input:"/"Final Answer:" literals, so changing
    # these does not change what the model is asked to emit or how it is parsed.
    thought_prefix: str = "Thought: "
    action_prefix: str = "Action: "
    observation_prefix: str = "Observation: "
    final_answer_prefix: str = "Final Answer: "

    # Memory settings
    memory_limit: int = 20  # Max steps to keep in memory

    # Error handling
    retry_on_error: bool = True
    max_retries: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary representation.

        Returns a dictionary containing the key configuration parameters.
        Useful for serialization, logging, or passing configuration to
        external systems.

        Returns:
            dict[str, Any]: Dictionary with configuration key-value pairs.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentConfig
            >>>
            >>> config = AgentConfig(max_iterations=5, verbose=True)
            >>> d = config.to_dict()
            >>> print(d)
            {'max_iterations': 5, 'max_execution_time': 300, ...}
        """
        return {
            "max_iterations": self.max_iterations,
            "max_execution_time": self.max_execution_time,
            "early_stop_on_finish": self.early_stop_on_finish,
            "verbose": self.verbose,
            "memory_limit": self.memory_limit,
            "max_retries": self.max_retries,
        }


@dataclass
class ToolParameter:
    """Definition of a single parameter for a Tool.

    This dataclass describes a parameter that a tool accepts. It includes
    the parameter name, type information, description, and whether it's
    required or optional with a default value.

    Attributes:
        name: The parameter name as it appears in the function signature.
        type: String representation of the parameter type (e.g., "str", "int", "list").
        description: Human-readable description of what this parameter does.
        required: If True, the parameter must be provided. If False, it's optional.
            Default is True.
        default: The default value to use if the parameter is not provided.
            Only meaningful when required=False. Default is None.

    Examples:
        Creating a required string parameter:

            >>> from insideLLMs.contrib.agents import ToolParameter
            >>>
            >>> query_param = ToolParameter(
            ...     name="query",
            ...     type="str",
            ...     description="The search query to execute"
            ... )
            >>> print(query_param.required)
            True

        Creating an optional integer parameter with default:

            >>> from insideLLMs.contrib.agents import ToolParameter
            >>>
            >>> limit_param = ToolParameter(
            ...     name="limit",
            ...     type="int",
            ...     description="Maximum number of results to return",
            ...     required=False,
            ...     default=10
            ... )
            >>> print(limit_param.default)
            10

        Using parameters to define a Tool:

            >>> from insideLLMs.contrib.agents import Tool, ToolParameter
            >>>
            >>> def search(query: str, limit: int = 10) -> str:
            ...     return f"Searching for {query} with limit {limit}"
            >>>
            >>> params = [
            ...     ToolParameter("query", "str", "Search query", required=True),
            ...     ToolParameter("limit", "int", "Max results", required=False, default=10)
            ... ]
            >>> tool = Tool("search", search, "Search for information", params)

        Accessing parameter properties:

            >>> from insideLLMs.contrib.agents import ToolParameter
            >>>
            >>> param = ToolParameter("name", "str", "User's name")
            >>> print(f"{param.name}: {param.type} - {param.description}")
            name: str - User's name

    See Also:
        :class:`Tool`: Uses ToolParameter to define its interface.
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """Result from executing a tool.

    This dataclass encapsulates the complete result of a tool execution,
    including the output (on success), error information (on failure),
    timing metrics, and optional metadata.

    Attributes:
        tool_name: Name of the tool that was executed.
        input: The input that was passed to the tool.
        output: The output returned by the tool (None if execution failed).
        success: True if tool executed successfully, False otherwise.
        error: Error message if execution failed, None otherwise.
        execution_time_ms: Time taken to execute the tool in milliseconds.
        metadata: Additional metadata about the execution (e.g., API call info).

    Examples:
        Creating a successful result:

            >>> from insideLLMs.contrib.agents import ToolResult
            >>>
            >>> result = ToolResult(
            ...     tool_name="calculator",
            ...     input="2 + 2",
            ...     output="4",
            ...     success=True,
            ...     execution_time_ms=1.5
            ... )
            >>> print(result.output)
            4

        Creating a failed result:

            >>> from insideLLMs.contrib.agents import ToolResult
            >>>
            >>> result = ToolResult(
            ...     tool_name="calculator",
            ...     input="2 / 0",
            ...     output=None,
            ...     success=False,
            ...     error="ZeroDivisionError: division by zero",
            ...     execution_time_ms=0.5
            ... )
            >>> if not result.success:
            ...     print(f"Error: {result.error}")
            Error: ZeroDivisionError: division by zero

        Converting result to dictionary for logging:

            >>> from insideLLMs.contrib.agents import ToolResult
            >>>
            >>> result = ToolResult(
            ...     tool_name="search",
            ...     input="python tutorials",
            ...     output=["result1", "result2"],
            ...     success=True,
            ...     execution_time_ms=150.3,
            ...     metadata={"source": "web", "count": 2}
            ... )
            >>> d = result.to_dict()
            >>> print(d["metadata"]["source"])
            web

        Using with tool execution:

            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> tool = Tool("add", add, "Add two numbers")
            >>> result = tool.execute({"a": 5, "b": 3})
            >>> print(result.success, result.output)
            True 8

    See Also:
        :class:`Tool`: Produces ToolResult objects via its execute method.
        :class:`AgentStep`: Contains ToolResult as part of execution trace.
    """

    tool_name: str
    input: Any
    output: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all result fields.

        Examples:
            >>> from insideLLMs.contrib.agents import ToolResult
            >>>
            >>> result = ToolResult("calc", "1+1", "2", True)
            >>> d = result.to_dict()
            >>> print(d["tool_name"], d["output"])
            calc 2
        """
        return {
            "tool_name": self.tool_name,
            "input": self.input,
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class AgentStep:
    """A single step in agent execution representing one iteration of the agent loop.

    This dataclass captures all information about a single step in the agent's
    reasoning and action cycle. Each step includes the agent's thought process,
    any action taken, the tool input/output, and timing information.

    Attributes:
        step_number: The sequential number of this step (1-indexed).
        thought: The agent's reasoning/thinking for this step (may be None).
        action: The name of the tool/action to execute (may be None if final answer).
        action_input: The input provided to the tool (may be None).
        observation: The result/observation from executing the action (may be None).
        tool_result: The complete ToolResult object if a tool was executed.
        timestamp: When this step was created.

    Examples:
        Creating a thinking step:

            >>> from insideLLMs.contrib.agents import AgentStep
            >>>
            >>> step = AgentStep(
            ...     step_number=1,
            ...     thought="I need to calculate 2 + 2",
            ...     action="calculator",
            ...     action_input="2 + 2"
            ... )
            >>> print(step.thought)
            I need to calculate 2 + 2

        Creating a step with observation:

            >>> from insideLLMs.contrib.agents import AgentStep, ToolResult
            >>>
            >>> tool_result = ToolResult("calculator", "2 + 2", "4", True)
            >>> step = AgentStep(
            ...     step_number=1,
            ...     thought="Let me calculate this",
            ...     action="calculator",
            ...     action_input="2 + 2",
            ...     observation="4",
            ...     tool_result=tool_result
            ... )
            >>> print(step.observation)
            4

        Converting step to dictionary for logging:

            >>> from insideLLMs.contrib.agents import AgentStep
            >>>
            >>> step = AgentStep(
            ...     step_number=1,
            ...     thought="Final step",
            ...     action=None,
            ...     observation="Task complete"
            ... )
            >>> d = step.to_dict()
            >>> print(d["step_number"], d["thought"])
            1 Final step

        Iterating through agent steps:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 5 + 3?")
            >>> for step in result.steps:
            ...     print(f"Step {step.step_number}: {step.action or 'No action'}")

    See Also:
        :class:`AgentResult`: Contains a list of AgentStep objects.
        :class:`AgentMemory`: Stores AgentStep objects for context.
    """

    step_number: int
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Any] = None
    observation: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert the step to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all step fields, with
                timestamp as ISO format string and tool_result as dict.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentStep
            >>>
            >>> step = AgentStep(1, thought="Thinking...")
            >>> d = step.to_dict()
            >>> print(d["step_number"])
            1
        """
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentResult:
    """Complete result of agent execution.

    This dataclass contains the full result of running an agent, including
    the final answer, execution status, complete step trace, timing metrics,
    and token usage statistics.

    Attributes:
        query: The original query/task given to the agent.
        answer: The final answer produced by the agent (None if not completed).
        status: The final status of execution (FINISHED, ERROR, MAX_ITERATIONS, etc.).
        steps: List of AgentStep objects representing the execution trace.
        total_iterations: Number of iterations the agent performed.
        execution_time_ms: Total execution time in milliseconds.
        token_usage: Dictionary tracking token usage (e.g., prompt, completion tokens).
        metadata: Additional metadata about the execution.

    Examples:
        Accessing basic result properties:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentStatus
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> result = agent.run("What is 2 + 2?")
            >>> print(f"Answer: {result.answer}")
            >>> print(f"Status: {result.status}")
            >>> print(f"Iterations: {result.total_iterations}")

        Checking execution status:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentStatus
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> result = agent.run("Complex task")
            >>> if result.status == AgentStatus.FINISHED:
            ...     print(f"Success: {result.answer}")
            ... elif result.status == AgentStatus.MAX_ITERATIONS:
            ...     print("Agent reached iteration limit")
            ... elif result.status == AgentStatus.ERROR:
            ...     print("Agent encountered an error")

        Analyzing execution trace:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("Calculate 10 * 5")
            >>> for step in result.steps:
            ...     print(f"Step {step.step_number}:")
            ...     if step.thought:
            ...         print(f"  Thought: {step.thought[:50]}...")
            ...     if step.action:
            ...         print(f"  Action: {step.action}")

        Serializing result to JSON:

            >>> from insideLLMs.contrib.agents import ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> result = agent.run("Hello")
            >>> json_str = result.to_json(indent=2)
            >>> print(json_str[:100])
            {
              "query": "Hello",
              ...

    See Also:
        :class:`AgentStep`: Individual steps within the result.
        :class:`AgentStatus`: Possible status values.
    """

    query: str
    answer: Optional[str]
    status: AgentStatus
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    execution_time_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all result fields, with
                nested objects converted to dictionaries.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentResult, AgentStatus
            >>>
            >>> result = AgentResult("query", "answer", AgentStatus.FINISHED)
            >>> d = result.to_dict()
            >>> print(d["status"])
            finished
        """
        return {
            "query": self.query,
            "answer": self.answer,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "total_iterations": self.total_iterations,
            "execution_time_ms": self.execution_time_ms,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the result to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation. Default is 2.

        Returns:
            str: JSON string representation of the result.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentResult, AgentStatus
            >>>
            >>> result = AgentResult("Hello", "World", AgentStatus.FINISHED)
            >>> json_str = result.to_json()
            >>> print('"query": "Hello"' in json_str)
            True

            >>> # With no indentation
            >>> compact = result.to_json(indent=0)
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
