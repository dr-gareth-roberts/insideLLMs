"""Autonomous Agents Module for insideLLMs.

This module provides a comprehensive framework for building autonomous LLM agents
that can reason, use tools, and accomplish complex tasks. It implements several
agent paradigms including ReAct (Reasoning + Acting), Chain-of-Thought, and
simple tool-execution agents.

Key Components:
    - **Tool System**: Define and register tools that agents can use to interact
      with external systems, perform calculations, search, etc.
    - **ReAct Agent**: Implements the ReAct paradigm where the agent reasons about
      what to do, takes an action using a tool, observes the result, and repeats.
    - **Chain-of-Thought Agent**: Breaks down complex problems into reasoning steps.
    - **Simple Agent**: Direct tool execution without iterative reasoning.
    - **Agent Executor**: Provides hooks, logging, and batch execution capabilities.
    - **Memory Management**: Maintains execution history and context for agents.

Architecture Overview:
    The module follows a layered architecture:

    1. **Configuration Layer**: AgentConfig, AgentStatus for behavior control
    2. **Tool Layer**: Tool, ToolParameter, ToolResult, ToolRegistry for capabilities
    3. **Memory Layer**: AgentMemory for maintaining execution state
    4. **Agent Layer**: BaseAgent, ReActAgent, SimpleAgent, ChainOfThoughtAgent
    5. **Execution Layer**: AgentExecutor for running agents with advanced features

Examples:
    Basic ReAct Agent with Calculator Tool:

        >>> from insideLLMs.contrib.agents import Tool, ReActAgent, create_calculator_tool
        >>> from insideLLMs import DummyModel
        >>>
        >>> # Create a model and tools
        >>> model = DummyModel()
        >>> calculator = create_calculator_tool()
        >>>
        >>> # Create and run the agent
        >>> agent = ReActAgent(model, tools=[calculator])
        >>> result = agent.run("What is 25 * 4?")
        >>> print(result.answer)

    Using the @tool Decorator:

        >>> from insideLLMs.contrib.agents import tool, ReActAgent
        >>> from insideLLMs import DummyModel
        >>>
        >>> @tool(name="greet", description="Generate a greeting")
        ... def greet(name: str) -> str:
        ...     '''Greet someone by name.'''
        ...     return f"Hello, {name}!"
        >>>
        >>> model = DummyModel()
        >>> agent = ReActAgent(model, tools=[greet])
        >>> result = agent.run("Greet Alice")

    Chain-of-Thought Reasoning:

        >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
        >>> from insideLLMs import DummyModel
        >>>
        >>> model = DummyModel()
        >>> agent = ChainOfThoughtAgent(model)
        >>> result = agent.run("If I have 3 apples and buy 5 more, then eat 2, how many do I have?")
        >>> print(result.answer)

    Using AgentExecutor with Hooks:

        >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor, create_calculator_tool
        >>> from insideLLMs import DummyModel
        >>>
        >>> def log_start(query, kwargs):
        ...     print(f"Starting query: {query}")
        >>>
        >>> def log_end(result):
        ...     print(f"Finished with status: {result.status}")
        >>>
        >>> model = DummyModel()
        >>> agent = ReActAgent(model, tools=[create_calculator_tool()])
        >>> executor = AgentExecutor(agent, pre_hooks=[log_start], post_hooks=[log_end])
        >>> result = executor.run("Calculate 10 + 20")

See Also:
    - :class:`Tool`: For creating custom tools
    - :class:`ReActAgent`: For the main agent implementation
    - :class:`AgentExecutor`: For advanced execution features
    - :func:`create_react_agent`: Convenience function for creating agents
"""

import ast
import contextlib
import inspect
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

# =============================================================================
# Configuration and Types
# =============================================================================


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


# =============================================================================
# Tool Definition
# =============================================================================


class Tool:
    """A tool that an agent can use to perform actions.

    Tools are callable objects that wrap functions, providing a consistent
    interface for agents to execute actions. They automatically infer
    parameters from function signatures and generate descriptions from
    docstrings.

    Args:
        name: The name of the tool (used by agents to invoke it).
        func: The underlying function to execute when the tool is called.
        description: Human-readable description of what the tool does.
            If not provided, will be inferred from the function's docstring.
        parameters: List of ToolParameter definitions describing the function's
            arguments. If not provided, will be inferred from the signature.
        return_type: String describing the return type. Default is "str".

    Attributes:
        name: Tool name.
        func: The wrapped function.
        description: Tool description.
        parameters: List of ToolParameter objects.
        return_type: Return type string.

    Examples:
        Creating a tool from a function:

            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def greet(name: str) -> str:
            ...     '''Greet someone by name.'''
            ...     return f"Hello, {name}!"
            >>>
            >>> tool = Tool("greet", greet)
            >>> print(tool.name)
            greet
            >>> print(tool.description)
            Greet someone by name.

        Creating a tool with explicit parameters:

            >>> from insideLLMs.contrib.agents import Tool, ToolParameter
            >>>
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> params = [
            ...     ToolParameter("a", "int", "First number"),
            ...     ToolParameter("b", "int", "Second number")
            ... ]
            >>> tool = Tool("add", add, "Add two numbers", params)

        Executing a tool directly:

            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def multiply(x: int, y: int) -> int:
            ...     return x * y
            >>>
            >>> tool = Tool("multiply", multiply, "Multiply two numbers")
            >>> result = tool(3, 4)  # Direct call
            >>> print(result)
            12

        Using the execute method (preferred for agents):

            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def divide(a: float, b: float) -> float:
            ...     return a / b
            >>>
            >>> tool = Tool("divide", divide, "Divide two numbers")
            >>> result = tool.execute({"a": 10, "b": 2})
            >>> print(result.success, result.output)
            True 5.0
            >>>
            >>> # Handling errors
            >>> result = tool.execute({"a": 10, "b": 0})
            >>> print(result.success, result.error)
            False division by zero

    See Also:
        :func:`tool`: Decorator to create tools from functions.
        :class:`ToolRegistry`: Container for managing multiple tools.
        :class:`ToolResult`: Return type from the execute method.
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[list[ToolParameter]] = None,
        return_type: str = "str",
    ):
        self.name = name
        self.func = func
        self.description = description or self._infer_description(func)
        self.parameters = parameters or self._infer_parameters(func)
        self.return_type = return_type

    def _infer_description(self, func: Callable) -> str:
        """Infer tool description from the function's docstring.

        Args:
            func: The function to inspect.

        Returns:
            str: First line of docstring, or default description.

        Examples:
            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def my_func():
            ...     '''This is the description.
            ...     More details here.
            ...     '''
            ...     pass
            >>>
            >>> tool = Tool("test", my_func)
            >>> print(tool.description)
            This is the description.
        """
        doc = func.__doc__
        if doc:
            # Get first line
            return doc.strip().split("\n")[0]
        return f"Execute {func.__name__}"

    def _infer_parameters(self, func: Callable) -> list[ToolParameter]:
        """Infer tool parameters from the function's signature.

        Args:
            func: The function to inspect.

        Returns:
            list[ToolParameter]: List of inferred parameter definitions.

        Examples:
            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def search(query: str, limit: int = 10) -> list:
            ...     pass
            >>>
            >>> tool = Tool("search", search)
            >>> print(len(tool.parameters))
            2
            >>> print(tool.parameters[0].name, tool.parameters[0].required)
            query True
            >>> print(tool.parameters[1].name, tool.parameters[1].required)
            limit False
        """
        sig = inspect.signature(func)
        params = []

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            # Get type annotation
            param_type = "str"
            if param.annotation != inspect.Parameter.empty:
                param_type = (
                    param.annotation.__name__
                    if hasattr(param.annotation, "__name__")
                    else str(param.annotation)
                )

            # Check if required
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            params.append(
                ToolParameter(
                    name=name,
                    type=param_type,
                    description=f"Parameter {name}",
                    required=required,
                    default=default,
                )
            )

        return params

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool by calling the underlying function directly.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The return value of the underlying function.

        Examples:
            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> tool = Tool("add", add)
            >>> print(tool(2, 3))
            5
            >>> print(tool(a=5, b=10))
            15
        """
        return self.func(*args, **kwargs)

    def execute(self, input_data: Any) -> ToolResult:
        """Execute the tool with input data and return a structured result.

        This is the preferred method for agent-based execution as it provides
        error handling, timing, and a consistent return type.

        Args:
            input_data: Input to the tool. Can be:
                - A string (passed as single argument, or parsed as JSON)
                - A dict (unpacked as keyword arguments)
                - Any other type (passed directly to the function)

        Returns:
            ToolResult: Structured result containing output or error information.

        Examples:
            Executing with a dictionary:

                >>> from insideLLMs.contrib.agents import Tool
                >>>
                >>> def greet(name: str, greeting: str = "Hello") -> str:
                ...     return f"{greeting}, {name}!"
                >>>
                >>> tool = Tool("greet", greet)
                >>> result = tool.execute({"name": "Alice"})
                >>> print(result.output)
                Hello, Alice!

            Executing with a JSON string:

                >>> from insideLLMs.contrib.agents import Tool
                >>>
                >>> tool = Tool("greet", greet)
                >>> result = tool.execute('{"name": "Bob", "greeting": "Hi"}')
                >>> print(result.output)
                Hi, Bob!

            Executing with a plain string:

                >>> from insideLLMs.contrib.agents import Tool
                >>>
                >>> def echo(text: str) -> str:
                ...     return text
                >>>
                >>> tool = Tool("echo", echo)
                >>> result = tool.execute("Hello World")
                >>> print(result.output)
                Hello World

            Handling execution errors:

                >>> from insideLLMs.contrib.agents import Tool
                >>>
                >>> def divide(a: int, b: int) -> float:
                ...     return a / b
                >>>
                >>> tool = Tool("divide", divide)
                >>> result = tool.execute({"a": 1, "b": 0})
                >>> print(result.success)
                False
                >>> print("division" in result.error)
                True
        """
        start_time = time.time()

        try:
            # Parse input
            if isinstance(input_data, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(input_data)
                    output = self.func(**parsed) if isinstance(parsed, dict) else self.func(parsed)
                except json.JSONDecodeError:
                    # Treat as single string argument
                    output = self.func(input_data)
            elif isinstance(input_data, dict):
                output = self.func(**input_data)
            else:
                output = self.func(input_data)

            return ToolResult(
                tool_name=self.name,
                input=input_data,
                output=output,
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input=input_data,
                output=None,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the tool to a dictionary representation.

        Useful for serialization or for including tool information in prompts.

        Returns:
            dict[str, Any]: Dictionary with tool name, description, parameters,
                and return type.

        Examples:
            >>> from insideLLMs.contrib.agents import Tool, ToolParameter
            >>>
            >>> def search(query: str) -> str:
            ...     return f"Results for {query}"
            >>>
            >>> params = [ToolParameter("query", "str", "Search query")]
            >>> tool = Tool("search", search, "Search the web", params)
            >>> d = tool.to_dict()
            >>> print(d["name"])
            search
            >>> print(d["parameters"][0]["name"])
            query
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
        }

    def format_for_prompt(self) -> str:
        """Format the tool for inclusion in an agent prompt.

        Returns:
            str: A human-readable string describing the tool and its parameters.

        Examples:
            >>> from insideLLMs.contrib.agents import Tool
            >>>
            >>> def search(query: str, limit: int = 10) -> str:
            ...     '''Search for information.'''
            ...     return "results"
            >>>
            >>> tool = Tool("search", search)
            >>> print(tool.format_for_prompt())
            - search(query: str, limit: int = 10): Search for information.
        """
        params_str = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else f" = {p.default}")
            for p in self.parameters
        )
        return f"- {self.name}({params_str}): {self.description}"


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[list[ToolParameter]] = None,
) -> Callable:
    """Decorator to create a Tool from a function.

    This decorator provides a convenient way to convert regular Python functions
    into Tool objects that can be used by agents. It automatically infers
    parameters from the function signature and can use the docstring for
    the description.

    Args:
        name: The name of the tool. If not provided, uses the function's name.
        description: Human-readable description of the tool. If not provided,
            will be inferred from the function's docstring.
        parameters: List of ToolParameter definitions. If not provided, will
            be inferred from the function signature.

    Returns:
        Callable: A decorator that converts a function into a Tool object.

    Examples:
        Basic usage with automatic inference:

            >>> from insideLLMs.contrib.agents import tool
            >>>
            >>> @tool()
            ... def greet(name: str) -> str:
            ...     '''Greet someone by name.'''
            ...     return f"Hello, {name}!"
            >>>
            >>> print(greet.name)
            greet
            >>> print(greet.description)
            Greet someone by name.
            >>> result = greet.execute("World")
            >>> print(result.output)
            Hello, World!

        Specifying a custom name:

            >>> from insideLLMs.contrib.agents import tool
            >>>
            >>> @tool(name="say_hello")
            ... def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>>
            >>> print(greet.name)
            say_hello

        With custom description:

            >>> from insideLLMs.contrib.agents import tool
            >>>
            >>> @tool(description="Perform addition of two integers")
            ... def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> print(add.description)
            Perform addition of two integers

        With explicit parameters:

            >>> from insideLLMs.contrib.agents import tool, ToolParameter
            >>>
            >>> @tool(
            ...     name="search",
            ...     description="Search for documents",
            ...     parameters=[
            ...         ToolParameter("query", "str", "The search query"),
            ...         ToolParameter("limit", "int", "Max results", required=False, default=10)
            ...     ]
            ... )
            ... def search_docs(query: str, limit: int = 10) -> list:
            ...     return [f"Result for {query}"]
            >>>
            >>> print(search_docs.parameters[0].description)
            The search query

        Using decorated tools with an agent:

            >>> from insideLLMs.contrib.agents import tool, ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> @tool(name="calculator")
            ... def calculate(expression: str) -> str:
            ...     '''Evaluate a math expression safely.'''
            ...     from insideLLMs.contrib.agents import safe_eval
            ...     return str(safe_eval(expression))
            >>>
            >>> model = DummyModel()
            >>> agent = ReActAgent(model, tools=[calculate])
            >>> result = agent.run("What is 5 * 5?")

    See Also:
        :class:`Tool`: The class that decorated functions become.
        :class:`ToolParameter`: For defining explicit parameters.
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        return Tool(
            name=tool_name,
            func=func,
            description=description or "",
            parameters=parameters,
        )

    return decorator


class ToolRegistry:
    """Registry for managing and organizing multiple tools.

    The ToolRegistry provides a central place to register, retrieve, and
    manage tools that agents can use. It supports both pre-built Tool objects
    and automatic conversion of functions to tools.

    Attributes:
        _tools: Internal dictionary mapping tool names to Tool objects.

    Examples:
        Creating a registry and registering tools:

            >>> from insideLLMs.contrib.agents import ToolRegistry, Tool
            >>>
            >>> registry = ToolRegistry()
            >>>
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> tool = Tool("add", add, "Add two numbers")
            >>> registry.register(tool)
            >>> print(registry.get("add").name)
            add

        Registering functions directly:

            >>> from insideLLMs.contrib.agents import ToolRegistry
            >>>
            >>> registry = ToolRegistry()
            >>>
            >>> def multiply(x: int, y: int) -> int:
            ...     '''Multiply two numbers.'''
            ...     return x * y
            >>>
            >>> tool = registry.register_function(multiply)
            >>> print(tool.name)
            multiply
            >>> print(registry.get("multiply").description)
            Multiply two numbers.

        Registering with custom name and description:

            >>> from insideLLMs.contrib.agents import ToolRegistry
            >>>
            >>> registry = ToolRegistry()
            >>>
            >>> def div(a: float, b: float) -> float:
            ...     return a / b
            >>>
            >>> tool = registry.register_function(
            ...     div,
            ...     name="divide",
            ...     description="Divide two numbers"
            ... )
            >>> print(tool.name)
            divide

        Listing all tools:

            >>> from insideLLMs.contrib.agents import ToolRegistry, Tool
            >>>
            >>> registry = ToolRegistry()
            >>> registry.register(Tool("tool1", lambda: None))
            >>> registry.register(Tool("tool2", lambda: None))
            >>> tools = registry.list_tools()
            >>> print(len(tools))
            2
            >>> print([t.name for t in tools])
            ['tool1', 'tool2']

    See Also:
        :class:`Tool`: The objects managed by the registry.
        :class:`BaseAgent`: Uses ToolRegistry internally.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a Tool object in the registry.

        Args:
            tool: The Tool object to register.

        Examples:
            >>> from insideLLMs.contrib.agents import ToolRegistry, Tool
            >>>
            >>> registry = ToolRegistry()
            >>> tool = Tool("greet", lambda name: f"Hi {name}")
            >>> registry.register(tool)
            >>> print(registry.get("greet") is not None)
            True
        """
        self._tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tool:
        """Register a function as a tool, automatically creating a Tool object.

        Args:
            func: The function to wrap as a tool.
            name: Custom name for the tool (defaults to function name).
            description: Custom description (defaults to docstring).

        Returns:
            Tool: The created and registered Tool object.

        Examples:
            >>> from insideLLMs.contrib.agents import ToolRegistry
            >>>
            >>> registry = ToolRegistry()
            >>>
            >>> def greet(name: str) -> str:
            ...     '''Say hello.'''
            ...     return f"Hello, {name}!"
            >>>
            >>> tool = registry.register_function(greet)
            >>> print(tool.name)
            greet
            >>> print(registry.get("greet")("World"))
            Hello, World!
        """
        t = Tool(
            name=name or func.__name__,
            func=func,
            description=description or "",
        )
        self.register(t)
        return t

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            Optional[Tool]: The Tool if found, None otherwise.

        Examples:
            >>> from insideLLMs.contrib.agents import ToolRegistry, Tool
            >>>
            >>> registry = ToolRegistry()
            >>> registry.register(Tool("calc", lambda x: x))
            >>>
            >>> tool = registry.get("calc")
            >>> print(tool is not None)
            True
            >>>
            >>> missing = registry.get("nonexistent")
            >>> print(missing is None)
            True
        """
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools.

        Returns:
            list[Tool]: List of all Tool objects in the registry.

        Examples:
            >>> from insideLLMs.contrib.agents import ToolRegistry, Tool
            >>>
            >>> registry = ToolRegistry()
            >>> registry.register(Tool("a", lambda: 1))
            >>> registry.register(Tool("b", lambda: 2))
            >>>
            >>> tools = registry.list_tools()
            >>> print(len(tools))
            2
        """
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        """Format all tools for inclusion in an agent prompt.

        Returns:
            str: Newline-separated descriptions of all tools.

        Examples:
            >>> from insideLLMs.contrib.agents import ToolRegistry, Tool
            >>>
            >>> registry = ToolRegistry()
            >>>
            >>> def add(a: int, b: int) -> int:
            ...     '''Add numbers.'''
            ...     return a + b
            >>>
            >>> registry.register_function(add)
            >>> print(registry.format_for_prompt())
            - add(a: int, b: int): Add numbers.
        """
        return "\n".join(t.format_for_prompt() for t in self._tools.values())


# =============================================================================
# Memory
# =============================================================================


class AgentMemory:
    """Memory system for agent execution history and context.

    AgentMemory maintains the execution history (steps) and arbitrary context
    data for an agent. It automatically manages memory limits by evicting
    older steps when the limit is exceeded.

    Attributes:
        max_steps: Maximum number of steps to retain in memory.
        _steps: Internal list of AgentStep objects.
        _context: Internal dictionary for arbitrary context data.

    Examples:
        Creating memory and adding steps:

            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep
            >>>
            >>> memory = AgentMemory(max_steps=10)
            >>>
            >>> step1 = AgentStep(1, thought="Starting task")
            >>> memory.add_step(step1)
            >>>
            >>> step2 = AgentStep(2, thought="Continuing", action="search")
            >>> memory.add_step(step2)
            >>>
            >>> print(len(memory.get_steps()))
            2

        Memory eviction when limit exceeded:

            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep
            >>>
            >>> memory = AgentMemory(max_steps=3)
            >>> for i in range(5):
            ...     memory.add_step(AgentStep(i + 1, thought=f"Step {i + 1}"))
            >>>
            >>> steps = memory.get_steps()
            >>> print(len(steps))
            3
            >>> print(steps[0].thought)  # Oldest retained step
            Step 3

        Using context storage:

            >>> from insideLLMs.contrib.agents import AgentMemory
            >>>
            >>> memory = AgentMemory()
            >>> memory.set_context("user_name", "Alice")
            >>> memory.set_context("session_id", 12345)
            >>>
            >>> print(memory.get_context("user_name"))
            Alice
            >>> print(memory.get_context("missing", "default"))
            default

        Formatting scratchpad for prompts:

            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep, AgentConfig
            >>>
            >>> memory = AgentMemory()
            >>> memory.add_step(AgentStep(
            ...     1,
            ...     thought="I should calculate this",
            ...     action="calculator",
            ...     action_input="2 + 2",
            ...     observation="4"
            ... ))
            >>>
            >>> config = AgentConfig()
            >>> scratchpad = memory.format_scratchpad(config)
            >>> print("Thought:" in scratchpad)
            True

    See Also:
        :class:`AgentStep`: The steps stored in memory.
        :class:`AgentConfig`: Configuration for scratchpad formatting.
    """

    def __init__(self, max_steps: int = 20):
        """Initialize agent memory.

        Args:
            max_steps: Maximum number of steps to retain. Default is 20.
        """
        self.max_steps = max_steps
        self._steps: list[AgentStep] = []
        self._context: dict[str, Any] = {}

    def add_step(self, step: AgentStep) -> None:
        """Add a step to memory, evicting old steps if necessary.

        Args:
            step: The AgentStep to add.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep
            >>>
            >>> memory = AgentMemory(max_steps=2)
            >>> memory.add_step(AgentStep(1))
            >>> memory.add_step(AgentStep(2))
            >>> memory.add_step(AgentStep(3))  # Evicts step 1
            >>> print(len(memory.get_steps()))
            2
        """
        self._steps.append(step)
        # Evict old steps if needed
        if len(self._steps) > self.max_steps:
            self._steps = self._steps[-self.max_steps :]

    def get_steps(self) -> list[AgentStep]:
        """Get all steps in memory.

        Returns:
            list[AgentStep]: Copy of all steps (to prevent external modification).

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep
            >>>
            >>> memory = AgentMemory()
            >>> memory.add_step(AgentStep(1))
            >>> steps = memory.get_steps()
            >>> print(len(steps))
            1
        """
        return self._steps.copy()

    def get_last_step(self) -> Optional[AgentStep]:
        """Get the most recent step.

        Returns:
            Optional[AgentStep]: The last step, or None if memory is empty.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep
            >>>
            >>> memory = AgentMemory()
            >>> print(memory.get_last_step())
            None
            >>> memory.add_step(AgentStep(1, thought="First"))
            >>> memory.add_step(AgentStep(2, thought="Second"))
            >>> print(memory.get_last_step().thought)
            Second
        """
        return self._steps[-1] if self._steps else None

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: The context key.
            value: The value to store.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory
            >>>
            >>> memory = AgentMemory()
            >>> memory.set_context("task_id", "abc123")
            >>> print(memory.get_context("task_id"))
            abc123
        """
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: The context key to retrieve.
            default: Default value if key not found.

        Returns:
            The context value, or default if not found.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory
            >>>
            >>> memory = AgentMemory()
            >>> memory.set_context("key", "value")
            >>> print(memory.get_context("key"))
            value
            >>> print(memory.get_context("missing", "default"))
            default
        """
        return self._context.get(key, default)

    def clear(self) -> None:
        """Clear all memory (steps and context).

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep
            >>>
            >>> memory = AgentMemory()
            >>> memory.add_step(AgentStep(1))
            >>> memory.set_context("key", "value")
            >>> memory.clear()
            >>> print(len(memory.get_steps()))
            0
            >>> print(memory.get_context("key"))
            None
        """
        self._steps.clear()
        self._context.clear()

    def format_scratchpad(self, config: AgentConfig) -> str:
        """Format memory as a scratchpad for inclusion in agent prompts.

        Creates a string representation of all steps using the configured
        prefixes for thoughts, actions, and observations.

        Args:
            config: AgentConfig with prefix settings.

        Returns:
            str: Formatted scratchpad string.

        Examples:
            >>> from insideLLMs.contrib.agents import AgentMemory, AgentStep, AgentConfig
            >>>
            >>> memory = AgentMemory()
            >>> memory.add_step(AgentStep(
            ...     1,
            ...     thought="Need to search",
            ...     action="search",
            ...     action_input="python tutorial",
            ...     observation="Found 10 results"
            ... ))
            >>>
            >>> config = AgentConfig()
            >>> scratchpad = memory.format_scratchpad(config)
            >>> print("Thought: Need to search" in scratchpad)
            True
            >>> print("Action: search" in scratchpad)
            True
        """
        lines = []
        for step in self._steps:
            if step.thought:
                lines.append(f"{config.thought_prefix}{step.thought}")
            if step.action:
                lines.append(f"{config.action_prefix}{step.action}")
                if step.action_input:
                    lines.append(f"Action Input: {step.action_input}")
            if step.observation:
                lines.append(f"{config.observation_prefix}{step.observation}")
        return "\n".join(lines)


# =============================================================================
# Base Agent
# =============================================================================


class BaseAgent(ABC):
    """Abstract base class for all agent implementations.

    BaseAgent provides the common infrastructure for agents including:
    - Model integration for LLM reasoning
    - Tool registry management
    - Memory for execution history
    - Configuration handling

    Subclasses must implement the `run` and `_plan` methods.

    Args:
        model: The LLM model to use for reasoning. Must have a `generate` method.
        tools: Optional list of tools available to the agent. Can be Tool objects
            or callable functions (which will be converted to Tools).
        config: Optional AgentConfig for customizing behavior.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory for execution history.

    Examples:
        Creating a custom agent by subclassing:

            >>> from insideLLMs.contrib.agents import BaseAgent, AgentResult, AgentStatus
            >>> from abc import ABC
            >>>
            >>> class MyAgent(BaseAgent):
            ...     def run(self, query: str, **kwargs) -> AgentResult:
            ...         response = self._plan(query)
            ...         return AgentResult(
            ...             query=query,
            ...             answer=response,
            ...             status=AgentStatus.FINISHED
            ...         )
            ...
            ...     def _plan(self, query: str) -> str:
            ...         return self.model.generate(f"Answer: {query}")

        Using the tools property:

            >>> from insideLLMs.contrib.agents import ReActAgent, Tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[Tool("greet", greet)])
            >>> print(len(agent.tools))
            1
            >>> print(agent.tools[0].name)
            greet

        Adding tools dynamically:

            >>> from insideLLMs.contrib.agents import ReActAgent, Tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> print(len(agent.tools))
            0
            >>>
            >>> from insideLLMs.contrib.agents import safe_eval
            >>> def calculate(expr: str) -> str:
            ...     return str(safe_eval(expr))
            >>>
            >>> agent.add_tool(Tool("calc", calculate))
            >>> print(len(agent.tools))
            1

        Executing tools:

            >>> from insideLLMs.contrib.agents import ReActAgent, Tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> def double(x: int) -> int:
            ...     return x * 2
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[Tool("double", double)])
            >>> result = agent._execute_tool("double", {"x": 5})
            >>> print(result.output)
            10

    See Also:
        :class:`ReActAgent`: Concrete implementation using ReAct paradigm.
        :class:`SimpleAgent`: Simpler concrete implementation.
        :class:`ChainOfThoughtAgent`: Chain-of-thought reasoning implementation.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[list[Tool]] = None,
        config: Optional[AgentConfig] = None,
    ):
        """Initialize the base agent.

        Args:
            model: LLM model with a generate() method.
            tools: Optional list of tools.
            config: Optional configuration.
        """
        self.model = model
        self.config = config or AgentConfig()
        self.memory = AgentMemory(self.config.memory_limit)

        # Set up tools
        self._registry = ToolRegistry()
        if tools:
            for t in tools:
                if isinstance(t, Tool):
                    self._registry.register(t)
                elif callable(t):
                    self._registry.register_function(t)

    @property
    def tools(self) -> list[Tool]:
        """Get the list of registered tools.

        Returns:
            list[Tool]: All tools registered with this agent.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, Tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[
            ...     Tool("a", lambda: 1),
            ...     Tool("b", lambda: 2)
            ... ])
            >>> print([t.name for t in agent.tools])
            ['a', 'b']
        """
        return self._registry.list_tools()

    def add_tool(self, tool: Union[Tool, Callable]) -> None:
        """Add a tool to the agent.

        Args:
            tool: Either a Tool object or a callable function.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, Tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>>
            >>> # Add a Tool object
            >>> agent.add_tool(Tool("greet", lambda n: f"Hi {n}"))
            >>>
            >>> # Add a function (converted to Tool)
            >>> def farewell(name: str) -> str:
            ...     return f"Bye {name}"
            >>> agent.add_tool(farewell)
            >>>
            >>> print(len(agent.tools))
            2
        """
        if isinstance(tool, Tool):
            self._registry.register(tool)
        else:
            self._registry.register_function(tool)

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query.

        This is the main entry point for agent execution. Subclasses must
        implement this method.

        Args:
            query: The question or task for the agent.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            AgentResult: The result of agent execution.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> result = agent.run("What is 2 + 2?")
            >>> print(result.query)
            What is 2 + 2?
        """
        pass

    @abstractmethod
    def _plan(self, query: str) -> str:
        """Generate a plan or thought for the query.

        This method is called internally to get the model's reasoning.
        Subclasses must implement this method.

        Args:
            query: The query to plan for.

        Returns:
            str: The model's response/plan.
        """
        pass

    def _execute_tool(self, tool_name: str, tool_input: Any) -> ToolResult:
        """Execute a tool by name with the given input.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input to pass to the tool.

        Returns:
            ToolResult: Result of tool execution, including error if tool not found.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, Tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[
            ...     Tool("add", lambda a, b: a + b)
            ... ])
            >>>
            >>> # Successful execution
            >>> result = agent._execute_tool("add", {"a": 2, "b": 3})
            >>> print(result.success, result.output)
            True 5
            >>>
            >>> # Tool not found
            >>> result = agent._execute_tool("unknown", {})
            >>> print(result.success, "not found" in result.error)
            False True
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                input=tool_input,
                output=None,
                success=False,
                error=f"Tool '{tool_name}' not found",
            )
        return tool.execute(tool_input)


# =============================================================================
# ReAct Agent
# =============================================================================


REACT_PROMPT_TEMPLATE = """Answer the following question using the tools available. Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Available Tools:
{tools}

Question: {question}
{scratchpad}"""


class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) Agent implementation.

    Implements the ReAct paradigm from "ReAct: Synergizing Reasoning and Acting
    in Language Models" (Yao et al., 2022). The agent iteratively:
    1. Thinks about what to do next (Reasoning)
    2. Takes an action using a tool (Acting)
    3. Observes the result
    4. Repeats until a final answer is reached or max iterations exceeded

    This is the primary agent type for tasks requiring multi-step reasoning
    with tool use.

    Args:
        model: LLM to use for reasoning. Must have a `generate(prompt)` method.
        tools: List of tools available to the agent. Can be Tool objects or
            callable functions.
        config: AgentConfig for customizing behavior (iterations, timeouts, etc.).
        system_prompt: Optional custom system prompt to override the default.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory for execution history.
        _system_prompt: The system prompt used (if custom).

    Examples:
        Basic usage with calculator:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> model = DummyModel()
            >>> calc = create_calculator_tool()
            >>> agent = ReActAgent(model, tools=[calc])
            >>>
            >>> result = agent.run("What is 15 * 4?")
            >>> print(result.status)
            AgentStatus.FINISHED

        With custom configuration:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentConfig, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> config = AgentConfig(
            ...     max_iterations=5,
            ...     verbose=True,
            ...     max_execution_time=60
            ... )
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()], config=config)
            >>> result = agent.run("Calculate 100 / 4")

        With multiple tools:

            >>> from insideLLMs.contrib.agents import ReActAgent, Tool, create_calculator_tool, create_search_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> tools = [
            ...     create_calculator_tool(),
            ...     create_search_tool(),
            ...     Tool("greet", lambda name: f"Hello, {name}!")
            ... ]
            >>> agent = ReActAgent(DummyModel(), tools=tools)
            >>> print(len(agent.tools))
            3

        Analyzing execution steps:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 5 + 5?")
            >>>
            >>> # Examine the reasoning trace
            >>> for step in result.steps:
            ...     if step.thought:
            ...         print(f"Thought: {step.thought[:50]}...")
            ...     if step.action:
            ...         print(f"Action: {step.action}({step.action_input})")
            ...     if step.observation:
            ...         print(f"Observation: {step.observation}")

    Note:
        The agent uses a ReAct-style prompt that expects the model to output
        responses in a specific format with "Thought:", "Action:", "Action Input:",
        "Observation:", and "Final Answer:" prefixes.

    See Also:
        :class:`SimpleAgent`: A simpler agent without iterative reasoning.
        :class:`ChainOfThoughtAgent`: Agent focused on reasoning without tools.
        :func:`create_react_agent`: Convenience function for creating ReActAgent.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[list[Tool]] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the ReAct agent.

        Args:
            model: LLM model with generate() method.
            tools: Optional list of tools.
            config: Optional configuration.
            system_prompt: Optional custom system prompt.
        """
        super().__init__(model, tools, config)
        self._system_prompt = system_prompt

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query using the ReAct loop.

        Executes the ReAct loop: Think -> Act -> Observe -> Repeat until
        a final answer is found or limits are reached.

        Args:
            query: The question or task for the agent.
            **kwargs: Additional arguments (currently unused).

        Returns:
            AgentResult: Contains the answer, execution status, step trace,
                and timing information.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 2 + 2?")
            >>>
            >>> print(f"Query: {result.query}")
            Query: What is 2 + 2?
            >>> print(f"Status: {result.status}")
            >>> print(f"Iterations: {result.total_iterations}")
            >>> print(f"Time: {result.execution_time_ms:.2f}ms")
        """
        start_time = time.time()
        self.memory.clear()

        result = AgentResult(
            query=query,
            answer=None,
            status=AgentStatus.THINKING,
        )

        for iteration in range(self.config.max_iterations):
            # Check timeout
            if time.time() - start_time > self.config.max_execution_time:
                result.status = AgentStatus.MAX_ITERATIONS
                break

            step = AgentStep(step_number=iteration + 1)

            try:
                # Get agent response
                response = self._plan(query)

                # Parse response
                thought, action, action_input, final_answer = self._parse_response(response)

                step.thought = thought
                step.action = action
                step.action_input = action_input

                # Check for final answer
                if final_answer:
                    result.answer = final_answer
                    result.status = AgentStatus.FINISHED
                    self.memory.add_step(step)
                    result.steps.append(step)
                    break

                # Execute action
                if action:
                    result.status = AgentStatus.ACTING
                    tool_result = self._execute_tool(action, action_input)
                    step.tool_result = tool_result

                    if tool_result.success:
                        step.observation = str(tool_result.output)
                    else:
                        step.observation = f"Error: {tool_result.error}"

                    result.status = AgentStatus.OBSERVING

                self.memory.add_step(step)
                result.steps.append(step)

            except Exception as e:
                step.observation = f"Error: {str(e)}"
                self.memory.add_step(step)
                result.steps.append(step)

                if not self.config.retry_on_error:
                    result.status = AgentStatus.ERROR
                    break

        if result.status not in (AgentStatus.FINISHED, AgentStatus.ERROR):
            result.status = AgentStatus.MAX_ITERATIONS

        result.total_iterations = len(result.steps)
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Generate the next reasoning step using the model.

        Builds a prompt including the question and scratchpad (previous steps),
        then calls the model to generate the next thought/action.

        Args:
            query: The original query being processed.

        Returns:
            str: The model's response containing thought/action/answer.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> response = agent._plan("What is 2 + 2?")
            >>> print(type(response))
            <class 'str'>
        """
        prompt = self._build_prompt(query)
        return self.model.generate(prompt)

    def _build_prompt(self, query: str) -> str:
        """Build the ReAct prompt for the model.

        Constructs a prompt that includes the tool descriptions, the question,
        and any previous reasoning steps (scratchpad).

        Args:
            query: The question to answer.

        Returns:
            str: The complete prompt string.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> prompt = agent._build_prompt("What is 5 + 5?")
            >>> print("calculator" in prompt)
            True
            >>> print("What is 5 + 5?" in prompt)
            True
        """
        tool_names = ", ".join(t.name for t in self.tools)
        tools_desc = self._registry.format_for_prompt()
        scratchpad = ""

        if self.config.include_scratchpad and self.memory.get_steps():
            scratchpad = self.memory.format_scratchpad(self.config)

        return REACT_PROMPT_TEMPLATE.format(
            tool_names=tool_names or "none",
            tools=tools_desc or "No tools available",
            question=query,
            scratchpad=scratchpad,
        )

    def _parse_response(
        self,
        response: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse the model response into structured components.

        Extracts thought, action, action_input, and final_answer from the
        model's ReAct-formatted response using regex patterns.

        Args:
            response: The raw model response string.

        Returns:
            tuple: A tuple of (thought, action, action_input, final_answer),
                where any component may be None if not found.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>>
            >>> # Parse a thought + action response
            >>> response = '''Thought: I need to calculate this.
            ... Action: calculator
            ... Action Input: 2 + 2'''
            >>> thought, action, action_input, final = agent._parse_response(response)
            >>> print(thought)
            I need to calculate this.
            >>> print(action)
            calculator
            >>> print(action_input)
            2 + 2
            >>> print(final)
            None
            >>>
            >>> # Parse a final answer response
            >>> response = '''Thought: I now know the answer.
            ... Final Answer: The result is 4.'''
            >>> thought, action, action_input, final = agent._parse_response(response)
            >>> print(final)
            The result is 4.
        """
        thought = None
        action = None
        action_input = None
        final_answer = None

        # Extract thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract final answer
        final_match = re.search(
            r"Final Answer:\s*(.+?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, None, None, final_answer

        # Extract action
        action_match = re.search(
            r"Action:\s*(.+?)(?=Action Input:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=Observation:|Thought:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            action_input = input_match.group(1).strip()

        return thought, action, action_input, final_answer


# =============================================================================
# Simple Agent (No ReAct)
# =============================================================================


class SimpleAgent(BaseAgent):
    """Simple agent that executes tools based on direct instructions.

    Unlike the ReActAgent, the SimpleAgent does not iterate through multiple
    reasoning cycles. It makes a single decision about which tool to use
    (if any) and returns the result. This is suitable for simple tasks
    where complex multi-step reasoning is not required.

    Args:
        model: LLM to use for reasoning. Must have a `generate(prompt)` method.
        tools: Optional list of tools available to the agent.
        config: Optional AgentConfig for customizing behavior.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory (less utilized in SimpleAgent).

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import SimpleAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = SimpleAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("Calculate 5 + 3")
            >>> print(result.total_iterations)
            1

        Without tools (direct answer):

            >>> from insideLLMs.contrib.agents import SimpleAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = SimpleAgent(DummyModel())
            >>> result = agent.run("What is the capital of France?")
            >>> print(result.status)
            AgentStatus.FINISHED

        Handling tool errors:

            >>> from insideLLMs.contrib.agents import SimpleAgent, Tool, AgentStatus
            >>> from insideLLMs import DummyModel
            >>>
            >>> def faulty_tool(x: int) -> int:
            ...     raise ValueError("Something went wrong")
            >>>
            >>> agent = SimpleAgent(DummyModel(), tools=[Tool("faulty", faulty_tool)])
            >>> # If model chooses the faulty tool, result.status will be ERROR

        Comparing with ReActAgent:

            >>> from insideLLMs.contrib.agents import SimpleAgent, ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> # SimpleAgent: single step, no iteration
            >>> simple = SimpleAgent(DummyModel())
            >>> result = simple.run("Hello")
            >>> print(result.total_iterations)
            1
            >>>
            >>> # ReActAgent: may take multiple steps
            >>> react = ReActAgent(DummyModel())
            >>> result = react.run("Hello")
            >>> # May have multiple iterations

    Note:
        SimpleAgent is best for tasks where:
        - A single tool call is sufficient
        - No multi-step reasoning is needed
        - Speed is prioritized over complex problem-solving

    See Also:
        :class:`ReActAgent`: For tasks requiring multi-step reasoning.
        :class:`ChainOfThoughtAgent`: For reasoning-heavy tasks without tools.
    """

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query with a single decision step.

        Makes a single decision about which tool to use (if any) based on
        the query, executes the tool, and returns the result.

        Args:
            query: The question or task for the agent.
            **kwargs: Additional arguments (currently unused).

        Returns:
            AgentResult: Contains the answer, execution status, and single step.

        Examples:
            >>> from insideLLMs.contrib.agents import SimpleAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = SimpleAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 10 + 5?")
            >>>
            >>> print(f"Query: {result.query}")
            Query: What is 10 + 5?
            >>> print(f"Steps: {len(result.steps)}")
            Steps: 1
        """
        start_time = time.time()

        result = AgentResult(
            query=query,
            answer=None,
            status=AgentStatus.THINKING,
        )

        # Ask model what to do
        tool_prompt = f"""Given this query: "{query}"

Available tools:
{self._registry.format_for_prompt()}

Which tool should be used? Respond with:
Tool: <tool_name>
Input: <input_to_tool>

Or if no tool is needed, respond with:
Answer: <direct_answer>"""

        response = self.model.generate(tool_prompt)

        step = AgentStep(step_number=1, thought=response)

        # Parse response
        tool_match = re.search(r"Tool:\s*(\w+)", response, re.IGNORECASE)
        input_match = re.search(r"Input:\s*(.+?)(?=\n|$)", response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"Answer:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)

        if answer_match:
            result.answer = answer_match.group(1).strip()
            result.status = AgentStatus.FINISHED
        elif tool_match:
            tool_name = tool_match.group(1).strip()
            tool_input = input_match.group(1).strip() if input_match else ""

            step.action = tool_name
            step.action_input = tool_input

            tool_result = self._execute_tool(tool_name, tool_input)
            step.tool_result = tool_result

            if tool_result.success:
                result.answer = str(tool_result.output)
                step.observation = str(tool_result.output)
                result.status = AgentStatus.FINISHED
            else:
                result.status = AgentStatus.ERROR
                step.observation = f"Error: {tool_result.error}"
        else:
            result.answer = response
            result.status = AgentStatus.FINISHED

        result.steps.append(step)
        result.total_iterations = 1
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Not used in SimpleAgent.

        The SimpleAgent does not use a planning method as it makes direct
        tool decisions within the run method.

        Args:
            query: The query (unused).

        Returns:
            str: Always returns an empty string.
        """
        return ""


# =============================================================================
# Chain of Thought Agent
# =============================================================================


class ChainOfThoughtAgent(BaseAgent):
    """Agent that uses chain-of-thought (CoT) reasoning.

    This agent prompts the model to break down complex problems into explicit
    reasoning steps before arriving at a final answer. Unlike ReActAgent,
    it does not use tools - it relies purely on the model's reasoning ability.

    Chain-of-thought prompting has been shown to improve performance on
    complex reasoning tasks by encouraging step-by-step thinking.

    Args:
        model: LLM to use for reasoning. Must have a `generate(prompt)` method.
        tools: Optional list of tools (note: CoT agent doesn't typically use tools).
        config: Optional AgentConfig for customizing behavior.
        cot_prompt: Optional custom chain-of-thought prompt template.
            Must include {question} placeholder.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory for execution history.
        _cot_prompt: The chain-of-thought prompt template.

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> result = agent.run("If I have 3 apples and buy 5 more, how many do I have?")
            >>> print(result.status)
            AgentStatus.FINISHED

        With custom prompt:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> custom_prompt = '''Think carefully about this question:
            ... {question}
            ...
            ... Break it down:
            ... Step 1:
            ... Step 2:
            ... Step 3:
            ... Final Answer:'''
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel(), cot_prompt=custom_prompt)
            >>> result = agent.run("What is 15% of 200?")

        Examining the reasoning:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> result = agent.run("Solve: x + 5 = 12")
            >>>
            >>> # The reasoning is captured in the step's thought
            >>> if result.steps:
            ...     print(f"Reasoning: {result.steps[0].thought[:100]}...")

        Comparing with other agents:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent, ReActAgent, SimpleAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> # CoT: Best for reasoning without tools
            >>> cot = ChainOfThoughtAgent(DummyModel())
            >>>
            >>> # ReAct: Best for tasks requiring tool use
            >>> react = ReActAgent(DummyModel())
            >>>
            >>> # Simple: Best for straightforward tool invocations
            >>> simple = SimpleAgent(DummyModel())

    Note:
        Chain-of-thought reasoning works best for:
        - Mathematical problems
        - Logical reasoning tasks
        - Multi-step word problems
        - Tasks that benefit from explicit reasoning

    See Also:
        :class:`ReActAgent`: For tasks requiring tool use.
        :class:`SimpleAgent`: For simple, direct tasks.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[list[Tool]] = None,
        config: Optional[AgentConfig] = None,
        cot_prompt: Optional[str] = None,
    ):
        """Initialize the Chain-of-Thought agent.

        Args:
            model: LLM model with generate() method.
            tools: Optional list of tools (rarely used with CoT).
            config: Optional configuration.
            cot_prompt: Optional custom prompt template with {question} placeholder.
        """
        super().__init__(model, tools, config)
        self._cot_prompt = cot_prompt or self._default_cot_prompt()

    def _default_cot_prompt(self) -> str:
        """Return the default chain-of-thought prompt template.

        Returns:
            str: The default CoT prompt with {question} placeholder.

        Examples:
            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> prompt = agent._default_cot_prompt()
            >>> print("{question}" in prompt)
            True
        """
        return """Let's solve this step by step:

Question: {question}

Please think through this carefully:
1. First, let me understand what is being asked...
2. Then, I'll break down the problem...
3. Now, let me work through each part...
4. Finally, I'll combine my findings...

Show your reasoning at each step, then provide your final answer prefixed with "Final Answer:"
"""

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run chain-of-thought reasoning on a query.

        Prompts the model with the CoT template and extracts the final answer
        from the response.

        Args:
            query: The question or problem to reason through.
            **kwargs: Additional arguments (currently unused).

        Returns:
            AgentResult: Contains the answer, reasoning trace, and status.

        Examples:
            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> result = agent.run("What is 2 + 2?")
            >>>
            >>> print(f"Query: {result.query}")
            Query: What is 2 + 2?
            >>> print(f"Answer: {result.answer}")
            >>> print(f"Status: {result.status}")
            AgentStatus.FINISHED
        """
        start_time = time.time()

        result = AgentResult(
            query=query,
            answer=None,
            status=AgentStatus.THINKING,
        )

        prompt = self._cot_prompt.format(question=query)
        response = self.model.generate(prompt)

        step = AgentStep(step_number=1, thought=response)

        # Extract final answer
        final_match = re.search(
            r"Final Answer:\s*(.+?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if final_match:
            result.answer = final_match.group(1).strip()
        else:
            # Use last paragraph as answer
            paragraphs = response.strip().split("\n\n")
            result.answer = paragraphs[-1] if paragraphs else response

        result.status = AgentStatus.FINISHED
        result.steps.append(step)
        result.total_iterations = 1
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Generate the chain-of-thought reasoning for a query.

        Args:
            query: The question to reason about.

        Returns:
            str: The model's chain-of-thought response.

        Examples:
            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> response = agent._plan("What is 5 + 5?")
            >>> print(type(response))
            <class 'str'>
        """
        return self.model.generate(self._cot_prompt.format(question=query))


# =============================================================================
# Agent Executor
# =============================================================================


class AgentExecutor:
    """Executor for running agents with enhanced features.

    AgentExecutor wraps an agent to provide additional functionality:
    - Pre-execution hooks (logging, validation, setup)
    - Post-execution hooks (logging, cleanup, notifications)
    - Result post-processing and transformation
    - Execution history tracking
    - Batch execution support

    Args:
        agent: The agent instance to execute.
        pre_hooks: Optional list of callables to run before agent execution.
            Each receives (query, kwargs) arguments.
        post_hooks: Optional list of callables to run after agent execution.
            Each receives the AgentResult.
        result_processor: Optional callable to transform the AgentResult.
            Receives and returns an AgentResult.

    Attributes:
        agent: The wrapped agent instance.
        pre_hooks: List of pre-execution hooks.
        post_hooks: List of post-execution hooks.
        result_processor: Optional result transformation function.

    Examples:
        Basic usage with hooks:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> def log_start(query, kwargs):
            ...     print(f"Starting: {query}")
            >>>
            >>> def log_end(result):
            ...     print(f"Finished: {result.status}")
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> executor = AgentExecutor(agent, pre_hooks=[log_start], post_hooks=[log_end])
            >>> result = executor.run("Hello")
            Starting: Hello
            Finished: AgentStatus.FINISHED

        With result processor:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> def add_metadata(result):
            ...     result.metadata["processed"] = True
            ...     return result
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> executor = AgentExecutor(agent, result_processor=add_metadata)
            >>> result = executor.run("Test")
            >>> print(result.metadata.get("processed"))
            True

        Batch execution:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> executor = AgentExecutor(agent)
            >>> results = executor.batch_run(["Query 1", "Query 2", "Query 3"])
            >>> print(len(results))
            3

        Accessing execution history:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> executor = AgentExecutor(agent)
            >>> executor.run("First query")
            >>> executor.run("Second query")
            >>>
            >>> history = executor.get_history()
            >>> print(len(history))
            2
            >>> print(executor.execution_count)
            2

    See Also:
        :class:`ReActAgent`: Common agent to use with executor.
        :class:`AgentResult`: The result type returned and tracked.
    """

    def __init__(
        self,
        agent: BaseAgent,
        pre_hooks: Optional[list[Callable]] = None,
        post_hooks: Optional[list[Callable]] = None,
        result_processor: Optional[Callable[[AgentResult], AgentResult]] = None,
    ):
        """Initialize the AgentExecutor.

        Args:
            agent: The agent to execute.
            pre_hooks: Optional pre-execution hooks.
            post_hooks: Optional post-execution hooks.
            result_processor: Optional result transformation function.
        """
        self.agent = agent
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []
        self.result_processor = result_processor
        self._execution_count = 0
        self._results_history: list[AgentResult] = []

    def run(
        self,
        query: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent with hooks and processing.

        Runs pre-hooks, executes the agent, applies result processing,
        runs post-hooks, and records the result in history.

        Args:
            query: The query to run.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            AgentResult: The (possibly processed) result of agent execution.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> executor = AgentExecutor(ReActAgent(DummyModel()))
            >>> result = executor.run("What is 2 + 2?")
            >>> print(result.query)
            What is 2 + 2?
            >>> print(executor.execution_count)
            1
        """
        self._execution_count += 1

        # Pre-hooks
        for hook in self.pre_hooks:
            with contextlib.suppress(Exception):
                hook(query, kwargs)

        # Run agent
        result = self.agent.run(query, **kwargs)

        # Post-process
        if self.result_processor:
            result = self.result_processor(result)

        # Post-hooks
        for hook in self.post_hooks:
            with contextlib.suppress(Exception):
                hook(result)

        self._results_history.append(result)
        return result

    def batch_run(
        self,
        queries: list[str],
        **kwargs: Any,
    ) -> list[AgentResult]:
        """Run the agent on multiple queries sequentially.

        Args:
            queries: List of queries to execute.
            **kwargs: Additional arguments passed to each run.

        Returns:
            list[AgentResult]: List of results in query order.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> executor = AgentExecutor(ReActAgent(DummyModel()))
            >>> results = executor.batch_run(["Q1", "Q2", "Q3"])
            >>> print(len(results))
            3
            >>> print([r.query for r in results])
            ['Q1', 'Q2', 'Q3']
        """
        return [self.run(q, **kwargs) for q in queries]

    def get_history(self) -> list[AgentResult]:
        """Get the execution history.

        Returns:
            list[AgentResult]: Copy of all results from previous executions.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> executor = AgentExecutor(ReActAgent(DummyModel()))
            >>> executor.run("Query 1")
            >>> executor.run("Query 2")
            >>> history = executor.get_history()
            >>> print(len(history))
            2
        """
        return self._results_history.copy()

    def clear_history(self) -> None:
        """Clear the execution history.

        Note: This does not reset the execution_count.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> executor = AgentExecutor(ReActAgent(DummyModel()))
            >>> executor.run("Query")
            >>> print(len(executor.get_history()))
            1
            >>> executor.clear_history()
            >>> print(len(executor.get_history()))
            0
            >>> print(executor.execution_count)  # Still 1
            1
        """
        self._results_history.clear()

    @property
    def execution_count(self) -> int:
        """Get the total number of executions performed.

        Returns:
            int: Total execution count (does not reset when history is cleared).

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, AgentExecutor
            >>> from insideLLMs import DummyModel
            >>>
            >>> executor = AgentExecutor(ReActAgent(DummyModel()))
            >>> print(executor.execution_count)
            0
            >>> executor.run("Query")
            >>> print(executor.execution_count)
            1
        """
        return self._execution_count


# =============================================================================
# Built-in Tools
# =============================================================================


def _safe_eval_arithmetic(expression: str) -> Union[int, float]:
    """Evaluate arithmetic expressions without using eval/exec."""
    parsed = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
            return node.value
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("Unsupported arithmetic operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        raise ValueError("Unsupported expression")

    return _eval(parsed)


def create_calculator_tool() -> Tool:
    """Create a calculator tool for evaluating mathematical expressions.

    Creates a tool that can safely evaluate basic mathematical expressions
    containing only numbers and basic operators (+, -, *, /, parentheses).

    Returns:
        Tool: A calculator tool instance.

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import create_calculator_tool
            >>>
            >>> calc = create_calculator_tool()
            >>> print(calc.name)
            calculator
            >>> result = calc.execute("2 + 2")
            >>> print(result.output)
            4

        Using with an agent:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> print(len(agent.tools))
            1

        More complex expressions:

            >>> from insideLLMs.contrib.agents import create_calculator_tool
            >>>
            >>> calc = create_calculator_tool()
            >>> result = calc.execute("(10 + 5) * 2")
            >>> print(result.output)
            30
            >>> result = calc.execute("100 / 4")
            >>> print(result.output)
            25.0

        Error handling:

            >>> from insideLLMs.contrib.agents import create_calculator_tool
            >>>
            >>> calc = create_calculator_tool()
            >>> result = calc.execute("import os")  # Invalid characters
            >>> print("Error" in result.output)
            True

    Note:
        This tool only allows basic math operations for safety.
        It does not support functions like sin(), sqrt(), etc.

    See Also:
        :class:`Tool`: The base Tool class.
        :func:`create_search_tool`: For creating search tools.
    """

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        # Safe evaluation (basic math only)
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        try:
            result = _safe_eval_arithmetic(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    return Tool(
        name="calculator",
        func=calculate,
        description="Evaluate mathematical expressions",
        parameters=[ToolParameter("expression", "str", "Math expression to evaluate")],
    )


def create_search_tool(search_fn: Optional[Callable] = None) -> Tool:
    """Create a search tool with an optional custom search function.

    Creates a tool for searching/retrieving information. If no search function
    is provided, returns a default placeholder that indicates no search is
    configured.

    Args:
        search_fn: Optional callable that takes a query string and returns
            search results as a string. If None, a placeholder is used.

    Returns:
        Tool: A search tool instance.

    Examples:
        Basic usage with default (placeholder) function:

            >>> from insideLLMs.contrib.agents import create_search_tool
            >>>
            >>> search = create_search_tool()
            >>> result = search.execute("python tutorials")
            >>> print("No search function configured" in result.output)
            True

        With a custom search function:

            >>> from insideLLMs.contrib.agents import create_search_tool
            >>>
            >>> def my_search(query: str) -> str:
            ...     # In practice, this would call a real search API
            ...     return f"Results for '{query}': Result 1, Result 2"
            >>>
            >>> search = create_search_tool(my_search)
            >>> result = search.execute("machine learning")
            >>> print("Results for" in result.output)
            True

        Using with an agent:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_search_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> def web_search(q: str) -> str:
            ...     return f"Found 10 results for {q}"
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_search_tool(web_search)])
            >>> print(agent.tools[0].name)
            search

        With a lambda function:

            >>> from insideLLMs.contrib.agents import create_search_tool
            >>>
            >>> search = create_search_tool(lambda q: f"Searching: {q}")
            >>> result = search.execute("test")
            >>> print(result.output)
            Searching: test

    See Also:
        :class:`Tool`: The base Tool class.
        :func:`create_calculator_tool`: For creating calculator tools.
    """

    def default_search(query: str) -> str:
        return f"No search function configured. Query was: {query}"

    return Tool(
        name="search",
        func=search_fn or default_search,
        description="Search for information",
        parameters=[ToolParameter("query", "str", "Search query")],
    )


def create_python_tool(
    allow_exec: bool = False,
    *,
    sandbox_contract: Optional[str] = None,
) -> Tool:
    """Create a Python code execution tool.

    Creates a tool that can execute Python code. By default, execution is
    disabled for safety. Only enable execution in trusted environments.

    Args:
        allow_exec: Whether to allow actual code execution. Default is False.
            WARNING: Setting this to True is dangerous and should only be
            done in fully trusted, sandboxed environments.
        sandbox_contract: Explicit description/identifier of the sandbox boundary
            used to contain execution (required when allow_exec=True).

    Returns:
        Tool: A Python execution tool instance.

    Examples:
        Default (safe) mode - execution disabled:

            >>> from insideLLMs.contrib.agents import create_python_tool
            >>>
            >>> python_tool = create_python_tool(allow_exec=False)
            >>> result = python_tool.execute("result = 2 + 2")
            >>> print(result.output)
            Code execution disabled for safety

        With execution enabled (DANGEROUS):

            >>> from insideLLMs.contrib.agents import create_python_tool
            >>>
            >>> # WARNING: Only use in trusted environments!
            >>> python_tool = create_python_tool(
            ...     allow_exec=True,
            ...     sandbox_contract="isolated-unit-test-sandbox",
            ... )
            >>> result = python_tool.execute("result = 2 + 2")
            >>> print(result.output)
            4

        Using with an agent:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_python_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_python_tool()])
            >>> print(agent.tools[0].name)
            python

        Error handling:

            >>> from insideLLMs.contrib.agents import create_python_tool
            >>>
            >>> python_tool = create_python_tool(
            ...     allow_exec=True,
            ...     sandbox_contract="isolated-unit-test-sandbox",
            ... )
            >>> result = python_tool.execute("result = 1/0")
            >>> print("Error" in result.output)
            True

    Warning:
        Enabling code execution (allow_exec=True) is inherently dangerous
        and can lead to arbitrary code execution vulnerabilities. Only use
        in sandboxed environments with proper security controls, and always
        provide an explicit sandbox_contract.

    See Also:
        :class:`Tool`: The base Tool class.
        :func:`create_calculator_tool`: A safer alternative for math operations.
    """

    def execute_python(code: str) -> str:
        if not allow_exec:
            return "Code execution disabled for safety"
        if not sandbox_contract:
            return (
                "Code execution disabled: missing sandbox contract. "
                "Pass sandbox_contract when allow_exec=True."
            )
        try:
            # DANGEROUS: Only use in trusted environments
            local_vars: dict[str, Any] = {}
            exec(code, {"__builtins__": {}}, local_vars)
            return str(local_vars.get("result", "Code executed successfully"))
        except Exception as e:
            return f"Error: {str(e)}"

    return Tool(
        name="python",
        func=execute_python,
        description="Execute Python code (use result variable for output)",
        parameters=[ToolParameter("code", "str", "Python code to execute")],
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_react_agent(
    model: Any,
    tools: Optional[list[Union[Tool, Callable]]] = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> ReActAgent:
    """Create a configured ReAct agent with sensible defaults.

    A convenience function for creating ReActAgent instances with common
    configuration options. Automatically converts callable functions to
    Tool objects.

    Args:
        model: The LLM model to use. Must have a `generate(prompt)` method.
        tools: Optional list of tools. Can be Tool objects or callable functions.
            Functions are automatically wrapped in Tool objects.
        max_iterations: Maximum number of reasoning iterations. Default is 10.
        verbose: If True, enables verbose output for debugging. Default is False.

    Returns:
        ReActAgent: A configured ReActAgent ready to use.

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import create_react_agent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = create_react_agent(DummyModel())
            >>> print(type(agent).__name__)
            ReActAgent

        With tools:

            >>> from insideLLMs.contrib.agents import create_react_agent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> calc = create_calculator_tool()
            >>> agent = create_react_agent(DummyModel(), tools=[calc])
            >>> print(len(agent.tools))
            1

        With callable functions as tools:

            >>> from insideLLMs.contrib.agents import create_react_agent
            >>> from insideLLMs import DummyModel
            >>>
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>>
            >>> agent = create_react_agent(DummyModel(), tools=[greet])
            >>> print(agent.tools[0].name)
            greet

        With custom configuration:

            >>> from insideLLMs.contrib.agents import create_react_agent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = create_react_agent(
            ...     DummyModel(),
            ...     max_iterations=5,
            ...     verbose=True
            ... )
            >>> print(agent.config.max_iterations)
            5

    See Also:
        :class:`ReActAgent`: The agent class being created.
        :func:`create_simple_agent`: For creating simpler agents.
        :func:`quick_agent_run`: For one-shot agent execution.
    """
    config = AgentConfig(
        max_iterations=max_iterations,
        verbose=verbose,
    )

    # Convert callables to Tools
    tool_list = []
    if tools:
        for t in tools:
            if isinstance(t, Tool):
                tool_list.append(t)
            elif callable(t):
                tool_list.append(
                    Tool(
                        name=t.__name__,
                        func=t,
                    )
                )

    return ReActAgent(model, tool_list, config)


def create_simple_agent(
    model: Any,
    tools: Optional[list[Union[Tool, Callable]]] = None,
) -> SimpleAgent:
    """Create a configured SimpleAgent.

    A convenience function for creating SimpleAgent instances. Automatically
    converts callable functions to Tool objects.

    Args:
        model: The LLM model to use. Must have a `generate(prompt)` method.
        tools: Optional list of tools. Can be Tool objects or callable functions.
            Functions are automatically wrapped in Tool objects.

    Returns:
        SimpleAgent: A configured SimpleAgent ready to use.

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import create_simple_agent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = create_simple_agent(DummyModel())
            >>> print(type(agent).__name__)
            SimpleAgent

        With tools:

            >>> from insideLLMs.contrib.agents import create_simple_agent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = create_simple_agent(DummyModel(), tools=[create_calculator_tool()])
            >>> print(len(agent.tools))
            1

        With callable functions:

            >>> from insideLLMs.contrib.agents import create_simple_agent
            >>> from insideLLMs import DummyModel
            >>>
            >>> def double(x: int) -> int:
            ...     return x * 2
            >>>
            >>> agent = create_simple_agent(DummyModel(), tools=[double])
            >>> print(agent.tools[0].name)
            double

        Running the agent:

            >>> from insideLLMs.contrib.agents import create_simple_agent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = create_simple_agent(DummyModel())
            >>> result = agent.run("Hello")
            >>> print(result.total_iterations)
            1

    See Also:
        :class:`SimpleAgent`: The agent class being created.
        :func:`create_react_agent`: For creating ReAct agents.
        :func:`quick_agent_run`: For one-shot agent execution.
    """
    tool_list = []
    if tools:
        for t in tools:
            if isinstance(t, Tool):
                tool_list.append(t)
            elif callable(t):
                tool_list.append(Tool(name=t.__name__, func=t))

    return SimpleAgent(model, tool_list)


def quick_agent_run(
    query: str,
    model: Any,
    tools: Optional[list[Union[Tool, Callable]]] = None,
    agent_type: str = "react",
) -> AgentResult:
    """Quick helper to create and run an agent in one step.

    Creates an agent of the specified type and immediately runs it on the
    given query. Useful for one-shot tasks where you don't need to reuse
    the agent.

    Args:
        query: The query/task for the agent to execute.
        model: The LLM model to use. Must have a `generate(prompt)` method.
        tools: Optional list of tools. Can be Tool objects or callable functions.
        agent_type: Type of agent to create. Options are:
            - "react": ReActAgent with iterative reasoning (default)
            - "simple": SimpleAgent with single-step execution
            - "cot": ChainOfThoughtAgent for reasoning without tools

    Returns:
        AgentResult: The result of agent execution.

    Raises:
        ValueError: If agent_type is not one of "react", "simple", or "cot".

    Examples:
        Quick ReAct agent run:

            >>> from insideLLMs.contrib.agents import quick_agent_run, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> result = quick_agent_run(
            ...     "What is 5 + 5?",
            ...     DummyModel(),
            ...     tools=[create_calculator_tool()]
            ... )
            >>> print(result.query)
            What is 5 + 5?

        Using a simple agent:

            >>> from insideLLMs.contrib.agents import quick_agent_run
            >>> from insideLLMs import DummyModel
            >>>
            >>> result = quick_agent_run(
            ...     "Hello world",
            ...     DummyModel(),
            ...     agent_type="simple"
            ... )
            >>> print(result.total_iterations)
            1

        Using chain-of-thought:

            >>> from insideLLMs.contrib.agents import quick_agent_run
            >>> from insideLLMs import DummyModel
            >>>
            >>> result = quick_agent_run(
            ...     "If x + 5 = 10, what is x?",
            ...     DummyModel(),
            ...     agent_type="cot"
            ... )
            >>> print(result.status)
            AgentStatus.FINISHED

        With callable functions as tools:

            >>> from insideLLMs.contrib.agents import quick_agent_run
            >>> from insideLLMs import DummyModel
            >>>
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>>
            >>> result = quick_agent_run(
            ...     "Greet Alice",
            ...     DummyModel(),
            ...     tools=[greet]
            ... )

    See Also:
        :func:`create_react_agent`: For creating reusable ReAct agents.
        :func:`create_simple_agent`: For creating reusable simple agents.
        :class:`ChainOfThoughtAgent`: The CoT agent used when agent_type="cot".
    """
    if agent_type == "react":
        agent = create_react_agent(model, tools)
    elif agent_type == "simple":
        agent = create_simple_agent(model, tools)
    elif agent_type == "cot":
        agent = ChainOfThoughtAgent(model)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent.run(query)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    "AgentConfig",
    "AgentStatus",
    # Tools
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "tool",
    # Steps and Results
    "AgentStep",
    "AgentResult",
    # Memory
    "AgentMemory",
    # Agents
    "BaseAgent",
    "ReActAgent",
    "SimpleAgent",
    "ChainOfThoughtAgent",
    # Executor
    "AgentExecutor",
    # Built-in tools
    "create_calculator_tool",
    "create_search_tool",
    "create_python_tool",
    # Convenience functions
    "create_react_agent",
    "create_simple_agent",
    "quick_agent_run",
]
