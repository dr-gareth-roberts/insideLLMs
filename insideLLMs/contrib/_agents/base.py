"""Base contract shared by autonomous agent implementations."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from insideLLMs.contrib._agents.memory import AgentMemory
from insideLLMs.contrib._agents.models import AgentConfig, AgentResult
from insideLLMs.contrib._agents.tools import Tool, ToolRegistry, ToolResult


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
            >>> def calculate(expr: str) -> str:
            ...     return str(_safe_eval_arithmetic(expr))
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
