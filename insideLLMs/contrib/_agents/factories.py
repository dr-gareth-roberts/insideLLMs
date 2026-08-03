"""Convenience factories for common autonomous-agent configurations."""

from typing import Any, Callable, Optional, Union

from insideLLMs.contrib._agents.implementations import (
    ChainOfThoughtAgent,
    ReActAgent,
    SimpleAgent,
)
from insideLLMs.contrib._agents.models import AgentConfig, AgentResult
from insideLLMs.contrib._agents.tools import Tool


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
