"""Batch execution and hooks for autonomous agents."""

import contextlib
from typing import Any, Callable, Optional

from insideLLMs.contrib._agents.base import BaseAgent
from insideLLMs.contrib._agents.models import AgentResult


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
