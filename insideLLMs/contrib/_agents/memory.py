"""Bounded in-memory state for agent runs."""

from typing import Any, Optional

from insideLLMs.contrib._agents.models import AgentConfig, AgentStep


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
