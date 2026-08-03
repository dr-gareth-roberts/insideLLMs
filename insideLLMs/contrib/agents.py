"""Autonomous-agent APIs exposed through a stable compatibility facade."""

from insideLLMs.contrib._agents.base import BaseAgent
from insideLLMs.contrib._agents.builtins import (
    create_calculator_tool,
    create_python_tool,
    create_search_tool,
)
from insideLLMs.contrib._agents.executor import AgentExecutor
from insideLLMs.contrib._agents.factories import (
    create_react_agent,
    create_simple_agent,
    quick_agent_run,
)
from insideLLMs.contrib._agents.implementations import (
    ChainOfThoughtAgent,
    ReActAgent,
    SimpleAgent,
)
from insideLLMs.contrib._agents.memory import AgentMemory
from insideLLMs.contrib._agents.models import (
    AgentConfig,
    AgentResult,
    AgentStatus,
    AgentStep,
    ToolParameter,
    ToolResult,
)
from insideLLMs.contrib._agents.tools import Tool, ToolRegistry, tool

__all__ = [
    "AgentConfig",
    "AgentStatus",
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "AgentStep",
    "AgentResult",
    "AgentMemory",
    "BaseAgent",
    "ReActAgent",
    "SimpleAgent",
    "ChainOfThoughtAgent",
    "AgentExecutor",
    "create_calculator_tool",
    "create_search_tool",
    "create_python_tool",
    "create_react_agent",
    "create_simple_agent",
    "quick_agent_run",
]

for _name in __all__:
    _value = globals()[_name]
    if callable(_value):
        _value.__module__ = __name__
del _name, _value
