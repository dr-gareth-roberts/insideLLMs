"""Autonomous Agents Module for insideLLMs.

This module provides tools for building autonomous LLM agents:
- Tool definition and registration
- ReAct (Reasoning + Acting) agent implementation
- Agent execution loops with observation handling
- Memory and state management
- Chain-of-thought reasoning

Example:
    >>> from insideLLMs.agents import Tool, ReActAgent, AgentExecutor
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Define tools
    >>> @tool("calculator")
    >>> def calculate(expression: str) -> str:
    ...     return str(eval(expression))
    >>>
    >>> # Create agent
    >>> model = DummyModel()
    >>> agent = ReActAgent(model, tools=[calculate])
    >>> result = agent.run("What is 25 * 4?")
"""

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
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import inspect


# =============================================================================
# Configuration and Types
# =============================================================================


class AgentStatus(Enum):
    """Status of agent execution."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    input: Any
    output: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """A single step in agent execution."""
    step_number: int
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Any] = None
    observation: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Result of agent execution."""
    query: str
    answer: Optional[str]
    status: AgentStatus
    steps: List[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    execution_time_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# Tool Definition
# =============================================================================


class Tool:
    """A tool that an agent can use.

    Args:
        name: Tool name
        func: The function to execute
        description: Human-readable description
        parameters: List of parameter definitions
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[List[ToolParameter]] = None,
        return_type: str = "str",
    ):
        self.name = name
        self.func = func
        self.description = description or self._infer_description(func)
        self.parameters = parameters or self._infer_parameters(func)
        self.return_type = return_type

    def _infer_description(self, func: Callable) -> str:
        """Infer description from docstring."""
        doc = func.__doc__
        if doc:
            # Get first line
            return doc.strip().split('\n')[0]
        return f"Execute {func.__name__}"

    def _infer_parameters(self, func: Callable) -> List[ToolParameter]:
        """Infer parameters from function signature."""
        sig = inspect.signature(func)
        params = []

        for name, param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue

            # Get type annotation
            param_type = "str"
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation.__name__ if hasattr(
                    param.annotation, '__name__'
                ) else str(param.annotation)

            # Check if required
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            params.append(ToolParameter(
                name=name,
                type=param_type,
                description=f"Parameter {name}",
                required=required,
                default=default,
            ))

        return params

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        return self.func(*args, **kwargs)

    def execute(self, input_data: Any) -> ToolResult:
        """Execute tool with input data and return result.

        Args:
            input_data: Input to the tool (string or dict)

        Returns:
            ToolResult with output or error
        """
        start_time = time.time()

        try:
            # Parse input
            if isinstance(input_data, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(input_data)
                    if isinstance(parsed, dict):
                        output = self.func(**parsed)
                    else:
                        output = self.func(parsed)
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompts."""
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
        """Format tool for inclusion in agent prompt."""
        params_str = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else f" = {p.default}")
            for p in self.parameters
        )
        return f"- {self.name}({params_str}): {self.description}"


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[ToolParameter]] = None,
) -> Callable:
    """Decorator to create a tool from a function.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description
        parameters: Parameter definitions

    Returns:
        Decorated function wrapped as Tool
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
    """Registry for managing tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tool:
        """Register a function as a tool."""
        t = Tool(
            name=name or func.__name__,
            func=func,
            description=description or "",
        )
        self.register(t)
        return t

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        """Format all tools for agent prompt."""
        return "\n".join(t.format_for_prompt() for t in self._tools.values())


# =============================================================================
# Memory
# =============================================================================


class AgentMemory:
    """Memory for agent execution history."""

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self._steps: List[AgentStep] = []
        self._context: Dict[str, Any] = {}

    def add_step(self, step: AgentStep) -> None:
        """Add a step to memory."""
        self._steps.append(step)
        # Evict old steps if needed
        if len(self._steps) > self.max_steps:
            self._steps = self._steps[-self.max_steps:]

    def get_steps(self) -> List[AgentStep]:
        """Get all steps."""
        return self._steps.copy()

    def get_last_step(self) -> Optional[AgentStep]:
        """Get the last step."""
        return self._steps[-1] if self._steps else None

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def clear(self) -> None:
        """Clear memory."""
        self._steps.clear()
        self._context.clear()

    def format_scratchpad(self, config: AgentConfig) -> str:
        """Format memory as scratchpad for agent prompt."""
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
    """Abstract base class for agents."""

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Tool]] = None,
        config: Optional[AgentConfig] = None,
    ):
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
    def tools(self) -> List[Tool]:
        """Get registered tools."""
        return self._registry.list_tools()

    def add_tool(self, tool: Union[Tool, Callable]) -> None:
        """Add a tool."""
        if isinstance(tool, Tool):
            self._registry.register(tool)
        else:
            self._registry.register_function(tool)

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query."""
        pass

    @abstractmethod
    def _plan(self, query: str) -> str:
        """Generate a plan/thought for the query."""
        pass

    def _execute_tool(self, tool_name: str, tool_input: Any) -> ToolResult:
        """Execute a tool by name."""
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
    """ReAct (Reasoning + Acting) Agent.

    Implements the ReAct paradigm where the agent reasons about what to do,
    takes an action using a tool, observes the result, and repeats.

    Args:
        model: LLM to use for reasoning
        tools: List of tools available to the agent
        config: Agent configuration
        system_prompt: Optional custom system prompt
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Tool]] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(model, tools, config)
        self._system_prompt = system_prompt

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query.

        Args:
            query: The question/task for the agent
            **kwargs: Additional arguments

        Returns:
            AgentResult with the answer and execution trace
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
        """Generate the next step using the model."""
        prompt = self._build_prompt(query)
        return self.model.generate(prompt)

    def _build_prompt(self, query: str) -> str:
        """Build the prompt for the model."""
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
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse the model response into thought, action, action_input, and final_answer."""
        thought = None
        action = None
        action_input = None
        final_answer = None

        # Extract thought
        thought_match = re.search(
            r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract final answer
        final_match = re.search(
            r'Final Answer:\s*(.+?)$',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, None, None, final_answer

        # Extract action
        action_match = re.search(
            r'Action:\s*(.+?)(?=Action Input:|$)',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(
            r'Action Input:\s*(.+?)(?=Observation:|Thought:|$)',
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

    Doesn't use the ReAct loop - just interprets instructions and
    executes appropriate tools.
    """

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent."""
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
        tool_match = re.search(r'Tool:\s*(\w+)', response, re.IGNORECASE)
        input_match = re.search(r'Input:\s*(.+?)(?=\n|$)', response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'Answer:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)

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
        """Not used in SimpleAgent."""
        return ""


# =============================================================================
# Chain of Thought Agent
# =============================================================================


class ChainOfThoughtAgent(BaseAgent):
    """Agent that uses chain-of-thought reasoning.

    Breaks down complex problems into steps and reasons through them.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Tool]] = None,
        config: Optional[AgentConfig] = None,
        cot_prompt: Optional[str] = None,
    ):
        super().__init__(model, tools, config)
        self._cot_prompt = cot_prompt or self._default_cot_prompt()

    def _default_cot_prompt(self) -> str:
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
        """Run chain-of-thought reasoning."""
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
            r'Final Answer:\s*(.+?)$',
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if final_match:
            result.answer = final_match.group(1).strip()
        else:
            # Use last paragraph as answer
            paragraphs = response.strip().split('\n\n')
            result.answer = paragraphs[-1] if paragraphs else response

        result.status = AgentStatus.FINISHED
        result.steps.append(step)
        result.total_iterations = 1
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Generate plan using CoT."""
        return self.model.generate(self._cot_prompt.format(question=query))


# =============================================================================
# Agent Executor
# =============================================================================


class AgentExecutor:
    """Executor for running agents with additional features.

    Provides:
    - Execution hooks (before/after)
    - Result post-processing
    - Error handling and recovery
    - Logging and tracing
    """

    def __init__(
        self,
        agent: BaseAgent,
        pre_hooks: Optional[List[Callable]] = None,
        post_hooks: Optional[List[Callable]] = None,
        result_processor: Optional[Callable[[AgentResult], AgentResult]] = None,
    ):
        self.agent = agent
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []
        self.result_processor = result_processor
        self._execution_count = 0
        self._results_history: List[AgentResult] = []

    def run(
        self,
        query: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent with hooks.

        Args:
            query: The query to run
            **kwargs: Additional arguments

        Returns:
            AgentResult
        """
        self._execution_count += 1

        # Pre-hooks
        for hook in self.pre_hooks:
            try:
                hook(query, kwargs)
            except Exception:
                pass

        # Run agent
        result = self.agent.run(query, **kwargs)

        # Post-process
        if self.result_processor:
            result = self.result_processor(result)

        # Post-hooks
        for hook in self.post_hooks:
            try:
                hook(result)
            except Exception:
                pass

        self._results_history.append(result)
        return result

    def batch_run(
        self,
        queries: List[str],
        **kwargs: Any,
    ) -> List[AgentResult]:
        """Run agent on multiple queries.

        Args:
            queries: List of queries
            **kwargs: Additional arguments

        Returns:
            List of AgentResults
        """
        return [self.run(q, **kwargs) for q in queries]

    def get_history(self) -> List[AgentResult]:
        """Get execution history."""
        return self._results_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._results_history.clear()

    @property
    def execution_count(self) -> int:
        """Get total execution count."""
        return self._execution_count


# =============================================================================
# Built-in Tools
# =============================================================================


def create_calculator_tool() -> Tool:
    """Create a calculator tool."""
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        # Safe evaluation (basic math only)
        allowed = set('0123456789+-*/.() ')
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        try:
            result = eval(expression)
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
    """Create a search tool.

    Args:
        search_fn: Optional custom search function
    """
    def default_search(query: str) -> str:
        return f"Search results for: {query} (placeholder - no real search)"

    return Tool(
        name="search",
        func=search_fn or default_search,
        description="Search for information",
        parameters=[ToolParameter("query", "str", "Search query")],
    )


def create_python_tool(allow_exec: bool = False) -> Tool:
    """Create a Python code execution tool.

    Args:
        allow_exec: Whether to allow actual code execution (DANGEROUS)
    """
    def execute_python(code: str) -> str:
        if not allow_exec:
            return "Code execution disabled for safety"
        try:
            # DANGEROUS: Only use in trusted environments
            local_vars: Dict[str, Any] = {}
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
    tools: Optional[List[Union[Tool, Callable]]] = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> ReActAgent:
    """Create a ReAct agent.

    Args:
        model: LLM to use
        tools: List of tools
        max_iterations: Maximum iterations
        verbose: Enable verbose output

    Returns:
        Configured ReActAgent
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
                tool_list.append(Tool(
                    name=t.__name__,
                    func=t,
                ))

    return ReActAgent(model, tool_list, config)


def create_simple_agent(
    model: Any,
    tools: Optional[List[Union[Tool, Callable]]] = None,
) -> SimpleAgent:
    """Create a simple agent.

    Args:
        model: LLM to use
        tools: List of tools

    Returns:
        Configured SimpleAgent
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
    tools: Optional[List[Union[Tool, Callable]]] = None,
    agent_type: str = "react",
) -> AgentResult:
    """Quick helper to run an agent.

    Args:
        query: The query to run
        model: LLM to use
        tools: List of tools
        agent_type: Type of agent ("react", "simple", "cot")

    Returns:
        AgentResult
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
