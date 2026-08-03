"""Agent tool definitions, decorators, and registry."""

import inspect
import json
import time
from typing import Any, Callable, Optional

from insideLLMs.contrib._agents.models import ToolParameter, ToolResult


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
            ...     '''Evaluate a math expression.'''
            ...     return str(_safe_eval_arithmetic(expression))
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
