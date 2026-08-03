"""Sandboxed built-in calculator, search, and Python-expression tools."""

import ast
from typing import Callable, Optional, Union

from insideLLMs.contrib._agents.tools import Tool, ToolParameter


def _bounded_pow(left: Union[int, float], right: Union[int, float]) -> Union[int, float]:
    """Exponentiate with operand bounds to prevent CPU/memory exhaustion.

    Unbounded ``**`` (e.g. ``10**10**10``) builds a multi-million-digit integer
    and hangs the process, so reject oversized operands instead.
    """
    if abs(right) > 1000 or abs(left) > 1_000_000:
        raise ValueError("Exponentiation operands too large")
    return left**right


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
                return _bounded_pow(left, right)
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


def _apply_binary_operator(
    op: ast.operator, left: Union[int, float], right: Union[int, float]
) -> Union[int, float]:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return left // right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.Pow):
        return _bounded_pow(left, right)
    raise ValueError("Unsupported arithmetic operator")


def _apply_unary_operator(op: ast.unaryop, operand: Union[int, float]) -> Union[int, float]:
    if isinstance(op, ast.UAdd):
        return +operand
    if isinstance(op, ast.USub):
        return -operand
    raise ValueError("Unsupported unary operator")


def _eval_exec_expression(
    node: ast.AST, variables: dict[str, Union[int, float]]
) -> Union[int, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown variable: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_exec_expression(node.left, variables)
        right = _eval_exec_expression(node.right, variables)
        return _apply_binary_operator(node.op, left, right)
    if isinstance(node, ast.UnaryOp):
        return _apply_unary_operator(node.op, _eval_exec_expression(node.operand, variables))
    raise ValueError("Unsupported expression")


_MAX_EXEC_CODE_LENGTH = 1024
_MAX_EXEC_AST_NODES = 100
_MAX_EXEC_AST_DEPTH = 20


def _ast_depth(node: ast.AST) -> int:
    """Return the maximum nesting depth of an AST node."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(c) for c in children)


def _safe_exec_python_subset(code: str) -> dict[str, Union[int, float]]:
    """Execute a tightly restricted Python subset (assignments + arithmetic)."""
    if len(code) > _MAX_EXEC_CODE_LENGTH:
        raise ValueError(f"Code exceeds maximum length of {_MAX_EXEC_CODE_LENGTH} characters")
    parsed = ast.parse(code, mode="exec")
    node_count = sum(1 for _ in ast.walk(parsed))
    if node_count > _MAX_EXEC_AST_NODES:
        raise ValueError(f"Code AST exceeds maximum of {_MAX_EXEC_AST_NODES} nodes")
    if _ast_depth(parsed) > _MAX_EXEC_AST_DEPTH:
        raise ValueError(f"Code AST exceeds maximum depth of {_MAX_EXEC_AST_DEPTH}")
    variables: dict[str, Union[int, float]] = {}
    for statement in parsed.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("Only simple variable assignments are supported")
            variables[statement.targets[0].id] = _eval_exec_expression(statement.value, variables)
            continue
        if isinstance(statement, ast.Expr):
            _eval_exec_expression(statement.value, variables)
            continue
        raise ValueError("Only arithmetic expressions and assignments are supported")
    return variables


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

    Creates a tool that represents Python code execution capability. Code execution
    is always disabled for security reasons. The parameters are kept for API
    compatibility but are ignored.

    Args:
        allow_exec: (Deprecated) Ignored. Code execution is always disabled.
        sandbox_contract: (Deprecated) Ignored. Code execution is always disabled.

    Returns:
        Tool: A Python execution tool instance that always returns a safe error message.

    Examples:
        Creating the tool:

            >>> from insideLLMs.contrib.agents import create_python_tool
            >>>
            >>> python_tool = create_python_tool()
            >>> result = python_tool.execute("result = 2 + 2")
            >>> print("disabled" in result.output.lower())
            True

        Using with an agent:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_python_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_python_tool()])
            >>> print(agent.tools[0].name)
            python

    Warning:
        Code execution is permanently disabled due to remote code execution
        vulnerabilities. Python's exec() cannot be safely sandboxed without
        additional process isolation (e.g., containers, separate processes).
        Use create_calculator_tool() for safe mathematical operations.

    See Also:
        :class:`Tool`: The base Tool class.
        :func:`create_calculator_tool`: A safer alternative for math operations.
    """

    def execute_python(code: str) -> str:
        # Code execution is disabled to prevent remote code execution vulnerabilities.
        # Even with restricted builtins, exec() can be bypassed through Python introspection.
        return (
            "Code execution is disabled for security reasons. "
            "Python's exec() cannot be safely sandboxed without additional process isolation."
        )

    return Tool(
        name="python",
        func=execute_python,
        description="Execute Python code (use result variable for output)",
        parameters=[ToolParameter("code", "str", "Python code to execute")],
    )
