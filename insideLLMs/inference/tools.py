"""Typed, allowlisted tool execution with objective feedback."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from typing import Awaitable

from ._callbacks import invoke_with_timeout

ToolCallback = Callable[[dict[str, object]], object | Awaitable[object]]
PolicyCallback = Callable[["ToolAction"], bool | Awaitable[bool]]
PostconditionCallback = Callable[[object], bool | Awaitable[bool]]


class ToolPolicyError(ValueError):
    """Raised before execution when an action violates tool policy."""


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: dict[str, object]
    idempotent: bool = False


@dataclass(frozen=True)
class ToolLimits:
    max_transport_attempts: int = 1
    timeout_seconds: float | None = None
    max_output_characters: int | None = None


@dataclass(frozen=True)
class Observation:
    output: object
    postcondition_passed: bool
    transport_attempts: int


async def execute_tool(
    action: ToolAction,
    *,
    tools: Mapping[str, ToolCallback],
    allowed_tools: Set[str],
    limits: ToolLimits,
    policy: PolicyCallback | None = None,
    postcondition: PostconditionCallback | None = None,
) -> Observation:
    """Execute one registered action; no command or shell adapter is provided."""

    if limits.max_transport_attempts < 1:
        raise ValueError("max_transport_attempts must be positive")
    if action.tool not in allowed_tools:
        raise ToolPolicyError(f"tool {action.tool!r} is not allowlisted")
    if action.tool not in tools:
        raise ToolPolicyError(f"tool {action.tool!r} is not registered")
    if limits.timeout_seconds is not None and not inspect.iscoroutinefunction(tools[action.tool]):
        raise ToolPolicyError("timeouts require an asynchronous tool callback")
    if policy is not None and not await invoke_with_timeout(
        policy, action, timeout=limits.timeout_seconds
    ):
        raise ToolPolicyError(f"action for {action.tool!r} was rejected by policy")

    output: object | None = None
    attempts = 0
    max_attempts = limits.max_transport_attempts if action.idempotent else 1
    for attempts in range(1, max_attempts + 1):
        try:
            output = await invoke_with_timeout(
                tools[action.tool],
                dict(action.arguments),
                timeout=limits.timeout_seconds,
            )
            break
        except (ConnectionError, OSError, TimeoutError):
            if attempts == max_attempts:
                raise

    if limits.max_output_characters is not None and len(str(output)) > limits.max_output_characters:
        raise ToolPolicyError("tool output exceeded max_output_characters")
    passed = (
        bool(await invoke_with_timeout(postcondition, output, timeout=limits.timeout_seconds))
        if postcondition is not None
        else True
    )
    return Observation(
        output=output,
        postcondition_passed=passed,
        transport_attempts=attempts,
    )
