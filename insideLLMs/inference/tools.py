"""Typed, allowlisted tool execution with objective feedback."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from typing import Awaitable

from ._callbacks import invoke_with_timeout, is_async_callable


def _backoff_seconds(attempt: int) -> float:
    """Deterministic exponential backoff so retries never hammer a flaky endpoint.

    Kept local rather than delegating to :mod:`insideLLMs.retry`: that engine
    owns its own attempt loop and exception classification, whereas the retry
    here must interleave with a per-attempt deadline and feed
    ``Observation.transport_attempts``. Threading those through RetryConfig
    would couple the two more tightly than the duplication costs.
    """
    return min(0.1 * 2 ** (attempt - 1), 2.0)


ToolCallback = Callable[[dict[str, object]], object | Awaitable[object]]
PolicyCallback = Callable[["ToolAction"], bool | Awaitable[bool]]
PostconditionCallback = Callable[[object], bool | Awaitable[bool]]


class ToolPolicyError(ValueError):
    """Raised before execution when an action violates tool policy.

    Callers may rely on this meaning the tool did **not** run, so it is safe to
    re-dispatch the action. Violations that can only be detected after the tool
    has run raise :class:`ToolOutputTooLarge` instead, which is a subclass so
    existing ``except ToolPolicyError`` handlers keep working.
    """


class ToolOutputTooLarge(ToolPolicyError):
    """Raised after a tool ran when its output exceeded ``max_output_characters``.

    Distinct from its parent because the action **has already executed** and any
    side effects are committed. Treating this as "rejected before execution" and
    retrying a non-idempotent action (a payment, say) would duplicate the effect.
    """


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
    if limits.timeout_seconds is not None:
        # A timed-out sync callback keeps running in its worker thread, so every
        # callback under the timeout must be async, not just the tool itself.
        for role, callback in (
            ("tool", tools[action.tool]),
            ("policy", policy),
            ("postcondition", postcondition),
        ):
            if callback is not None and not is_async_callable(callback):
                raise ToolPolicyError(f"timeouts require an asynchronous {role} callback")
    if policy is not None and not await invoke_with_timeout(
        policy, action, timeout=limits.timeout_seconds
    ):
        raise ToolPolicyError(f"action for {action.tool!r} was rejected by policy")

    output: object | None = None
    attempts = 0
    max_attempts = limits.max_transport_attempts if action.idempotent else 1
    for attempts in range(1, max_attempts + 1):
        attempt_started = time.monotonic()
        try:
            output = await invoke_with_timeout(
                tools[action.tool],
                dict(action.arguments),
                timeout=limits.timeout_seconds,
            )
            break
        except TimeoutError:
            # Distinguish our own per-attempt deadline from a transport read
            # timeout raised by the tool. Retrying our deadline would silently
            # multiply the timeout the caller declared (3 attempts at 5s is a
            # 15s wall clock), so it is never retried.
            if (
                limits.timeout_seconds is not None
                and time.monotonic() - attempt_started >= limits.timeout_seconds
            ):
                raise
            if attempts == max_attempts:
                raise
            await asyncio.sleep(_backoff_seconds(attempts))
        except ConnectionError:
            # Genuine transport faults only. Bare OSError would also cover
            # deterministic local failures (FileNotFoundError, PermissionError,
            # IsADirectoryError) that cannot succeed on retry, so retrying them
            # only delays the same error behind exponential backoff.
            if attempts == max_attempts:
                raise
            await asyncio.sleep(_backoff_seconds(attempts))

    if limits.max_output_characters is not None and len(str(output)) > limits.max_output_characters:
        raise ToolOutputTooLarge(
            "tool output exceeded max_output_characters (the tool already ran; "
            "do not re-dispatch a non-idempotent action)"
        )
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
