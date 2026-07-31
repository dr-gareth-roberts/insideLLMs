import pytest

from insideLLMs.inference.tools import (
    ToolAction,
    ToolLimits,
    ToolPolicyError,
    execute_tool,
)


async def test_typed_tool_execution_retries_transport_and_checks_postcondition() -> None:
    attempts = 0

    async def calculator(arguments: dict[str, object]) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary transport failure")
        return int(arguments["left"]) + int(arguments["right"])

    observation = await execute_tool(
        ToolAction("calculator", {"left": 2, "right": 2}, idempotent=True),
        tools={"calculator": calculator},
        allowed_tools={"calculator"},
        limits=ToolLimits(max_transport_attempts=2),
        postcondition=lambda value: value == 4,
    )

    assert observation.output == 4
    assert observation.postcondition_passed is True
    assert observation.transport_attempts == 2


async def test_typed_tool_execution_rejects_non_allowlisted_action_without_running_it() -> None:
    ran = False

    def forbidden(arguments: dict[str, object]) -> object:
        nonlocal ran
        ran = True
        return "unsafe"

    with pytest.raises(ToolPolicyError, match="not allowlisted"):
        await execute_tool(
            ToolAction("shell", {"command": "anything"}),
            tools={"shell": forbidden},
            allowed_tools={"calculator"},
            limits=ToolLimits(),
        )

    assert ran is False


async def test_non_idempotent_tool_is_not_retried_after_transport_failure() -> None:
    attempts = 0

    def charge(arguments: dict[str, object]) -> object:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("result unknown")

    with pytest.raises(TimeoutError, match="result unknown"):
        await execute_tool(
            ToolAction("charge", {"amount": 10}),
            tools={"charge": charge},
            allowed_tools={"charge"},
            limits=ToolLimits(max_transport_attempts=3),
        )

    assert attempts == 1


async def test_typed_tool_execution_awaits_async_policy_before_authorization() -> None:
    ran = False

    async def deny(action: ToolAction) -> bool:
        return False

    async def tool(arguments: dict[str, object]) -> object:
        nonlocal ran
        ran = True
        return "unsafe"

    with pytest.raises(ToolPolicyError, match="rejected by policy"):
        await execute_tool(
            ToolAction("lookup", {}),
            tools={"lookup": tool},
            allowed_tools={"lookup"},
            limits=ToolLimits(),
            policy=deny,
        )

    assert ran is False


async def test_sync_tool_with_timeout_is_rejected_before_execution() -> None:
    ran = False

    def tool(arguments: dict[str, object]) -> object:
        nonlocal ran
        ran = True
        return "late side effect"

    with pytest.raises(ToolPolicyError, match="asynchronous"):
        await execute_tool(
            ToolAction("write", {}),
            tools={"write": tool},
            allowed_tools={"write"},
            limits=ToolLimits(timeout_seconds=0.01),
        )

    assert ran is False
