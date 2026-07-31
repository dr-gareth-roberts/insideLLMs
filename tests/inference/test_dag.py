import asyncio

import pytest

from insideLLMs.inference import Budget
from insideLLMs.inference.dag import DagBudgetExceeded, PlanNode, execute_dag


async def test_plan_dag_runs_ready_branches_concurrently_then_reduces_deterministically() -> None:
    both_started = asyncio.Event()
    started: set[str] = set()

    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        if node.id in {"a", "b"}:
            started.add(node.id)
            if started == {"a", "b"}:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            return node.id.upper()
        return f"{dependencies['a']}+{dependencies['b']}"

    result = await execute_dag(
        (
            PlanNode("a"),
            PlanNode("b"),
            PlanNode("merge", dependencies=("a", "b")),
        ),
        execute=execute,
        reduce=lambda observations: observations["merge"],
    )

    assert result.answer == "A+B"
    assert result.provenance["levels"] == (("a", "b"), ("merge",))
    assert result.provenance["observations"] == {"a": "A", "b": "B", "merge": "A+B"}
    assert [event.parent_ids for event in result.trace] == [(), (), ("a", "b")]


async def test_plan_dag_fails_before_starting_a_level_that_exceeds_call_budget() -> None:
    calls: list[str] = []

    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        calls.append(node.id)
        return node.id

    with pytest.raises(DagBudgetExceeded, match="call budget"):
        await execute_dag(
            (PlanNode("a"), PlanNode("b")),
            execute=execute,
            reduce=lambda observations: observations,
            budget=Budget(max_calls=1),
        )

    assert calls == []


async def test_plan_dag_applies_token_budget_before_reduction() -> None:
    reduced = False

    def reduce(observations: dict[str, object]) -> object:
        nonlocal reduced
        reduced = True
        return observations

    with pytest.raises(DagBudgetExceeded, match="token budget"):
        await execute_dag(
            (PlanNode("a"),),
            execute=lambda node, dependencies: "one token",
            reduce=reduce,
            budget=Budget(max_tokens=1),
            observation_tokens=lambda value: len(str(value).split()),
        )

    assert reduced is False


async def test_plan_dag_deadline_covers_reduction() -> None:
    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        return "done"

    async def slow_reduce(observations: dict[str, object]) -> object:
        await asyncio.sleep(0.05)
        return observations

    with pytest.raises(DagBudgetExceeded, match="time budget"):
        await execute_dag(
            (PlanNode("a"),),
            execute=execute,
            reduce=slow_reduce,
            budget=Budget(max_seconds=0.01),
        )


async def test_plan_dag_preflights_checkpoint_token_budget() -> None:
    reduced = False

    def reduce(observations: dict[str, object]) -> object:
        nonlocal reduced
        reduced = True
        return observations

    with pytest.raises(DagBudgetExceeded, match="checkpoint token budget"):
        await execute_dag(
            (PlanNode("a"),),
            execute=lambda node, dependencies: "unused",
            reduce=reduce,
            checkpoint={"a": "two words"},
            budget=Budget(max_tokens=1),
            observation_tokens=lambda value: len(str(value).split()),
        )

    assert reduced is False


async def test_checkpoint_cannot_mask_plan_cycle() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        await execute_dag(
            (PlanNode("a", dependencies=("b",)), PlanNode("b", dependencies=("a",))),
            execute=lambda node, dependencies: node.id,
            reduce=lambda observations: observations,
            checkpoint={"a": "saved"},
        )


async def test_plan_dag_reserves_call_budget_for_reduction() -> None:
    executed = False

    def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        nonlocal executed
        executed = True
        return node.id

    with pytest.raises(DagBudgetExceeded, match="call budget"):
        await execute_dag(
            (PlanNode("a"),),
            execute=execute,
            reduce=lambda observations: observations,
            budget=Budget(max_calls=1),
        )

    assert executed is False


async def test_plan_dag_cancels_and_awaits_siblings_after_failure() -> None:
    side_effects: list[str] = []

    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        if node.id == "fail":
            raise RuntimeError("boom")
        await asyncio.sleep(0.05)
        side_effects.append(node.id)
        return node.id

    with pytest.raises(RuntimeError, match="boom"):
        await execute_dag(
            (PlanNode("fail"), PlanNode("slow")),
            execute=execute,
            reduce=lambda observations: observations,
        )
    await asyncio.sleep(0.08)

    assert side_effects == []


async def test_sync_dag_level_runs_sequentially_so_failure_leaves_no_sibling() -> None:
    side_effects: list[str] = []

    def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        if node.id == "fail":
            raise RuntimeError("boom")
        side_effects.append(node.id)
        return node.id

    with pytest.raises(RuntimeError, match="boom"):
        await execute_dag(
            (PlanNode("fail"), PlanNode("slow")),
            execute=execute,
            reduce=lambda observations: observations,
        )

    assert side_effects == []
