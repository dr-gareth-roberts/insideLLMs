"""Validated plan DAG execution with parallel ready branches."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Awaitable

from ._callbacks import invoke_with_timeout
from .schemas import Budget, InferenceResult, Spend, TraceEvent


class DagBudgetExceeded(RuntimeError):
    """Raised before a DAG level would exceed a hard budget."""


@dataclass(frozen=True)
class PlanNode:
    id: str
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None


async def execute_dag(
    nodes: Sequence[PlanNode],
    *,
    execute: Callable[[PlanNode, dict[str, object]], object | Awaitable[object]],
    reduce: Callable[[dict[str, object]], object | Awaitable[object]],
    checkpoint: Mapping[str, object] | None = None,
    max_concurrency: int | None = None,
    budget: Budget = Budget(),
    compact_observation: Callable[[object], object] = lambda value: value,
    observation_tokens: Callable[[object], int] = lambda value: len(str(value).split()),
) -> InferenceResult:
    """Execute topological levels concurrently and reduce stable keyed observations."""

    by_id = {node.id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("plan node ids must be unique")
    missing = {
        dependency for node in nodes for dependency in node.dependencies if dependency not in by_id
    }
    if missing:
        raise ValueError(f"unknown dependencies: {sorted(missing)}")
    checkpoint_keys = set(checkpoint or {})
    unknown_checkpoint_keys = checkpoint_keys - set(by_id)
    if unknown_checkpoint_keys:
        raise ValueError(f"unknown checkpoint node ids: {sorted(unknown_checkpoint_keys)}")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if budget.max_seconds is not None and not inspect.iscoroutinefunction(execute):
        raise ValueError("DAG time budgets require an asynchronous execute callback")

    # Validate the complete plan independently of checkpoint state. A cached
    # observation must never be able to hide a cycle in the submitted plan.
    remaining_dependencies = {node.id: set(node.dependencies) for node in nodes}
    ready_to_validate = sorted(
        node_id for node_id, dependencies in remaining_dependencies.items() if not dependencies
    )
    validated: set[str] = set()
    while ready_to_validate:
        node_id = ready_to_validate.pop(0)
        if node_id in validated:
            continue
        validated.add(node_id)
        for dependent_id, dependencies in remaining_dependencies.items():
            if node_id in dependencies:
                dependencies.remove(node_id)
                if not dependencies and dependent_id not in validated:
                    ready_to_validate.append(dependent_id)
        ready_to_validate.sort()
    if len(validated) != len(nodes):
        raise ValueError("plan contains a dependency cycle")
    observations = dict(checkpoint or {})
    used_tokens = sum(observation_tokens(value) for value in observations.values())
    if budget.max_tokens is not None and used_tokens > budget.max_tokens:
        raise DagBudgetExceeded("DAG checkpoint token budget would be exceeded")
    started = time.monotonic()
    pending = {node.id for node in nodes if node.id not in observations}
    levels: list[tuple[str, ...]] = []
    trace: list[TraceEvent] = []
    semaphore = asyncio.Semaphore(max_concurrency or max(1, len(nodes)))

    async def run_node(node: PlanNode) -> tuple[str, object]:
        dependencies = {key: observations[key] for key in node.dependencies}
        async with semaphore:
            remaining = (
                None
                if budget.max_seconds is None
                else budget.max_seconds - (time.monotonic() - started)
            )
            if remaining is not None and remaining <= 0:
                raise DagBudgetExceeded("DAG time budget exhausted")
            try:
                value = await invoke_with_timeout(execute, node, dependencies, timeout=remaining)
            except TimeoutError as error:
                raise DagBudgetExceeded("DAG time budget exhausted") from error
        return node.id, value

    while pending:
        ready = tuple(
            sorted(
                node_id
                for node_id in pending
                if all(dependency in observations for dependency in by_id[node_id].dependencies)
            )
        )
        if not ready:
            raise ValueError("plan contains a dependency cycle")
        if budget.max_calls is not None and len(trace) + len(ready) + 1 > budget.max_calls:
            raise DagBudgetExceeded("DAG call budget would be exceeded")
        if budget.max_nodes is not None and len(trace) + len(ready) > budget.max_nodes:
            raise DagBudgetExceeded("DAG node budget would be exceeded")
        if inspect.iscoroutinefunction(execute):
            tasks = [asyncio.create_task(run_node(by_id[node_id])) for node_id in ready]
            try:
                completed = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        else:
            completed = [await run_node(by_id[node_id]) for node_id in ready]
        levels.append(ready)
        for node_id, value in completed:
            used_tokens += observation_tokens(value)
            if budget.max_tokens is not None and used_tokens > budget.max_tokens:
                raise DagBudgetExceeded("DAG token budget would be exceeded")
            observations[node_id] = value
            pending.remove(node_id)
            trace.append(
                TraceEvent(
                    id=node_id,
                    kind="plan-node",
                    parent_ids=by_id[node_id].dependencies,
                    calls=1,
                )
            )

    remaining = (
        None if budget.max_seconds is None else budget.max_seconds - (time.monotonic() - started)
    )
    if remaining is not None and remaining <= 0:
        raise DagBudgetExceeded("DAG time budget exhausted before reduction")
    if budget.max_calls is not None and len(trace) + 1 > budget.max_calls:
        raise DagBudgetExceeded("DAG call budget would be exceeded before reduction")
    try:
        answer = await invoke_with_timeout(reduce, dict(observations), timeout=remaining)
    except TimeoutError as error:
        raise DagBudgetExceeded("DAG time budget exhausted during reduction") from error
    return InferenceResult(
        answer=answer,
        trace=tuple(trace),
        spend=Spend(calls=len(trace) + 1),
        provenance={
            "levels": tuple(levels),
            "observations": {
                key: compact_observation(value) for key, value in observations.items()
            },
        },
    )
