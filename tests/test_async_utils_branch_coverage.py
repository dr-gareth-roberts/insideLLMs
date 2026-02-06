"""Additional branch coverage for async_utils."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

import insideLLMs.async_utils as async_utils
from insideLLMs.async_utils import (
    AsyncProgress,
    async_timeout,
    first_completed,
    for_each_async,
    map_async,
    map_async_ordered,
    retry_until_success,
    run_async,
)


@pytest.mark.asyncio
async def test_map_async_return_exceptions_path():
    async def fail_on_two(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    result = await map_async(fail_on_two, [1, 2, 3], return_exceptions=True)
    assert isinstance(result.results[1], Exception)
    assert result.failed == 1


@pytest.mark.asyncio
async def test_map_async_ordered_exception_tuple_path():
    async def maybe_fail(x: int) -> int:
        if x == 1:
            raise RuntimeError("bad item")
        return x * 2

    seen = []
    async for idx, value, error in map_async_ordered(maybe_fail, [0, 1, 2]):
        seen.append((idx, value, error))

    assert seen[1][1] is None
    assert isinstance(seen[1][2], RuntimeError)


@pytest.mark.asyncio
async def test_for_each_async_stop_flag_short_circuit_branch():
    calls = []

    async def worker(x: int) -> None:
        calls.append(x)
        await asyncio.sleep(0.001 * x)
        if x == 1:
            raise ValueError("stop")

    errors = await for_each_async(worker, [1, 2, 3, 4], stop_on_error=True)
    assert errors
    assert 1 in calls


def test_async_progress_edge_properties_before_start_and_zero_total():
    progress = AsyncProgress(total=0)
    assert progress.elapsed_time == 0.0
    assert progress.items_per_second == 0.0
    assert progress.estimated_remaining == float("inf")
    assert progress.percent_complete == 100.0


@pytest.mark.asyncio
async def test_first_completed_error_cancels_remaining_tasks():
    async def fail_fast():
        await asyncio.sleep(0.01)
        raise RuntimeError("fast fail")

    async def slow():
        await asyncio.sleep(1)
        return "slow"

    with pytest.raises(RuntimeError, match="fast fail"):
        await first_completed([fail_fast(), slow()])


@pytest.mark.asyncio
async def test_retry_until_success_no_coroutines_raises_value_error():
    with pytest.raises(ValueError, match="No coroutines provided"):
        await retry_until_success([])


@pytest.mark.asyncio
async def test_async_timeout_handles_missing_current_task(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(asyncio, "current_task", lambda: None)
    async with async_timeout(0.01):
        await asyncio.sleep(0)


def test_run_async_without_running_loop_branch():
    async def coro() -> str:
        return "ok"

    assert run_async(coro()) == "ok"


def test_run_async_with_running_loop_branch(monkeypatch: pytest.MonkeyPatch):
    async def coro() -> int:
        await asyncio.sleep(0)
        return 42

    class FakeRunningLoop:
        def run_until_complete(self, task):
            task.close()
            return 42

    monkeypatch.setitem(sys.modules, "nest_asyncio", types.SimpleNamespace(apply=lambda: None))
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: FakeRunningLoop())

    assert run_async(coro()) == 42
