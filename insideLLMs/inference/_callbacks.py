"""Internal helpers for accepting plain and asynchronous callbacks."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

T = TypeVar("T")


def budget_elapsed(started: float, max_seconds: float | None) -> bool:
    """True when the wall-clock budget armed at ``started`` has actually run out.

    This is the discriminator every budgeted strategy needs when it catches a
    ``TimeoutError``: only a deadline that genuinely elapsed means the budget is
    exhausted. A callback raising its own ``TimeoutError`` early — a provider or
    network timeout, say — must propagate unchanged rather than be relabelled as
    budget exhaustion, which would both hide the real fault and misreport why the
    run stopped.

    Checking ``max_seconds is None`` alone is not sufficient: with a budget armed
    the old check treated *any* ``TimeoutError`` as exhaustion even when nearly
    the whole budget remained. ``invoke_with_timeout`` only raises after the
    remaining budget has passed, so recomputing it here is exact and needs no
    tolerance.
    """
    if max_seconds is None:
        return False
    return (max_seconds - (time.monotonic() - started)) <= 0


def is_async_callable(callback: object) -> bool:
    """True for coroutine functions and objects with an ``async def __call__``."""

    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


async def gather_cancelling(*awaitables: Awaitable[T]) -> list[T]:
    """``asyncio.gather`` that cancels and drains siblings when one child fails.

    Bare ``gather`` propagates the first exception immediately while every other
    child keeps running. For model-backed callbacks that means a failed
    verification or rerank leaves the remaining provider calls executing
    unobserved after the caller has already raised, and a second failure
    surfaces only as a "Task exception was never retrieved" warning at garbage
    collection.
    """
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def resolve(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


async def invoke(callback: Callable[..., T | Awaitable[T]], *args: object) -> T:
    """Call async callbacks directly and keep synchronous callbacks off the loop."""

    if is_async_callable(callback):
        async_callback = cast(Callable[..., Awaitable[T]], callback)
        return await async_callback(*args)
    return await resolve(await asyncio.to_thread(callback, *args))


async def invoke_with_timeout(
    callback: Callable[..., T | Awaitable[T]],
    *args: object,
    timeout: float | None,
) -> T:
    # Deadline enforcement (and the pre-3.11 asyncio.TimeoutError normalization)
    # lives in the shared async layer so every layer raises one timeout type.
    from insideLLMs.async_utils import wait_for

    return await wait_for(invoke(callback, *args), timeout)
