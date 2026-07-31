"""Internal helpers for accepting plain and asynchronous callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

T = TypeVar("T")


def is_async_callable(callback: object) -> bool:
    """True for coroutine functions and objects with an ``async def __call__``."""

    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


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
