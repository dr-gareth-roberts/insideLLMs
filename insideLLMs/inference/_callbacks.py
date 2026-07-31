"""Internal helpers for accepting plain and asynchronous callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

T = TypeVar("T")


async def resolve(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


async def invoke(callback: Callable[..., T | Awaitable[T]], *args: object) -> T:
    """Call async callbacks directly and keep synchronous callbacks off the loop."""

    if inspect.iscoroutinefunction(callback):
        async_callback = cast(Callable[..., Awaitable[T]], callback)
        return await async_callback(*args)
    return await resolve(await asyncio.to_thread(callback, *args))


async def invoke_with_timeout(
    callback: Callable[..., T | Awaitable[T]],
    *args: object,
    timeout: float | None,
) -> T:
    pending = invoke(callback, *args)
    if timeout is None:
        return await pending
    try:
        return await asyncio.wait_for(pending, timeout)
    except asyncio.TimeoutError as error:
        # On Python < 3.11 asyncio.TimeoutError is not the builtin TimeoutError;
        # normalize so callers can catch TimeoutError on every supported version.
        raise TimeoutError(str(error)) from error
