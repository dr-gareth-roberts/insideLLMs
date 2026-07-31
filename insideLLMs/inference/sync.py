"""Safe synchronous entry point for async-first inference APIs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_sync(awaitable: Awaitable[T]) -> T:
    """Run outside an event loop; async callers must await the underlying API."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if inspect.iscoroutine(awaitable):
            return asyncio.run(awaitable)

        # asyncio.run only accepts coroutines; wrap Tasks/Futures/custom
        # __await__ objects so the declared Awaitable contract holds.
        async def _await() -> T:
            return await awaitable

        return asyncio.run(_await())
    if inspect.iscoroutine(awaitable):
        awaitable.close()
    raise RuntimeError("run_sync cannot run inside an event loop; await the async API instead")
