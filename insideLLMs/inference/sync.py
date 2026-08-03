"""Safe synchronous entry point for async-first inference APIs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_sync(awaitable: Awaitable[T]) -> T:
    """Run outside an event loop; async callers must await the underlying API.

    Deliberately *not* consolidated with :func:`insideLLMs.async_utils.run_async`,
    which the two look like duplicates of. Called from inside a running loop,
    ``run_async`` applies ``nest_asyncio`` to re-enter that loop; this function
    refuses instead, because loop re-entrancy is a global monkey-patch with an
    optional third-party dependency and inference must not require either. The
    contracts differ on purpose: use ``run_async`` where nesting is acceptable,
    and this where a clear failure is preferred over silently nesting.
    """

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
