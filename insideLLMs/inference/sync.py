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
        return asyncio.run(awaitable)
    if inspect.iscoroutine(awaitable):
        awaitable.close()
    raise RuntimeError("run_sync cannot run inside an event loop; await the async API instead")
