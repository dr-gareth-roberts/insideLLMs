"""Timeout wrapper for async probe execution.

This module provides timeout handling for probe execution to prevent
indefinite hangs in production environments.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from insideLLMs.exceptions import ProbeExecutionError

logger = logging.getLogger(__name__)


async def run_with_timeout(
    coro_func: Callable[[], Any],
    timeout: Optional[float] = None,
    context: Optional[dict[str, Any]] = None,
) -> Any:
    """Execute a coroutine with optional timeout.

    Args:
        coro_func: Callable that returns a coroutine
        timeout: Timeout in seconds (None = no timeout)
        context: Additional context for error messages

    Returns:
        Result of the coroutine

    Raises:
        ProbeExecutionError: If timeout exceeded
    """
    if timeout is None:
        return await coro_func()

    try:
        return await asyncio.wait_for(coro_func(), timeout=timeout)
    except asyncio.TimeoutError:
        context_str = f" (context: {context})" if context else ""
        logger.error(
            f"Probe execution timed out after {timeout}s{context_str}", extra=context or {}
        )
        raise ProbeExecutionError(
            f"Probe execution timed out after {timeout}s", details=context or {}
        )
