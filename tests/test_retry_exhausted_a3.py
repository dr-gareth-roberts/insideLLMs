"""A3: exhaustion paths always carry a captured retryable exception."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from insideLLMs.retry import (
    BackoffStrategy,
    RateLimitError,
    RetryConfig,
    RetryExhaustedError,
    execute_with_retry,
    execute_with_retry_async,
)


def test_sync_exhaustion_preserves_last_exception() -> None:
    cfg = RetryConfig(
        max_retries=0, strategy=BackoffStrategy.CONSTANT, initial_delay=0.0, jitter=False
    )

    def boom():
        raise RateLimitError("m", retry_after=0.0)

    with pytest.raises(RetryExhaustedError) as ei:
        execute_with_retry(boom, (), {}, cfg)
    assert isinstance(ei.value.last_exception, RateLimitError)


def test_async_exhaustion_preserves_last_exception() -> None:
    import asyncio

    cfg = RetryConfig(
        max_retries=0, strategy=BackoffStrategy.CONSTANT, initial_delay=0.0, jitter=False
    )

    async def boom():
        raise RateLimitError("m", retry_after=0.0)

    async def _run():
        with pytest.raises(RetryExhaustedError) as ei:
            await execute_with_retry_async(boom, (), {}, cfg)
        assert isinstance(ei.value.last_exception, RateLimitError)

    asyncio.run(_run())
