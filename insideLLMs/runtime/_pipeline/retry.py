"""Transient-failure retry middleware."""

import asyncio
import time
from typing import Any, Optional

from insideLLMs.exceptions import ModelError, ModelTimeoutError, RateLimitError
from insideLLMs.models.base import AsyncModelProtocol
from insideLLMs.runtime._pipeline.middleware import Middleware


class RetryMiddleware(Middleware):
    """Middleware for retrying failed requests with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        """Initialize the retry middleware."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_count = 0
        self.total_retries = 0

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)
        # Add jitter
        jitter = random.uniform(0, delay * 0.1)  # noqa: S311
        return delay + jitter

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Retry on failure with exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.next_middleware:
                    return self.next_middleware.process_generate(prompt, **kwargs)
                if self.model:
                    return self.model.generate(prompt, **kwargs)
                raise ModelError("No model available in pipeline")
            except (RateLimitError, ModelTimeoutError, ModelError) as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.retry_count += 1
                    self.total_retries += 1
                    time.sleep(delay)
                    continue

                # Max retries exceeded
                break

        # All retries failed
        raise ModelError(
            f"Failed after {self.max_retries} retries", details={"original_error": str(last_error)}
        )

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async retry on failure with exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.next_middleware:
                    return await self.next_middleware.aprocess_generate(prompt, **kwargs)
                if self.model:
                    if isinstance(self.model, AsyncModelProtocol):
                        return await self.model.agenerate(prompt, **kwargs)
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, lambda: self.model.generate(prompt, **kwargs)
                    )
                raise ModelError("No model available in pipeline")
            except (RateLimitError, ModelTimeoutError, ModelError) as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.retry_count += 1
                    self.total_retries += 1
                    await asyncio.sleep(delay)
                    continue

                # Max retries exceeded
                break

        # All retries failed
        raise ModelError(
            f"Failed after {self.max_retries} retries", details={"original_error": str(last_error)}
        )
