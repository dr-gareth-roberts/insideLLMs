"""Token-bucket rate-limit middleware."""

import asyncio
import threading
import time
from typing import Any, Optional

from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import AsyncModelProtocol
from insideLLMs.runtime._pipeline.middleware import Middleware


class RateLimitMiddleware(Middleware):
    """Middleware for rate limiting model requests using a token bucket algorithm.

    Enforces a maximum request rate to prevent exceeding API rate limits or
    to control costs. The token bucket algorithm allows for configurable burst
    behavior: requests can be made at a higher rate temporarily if tokens have
    accumulated during idle periods.

    The algorithm works as follows:
    1. Tokens are added to the bucket at a steady rate (requests_per_minute / 60)
    2. Each request consumes one token
    3. If no tokens are available, the request waits until one becomes available
    4. The bucket has a maximum capacity (burst_size) to limit bursts

    Parameters
    ----------
    requests_per_minute : int, default=60
        Maximum sustained request rate. Internally converted to requests
        per second for finer-grained control.
    burst_size : int, optional
        Maximum number of requests that can be made in a burst when tokens
        have accumulated. Defaults to requests_per_minute (allowing a full
        minute's worth of requests to be made immediately after idle time).

    Attributes
    ----------
    rate : float
        The token refill rate in tokens per second.
    burst_size : int
        The maximum number of tokens the bucket can hold.
    tokens : float
        Current number of available tokens.
    last_update : float
        Timestamp of the last token update.

    Examples
    --------
    Basic rate limiting:

        >>> from insideLLMs.runtime.pipeline import RateLimitMiddleware, ModelPipeline
        >>>
        >>> # Allow 60 requests per minute (1 per second)
        >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)
        >>> pipeline = ModelPipeline(model, middlewares=[rate_mw])
        >>>
        >>> # This will process at ~1 request per second
        >>> for prompt in prompts:
        ...     response = pipeline.generate(prompt)  # May wait if too fast

    High burst allowance:

        >>> # Allow 30/minute but permit bursts of up to 10 requests
        >>> rate_mw = RateLimitMiddleware(
        ...     requests_per_minute=30,
        ...     burst_size=10
        ... )
        >>>
        >>> # First 10 requests execute immediately
        >>> # Subsequent requests are spaced at 2 seconds apart

    Strict rate limiting (no bursts):

        >>> # Exactly 1 request per second, no accumulation
        >>> rate_mw = RateLimitMiddleware(
        ...     requests_per_minute=60,
        ...     burst_size=1
        ... )

    Async rate limiting with concurrent requests:

        >>> async def rate_limited_batch():
        ...     rate_mw = RateLimitMiddleware(requests_per_minute=120)
        ...     pipeline = AsyncModelPipeline(
        ...         model,
        ...         middlewares=[rate_mw]
        ...     )
        ...
        ...     # These will be rate limited even with concurrency
        ...     results = await pipeline.abatch_generate(
        ...         prompts,
        ...         max_concurrency=10
        ...     )
        ...     return results

    Combining with other middleware:

        >>> # Place rate limiting before retry to count retries correctly
        >>> pipeline = ModelPipeline(
        ...     model,
        ...     middlewares=[
        ...         CacheMiddleware(),  # Cache hits bypass rate limit
        ...         RateLimitMiddleware(requests_per_minute=60),
        ...         RetryMiddleware(max_retries=3),  # Retries consume tokens
        ...     ],
        ... )

    Notes
    -----
    - The token bucket algorithm provides smooth rate limiting with controlled
      burst behavior, unlike simple time-window approaches.
    - Async operations use an asyncio.Lock to ensure thread-safe token access
      across concurrent requests.
    - The middleware waits (sleeps) when rate limited rather than raising an
      exception, ensuring requests eventually complete.
    - Place CacheMiddleware before RateLimitMiddleware so cache hits don't
      consume rate limit tokens.

    See Also
    --------
    Middleware : The base middleware class
    RetryMiddleware : Often used together with rate limiting
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ) -> None:
        """Initialize the rate limiter with rate and burst configuration.

        Args
        ----
        requests_per_minute : int, default=60
            Maximum sustained request rate. This is the average rate that
            can be maintained over time.
        burst_size : int, optional
            Maximum burst capacity. If not specified, defaults to
            requests_per_minute, allowing accumulated tokens from idle
            time to be used in a burst.

        Examples
        --------
        Standard rate limiting:

            >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)

        Strict rate limiting (no bursts):

            >>> rate_mw = RateLimitMiddleware(
            ...     requests_per_minute=30,
            ...     burst_size=1
            ... )

        High burst tolerance:

            >>> rate_mw = RateLimitMiddleware(
            ...     requests_per_minute=10,
            ...     burst_size=50
            ... )
        """
        super().__init__()
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        if burst_size is not None and burst_size <= 0:
            raise ValueError("burst_size must be > 0")
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst_size = burst_size if burst_size is not None else requests_per_minute
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self._sync_lock = threading.Lock()

    def _acquire_token(self) -> None:
        """Acquire a token from the bucket, waiting if necessary.

        Updates the token count based on elapsed time, then either
        consumes a token immediately (if available) or waits until
        a token becomes available.

        This method modifies the token state and may block (sleep)
        if rate limited.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)
            >>> rate_mw._acquire_token()  # May wait if no tokens
        """
        with self._sync_lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 1.0
                self.last_update = time.time()

            self.tokens -= 1.0

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a generate request with rate limiting.

        Acquires a rate limit token (waiting if necessary) before
        delegating to the next middleware or model.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated response.

        Raises
        ------
        ModelError
            If generation fails.

        Examples
        --------
        Rate-limited generation:

            >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)
            >>> rate_mw.model = model
            >>>
            >>> # First request: immediate (tokens available)
            >>> r1 = rate_mw.process_generate("Q1")
            >>>
            >>> # Rapid follow-up: may wait for token
            >>> r2 = rate_mw.process_generate("Q2")
        """
        self._acquire_token()

        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available in pipeline")

    async def _aacquire_token(self) -> None:
        """Asynchronously acquire a token, waiting if necessary.

        The async counterpart to _acquire_token. Uses asyncio.sleep for
        non-blocking waits and an asyncio.Lock for thread-safe token
        access across concurrent requests.

        Examples
        --------
        Internal async usage:

            >>> async def example():
            ...     rate_mw = RateLimitMiddleware(requests_per_minute=60)
            ...     await rate_mw._aacquire_token()
        """
        async with self._get_lock():
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 1.0
                self.last_update = time.time()

            self.tokens -= 1.0

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an asyncio.Lock for thread-safe token access.

        Lazily creates the lock on first access to avoid issues with
        asyncio.Lock creation outside of async contexts.

        Returns
        -------
        asyncio.Lock
            The lock instance for synchronizing token access.
        """
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a generate request with rate limiting.

        The async counterpart to process_generate. Acquires a token using
        async-safe methods before delegating to the next middleware or model.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated response.

        Raises
        ------
        ModelError
            If generation fails.

        Examples
        --------
        Async rate-limited generation:

            >>> async def example():
            ...     rate_mw = RateLimitMiddleware(requests_per_minute=120)
            ...     rate_mw.model = async_model
            ...
            ...     # Concurrent but rate-limited
            ...     tasks = [
            ...         rate_mw.aprocess_generate(f"Question {i}")
            ...         for i in range(10)
            ...     ]
            ...     results = await asyncio.gather(*tasks)
            ...     return results
        """
        await self._aacquire_token()

        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")
