"""Model pipeline with composable middleware for advanced model capabilities.

This module implements the proposed architecture from ARCHITECTURE.md, providing
a clean way to compose model capabilities like caching, rate limiting, retry logic,
cost tracking, and more through a middleware pattern.

Supports both synchronous and asynchronous execution patterns.

Example (sync):
    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.pipeline import (
    ...     ModelPipeline,
    ...     CacheMiddleware,
    ...     RateLimitMiddleware,
    ...     RetryMiddleware,
    ...     CostTrackingMiddleware,
    ... )
    >>>
    >>> base_model = OpenAIModel(model_name="gpt-4")
    >>> pipeline = ModelPipeline(
    ...     base_model,
    ...     middlewares=[
    ...         CacheMiddleware(),
    ...         RateLimitMiddleware(requests_per_minute=60),
    ...         RetryMiddleware(max_retries=3),
    ...         CostTrackingMiddleware(),
    ...     ],
    ... )
    >>> response = pipeline.generate("Explain transformers")

Example (async):
    >>> async def main():
    ...     pipeline = AsyncModelPipeline(base_model, middlewares=[...])
    ...     response = await pipeline.agenerate("Explain transformers")
    ...     batch = await pipeline.abatch_generate(["Q1", "Q2", "Q3"])
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional

from insideLLMs.exceptions import ModelError, RateLimitError, TimeoutError
from insideLLMs.models.base import (
    AsyncModelProtocol,
    ChatMessage,
    Model,
    ModelProtocol,
)

if TYPE_CHECKING:
    from insideLLMs.tracing import TraceRecorder


class Middleware(ABC):
    """Base class for model pipeline middleware.

    Middleware can intercept and modify requests/responses, add side effects,
    or prevent execution entirely (e.g., cache hits, rate limiting).

    Middleware is executed in the order it's added to the pipeline for requests,
    and in reverse order for responses.
    """

    def __init__(self):
        """Initialize the middleware."""
        self.next_middleware: Optional["Middleware"] = None
        self.model: Optional[ModelProtocol] = None

    @abstractmethod
    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a generate request.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Returns:
            The generated response.

        Raises:
            ModelError: If generation fails.
        """
        raise NotImplementedError

    def process_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Process a chat request.

        Args:
            messages: The conversation history.
            **kwargs: Additional arguments for chat.

        Returns:
            The chat response.

        Raises:
            ModelError: If chat fails.
        """
        # Default implementation delegates to next middleware or model
        if self.next_middleware:
            return self.next_middleware.process_chat(messages, **kwargs)
        if self.model and hasattr(self.model, "chat"):
            return self.model.chat(messages, **kwargs)
        raise ModelError("No chat implementation available")

    def process_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Process a streaming request.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for streaming.

        Yields:
            Response chunks.

        Raises:
            ModelError: If streaming fails.
        """
        # Default implementation delegates to next middleware or model
        if self.next_middleware:
            yield from self.next_middleware.process_stream(prompt, **kwargs)
        elif self.model and hasattr(self.model, "stream"):
            yield from self.model.stream(prompt, **kwargs)
        else:
            raise ModelError("No streaming implementation available")

    # Async methods - default implementations delegate to sync or use executor

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a generate request.

        Default implementation runs sync method in executor. Subclasses
        should override for true async behavior.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Returns:
            The generated response.

        Raises:
            ModelError: If generation fails.
        """
        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            # Fall back to running sync method in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")

    async def aprocess_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously process a chat request.

        Args:
            messages: The conversation history.
            **kwargs: Additional arguments for chat.

        Returns:
            The chat response.

        Raises:
            ModelError: If chat fails.
        """
        if self.next_middleware:
            return await self.next_middleware.aprocess_chat(messages, **kwargs)
        if self.model:
            if hasattr(self.model, "achat"):
                return await self.model.achat(messages, **kwargs)
            if hasattr(self.model, "chat"):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: self.model.chat(messages, **kwargs))
        raise ModelError("No chat implementation available")

    async def aprocess_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously process a streaming request.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for streaming.

        Yields:
            Response chunks.

        Raises:
            ModelError: If streaming fails.
        """
        if self.next_middleware:
            async for chunk in self.next_middleware.aprocess_stream(prompt, **kwargs):
                yield chunk
        elif self.model:
            if hasattr(self.model, "astream"):
                async for chunk in self.model.astream(prompt, **kwargs):
                    yield chunk
            elif hasattr(self.model, "stream"):
                loop = asyncio.get_running_loop()
                sync_iter = await loop.run_in_executor(
                    None, lambda: list(self.model.stream(prompt, **kwargs))
                )
                for chunk in sync_iter:
                    yield chunk
            else:
                raise ModelError("No streaming implementation available")
        else:
            raise ModelError("No streaming implementation available")


class PassthroughMiddleware(Middleware):
    """Middleware that passes requests through unchanged.

    Useful as a base for middleware that only needs to observe requests.
    """

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Pass through to next middleware or model."""
        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available in pipeline")

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async pass through to next middleware or model."""
        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")


class TraceMiddleware(PassthroughMiddleware):
    """Middleware for capturing execution traces.

    Records trace events for generate, chat, and stream operations.
    Uses the TraceRecorder from insideLLMs.tracing for deterministic
    event sequencing (no wall-clock time).

    The recorder can be accessed after execution to get trace data
    for validation or storage in ResultRecord.custom.

    Args:
        run_id: Optional run ID for trace context.
        example_id: Optional example ID for trace context.

    Example:
        >>> from insideLLMs.tracing import TraceRecorder
        >>> trace_mw = TraceMiddleware(run_id="run_001")
        >>> pipeline = ModelPipeline(model, middlewares=[trace_mw])
        >>> response = pipeline.generate("Hello")
        >>> events = trace_mw.recorder.events
        >>> print(f"Captured {len(events)} trace events")
    """

    # Reserved kwargs that should not leak to providers
    _RESERVED_KWARGS = {"_trace", "_trace_recorder", "_run_id", "_example_id"}

    def __init__(
        self,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ):
        """Initialize the trace middleware.

        Args:
            run_id: Optional run ID for trace context.
            example_id: Optional example ID for trace context.
        """
        super().__init__()
        # Lazy import to avoid circular dependencies
        from insideLLMs.tracing import TraceRecorder

        self._recorder = TraceRecorder(run_id=run_id, example_id=example_id)

    @property
    def recorder(self) -> "TraceRecorder":
        """Get the trace recorder."""
        return self._recorder

    def reset(
        self,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ) -> None:
        """Reset the recorder for a new execution.

        Args:
            run_id: Optional new run ID.
            example_id: Optional new example ID.
        """
        from insideLLMs.tracing import TraceRecorder

        self._recorder = TraceRecorder(run_id=run_id, example_id=example_id)

    def _strip_reserved_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove reserved trace kwargs before passing to providers."""
        return {k: v for k, v in kwargs.items() if k not in self._RESERVED_KWARGS}

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Trace-wrapped generate."""
        # Strip reserved kwargs
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record start
        self._recorder.record_generate_start(prompt, **clean_kwargs)

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                response = self.next_middleware.process_generate(prompt, **clean_kwargs)
            elif self.model:
                response = self.model.generate(prompt, **clean_kwargs)
            else:
                raise ModelError("No model available in pipeline")

            # Record end
            self._recorder.record_generate_end(response)
            return response

        except Exception as e:
            # Record error
            self._recorder.record_error(
                str(e),
                error_type=type(e).__name__,
            )
            raise

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async trace-wrapped generate."""
        # Strip reserved kwargs
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record start
        self._recorder.record_generate_start(prompt, **clean_kwargs)

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                response = await self.next_middleware.aprocess_generate(prompt, **clean_kwargs)
            elif self.model:
                if isinstance(self.model, AsyncModelProtocol):
                    response = await self.model.agenerate(prompt, **clean_kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, lambda: self.model.generate(prompt, **clean_kwargs)
                    )
            else:
                raise ModelError("No model available in pipeline")

            # Record end
            self._recorder.record_generate_end(response)
            return response

        except Exception as e:
            # Record error
            self._recorder.record_error(
                str(e),
                error_type=type(e).__name__,
            )
            raise

    def process_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Trace-wrapped chat."""
        from insideLLMs.tracing import TraceEventKind

        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record chat start
        self._recorder.record(
            TraceEventKind.CHAT_START,
            {"message_count": len(messages), "params": clean_kwargs},
        )

        try:
            # Delegate
            if self.next_middleware:
                response = self.next_middleware.process_chat(messages, **clean_kwargs)
            elif self.model and hasattr(self.model, "chat"):
                response = self.model.chat(messages, **clean_kwargs)
            else:
                raise ModelError("No chat implementation available")

            # Record chat end
            self._recorder.record(
                TraceEventKind.CHAT_END,
                {"response": response},
            )
            return response

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise

    async def aprocess_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Async trace-wrapped chat."""
        from insideLLMs.tracing import TraceEventKind

        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record chat start
        self._recorder.record(
            TraceEventKind.CHAT_START,
            {"message_count": len(messages), "params": clean_kwargs},
        )

        try:
            # Delegate
            if self.next_middleware:
                response = await self.next_middleware.aprocess_chat(messages, **clean_kwargs)
            elif self.model:
                if hasattr(self.model, "achat"):
                    response = await self.model.achat(messages, **clean_kwargs)
                elif hasattr(self.model, "chat"):
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, lambda: self.model.chat(messages, **clean_kwargs)
                    )
                else:
                    raise ModelError("No chat implementation available")
            else:
                raise ModelError("No chat implementation available")

            # Record chat end
            self._recorder.record(
                TraceEventKind.CHAT_END,
                {"response": response},
            )
            return response

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise

    def process_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Trace-wrapped streaming."""
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record stream start
        self._recorder.record_stream_start(prompt, **clean_kwargs)

        chunk_index = 0
        accumulated = []

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                stream = self.next_middleware.process_stream(prompt, **clean_kwargs)
            elif self.model and hasattr(self.model, "stream"):
                stream = self.model.stream(prompt, **clean_kwargs)
            else:
                raise ModelError("No streaming implementation available")

            for chunk in stream:
                # Record each chunk
                self._recorder.record_stream_chunk(chunk, chunk_index)
                accumulated.append(chunk)
                chunk_index += 1
                yield chunk

            # Record stream end
            self._recorder.record_stream_end(
                full_response="".join(accumulated),
                chunk_count=chunk_index,
            )

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise

    async def aprocess_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Async trace-wrapped streaming."""
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record stream start
        self._recorder.record_stream_start(prompt, **clean_kwargs)

        chunk_index = 0
        accumulated = []

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                async for chunk in self.next_middleware.aprocess_stream(prompt, **clean_kwargs):
                    self._recorder.record_stream_chunk(chunk, chunk_index)
                    accumulated.append(chunk)
                    chunk_index += 1
                    yield chunk
            elif self.model:
                if hasattr(self.model, "astream"):
                    async for chunk in self.model.astream(prompt, **clean_kwargs):
                        self._recorder.record_stream_chunk(chunk, chunk_index)
                        accumulated.append(chunk)
                        chunk_index += 1
                        yield chunk
                elif hasattr(self.model, "stream"):
                    loop = asyncio.get_running_loop()
                    sync_chunks = await loop.run_in_executor(
                        None, lambda: list(self.model.stream(prompt, **clean_kwargs))
                    )
                    for chunk in sync_chunks:
                        self._recorder.record_stream_chunk(chunk, chunk_index)
                        accumulated.append(chunk)
                        chunk_index += 1
                        yield chunk
                else:
                    raise ModelError("No streaming implementation available")
            else:
                raise ModelError("No streaming implementation available")

            # Record stream end
            self._recorder.record_stream_end(
                full_response="".join(accumulated),
                chunk_count=chunk_index,
            )

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise


class CacheMiddleware(Middleware):
    """Middleware for caching model responses.

    Caches responses based on prompt and parameters to avoid redundant API calls.

    Args:
        cache_size: Maximum number of cached entries (LRU eviction).
        ttl_seconds: Time-to-live for cache entries (None = no expiration).
    """

    def __init__(self, cache_size: int = 1000, ttl_seconds: Optional[float] = None):
        """Initialize the cache middleware."""
        super().__init__()
        self.cache: dict[str, tuple[str, float]] = {}
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        """Generate a cache key from prompt and parameters."""
        import hashlib
        import json

        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self.ttl_seconds

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Check cache before generating."""
        key = self._cache_key(prompt, **kwargs)

        # Check cache
        if key in self.cache:
            response, timestamp = self.cache[key]
            if not self._is_expired(timestamp):
                self.hits += 1
                return response
            # Expired, remove from cache
            del self.cache[key]

        # Cache miss - generate and cache
        self.misses += 1
        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")

        # Store in cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = (response, time.time())
        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async check cache before generating."""
        key = self._cache_key(prompt, **kwargs)

        # Check cache (async-safe since cache is just a dict read)
        async with self._get_lock():
            if key in self.cache:
                response, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    self.hits += 1
                    return response
                # Expired, remove from cache
                del self.cache[key]

            self.misses += 1

        # Cache miss - generate and cache
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if isinstance(self.model, AsyncModelProtocol):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")

        # Store in cache with LRU eviction
        async with self._get_lock():
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = (response, time.time())

        return response

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an async lock for thread-safe cache access."""
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class RateLimitMiddleware(Middleware):
    """Middleware for rate limiting model requests.

    Enforces a maximum request rate using a token bucket algorithm.

    Args:
        requests_per_minute: Maximum requests per minute.
        burst_size: Maximum burst size (default: same as rate).
    """

    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        """Initialize the rate limiter."""
        super().__init__()
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst_size = burst_size or requests_per_minute
        self.tokens = float(self.burst_size)
        self.last_update = time.time()

    def _acquire_token(self) -> None:
        """Acquire a token, waiting if necessary."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens < 1.0:
            # Wait for token to be available
            wait_time = (1.0 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 1.0
            self.last_update = time.time()

        self.tokens -= 1.0

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Rate limit before generating."""
        self._acquire_token()

        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available in pipeline")

    async def _aacquire_token(self) -> None:
        """Asynchronously acquire a token, waiting if necessary."""
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
        """Get or create an async lock for thread-safe token access."""
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async rate limit before generating."""
        await self._aacquire_token()

        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")


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
        jitter = random.uniform(0, delay * 0.1)
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
            except (RateLimitError, TimeoutError, ModelError) as e:
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
            except (RateLimitError, TimeoutError, ModelError) as e:
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


class CostTrackingMiddleware(Middleware):
    """Middleware for tracking API costs and token usage.

    Tracks requests, tokens, and estimated costs for model usage.
    """

    # Approximate costs per 1K tokens (as of 2024, subject to change)
    COST_PER_1K_TOKENS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self):
        """Initialize cost tracking."""
        super().__init__()
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0

    def _estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model and token counts."""
        # Try to match model name to known pricing
        model_key = None
        for key in self.COST_PER_1K_TOKENS:
            if key in model_name.lower():
                model_key = key
                break

        if not model_key:
            return 0.0  # Unknown model

        pricing = self.COST_PER_1K_TOKENS[model_key]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Track costs for generation."""
        # Generate response
        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")

        # Track usage
        self.total_requests += 1

        # Rough token estimation (4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Estimate cost
        if self.model:
            model_name = getattr(self.model, "model_id", "unknown")
            cost = self._estimate_cost(model_name, input_tokens, output_tokens)
            self.estimated_cost += cost

        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async track costs for generation."""
        # Generate response
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if isinstance(self.model, AsyncModelProtocol):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")

        # Track usage (thread-safe via lock)
        async with self._get_lock():
            self.total_requests += 1

            # Rough token estimation (4 chars ≈ 1 token)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # Estimate cost
            if self.model:
                model_name = getattr(self.model, "model_id", "unknown")
                cost = self._estimate_cost(model_name, input_tokens, output_tokens)
                self.estimated_cost += cost

        return response

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an async lock for thread-safe tracking."""
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    def get_stats(self) -> dict[str, Any]:
        """Get cost tracking statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }


class ModelPipeline(Model):
    """A model wrapper that composes multiple middleware for enhanced capabilities.

    The pipeline executes middleware in order for requests and reverse order
    for responses, following the chain of responsibility pattern.

    Args:
        base_model: The underlying model to wrap.
        middlewares: List of middleware to apply (in order).
        name: Optional name for the pipeline (defaults to base model name).

    Example:
        >>> pipeline = ModelPipeline(
        ...     OpenAIModel("gpt-4"),
        ...     middlewares=[
        ...         CacheMiddleware(cache_size=500),
        ...         RateLimitMiddleware(requests_per_minute=60),
        ...         RetryMiddleware(max_retries=3),
        ...         CostTrackingMiddleware(),
        ...     ],
        ... )
        >>> response = pipeline.generate("Hello, world!")
    """

    def __init__(
        self,
        base_model: ModelProtocol,
        middlewares: Optional[list[Middleware]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the pipeline."""
        self.base_model = base_model
        self.middlewares = middlewares or []

        # Chain middleware together
        prev: Optional[Middleware] = None
        for middleware in self.middlewares:
            middleware.model = base_model
            if prev:
                prev.next_middleware = middleware
            prev = middleware

        # Initialize base Model
        pipeline_name = name or f"{base_model.name}_pipeline"
        model_id = getattr(base_model, "model_id", base_model.name)
        super().__init__(name=pipeline_name, model_id=model_id)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate through the middleware pipeline."""
        if self.middlewares:
            return self.middlewares[0].process_generate(prompt, **kwargs)
        return self.base_model.generate(prompt, **kwargs)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Chat through the middleware pipeline."""
        if self.middlewares:
            return self.middlewares[0].process_chat(messages, **kwargs)
        if hasattr(self.base_model, "chat"):
            return self.base_model.chat(messages, **kwargs)
        raise ModelError("Base model does not support chat")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream through the middleware pipeline."""
        if self.middlewares:
            yield from self.middlewares[0].process_stream(prompt, **kwargs)
        elif hasattr(self.base_model, "stream"):
            yield from self.base_model.stream(prompt, **kwargs)
        else:
            raise ModelError("Base model does not support streaming")

    # Async methods

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate through the middleware pipeline."""
        if self.middlewares:
            return await self.middlewares[0].aprocess_generate(prompt, **kwargs)
        if isinstance(self.base_model, AsyncModelProtocol):
            return await self.base_model.agenerate(prompt, **kwargs)
        # Fall back to executor for sync model
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.base_model.generate(prompt, **kwargs))

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously chat through the middleware pipeline."""
        if self.middlewares:
            return await self.middlewares[0].aprocess_chat(messages, **kwargs)
        if hasattr(self.base_model, "achat"):
            return await self.base_model.achat(messages, **kwargs)
        if hasattr(self.base_model, "chat"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.base_model.chat(messages, **kwargs)
            )
        raise ModelError("Base model does not support chat")

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream through the middleware pipeline."""
        if self.middlewares:
            async for chunk in self.middlewares[0].aprocess_stream(prompt, **kwargs):
                yield chunk
        elif hasattr(self.base_model, "astream"):
            async for chunk in self.base_model.astream(prompt, **kwargs):
                yield chunk
        elif hasattr(self.base_model, "stream"):
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                None, lambda: list(self.base_model.stream(prompt, **kwargs))
            )
            for chunk in chunks:
                yield chunk
        else:
            raise ModelError("Base model does not support streaming")

    async def abatch_generate(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            return_exceptions: If True, include exceptions in results as strings.
            **kwargs: Additional arguments for generation.

        Returns:
            List of responses in same order as prompts.

        Example:
            >>> results = await pipeline.abatch_generate(
            ...     ["Q1", "Q2", "Q3"],
            ...     max_concurrency=5
            ... )
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(prompts)

        async def process(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    results[index] = await self.agenerate(prompt, **kwargs)
                except Exception as e:
                    if return_exceptions:
                        results[index] = f"Error: {e}"
                    else:
                        raise

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return results

    def info(self) -> dict[str, Any]:
        """Get pipeline information including middleware stats."""
        from dataclasses import asdict, is_dataclass

        base_info = self.base_model.info()

        # Convert ModelInfo dataclass to dict if needed
        if is_dataclass(base_info):
            base_info_dict = asdict(base_info)
        else:
            base_info_dict = base_info

        pipeline_info = {
            **base_info_dict,
            "pipeline": True,
            "middleware_count": len(self.middlewares),
            "middlewares": [type(m).__name__ for m in self.middlewares],
        }

        # Add middleware-specific stats
        for middleware in self.middlewares:
            if isinstance(middleware, CacheMiddleware):
                pipeline_info["cache_hit_rate"] = middleware.hit_rate
                pipeline_info["cache_hits"] = middleware.hits
                pipeline_info["cache_misses"] = middleware.misses
            elif isinstance(middleware, RetryMiddleware):
                pipeline_info["total_retries"] = middleware.total_retries
            elif isinstance(middleware, CostTrackingMiddleware):
                pipeline_info["cost_stats"] = middleware.get_stats()

        return pipeline_info


class AsyncModelPipeline(ModelPipeline):
    """Async-first model pipeline optimized for concurrent workloads.

    Extends ModelPipeline with additional async features like batch processing
    with progress tracking and advanced concurrency control.

    Example:
        >>> async def main():
        ...     pipeline = AsyncModelPipeline(
        ...         base_model,
        ...         middlewares=[CacheMiddleware(), RateLimitMiddleware()],
        ...     )
        ...     # Single request
        ...     response = await pipeline.agenerate("Hello")
        ...
        ...     # Batch processing with progress
        ...     async for result in pipeline.agenerate_stream_results(prompts):
        ...         print(f"Got result: {result}")
    """

    async def agenerate_with_callback(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_result: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate with progress and result callbacks.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            on_progress: Callback(completed, total) for progress updates.
            on_result: Callback(index, result) when each result completes.
            **kwargs: Additional arguments for generation.

        Returns:
            List of responses in order.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(prompts)
        completed = 0
        total = len(prompts)

        async def process(index: int, prompt: str) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.agenerate(prompt, **kwargs)
                    results[index] = result
                    if on_result:
                        on_result(index, result)
                except Exception as e:
                    results[index] = f"Error: {e}"
                    if on_result:
                        on_result(index, f"Error: {e}")
                finally:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    async def agenerate_stream_results(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, str]]:
        """Generate and yield results as they complete.

        Yields results in completion order (not input order).

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            **kwargs: Additional arguments for generation.

        Yields:
            Tuples of (index, result) as each completes.

        Example:
            >>> async for idx, result in pipeline.agenerate_stream_results(prompts):
            ...     print(f"Prompt {idx}: {result[:50]}...")
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

        async def process(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.agenerate(prompt, **kwargs)
                    await queue.put((index, result))
                except Exception as e:
                    await queue.put((index, f"Error: {e}"))

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]

        # Yield results as they complete
        for _ in range(len(prompts)):
            result = await queue.get()
            yield result

        # Ensure all tasks complete
        await asyncio.gather(*tasks)

    async def amap(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> list[tuple[str, Optional[str], Optional[Exception]]]:
        """Map prompts to responses with detailed error handling.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            timeout: Optional timeout per request in seconds.
            **kwargs: Additional arguments for generation.

        Returns:
            List of (prompt, response, error) tuples.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[tuple[str, Optional[str], Optional[Exception]]] = []

        async def process(prompt: str) -> tuple[str, Optional[str], Optional[Exception]]:
            async with semaphore:
                try:
                    if timeout:
                        result = await asyncio.wait_for(
                            self.agenerate(prompt, **kwargs), timeout=timeout
                        )
                    else:
                        result = await self.agenerate(prompt, **kwargs)
                    return (prompt, result, None)
                except Exception as e:
                    return (prompt, None, e)

        tasks = [asyncio.create_task(process(prompt)) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return list(results)
