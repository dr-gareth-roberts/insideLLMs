"""
Rate Limiting and Retry Module

Production-grade rate limiting and retry mechanisms for LLM APIs:
- Token bucket and sliding window rate limiters
- Exponential backoff with jitter
- Circuit breaker pattern
- Request queuing and prioritization
- Concurrent request management
- Provider-specific rate limit handling
"""

import asyncio
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RetryStrategy(Enum):
    """Retry strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


# Import CircuitState from retry module to avoid duplication
from insideLLMs.retry import CircuitState  # noqa: E402


class RequestPriority(Enum):
    """Request priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    tokens_per_minute: int = 100000
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "burst_size": self.burst_size,
            "strategy": self.strategy.value,
        }


@dataclass
class RateLimitRetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_errors: list[type] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "strategy": self.strategy.value,
            "jitter": self.jitter,
            "jitter_factor": self.jitter_factor,
        }


@dataclass
class RateLimitState:
    """Current state of rate limiter."""

    available_tokens: float
    requests_in_window: int
    tokens_in_window: int
    last_request_time: Optional[datetime]
    is_limited: bool
    wait_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "available_tokens": self.available_tokens,
            "requests_in_window": self.requests_in_window,
            "tokens_in_window": self.tokens_in_window,
            "last_request_time": self.last_request_time.isoformat()
            if self.last_request_time
            else None,
            "is_limited": self.is_limited,
            "wait_time_ms": self.wait_time_ms,
        }


@dataclass
class RateLimitRetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any
    attempts: int
    total_time_ms: float
    errors: list[str]
    final_error: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_time_ms": self.total_time_ms,
            "errors": self.errors,
            "final_error": self.final_error,
        }


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    half_open_successes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time
            else None,
            "last_success_time": self.last_success_time.isoformat()
            if self.last_success_time
            else None,
            "half_open_successes": self.half_open_successes,
        }


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""

    total_requests: int = 0
    allowed_requests: int = 0
    throttled_requests: int = 0
    total_wait_time_ms: float = 0.0
    tokens_consumed: int = 0

    @property
    def throttle_rate(self) -> float:
        """Calculate throttle rate."""
        if self.total_requests == 0:
            return 0.0
        return self.throttled_requests / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "throttled_requests": self.throttled_requests,
            "throttle_rate": self.throttle_rate,
            "total_wait_time_ms": self.total_wait_time_ms,
            "tokens_consumed": self.tokens_consumed,
        }


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining average rate.
    """

    def __init__(
        self,
        rate: float = 10.0,  # tokens per second
        capacity: int = 20,  # bucket capacity
    ):
        """
        Initialize token bucket.

        Args:
            rate: Token refill rate per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()

    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            block: Whether to wait for tokens

        Returns:
            True if tokens acquired
        """
        with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.tokens_consumed += tokens
                return True

            if not block:
                self._stats.throttled_requests += 1
                return False

            # Calculate wait time
            needed = tokens - self._tokens
            wait_time = needed / self.rate

        # Wait outside lock
        self._stats.total_wait_time_ms += wait_time * 1000
        time.sleep(wait_time)

        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.tokens_consumed += tokens
                return True

            self._stats.throttled_requests += 1
            return False

    async def acquire_async(self, tokens: int = 1, block: bool = True) -> bool:
        """Async version of acquire."""
        with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.tokens_consumed += tokens
                return True

            if not block:
                self._stats.throttled_requests += 1
                return False

            needed = tokens - self._tokens
            wait_time = needed / self.rate

        self._stats.total_wait_time_ms += wait_time * 1000
        await asyncio.sleep(wait_time)

        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.tokens_consumed += tokens
                return True

            self._stats.throttled_requests += 1
            return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def get_state(self) -> RateLimitState:
        """Get current state."""
        with self._lock:
            self._refill()
            wait_time = 0.0
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate * 1000

            return RateLimitState(
                available_tokens=self._tokens,
                requests_in_window=0,
                tokens_in_window=0,
                last_request_time=None,
                is_limited=self._tokens < 1,
                wait_time_ms=wait_time,
            )

    def get_stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats

    def reset(self):
        """Reset rate limiter."""
        with self._lock:
            self._tokens = float(self.capacity)
            self._last_update = time.monotonic()


class ThreadSafeSlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in a sliding time window.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        window_size_seconds: float = 1.0,
    ):
        """
        Initialize sliding window.

        Args:
            requests_per_second: Maximum requests per second
            window_size_seconds: Window size in seconds
        """
        self.max_requests = int(requests_per_second * window_size_seconds)
        self.window_size = window_size_seconds
        self._requests: deque = deque()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()

    def acquire(self, block: bool = True) -> bool:
        """
        Try to acquire a request slot.

        Args:
            block: Whether to wait for slot

        Returns:
            True if acquired
        """
        with self._lock:
            self._cleanup()
            self._stats.total_requests += 1

            if len(self._requests) < self.max_requests:
                self._requests.append(time.monotonic())
                self._stats.allowed_requests += 1
                return True

            if not block:
                self._stats.throttled_requests += 1
                return False

            # Calculate wait time
            oldest = self._requests[0]
            wait_time = self.window_size - (time.monotonic() - oldest)

        if wait_time > 0:
            self._stats.total_wait_time_ms += wait_time * 1000
            time.sleep(wait_time)

        with self._lock:
            self._cleanup()
            if len(self._requests) < self.max_requests:
                self._requests.append(time.monotonic())
                self._stats.allowed_requests += 1
                return True

            self._stats.throttled_requests += 1
            return False

    async def acquire_async(self, block: bool = True) -> bool:
        """Async version of acquire."""
        with self._lock:
            self._cleanup()
            self._stats.total_requests += 1

            if len(self._requests) < self.max_requests:
                self._requests.append(time.monotonic())
                self._stats.allowed_requests += 1
                return True

            if not block:
                self._stats.throttled_requests += 1
                return False

            oldest = self._requests[0]
            wait_time = self.window_size - (time.monotonic() - oldest)

        if wait_time > 0:
            self._stats.total_wait_time_ms += wait_time * 1000
            await asyncio.sleep(wait_time)

        with self._lock:
            self._cleanup()
            if len(self._requests) < self.max_requests:
                self._requests.append(time.monotonic())
                self._stats.allowed_requests += 1
                return True

            self._stats.throttled_requests += 1
            return False

    def _cleanup(self):
        """Remove expired requests."""
        cutoff = time.monotonic() - self.window_size
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def get_state(self) -> RateLimitState:
        """Get current state."""
        with self._lock:
            self._cleanup()
            requests_in_window = len(self._requests)
            wait_time = 0.0

            if requests_in_window >= self.max_requests and self._requests:
                oldest = self._requests[0]
                wait_time = (self.window_size - (time.monotonic() - oldest)) * 1000

            return RateLimitState(
                available_tokens=self.max_requests - requests_in_window,
                requests_in_window=requests_in_window,
                tokens_in_window=0,
                last_request_time=datetime.now() if self._requests else None,
                is_limited=requests_in_window >= self.max_requests,
                wait_time_ms=max(0, wait_time),
            )

    def get_stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats

    def reset(self):
        """Reset rate limiter."""
        with self._lock:
            self._requests.clear()


class RetryHandler:
    """
    Handles retries with configurable strategies.
    """

    def __init__(self, config: Optional[RateLimitRetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config or RateLimitRetryConfig()
        self._fibonacci_cache = [0, 1]

    def execute(
        self,
        func: Callable[[], T],
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ) -> RateLimitRetryResult:
        """
        Execute function with retries.

        Args:
            func: Function to execute
            on_retry: Optional callback on retry

        Returns:
            RateLimitRetryResult with outcome
        """
        start_time = time.time()
        errors = []
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func()
                return RateLimitRetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_time_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                    final_error=None,
                )
            except Exception as e:
                last_error = str(e)
                errors.append(last_error)

                # Check if error is retryable
                if self.config.retryable_errors and not any(
                    isinstance(e, err_type) for err_type in self.config.retryable_errors
                ):
                    break

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    if on_retry:
                        on_retry(attempt + 1, e)
                    time.sleep(delay)

        return RateLimitRetryResult(
            success=False,
            result=None,
            attempts=self.config.max_retries + 1,
            total_time_ms=(time.time() - start_time) * 1000,
            errors=errors,
            final_error=last_error,
        )

    async def execute_async(
        self,
        func: Callable[[], T],
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ) -> RateLimitRetryResult:
        """Async version of execute."""
        start_time = time.time()
        errors = []
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()

                return RateLimitRetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_time_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                    final_error=None,
                )
            except Exception as e:
                last_error = str(e)
                errors.append(last_error)

                if self.config.retryable_errors and not any(
                    isinstance(e, err_type) for err_type in self.config.retryable_errors
                ):
                    break

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    if on_retry:
                        on_retry(attempt + 1, e)
                    await asyncio.sleep(delay)

        return RateLimitRetryResult(
            success=False,
            result=None,
            attempts=self.config.max_retries + 1,
            total_time_ms=(time.time() - start_time) * 1000,
            errors=errors,
            final_error=last_error,
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt."""
        if self.config.strategy == RetryStrategy.CONSTANT:
            delay = self.config.base_delay

        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)

        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2**attempt)

        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._get_fibonacci(attempt + 1)

        else:
            delay = self.config.base_delay

        # Apply cap
        delay = min(delay, self.config.max_delay)

        # Apply jitter
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(self._fibonacci_cache[-1] + self._fibonacci_cache[-2])
        return self._fibonacci_cache[n]


class RateLimitCircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascading failures by stopping requests to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying recovery
            half_open_max_calls: Successful calls needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_successes = 0
                        return True
                return False

            # Half-open state allows limited calls
            return True

    def record_success(self):
        """Record a successful execution."""
        with self._lock:
            self._success_count += 1
            self._last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self):
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._state = CircuitState.OPEN

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN

    def execute(self, func: Callable[[], T]) -> T:
        """
        Execute function with circuit breaker.

        Args:
            func: Function to execute

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self.can_execute():
            raise CircuitOpenError("Circuit breaker is open")

        try:
            result = func()
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    async def execute_async(self, func: Callable[[], T]) -> T:
        """Async version of execute."""
        if not self.can_execute():
            raise CircuitOpenError("Circuit breaker is open")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def get_state(self) -> CircuitBreakerState:
        """Get current state."""
        with self._lock:
            return CircuitBreakerState(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                half_open_successes=self._half_open_successes,
            )

    def reset(self):
        """Reset circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_successes = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RequestQueue:
    """
    Priority queue for requests with rate limiting.
    """

    def __init__(
        self,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        max_queue_size: int = 1000,
    ):
        """
        Initialize request queue.

        Args:
            rate_limiter: Rate limiter to use
            max_queue_size: Maximum queue size
        """
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self.max_queue_size = max_queue_size
        self._queue: list[tuple[int, float, Callable]] = []  # (priority, timestamp, func)
        self._lock = threading.Lock()
        self._processing = False
        self._processed_count = 0
        self._dropped_count = 0

    def enqueue(
        self,
        func: Callable[[], T],
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> bool:
        """
        Add request to queue.

        Args:
            func: Function to execute
            priority: Request priority

        Returns:
            True if enqueued
        """
        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                self._dropped_count += 1
                return False

            # Add with priority and timestamp for ordering
            entry = (priority.value, time.monotonic(), func)
            self._queue.append(entry)
            self._queue.sort(key=lambda x: (x[0], x[1]))
            return True

    def process_one(self) -> Optional[Any]:
        """Process one request from queue."""
        with self._lock:
            if not self._queue:
                return None

            _, _, func = self._queue.pop(0)

        self.rate_limiter.acquire(block=True)

        try:
            result = func()
            self._processed_count += 1
            return result
        except Exception:
            raise

    async def process_one_async(self) -> Optional[Any]:
        """Async version of process_one."""
        with self._lock:
            if not self._queue:
                return None

            _, _, func = self._queue.pop(0)

        await self.rate_limiter.acquire_async(block=True)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            self._processed_count += 1
            return result
        except Exception:
            raise

    def process_all(self) -> list[Any]:
        """Process all queued requests."""
        results = []
        while True:
            with self._lock:
                if not self._queue:
                    break
            result = self.process_one()
            if result is not None:
                results.append(result)
        return results

    def get_queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "processed_count": self._processed_count,
                "dropped_count": self._dropped_count,
                "rate_limiter_stats": self.rate_limiter.get_stats().to_dict(),
            }

    def clear(self):
        """Clear the queue."""
        with self._lock:
            self._queue.clear()


class ConcurrencyLimiter:
    """
    Limits concurrent executions.
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum concurrent executions
        """
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._current_count = 0
        self._lock = threading.Lock()

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a slot."""
        acquired = self._semaphore.acquire(blocking=block, timeout=timeout)
        if acquired:
            with self._lock:
                self._current_count += 1
        return acquired

    def release(self):
        """Release a slot."""
        self._semaphore.release()
        with self._lock:
            self._current_count = max(0, self._current_count - 1)

    async def acquire_async(self) -> bool:
        """Async acquire."""
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_concurrent)
        await self._async_semaphore.acquire()
        with self._lock:
            self._current_count += 1
        return True

    async def release_async(self):
        """Async release."""
        if self._async_semaphore:
            self._async_semaphore.release()
        with self._lock:
            self._current_count = max(0, self._current_count - 1)

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release_async()
        return False

    def get_current_count(self) -> int:
        """Get current concurrent count."""
        with self._lock:
            return self._current_count

    def get_available(self) -> int:
        """Get available slots."""
        with self._lock:
            return self.max_concurrent - self._current_count


class RateLimitedExecutor:
    """
    Combined rate limiting, retry, and circuit breaker execution.
    """

    def __init__(
        self,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        retry_handler: Optional[RetryHandler] = None,
        circuit_breaker: Optional[RateLimitCircuitBreaker] = None,
        concurrency_limiter: Optional[ConcurrencyLimiter] = None,
    ):
        """
        Initialize executor.

        Args:
            rate_limiter: Rate limiter
            retry_handler: Retry handler
            circuit_breaker: Circuit breaker
            concurrency_limiter: Concurrency limiter
        """
        self.rate_limiter = rate_limiter
        self.retry_handler = retry_handler
        self.circuit_breaker = circuit_breaker
        self.concurrency_limiter = concurrency_limiter

    def execute(
        self,
        func: Callable[[], T],
        tokens: int = 1,
    ) -> T:
        """
        Execute with all protection mechanisms.

        Args:
            func: Function to execute
            tokens: Tokens to consume

        Returns:
            Function result
        """
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise CircuitOpenError("Circuit breaker is open")

        # Acquire rate limit
        if self.rate_limiter:
            self.rate_limiter.acquire(tokens, block=True)

        # Acquire concurrency slot
        if self.concurrency_limiter:
            self.concurrency_limiter.acquire()

        try:
            # Execute with retry
            if self.retry_handler:
                result = self.retry_handler.execute(func)
                if not result.success:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                    raise Exception(result.final_error)
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                return result.result
            else:
                try:
                    result = func()
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                    return result
                except Exception:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                    raise
        finally:
            if self.concurrency_limiter:
                self.concurrency_limiter.release()

    async def execute_async(
        self,
        func: Callable[[], T],
        tokens: int = 1,
    ) -> T:
        """Async version of execute."""
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise CircuitOpenError("Circuit breaker is open")

        if self.rate_limiter:
            await self.rate_limiter.acquire_async(tokens, block=True)

        if self.concurrency_limiter:
            await self.concurrency_limiter.acquire_async()

        try:
            if self.retry_handler:
                result = await self.retry_handler.execute_async(func)
                if not result.success:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                    raise Exception(result.final_error)
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                return result.result
            else:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func()
                    else:
                        result = func()
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                    return result
                except Exception:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                    raise
        finally:
            if self.concurrency_limiter:
                await self.concurrency_limiter.release_async()


# Decorators


def rate_limited(
    rate: float = 10.0,
    capacity: int = 20,
):
    """
    Decorator for rate limiting.

    Args:
        rate: Tokens per second
        capacity: Bucket capacity
    """
    limiter = TokenBucketRateLimiter(rate=rate, capacity=capacity)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await limiter.acquire_async()
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
):
    """
    Decorator for retry behavior.

    Args:
        max_retries: Maximum retries
        base_delay: Base delay seconds
        strategy: Retry strategy
    """
    config = RateLimitRetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
    )
    handler = RetryHandler(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = handler.execute(lambda: func(*args, **kwargs))
            if result.success:
                return result.result
            raise Exception(result.final_error)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await handler.execute_async(lambda: func(*args, **kwargs))
            if result.success:
                return result.result
            raise Exception(result.final_error)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def circuit_protected(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
):
    """
    Decorator for circuit breaker protection.

    Args:
        failure_threshold: Failures before opening
        recovery_timeout: Recovery timeout seconds
    """
    breaker = RateLimitCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.execute(lambda: func(*args, **kwargs))

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.execute_async(lambda: func(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# Convenience functions


def create_rate_limiter(
    rate: float = 10.0,
    capacity: int = 20,
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
) -> TokenBucketRateLimiter | ThreadSafeSlidingWindowRateLimiter:
    """
    Create a rate limiter.

    Args:
        rate: Rate limit
        capacity: Capacity/burst size
        strategy: Rate limiting strategy
    """
    if strategy == RateLimitStrategy.TOKEN_BUCKET:
        return TokenBucketRateLimiter(rate=rate, capacity=capacity)
    elif strategy == RateLimitStrategy.SLIDING_WINDOW:
        return ThreadSafeSlidingWindowRateLimiter(requests_per_second=rate)
    else:
        return TokenBucketRateLimiter(rate=rate, capacity=capacity)


def create_retry_handler(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
) -> RetryHandler:
    """
    Create a retry handler.

    Args:
        max_retries: Maximum retries
        base_delay: Base delay
        strategy: Retry strategy
    """
    config = RateLimitRetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
    )
    return RetryHandler(config)


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> RateLimitCircuitBreaker:
    """
    Create a circuit breaker.

    Args:
        failure_threshold: Failures before opening
        recovery_timeout: Recovery timeout
    """
    return RateLimitCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


def create_executor(
    rate: float = 10.0,
    max_retries: int = 3,
    failure_threshold: int = 5,
    max_concurrent: int = 10,
) -> RateLimitedExecutor:
    """
    Create a fully configured executor.

    Args:
        rate: Rate limit
        max_retries: Maximum retries
        failure_threshold: Circuit breaker threshold
        max_concurrent: Max concurrent requests
    """
    return RateLimitedExecutor(
        rate_limiter=TokenBucketRateLimiter(rate=rate),
        retry_handler=create_retry_handler(max_retries=max_retries),
        circuit_breaker=create_circuit_breaker(failure_threshold=failure_threshold),
        concurrency_limiter=ConcurrencyLimiter(max_concurrent=max_concurrent),
    )


def execute_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """
    Execute function with exponential backoff.

    Args:
        func: Function to execute
        max_retries: Maximum retries
        base_delay: Base delay

    Returns:
        Function result
    """
    handler = create_retry_handler(max_retries, base_delay)
    result = handler.execute(func)
    if result.success:
        return result.result
    raise Exception(result.final_error)
