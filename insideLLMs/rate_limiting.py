"""
Rate Limiting and Retry Module
==============================

Production-grade rate limiting and retry mechanisms for LLM APIs. This module
provides comprehensive tools for managing API request rates, implementing
fault tolerance patterns, and handling transient failures gracefully.

Features
--------
- **Token bucket rate limiter**: Allows controlled bursts while maintaining average rate
- **Sliding window rate limiter**: Tracks requests in a rolling time window
- **Exponential backoff with jitter**: Smart retry delays to avoid thundering herd
- **Circuit breaker pattern**: Prevents cascading failures to failing services
- **Request queuing**: Priority-based request queue with rate limiting
- **Concurrent request management**: Limits parallel executions

Examples
--------
Basic token bucket rate limiting:

    >>> from insideLLMs.rate_limiting import TokenBucketRateLimiter
    >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
    >>> if limiter.acquire(tokens=1, block=False):
    ...     print("Request allowed")
    ... else:
    ...     print("Rate limited")
    Request allowed

Using the rate_limited decorator:

    >>> from insideLLMs.rate_limiting import rate_limited
    >>> @rate_limited(rate=5.0, capacity=10)
    ... def call_api():
    ...     return "API response"
    >>> result = call_api()  # Automatically rate limited

Retry with exponential backoff:

    >>> from insideLLMs.rate_limiting import RetryHandler, RateLimitRetryConfig, RetryStrategy
    >>> config = RateLimitRetryConfig(
    ...     max_retries=3,
    ...     base_delay=1.0,
    ...     strategy=RetryStrategy.EXPONENTIAL,
    ...     jitter=True
    ... )
    >>> handler = RetryHandler(config)
    >>> result = handler.execute(lambda: some_flaky_function())

Circuit breaker for fault tolerance:

    >>> from insideLLMs.rate_limiting import RateLimitCircuitBreaker
    >>> breaker = RateLimitCircuitBreaker(
    ...     failure_threshold=5,
    ...     recovery_timeout=30.0
    ... )
    >>> try:
    ...     result = breaker.execute(lambda: external_service_call())
    ... except CircuitOpenError:
    ...     print("Circuit is open, using fallback")

Full executor with all protections:

    >>> from insideLLMs.rate_limiting import create_executor
    >>> executor = create_executor(
    ...     rate=10.0,
    ...     max_retries=3,
    ...     failure_threshold=5,
    ...     max_concurrent=10
    ... )
    >>> result = executor.execute(lambda: api_call())
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
    """
    Rate limiting strategy enumeration.

    Defines the algorithm used for rate limiting requests. Each strategy
    has different characteristics for handling bursts and maintaining
    rate limits.

    Attributes
    ----------
    TOKEN_BUCKET : str
        Token bucket algorithm. Allows bursts up to bucket capacity
        while maintaining a long-term average rate. Good for APIs that
        allow occasional bursts but need rate control.
    SLIDING_WINDOW : str
        Sliding window algorithm. Tracks requests in a moving time window.
        Provides smoother rate limiting without allowing large bursts.
    FIXED_WINDOW : str
        Fixed window algorithm. Resets counters at fixed intervals.
        Simple but can allow 2x burst at window boundaries.
    LEAKY_BUCKET : str
        Leaky bucket algorithm. Processes requests at a constant rate,
        queuing excess requests. Provides very smooth output rate.

    Examples
    --------
    Selecting a rate limiting strategy:

        >>> from insideLLMs.rate_limiting import RateLimitStrategy, create_rate_limiter
        >>> # Token bucket for bursty workloads
        >>> limiter = create_rate_limiter(
        ...     rate=10.0,
        ...     capacity=20,
        ...     strategy=RateLimitStrategy.TOKEN_BUCKET
        ... )

        >>> # Sliding window for smooth rate limiting
        >>> limiter = create_rate_limiter(
        ...     rate=10.0,
        ...     strategy=RateLimitStrategy.SLIDING_WINDOW
        ... )

    Checking strategy type:

        >>> strategy = RateLimitStrategy.TOKEN_BUCKET
        >>> strategy.value
        'token_bucket'
        >>> strategy == RateLimitStrategy.TOKEN_BUCKET
        True
    """

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RetryStrategy(Enum):
    """
    Retry strategy enumeration.

    Defines how delay between retry attempts is calculated. Different
    strategies are suited for different failure patterns and recovery
    scenarios.

    Attributes
    ----------
    EXPONENTIAL : str
        Exponential backoff (2^attempt * base_delay). Best for network
        failures and overloaded services. Quickly backs off to reduce
        load on struggling services.
    LINEAR : str
        Linear backoff (attempt * base_delay). Provides steady increase
        in delay. Good when you want predictable retry timing.
    CONSTANT : str
        Constant delay (base_delay). Same delay between all attempts.
        Useful for known recovery times or simple retry logic.
    FIBONACCI : str
        Fibonacci sequence backoff. Grows slower than exponential but
        faster than linear. Good middle ground for uncertain scenarios.

    Examples
    --------
    Using different retry strategies:

        >>> from insideLLMs.rate_limiting import RetryStrategy, RateLimitRetryConfig
        >>> # Exponential for API rate limits
        >>> config = RateLimitRetryConfig(
        ...     max_retries=5,
        ...     base_delay=1.0,
        ...     strategy=RetryStrategy.EXPONENTIAL
        ... )
        >>> # Delays: 1s, 2s, 4s, 8s, 16s

        >>> # Linear for predictable services
        >>> config = RateLimitRetryConfig(
        ...     max_retries=5,
        ...     base_delay=2.0,
        ...     strategy=RetryStrategy.LINEAR
        ... )
        >>> # Delays: 2s, 4s, 6s, 8s, 10s

        >>> # Constant for quick retries
        >>> config = RateLimitRetryConfig(
        ...     max_retries=3,
        ...     base_delay=0.5,
        ...     strategy=RetryStrategy.CONSTANT
        ... )
        >>> # Delays: 0.5s, 0.5s, 0.5s

    Accessing strategy value:

        >>> strategy = RetryStrategy.FIBONACCI
        >>> strategy.value
        'fibonacci'
    """

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


# Import CircuitState from retry module to avoid duplication
from insideLLMs.retry import CircuitState  # noqa: E402


class RequestPriority(Enum):
    """
    Request priority levels for queue ordering.

    Used by RequestQueue to order requests by importance. Lower numeric
    values indicate higher priority and are processed first.

    Attributes
    ----------
    CRITICAL : int
        Priority 1. Mission-critical requests that must be processed
        immediately. Use sparingly for truly urgent operations.
    HIGH : int
        Priority 2. Important requests that should be processed soon.
        Good for user-facing operations.
    NORMAL : int
        Priority 3. Default priority for standard requests.
        Most API calls should use this level.
    LOW : int
        Priority 4. Less urgent requests that can wait.
        Good for batch processing or prefetching.
    BACKGROUND : int
        Priority 5. Lowest priority for background tasks.
        Processed only when queue is otherwise empty.

    Examples
    --------
    Enqueueing requests with different priorities:

        >>> from insideLLMs.rate_limiting import RequestQueue, RequestPriority
        >>> queue = RequestQueue()
        >>> # High priority user request
        >>> queue.enqueue(
        ...     lambda: user_query("urgent question"),
        ...     priority=RequestPriority.HIGH
        ... )
        True
        >>> # Background analytics
        >>> queue.enqueue(
        ...     lambda: log_analytics(),
        ...     priority=RequestPriority.BACKGROUND
        ... )
        True

    Comparing priorities:

        >>> RequestPriority.CRITICAL.value < RequestPriority.NORMAL.value
        True
        >>> # CRITICAL (1) is processed before NORMAL (3)
    """

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting behavior.

    Dataclass containing all parameters needed to configure a rate limiter.
    Supports both request-based and token-based rate limiting commonly
    used by LLM API providers.

    Attributes
    ----------
    requests_per_second : float
        Maximum requests allowed per second. Default is 10.0.
    requests_per_minute : float
        Maximum requests allowed per minute. Default is 600.0.
        Used as a secondary limit alongside per-second limits.
    tokens_per_minute : int
        Maximum tokens (API tokens, not rate limiter tokens) per minute.
        Default is 100000. Relevant for LLM APIs with token limits.
    burst_size : int
        Maximum burst size allowed. Default is 20.
        Controls how many requests can be made in quick succession.
    strategy : RateLimitStrategy
        Rate limiting algorithm to use. Default is TOKEN_BUCKET.

    Examples
    --------
    Creating a configuration for OpenAI-like limits:

        >>> config = RateLimitConfig(
        ...     requests_per_second=3.0,
        ...     requests_per_minute=60.0,
        ...     tokens_per_minute=90000,
        ...     burst_size=5
        ... )
        >>> config.requests_per_second
        3.0

    Using with sliding window strategy:

        >>> config = RateLimitConfig(
        ...     requests_per_second=10.0,
        ...     strategy=RateLimitStrategy.SLIDING_WINDOW
        ... )

    Converting to dictionary for serialization:

        >>> config = RateLimitConfig()
        >>> data = config.to_dict()
        >>> data['strategy']
        'token_bucket'

    Using default configuration:

        >>> config = RateLimitConfig()
        >>> config.burst_size
        20
    """

    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    tokens_per_minute: int = 100000
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the configuration with
            string values for enum fields.

        Examples
        --------
            >>> config = RateLimitConfig(requests_per_second=5.0)
            >>> d = config.to_dict()
            >>> d['requests_per_second']
            5.0
        """
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "burst_size": self.burst_size,
            "strategy": self.strategy.value,
        }


@dataclass
class RateLimitRetryConfig:
    """
    Configuration for retry behavior with backoff strategies.

    Dataclass containing all parameters to configure retry logic including
    backoff strategy, delay timing, and error filtering.

    Attributes
    ----------
    max_retries : int
        Maximum number of retry attempts. Default is 3.
        Total attempts = max_retries + 1 (initial attempt).
    base_delay : float
        Base delay in seconds between retries. Default is 1.0.
        Actual delay depends on the strategy used.
    max_delay : float
        Maximum delay cap in seconds. Default is 60.0.
        Prevents exponential backoff from growing too large.
    strategy : RetryStrategy
        Backoff strategy to use. Default is EXPONENTIAL.
    jitter : bool
        Whether to add random jitter to delays. Default is True.
        Helps prevent thundering herd problem.
    jitter_factor : float
        Jitter range as fraction of delay. Default is 0.1.
        With 0.1, delay varies by +/- 10%.
    retryable_errors : list[type]
        Exception types that should trigger retry. Default is empty list.
        Empty list means all exceptions are retryable.

    Examples
    --------
    Basic exponential backoff configuration:

        >>> config = RateLimitRetryConfig(
        ...     max_retries=5,
        ...     base_delay=1.0,
        ...     strategy=RetryStrategy.EXPONENTIAL
        ... )
        >>> config.max_retries
        5

    Configuration for specific error types:

        >>> config = RateLimitRetryConfig(
        ...     max_retries=3,
        ...     retryable_errors=[ConnectionError, TimeoutError]
        ... )
        >>> # Only retries ConnectionError and TimeoutError

    High-frequency retry with low jitter:

        >>> config = RateLimitRetryConfig(
        ...     max_retries=10,
        ...     base_delay=0.1,
        ...     max_delay=5.0,
        ...     strategy=RetryStrategy.CONSTANT,
        ...     jitter_factor=0.05
        ... )

    Converting to dictionary:

        >>> config = RateLimitRetryConfig()
        >>> d = config.to_dict()
        >>> d['jitter']
        True
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_errors: list[type] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the configuration.
            Note: retryable_errors is not included in output.

        Examples
        --------
            >>> config = RateLimitRetryConfig(max_retries=5)
            >>> d = config.to_dict()
            >>> d['max_retries']
            5
        """
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
    """
    Current state snapshot of a rate limiter.

    Immutable snapshot of rate limiter state at a point in time.
    Useful for monitoring, debugging, and making decisions about
    whether to proceed with requests.

    Attributes
    ----------
    available_tokens : float
        Number of tokens currently available for use.
        For token bucket, this is remaining bucket capacity.
    requests_in_window : int
        Number of requests made in the current time window.
        Relevant for sliding/fixed window strategies.
    tokens_in_window : int
        Number of API tokens consumed in current window.
        Relevant for LLM token-based rate limiting.
    last_request_time : Optional[datetime]
        Timestamp of the most recent request, or None if no
        requests have been made.
    is_limited : bool
        Whether the rate limiter is currently blocking requests.
        True means requests would be delayed or rejected.
    wait_time_ms : float
        Estimated milliseconds to wait before tokens available.
        0.0 if not currently limited.

    Examples
    --------
    Checking rate limiter state:

        >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        >>> state = limiter.get_state()
        >>> if not state.is_limited:
        ...     print(f"Can proceed, {state.available_tokens} tokens available")

    Monitoring rate limit usage:

        >>> state = limiter.get_state()
        >>> if state.available_tokens < 5:
        ...     print(f"Running low, wait {state.wait_time_ms}ms for refill")

    Serializing state for logging:

        >>> state = limiter.get_state()
        >>> log_data = state.to_dict()
        >>> print(f"Rate limit state: {log_data}")

    Checking window-based limits:

        >>> state = sliding_limiter.get_state()
        >>> print(f"{state.requests_in_window} requests in current window")
    """

    available_tokens: float
    requests_in_window: int
    tokens_in_window: int
    last_request_time: Optional[datetime]
    is_limited: bool
    wait_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert state to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with ISO format datetime.

        Examples
        --------
            >>> state = limiter.get_state()
            >>> d = state.to_dict()
            >>> 'available_tokens' in d
            True
        """
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
    """
    Result of a retry operation.

    Contains the outcome of a function execution with retry logic,
    including success status, result value, timing, and error history.

    Attributes
    ----------
    success : bool
        Whether the operation ultimately succeeded.
    result : Any
        The return value of the function if successful, None otherwise.
    attempts : int
        Total number of attempts made (including initial attempt).
    total_time_ms : float
        Total elapsed time in milliseconds including all retries.
    errors : list[str]
        List of error messages from each failed attempt.
    final_error : Optional[str]
        The last error message if operation failed, None if successful.

    Examples
    --------
    Handling a successful retry result:

        >>> handler = RetryHandler()
        >>> result = handler.execute(lambda: api_call())
        >>> if result.success:
        ...     print(f"Got {result.result} after {result.attempts} attempts")
        ... else:
        ...     print(f"Failed: {result.final_error}")

    Checking retry statistics:

        >>> result = handler.execute(flaky_function)
        >>> print(f"Completed in {result.total_time_ms:.2f}ms")
        >>> if result.attempts > 1:
        ...     print(f"Required {result.attempts - 1} retries")

    Logging all errors encountered:

        >>> result = handler.execute(unstable_service)
        >>> for i, error in enumerate(result.errors):
        ...     print(f"Attempt {i+1} failed: {error}")

    Converting to dictionary for metrics:

        >>> result = handler.execute(api_call)
        >>> metrics = result.to_dict()
        >>> send_to_monitoring(metrics)
    """

    success: bool
    result: Any
    attempts: int
    total_time_ms: float
    errors: list[str]
    final_error: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation. Note: result field is not
            included to avoid serialization issues.

        Examples
        --------
            >>> result = handler.execute(some_func)
            >>> d = result.to_dict()
            >>> 'success' in d and 'attempts' in d
            True
        """
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_time_ms": self.total_time_ms,
            "errors": self.errors,
            "final_error": self.final_error,
        }


@dataclass
class CircuitBreakerState:
    """
    State snapshot of a circuit breaker.

    Represents the current state of a circuit breaker including
    its operational state, failure/success counts, and timing information.

    Attributes
    ----------
    state : CircuitState
        Current circuit state: CLOSED (normal), OPEN (blocking),
        or HALF_OPEN (testing recovery).
    failure_count : int
        Number of consecutive failures recorded.
    success_count : int
        Total number of successful executions.
    last_failure_time : Optional[datetime]
        Timestamp of most recent failure, None if never failed.
    last_success_time : Optional[datetime]
        Timestamp of most recent success, None if never succeeded.
    half_open_successes : int
        Number of successful calls in half-open state.
        Used to determine when to close the circuit.

    Examples
    --------
    Monitoring circuit breaker health:

        >>> breaker = RateLimitCircuitBreaker(failure_threshold=5)
        >>> state = breaker.get_state()
        >>> if state.state == CircuitState.OPEN:
        ...     print(f"Circuit open! {state.failure_count} failures")

    Checking recovery progress:

        >>> state = breaker.get_state()
        >>> if state.state == CircuitState.HALF_OPEN:
        ...     print(f"{state.half_open_successes} successful test calls")

    Logging circuit breaker metrics:

        >>> state = breaker.get_state()
        >>> metrics = state.to_dict()
        >>> log_metrics("circuit_breaker", metrics)

    Calculating failure rate:

        >>> state = breaker.get_state()
        >>> total = state.failure_count + state.success_count
        >>> if total > 0:
        ...     failure_rate = state.failure_count / total
    """

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    half_open_successes: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert state to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with ISO format datetimes.

        Examples
        --------
            >>> state = breaker.get_state()
            >>> d = state.to_dict()
            >>> d['state'] in ['closed', 'open', 'half_open']
            True
        """
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
    """
    Statistics for rate limiting operations.

    Tracks cumulative metrics about rate limiter usage including
    request counts, throttling, and token consumption.

    Attributes
    ----------
    total_requests : int
        Total number of acquire attempts. Default is 0.
    allowed_requests : int
        Number of requests that were allowed through. Default is 0.
    throttled_requests : int
        Number of requests that were rate limited. Default is 0.
    total_wait_time_ms : float
        Cumulative wait time in milliseconds. Default is 0.0.
    tokens_consumed : int
        Total tokens consumed from the bucket. Default is 0.

    Examples
    --------
    Getting and analyzing rate limiter statistics:

        >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        >>> for _ in range(100):
        ...     limiter.acquire(block=False)
        >>> stats = limiter.get_stats()
        >>> print(f"Allowed: {stats.allowed_requests}/{stats.total_requests}")

    Monitoring throttle rate:

        >>> stats = limiter.get_stats()
        >>> if stats.throttle_rate > 0.1:
        ...     print(f"Warning: {stats.throttle_rate:.1%} requests throttled")

    Tracking wait time:

        >>> stats = limiter.get_stats()
        >>> avg_wait = stats.total_wait_time_ms / max(1, stats.allowed_requests)
        >>> print(f"Average wait: {avg_wait:.2f}ms")

    Exporting stats for monitoring:

        >>> stats = limiter.get_stats()
        >>> prometheus_metrics.update(stats.to_dict())
    """

    total_requests: int = 0
    allowed_requests: int = 0
    throttled_requests: int = 0
    total_wait_time_ms: float = 0.0
    tokens_consumed: int = 0

    @property
    def throttle_rate(self) -> float:
        """
        Calculate the throttle rate.

        Returns
        -------
        float
            Fraction of requests that were throttled (0.0 to 1.0).
            Returns 0.0 if no requests have been made.

        Examples
        --------
            >>> stats = RateLimitStats(total_requests=100, throttled_requests=25)
            >>> stats.throttle_rate
            0.25
        """
        if self.total_requests == 0:
            return 0.0
        return self.throttled_requests / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """
        Convert statistics to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation including computed throttle_rate.

        Examples
        --------
            >>> stats = limiter.get_stats()
            >>> d = stats.to_dict()
            >>> 'throttle_rate' in d
            True
        """
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

    Implements the token bucket algorithm for rate limiting. Tokens are
    added to a bucket at a constant rate, and requests consume tokens.
    Allows bursts up to bucket capacity while maintaining a long-term
    average rate.

    This implementation is thread-safe and supports both synchronous
    and asynchronous usage patterns.

    Attributes
    ----------
    rate : float
        Token refill rate per second.
    capacity : int
        Maximum bucket capacity (burst size).

    Examples
    --------
    Basic rate limiting:

        >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        >>> # Make a request
        >>> if limiter.acquire(tokens=1, block=False):
        ...     make_api_call()
        ... else:
        ...     handle_rate_limit()

    Blocking acquisition (waits for tokens):

        >>> limiter = TokenBucketRateLimiter(rate=5.0, capacity=10)
        >>> # This will block until tokens are available
        >>> limiter.acquire(tokens=1, block=True)
        True
        >>> make_api_call()

    Async usage:

        >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        >>> async def rate_limited_call():
        ...     await limiter.acquire_async(tokens=1)
        ...     return await async_api_call()

    Checking rate limiter state:

        >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        >>> state = limiter.get_state()
        >>> print(f"Available: {state.available_tokens}, Limited: {state.is_limited}")
    """

    def __init__(
        self,
        rate: float = 10.0,  # tokens per second
        capacity: int = 20,  # bucket capacity
    ):
        """
        Initialize token bucket rate limiter.

        Parameters
        ----------
        rate : float, optional
            Token refill rate per second. Default is 10.0.
            Higher values allow more requests per second.
        capacity : int, optional
            Maximum bucket capacity. Default is 20.
            Controls maximum burst size allowed.

        Examples
        --------
            >>> # High-throughput limiter
            >>> limiter = TokenBucketRateLimiter(rate=100.0, capacity=200)

            >>> # Conservative limiter
            >>> limiter = TokenBucketRateLimiter(rate=1.0, capacity=5)
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

        Attempts to acquire the specified number of tokens. If blocking
        is enabled and tokens are not available, waits until they are.

        Parameters
        ----------
        tokens : int, optional
            Number of tokens to acquire. Default is 1.
        block : bool, optional
            Whether to wait for tokens if not immediately available.
            Default is True.

        Returns
        -------
        bool
            True if tokens were acquired, False if non-blocking and
            tokens were not available.

        Examples
        --------
        Non-blocking check:

            >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
            >>> if limiter.acquire(tokens=1, block=False):
            ...     print("Token acquired")
            ... else:
            ...     print("Rate limited, try later")

        Blocking acquisition:

            >>> limiter.acquire(tokens=1, block=True)  # Waits if needed
            True

        Acquiring multiple tokens:

            >>> # For requests that count as multiple units
            >>> limiter.acquire(tokens=5, block=True)
            True
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
        """
        Asynchronously acquire tokens from bucket.

        Async version of acquire() that uses asyncio.sleep() for non-blocking
        waits, suitable for use in async/await code.

        Parameters
        ----------
        tokens : int, optional
            Number of tokens to acquire. Default is 1.
        block : bool, optional
            Whether to wait for tokens if not immediately available.
            Default is True.

        Returns
        -------
        bool
            True if tokens were acquired, False if non-blocking and
            tokens were not available.

        Examples
        --------
        Basic async usage:

            >>> async def make_request():
            ...     limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
            ...     await limiter.acquire_async(tokens=1)
            ...     return await async_api_call()

        Non-blocking async check:

            >>> async def try_request():
            ...     if await limiter.acquire_async(tokens=1, block=False):
            ...         return await async_api_call()
            ...     return None

        Rate-limited async loop:

            >>> async def process_items(items):
            ...     limiter = TokenBucketRateLimiter(rate=5.0)
            ...     for item in items:
            ...         await limiter.acquire_async()
            ...         await process(item)
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
        """
        Refill tokens based on elapsed time.

        Internal method that adds tokens to the bucket based on time
        elapsed since last update. Called automatically before token
        checks.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def get_state(self) -> RateLimitState:
        """
        Get current rate limiter state.

        Returns a snapshot of the rate limiter's current state including
        available tokens and whether the limiter is currently blocking.

        Returns
        -------
        RateLimitState
            Current state snapshot.

        Examples
        --------
            >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
            >>> state = limiter.get_state()
            >>> print(f"Tokens: {state.available_tokens}, Limited: {state.is_limited}")
        """
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
        """
        Get cumulative statistics.

        Returns
        -------
        RateLimitStats
            Statistics about rate limiter usage.

        Examples
        --------
            >>> limiter = TokenBucketRateLimiter()
            >>> limiter.acquire()
            True
            >>> stats = limiter.get_stats()
            >>> stats.allowed_requests
            1
        """
        return self._stats

    def reset(self):
        """
        Reset rate limiter to initial state.

        Refills the token bucket to capacity and resets timing.
        Does not reset statistics.

        Examples
        --------
            >>> limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
            >>> # Exhaust tokens
            >>> for _ in range(20):
            ...     limiter.acquire(block=False)
            >>> state = limiter.get_state()
            >>> state.available_tokens < 1
            True
            >>> limiter.reset()
            >>> state = limiter.get_state()
            >>> state.available_tokens == 20.0
            True
        """
        with self._lock:
            self._tokens = float(self.capacity)
            self._last_update = time.monotonic()


class ThreadSafeSlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in a sliding time window, providing smoother rate
    limiting than fixed windows. Prevents the boundary burst problem
    where requests cluster at window edges.

    Thread-safe implementation using a deque to track request timestamps.

    Attributes
    ----------
    max_requests : int
        Maximum requests allowed in the window.
    window_size : float
        Window size in seconds.

    Examples
    --------
    Basic sliding window limiting:

        >>> limiter = ThreadSafeSlidingWindowRateLimiter(
        ...     requests_per_second=10.0,
        ...     window_size_seconds=1.0
        ... )
        >>> if limiter.acquire(block=False):
        ...     make_api_call()

    Blocking acquisition:

        >>> limiter = ThreadSafeSlidingWindowRateLimiter(requests_per_second=5.0)
        >>> limiter.acquire(block=True)  # Waits if at limit
        True

    Async usage:

        >>> async def rate_limited_call():
        ...     await limiter.acquire_async()
        ...     return await async_api_call()

    Monitoring window state:

        >>> limiter = ThreadSafeSlidingWindowRateLimiter(requests_per_second=10.0)
        >>> state = limiter.get_state()
        >>> print(f"{state.requests_in_window} requests in window")
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        window_size_seconds: float = 1.0,
    ):
        """
        Initialize sliding window rate limiter.

        Parameters
        ----------
        requests_per_second : float, optional
            Maximum requests per second. Default is 10.0.
        window_size_seconds : float, optional
            Window size in seconds. Default is 1.0.
            Larger windows provide smoother limiting.

        Examples
        --------
            >>> # Standard rate limiter
            >>> limiter = ThreadSafeSlidingWindowRateLimiter(
            ...     requests_per_second=10.0
            ... )

            >>> # Longer window for smoother limiting
            >>> limiter = ThreadSafeSlidingWindowRateLimiter(
            ...     requests_per_second=100.0,
            ...     window_size_seconds=10.0
            ... )
        """
        self.max_requests = int(requests_per_second * window_size_seconds)
        self.window_size = window_size_seconds
        self._requests: deque = deque()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()

    def acquire(self, block: bool = True) -> bool:
        """
        Try to acquire a request slot.

        Attempts to record a request in the sliding window. If the window
        is full and blocking is enabled, waits until a slot opens.

        Parameters
        ----------
        block : bool, optional
            Whether to wait for a slot if not immediately available.
            Default is True.

        Returns
        -------
        bool
            True if request was recorded, False if non-blocking and
            window was full.

        Examples
        --------
        Non-blocking check:

            >>> limiter = ThreadSafeSlidingWindowRateLimiter(requests_per_second=5.0)
            >>> if limiter.acquire(block=False):
            ...     make_request()
            ... else:
            ...     queue_for_later()

        Blocking wait:

            >>> limiter.acquire(block=True)  # Waits until slot available
            True
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
        """
        Asynchronously acquire a request slot.

        Async version of acquire() that uses asyncio.sleep() for
        non-blocking waits.

        Parameters
        ----------
        block : bool, optional
            Whether to wait for a slot if not immediately available.
            Default is True.

        Returns
        -------
        bool
            True if request was recorded, False if non-blocking and
            window was full.

        Examples
        --------
            >>> async def rate_limited_request():
            ...     limiter = ThreadSafeSlidingWindowRateLimiter(
            ...         requests_per_second=10.0
            ...     )
            ...     await limiter.acquire_async()
            ...     return await make_async_request()
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
        """
        Remove expired requests from the window.

        Internal method that removes request timestamps older than the
        window size. Called automatically before window checks.
        """
        cutoff = time.monotonic() - self.window_size
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def get_state(self) -> RateLimitState:
        """
        Get current rate limiter state.

        Returns a snapshot of the sliding window state including
        request count and wait time.

        Returns
        -------
        RateLimitState
            Current state snapshot.

        Examples
        --------
            >>> limiter = ThreadSafeSlidingWindowRateLimiter(requests_per_second=10.0)
            >>> state = limiter.get_state()
            >>> print(f"Requests in window: {state.requests_in_window}")
        """
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
        """
        Get cumulative statistics.

        Returns
        -------
        RateLimitStats
            Statistics about rate limiter usage.

        Examples
        --------
            >>> limiter = ThreadSafeSlidingWindowRateLimiter()
            >>> stats = limiter.get_stats()
            >>> print(f"Throttle rate: {stats.throttle_rate:.1%}")
        """
        return self._stats

    def reset(self):
        """
        Reset rate limiter.

        Clears all recorded requests, allowing immediate full capacity.
        Does not reset statistics.

        Examples
        --------
            >>> limiter = ThreadSafeSlidingWindowRateLimiter(requests_per_second=5.0)
            >>> limiter.reset()  # Clear all recorded requests
        """
        with self._lock:
            self._requests.clear()


class RetryHandler:
    """
    Handles retries with configurable backoff strategies.

    Provides flexible retry logic with multiple backoff strategies,
    jitter support, and error filtering. Suitable for handling
    transient failures in network operations and API calls.

    Attributes
    ----------
    config : RateLimitRetryConfig
        Configuration for retry behavior.

    Examples
    --------
    Basic retry with default configuration:

        >>> handler = RetryHandler()
        >>> result = handler.execute(lambda: flaky_api_call())
        >>> if result.success:
        ...     print(f"Got {result.result}")
        ... else:
        ...     print(f"Failed after {result.attempts} attempts")

    Custom exponential backoff:

        >>> config = RateLimitRetryConfig(
        ...     max_retries=5,
        ...     base_delay=0.5,
        ...     max_delay=30.0,
        ...     strategy=RetryStrategy.EXPONENTIAL,
        ...     jitter=True
        ... )
        >>> handler = RetryHandler(config)
        >>> result = handler.execute(api_call)

    Retry only specific errors:

        >>> config = RateLimitRetryConfig(
        ...     max_retries=3,
        ...     retryable_errors=[ConnectionError, TimeoutError]
        ... )
        >>> handler = RetryHandler(config)
        >>> # Only retries ConnectionError and TimeoutError

    With retry callback:

        >>> def on_retry(attempt, error):
        ...     print(f"Retry {attempt}: {error}")
        >>> result = handler.execute(api_call, on_retry=on_retry)
    """

    def __init__(self, config: Optional[RateLimitRetryConfig] = None):
        """
        Initialize retry handler.

        Parameters
        ----------
        config : RateLimitRetryConfig, optional
            Retry configuration. If None, uses default configuration
            with exponential backoff and 3 retries.

        Examples
        --------
            >>> # Default configuration
            >>> handler = RetryHandler()

            >>> # Custom configuration
            >>> config = RateLimitRetryConfig(max_retries=5)
            >>> handler = RetryHandler(config)
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

        Executes the provided function, retrying on failure according
        to the configured strategy until success or max retries reached.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute. Should take no arguments.
        on_retry : Callable[[int, Exception], None], optional
            Callback invoked before each retry. Receives attempt number
            (1-indexed) and the exception that triggered the retry.

        Returns
        -------
        RateLimitRetryResult
            Result object containing success status, result/error,
            attempt count, and timing information.

        Examples
        --------
        Basic execution:

            >>> handler = RetryHandler()
            >>> result = handler.execute(lambda: api_call())
            >>> if result.success:
            ...     data = result.result

        With retry callback:

            >>> def log_retry(attempt, error):
            ...     logging.warning(f"Attempt {attempt} failed: {error}")
            >>> result = handler.execute(api_call, on_retry=log_retry)

        Handling failure:

            >>> result = handler.execute(unreliable_function)
            >>> if not result.success:
            ...     print(f"All {result.attempts} attempts failed")
            ...     raise Exception(result.final_error)
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
        """
        Asynchronously execute function with retries.

        Async version of execute() that properly handles coroutine
        functions and uses asyncio.sleep() for delays.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute. Can be sync or async.
        on_retry : Callable[[int, Exception], None], optional
            Callback invoked before each retry.

        Returns
        -------
        RateLimitRetryResult
            Result object with outcome details.

        Examples
        --------
            >>> async def main():
            ...     handler = RetryHandler()
            ...     result = await handler.execute_async(async_api_call)
            ...     return result.result if result.success else None
        """
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
        """
        Calculate delay for retry attempt.

        Computes the delay in seconds based on the configured strategy,
        applying jitter if enabled.

        Parameters
        ----------
        attempt : int
            Zero-indexed attempt number.

        Returns
        -------
        float
            Delay in seconds before next retry.

        Examples
        --------
        Exponential delays (base_delay=1.0):
            - attempt 0: 1s
            - attempt 1: 2s
            - attempt 2: 4s
            - attempt 3: 8s
        """
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
        """
        Get nth Fibonacci number.

        Uses cached computation for efficiency across multiple calls.

        Parameters
        ----------
        n : int
            Index in Fibonacci sequence (0-indexed).

        Returns
        -------
        int
            The nth Fibonacci number.
        """
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(self._fibonacci_cache[-1] + self._fibonacci_cache[-2])
        return self._fibonacci_cache[n]


class RateLimitCircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Implements the circuit breaker pattern to prevent cascading failures.
    After a threshold of failures, the circuit "opens" and rejects requests
    immediately, giving the failing service time to recover.

    States
    ------
    CLOSED : Normal operation, requests pass through
    OPEN : Circuit tripped, requests rejected immediately
    HALF_OPEN : Testing recovery, limited requests allowed

    Attributes
    ----------
    failure_threshold : int
        Number of failures before opening circuit.
    recovery_timeout : float
        Seconds before attempting recovery from open state.
    half_open_max_calls : int
        Successful calls needed in half-open state to close circuit.

    Examples
    --------
    Basic circuit breaker usage:

        >>> breaker = RateLimitCircuitBreaker(
        ...     failure_threshold=5,
        ...     recovery_timeout=30.0
        ... )
        >>> try:
        ...     result = breaker.execute(lambda: api_call())
        ... except CircuitOpenError:
        ...     result = use_fallback()

    Checking circuit state:

        >>> breaker = RateLimitCircuitBreaker()
        >>> if breaker.can_execute():
        ...     try:
        ...         result = make_request()
        ...         breaker.record_success()
        ...     except Exception:
        ...         breaker.record_failure()
        ...         raise

    Monitoring circuit health:

        >>> state = breaker.get_state()
        >>> if state.state == CircuitState.OPEN:
        ...     alert("Circuit breaker open for service X")

    Async usage:

        >>> async def protected_call():
        ...     try:
        ...         return await breaker.execute_async(async_api_call)
        ...     except CircuitOpenError:
        ...         return await fallback()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Parameters
        ----------
        failure_threshold : int, optional
            Number of consecutive failures before opening circuit.
            Default is 5.
        recovery_timeout : float, optional
            Seconds to wait before testing recovery. Default is 30.0.
        half_open_max_calls : int, optional
            Successful calls in half-open state needed to close
            the circuit. Default is 3.

        Examples
        --------
            >>> # Sensitive circuit breaker
            >>> breaker = RateLimitCircuitBreaker(
            ...     failure_threshold=3,
            ...     recovery_timeout=60.0
            ... )

            >>> # Resilient circuit breaker
            >>> breaker = RateLimitCircuitBreaker(
            ...     failure_threshold=10,
            ...     recovery_timeout=10.0,
            ...     half_open_max_calls=5
            ... )
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
        """
        Check if execution is allowed.

        Determines if a request can proceed based on circuit state.
        In OPEN state, checks if recovery timeout has elapsed.

        Returns
        -------
        bool
            True if request is allowed, False if circuit is open.

        Examples
        --------
            >>> breaker = RateLimitCircuitBreaker()
            >>> if breaker.can_execute():
            ...     make_request()
        """
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
        """
        Record a successful execution.

        Should be called after a successful operation. In half-open state,
        accumulates successes toward closing the circuit.

        Examples
        --------
            >>> breaker = RateLimitCircuitBreaker()
            >>> try:
            ...     result = make_request()
            ...     breaker.record_success()
            ... except Exception:
            ...     breaker.record_failure()
        """
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
        """
        Record a failed execution.

        Should be called after a failed operation. Accumulates toward
        opening the circuit in closed state, or immediately reopens
        in half-open state.

        Examples
        --------
            >>> breaker = RateLimitCircuitBreaker(failure_threshold=3)
            >>> for _ in range(3):
            ...     breaker.record_failure()
            >>> breaker.get_state().state == CircuitState.OPEN
            True
        """
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
        Execute function with circuit breaker protection.

        Executes the function if circuit is closed or half-open,
        automatically recording success or failure.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute.

        Returns
        -------
        T
            Function result.

        Raises
        ------
        CircuitOpenError
            If circuit is open and not accepting requests.

        Examples
        --------
            >>> breaker = RateLimitCircuitBreaker()
            >>> try:
            ...     result = breaker.execute(lambda: api_call())
            ... except CircuitOpenError:
            ...     result = use_fallback()
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
        """
        Asynchronously execute function with circuit breaker.

        Async version of execute() that handles coroutine functions.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute. Can be sync or async.

        Returns
        -------
        T
            Function result.

        Raises
        ------
        CircuitOpenError
            If circuit is open.

        Examples
        --------
            >>> async def protected_call():
            ...     return await breaker.execute_async(async_api_call)
        """
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
        """
        Get current circuit breaker state.

        Returns
        -------
        CircuitBreakerState
            Current state snapshot.

        Examples
        --------
            >>> breaker = RateLimitCircuitBreaker()
            >>> state = breaker.get_state()
            >>> print(f"State: {state.state}, Failures: {state.failure_count}")
        """
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
        """
        Reset circuit breaker to closed state.

        Clears failure count and closes the circuit. Use for manual
        recovery or testing.

        Examples
        --------
            >>> breaker = RateLimitCircuitBreaker()
            >>> # Force open then reset
            >>> for _ in range(10):
            ...     breaker.record_failure()
            >>> breaker.reset()
            >>> breaker.get_state().state == CircuitState.CLOSED
            True
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_successes = 0


class CircuitOpenError(Exception):
    """
    Raised when circuit breaker is open.

    This exception indicates that the circuit breaker has tripped due to
    too many failures, and requests are being rejected to protect the
    system from cascading failures.

    Examples
    --------
    Handling circuit open:

        >>> breaker = RateLimitCircuitBreaker(failure_threshold=3)
        >>> try:
        ...     result = breaker.execute(failing_function)
        ... except CircuitOpenError as e:
        ...     print(f"Circuit open: {e}")
        ...     result = fallback_value

    Checking before execution:

        >>> if not breaker.can_execute():
        ...     # Circuit is open, use alternative
        ...     pass
    """

    pass


class RequestQueue:
    """
    Priority queue for requests with rate limiting.

    Manages a queue of pending requests with priority ordering and
    integrated rate limiting. Higher priority requests are processed
    first, and all requests are subject to the configured rate limit.

    Attributes
    ----------
    rate_limiter : TokenBucketRateLimiter
        Rate limiter applied to dequeued requests.
    max_queue_size : int
        Maximum number of pending requests.

    Examples
    --------
    Basic queue usage:

        >>> queue = RequestQueue(max_queue_size=100)
        >>> queue.enqueue(lambda: api_call_1())
        True
        >>> queue.enqueue(lambda: api_call_2())
        True
        >>> result = queue.process_one()

    Priority ordering:

        >>> queue = RequestQueue()
        >>> queue.enqueue(lambda: low_priority(), priority=RequestPriority.LOW)
        True
        >>> queue.enqueue(lambda: high_priority(), priority=RequestPriority.HIGH)
        True
        >>> # High priority processed first
        >>> result = queue.process_one()

    Processing all queued requests:

        >>> queue = RequestQueue()
        >>> for task in tasks:
        ...     queue.enqueue(task)
        >>> results = queue.process_all()

    Monitoring queue:

        >>> stats = queue.get_stats()
        >>> print(f"Queue size: {stats['queue_size']}")
        >>> print(f"Dropped: {stats['dropped_count']}")
    """

    def __init__(
        self,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        max_queue_size: int = 1000,
    ):
        """
        Initialize request queue.

        Parameters
        ----------
        rate_limiter : TokenBucketRateLimiter, optional
            Rate limiter to use. If None, creates a default limiter.
        max_queue_size : int, optional
            Maximum queue size. Default is 1000.
            Requests beyond this limit are dropped.

        Examples
        --------
            >>> # Default configuration
            >>> queue = RequestQueue()

            >>> # Custom rate limiter
            >>> limiter = TokenBucketRateLimiter(rate=5.0, capacity=10)
            >>> queue = RequestQueue(rate_limiter=limiter, max_queue_size=500)
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

        Enqueues a function for later execution, ordered by priority
        and then by arrival time.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute when dequeued.
        priority : RequestPriority, optional
            Request priority. Default is NORMAL.

        Returns
        -------
        bool
            True if enqueued, False if queue is full.

        Examples
        --------
            >>> queue = RequestQueue()
            >>> queue.enqueue(lambda: api_call())
            True
            >>> queue.enqueue(
            ...     lambda: urgent_call(),
            ...     priority=RequestPriority.HIGH
            ... )
            True
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
        """
        Process one request from queue.

        Dequeues the highest priority request, acquires rate limit,
        and executes the function.

        Returns
        -------
        Optional[Any]
            Function result, or None if queue is empty.

        Examples
        --------
            >>> queue = RequestQueue()
            >>> queue.enqueue(lambda: "result")
            True
            >>> queue.process_one()
            'result'
        """
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
        """
        Asynchronously process one request from queue.

        Async version of process_one().

        Returns
        -------
        Optional[Any]
            Function result, or None if queue is empty.

        Examples
        --------
            >>> async def process():
            ...     queue = RequestQueue()
            ...     queue.enqueue(async_task)
            ...     return await queue.process_one_async()
        """
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
        """
        Process all queued requests.

        Processes requests in priority order until queue is empty.

        Returns
        -------
        list[Any]
            List of results from all processed requests.

        Examples
        --------
            >>> queue = RequestQueue()
            >>> queue.enqueue(lambda: 1)
            True
            >>> queue.enqueue(lambda: 2)
            True
            >>> queue.process_all()
            [1, 2]
        """
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
        """
        Get current queue size.

        Returns
        -------
        int
            Number of pending requests.

        Examples
        --------
            >>> queue = RequestQueue()
            >>> queue.get_queue_size()
            0
        """
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> dict[str, Any]:
        """
        Get queue statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with queue_size, processed_count, dropped_count,
            and rate_limiter_stats.

        Examples
        --------
            >>> queue = RequestQueue()
            >>> stats = queue.get_stats()
            >>> 'queue_size' in stats
            True
        """
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "processed_count": self._processed_count,
                "dropped_count": self._dropped_count,
                "rate_limiter_stats": self.rate_limiter.get_stats().to_dict(),
            }

    def clear(self):
        """
        Clear the queue.

        Removes all pending requests without processing them.

        Examples
        --------
            >>> queue = RequestQueue()
            >>> queue.enqueue(lambda: task())
            True
            >>> queue.clear()
            >>> queue.get_queue_size()
            0
        """
        with self._lock:
            self._queue.clear()


class ConcurrencyLimiter:
    """
    Limits concurrent executions.

    Controls the number of simultaneous executions using semaphores.
    Supports both sync and async usage, and can be used as a context
    manager.

    Attributes
    ----------
    max_concurrent : int
        Maximum number of concurrent executions allowed.

    Examples
    --------
    Basic concurrency limiting:

        >>> limiter = ConcurrencyLimiter(max_concurrent=5)
        >>> if limiter.acquire(block=False):
        ...     try:
        ...         result = make_request()
        ...     finally:
        ...         limiter.release()

    As context manager:

        >>> limiter = ConcurrencyLimiter(max_concurrent=10)
        >>> with limiter:
        ...     result = make_request()

    Async context manager:

        >>> async def concurrent_request():
        ...     limiter = ConcurrencyLimiter(max_concurrent=5)
        ...     async with limiter:
        ...         return await async_request()

    Monitoring concurrency:

        >>> limiter = ConcurrencyLimiter(max_concurrent=10)
        >>> print(f"Active: {limiter.get_current_count()}")
        >>> print(f"Available: {limiter.get_available()}")
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize concurrency limiter.

        Parameters
        ----------
        max_concurrent : int, optional
            Maximum concurrent executions allowed. Default is 10.

        Examples
        --------
            >>> # Limit to 5 concurrent requests
            >>> limiter = ConcurrencyLimiter(max_concurrent=5)

            >>> # High concurrency limit
            >>> limiter = ConcurrencyLimiter(max_concurrent=100)
        """
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._current_count = 0
        self._lock = threading.Lock()

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire a concurrency slot.

        Parameters
        ----------
        block : bool, optional
            Whether to wait for a slot. Default is True.
        timeout : float, optional
            Maximum seconds to wait. None means wait forever.

        Returns
        -------
        bool
            True if slot acquired, False if non-blocking and unavailable.

        Examples
        --------
            >>> limiter = ConcurrencyLimiter(max_concurrent=5)
            >>> limiter.acquire(block=True)
            True
        """
        acquired = self._semaphore.acquire(blocking=block, timeout=timeout)
        if acquired:
            with self._lock:
                self._current_count += 1
        return acquired

    def release(self):
        """
        Release a concurrency slot.

        Should be called after acquire() when done with the resource.

        Examples
        --------
            >>> limiter = ConcurrencyLimiter()
            >>> limiter.acquire()
            True
            >>> # do work
            >>> limiter.release()
        """
        self._semaphore.release()
        with self._lock:
            self._current_count = max(0, self._current_count - 1)

    async def acquire_async(self) -> bool:
        """
        Asynchronously acquire a concurrency slot.

        Returns
        -------
        bool
            Always True (async semaphore always waits).

        Examples
        --------
            >>> async def limited_work():
            ...     await limiter.acquire_async()
            ...     try:
            ...         await do_work()
            ...     finally:
            ...         await limiter.release_async()
        """
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_concurrent)
        await self._async_semaphore.acquire()
        with self._lock:
            self._current_count += 1
        return True

    async def release_async(self):
        """
        Asynchronously release a concurrency slot.

        Examples
        --------
            >>> await limiter.release_async()
        """
        if self._async_semaphore:
            self._async_semaphore.release()
        with self._lock:
            self._current_count = max(0, self._current_count - 1)

    def __enter__(self):
        """
        Context manager entry.

        Acquires a slot on entry.

        Examples
        --------
            >>> with ConcurrencyLimiter(max_concurrent=5) as limiter:
            ...     make_request()
        """
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.

        Releases the slot on exit.
        """
        self.release()
        return False

    async def __aenter__(self):
        """
        Async context manager entry.

        Examples
        --------
            >>> async with ConcurrencyLimiter(max_concurrent=5):
            ...     await async_request()
        """
        await self.acquire_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release_async()
        return False

    def get_current_count(self) -> int:
        """
        Get current concurrent execution count.

        Returns
        -------
        int
            Number of currently active executions.

        Examples
        --------
            >>> limiter = ConcurrencyLimiter(max_concurrent=10)
            >>> limiter.get_current_count()
            0
        """
        with self._lock:
            return self._current_count

    def get_available(self) -> int:
        """
        Get available slots.

        Returns
        -------
        int
            Number of slots available for new executions.

        Examples
        --------
            >>> limiter = ConcurrencyLimiter(max_concurrent=10)
            >>> limiter.get_available()
            10
        """
        with self._lock:
            return self.max_concurrent - self._current_count


class RateLimitedExecutor:
    """
    Combined rate limiting, retry, and circuit breaker execution.

    Provides a unified executor that combines multiple protection
    mechanisms: rate limiting, automatic retries, circuit breaker,
    and concurrency limiting.

    Attributes
    ----------
    rate_limiter : TokenBucketRateLimiter, optional
        Rate limiter for request throttling.
    retry_handler : RetryHandler, optional
        Handler for automatic retries.
    circuit_breaker : RateLimitCircuitBreaker, optional
        Circuit breaker for fault tolerance.
    concurrency_limiter : ConcurrencyLimiter, optional
        Limiter for concurrent executions.

    Examples
    --------
    Creating a fully configured executor:

        >>> executor = RateLimitedExecutor(
        ...     rate_limiter=TokenBucketRateLimiter(rate=10.0),
        ...     retry_handler=RetryHandler(),
        ...     circuit_breaker=RateLimitCircuitBreaker(),
        ...     concurrency_limiter=ConcurrencyLimiter(max_concurrent=5)
        ... )
        >>> result = executor.execute(lambda: api_call())

    Using the create_executor helper:

        >>> from insideLLMs.rate_limiting import create_executor
        >>> executor = create_executor(
        ...     rate=10.0,
        ...     max_retries=3,
        ...     failure_threshold=5,
        ...     max_concurrent=10
        ... )
        >>> result = executor.execute(api_call)

    Async execution:

        >>> async def protected_call():
        ...     executor = create_executor(rate=5.0)
        ...     return await executor.execute_async(async_api_call)

    Selective protection:

        >>> # Only rate limiting and retries
        >>> executor = RateLimitedExecutor(
        ...     rate_limiter=TokenBucketRateLimiter(rate=5.0),
        ...     retry_handler=RetryHandler()
        ... )
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

        Parameters
        ----------
        rate_limiter : TokenBucketRateLimiter, optional
            Rate limiter for request throttling. If None, no rate
            limiting is applied.
        retry_handler : RetryHandler, optional
            Handler for automatic retries. If None, no retries.
        circuit_breaker : RateLimitCircuitBreaker, optional
            Circuit breaker for fault tolerance. If None, no circuit
            breaking.
        concurrency_limiter : ConcurrencyLimiter, optional
            Limiter for concurrent executions. If None, no concurrency
            limit.

        Examples
        --------
            >>> # Full protection
            >>> executor = RateLimitedExecutor(
            ...     rate_limiter=TokenBucketRateLimiter(rate=10.0),
            ...     retry_handler=RetryHandler(),
            ...     circuit_breaker=RateLimitCircuitBreaker(),
            ...     concurrency_limiter=ConcurrencyLimiter()
            ... )

            >>> # Just rate limiting
            >>> executor = RateLimitedExecutor(
            ...     rate_limiter=TokenBucketRateLimiter(rate=5.0)
            ... )
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

        Applies configured protections in order: circuit breaker check,
        rate limiting, concurrency limiting, and retry logic.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute.
        tokens : int, optional
            Tokens to consume from rate limiter. Default is 1.

        Returns
        -------
        T
            Function result.

        Raises
        ------
        CircuitOpenError
            If circuit breaker is open.
        Exception
            If all retry attempts fail.

        Examples
        --------
            >>> executor = create_executor(rate=10.0, max_retries=3)
            >>> result = executor.execute(lambda: api_call())

            >>> # With token cost
            >>> result = executor.execute(
            ...     lambda: expensive_call(),
            ...     tokens=5
            ... )
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
        """
        Asynchronously execute with all protection mechanisms.

        Async version of execute() that properly handles coroutine
        functions.

        Parameters
        ----------
        func : Callable[[], T]
            Function to execute. Can be sync or async.
        tokens : int, optional
            Tokens to consume from rate limiter. Default is 1.

        Returns
        -------
        T
            Function result.

        Raises
        ------
        CircuitOpenError
            If circuit breaker is open.
        Exception
            If all retry attempts fail.

        Examples
        --------
            >>> async def protected_call():
            ...     executor = create_executor(rate=5.0)
            ...     return await executor.execute_async(async_api_call)
        """
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
    Decorator for rate limiting functions.

    Creates a token bucket rate limiter and applies it to the decorated
    function. Works with both sync and async functions.

    Parameters
    ----------
    rate : float, optional
        Tokens per second. Default is 10.0.
    capacity : int, optional
        Bucket capacity. Default is 20.

    Returns
    -------
    Callable
        Decorated function with rate limiting.

    Examples
    --------
    Rate limiting a sync function:

        >>> @rate_limited(rate=5.0, capacity=10)
        ... def api_call():
        ...     return requests.get("https://api.example.com")

    Rate limiting an async function:

        >>> @rate_limited(rate=10.0)
        ... async def async_api_call():
        ...     async with aiohttp.ClientSession() as session:
        ...         return await session.get("https://api.example.com")

    Strict rate limiting:

        >>> @rate_limited(rate=1.0, capacity=1)  # 1 request per second
        ... def slow_api():
        ...     return make_request()
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
    Decorator for automatic retry with backoff.

    Wraps a function with retry logic using the specified strategy.
    Works with both sync and async functions.

    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retries. Default is 3.
    base_delay : float, optional
        Base delay in seconds. Default is 1.0.
    strategy : RetryStrategy, optional
        Backoff strategy. Default is EXPONENTIAL.

    Returns
    -------
    Callable
        Decorated function with retry logic.

    Raises
    ------
    Exception
        If all retries fail.

    Examples
    --------
    Exponential backoff retry:

        >>> @with_retry(max_retries=5, base_delay=0.5)
        ... def flaky_api_call():
        ...     return external_service.call()

    Linear backoff:

        >>> @with_retry(
        ...     max_retries=3,
        ...     base_delay=2.0,
        ...     strategy=RetryStrategy.LINEAR
        ... )
        ... def database_operation():
        ...     return db.execute(query)

    Async with retry:

        >>> @with_retry(max_retries=3)
        ... async def async_fetch():
        ...     return await fetch_data()
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

    Wraps a function with circuit breaker logic. After the threshold
    of failures, the circuit opens and calls fail fast.

    Parameters
    ----------
    failure_threshold : int, optional
        Failures before opening circuit. Default is 5.
    recovery_timeout : float, optional
        Seconds before testing recovery. Default is 30.0.

    Returns
    -------
    Callable
        Decorated function with circuit breaker.

    Raises
    ------
    CircuitOpenError
        If circuit is open.

    Examples
    --------
    Basic circuit breaker:

        >>> @circuit_protected(failure_threshold=3)
        ... def external_api():
        ...     return requests.get("https://unstable-api.com")

    Longer recovery timeout:

        >>> @circuit_protected(
        ...     failure_threshold=5,
        ...     recovery_timeout=60.0
        ... )
        ... def critical_service():
        ...     return call_critical_service()

    With fallback handling:

        >>> @circuit_protected(failure_threshold=3)
        ... def api_call():
        ...     return external_service()
        >>>
        >>> try:
        ...     result = api_call()
        ... except CircuitOpenError:
        ...     result = use_cached_value()
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
    Create a rate limiter with the specified strategy.

    Factory function that creates the appropriate rate limiter based
    on the specified strategy.

    Parameters
    ----------
    rate : float, optional
        Rate limit (requests per second). Default is 10.0.
    capacity : int, optional
        Capacity/burst size. Default is 20.
    strategy : RateLimitStrategy, optional
        Rate limiting strategy. Default is TOKEN_BUCKET.

    Returns
    -------
    TokenBucketRateLimiter | ThreadSafeSlidingWindowRateLimiter
        Configured rate limiter.

    Examples
    --------
    Token bucket limiter:

        >>> limiter = create_rate_limiter(rate=10.0, capacity=20)
        >>> limiter.acquire()
        True

    Sliding window limiter:

        >>> limiter = create_rate_limiter(
        ...     rate=5.0,
        ...     strategy=RateLimitStrategy.SLIDING_WINDOW
        ... )

    High-throughput configuration:

        >>> limiter = create_rate_limiter(rate=100.0, capacity=200)
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
    Create a retry handler with the specified configuration.

    Factory function that creates a configured RetryHandler.

    Parameters
    ----------
    max_retries : int, optional
        Maximum number of retries. Default is 3.
    base_delay : float, optional
        Base delay in seconds. Default is 1.0.
    strategy : RetryStrategy, optional
        Backoff strategy. Default is EXPONENTIAL.

    Returns
    -------
    RetryHandler
        Configured retry handler.

    Examples
    --------
    Default exponential backoff:

        >>> handler = create_retry_handler()
        >>> result = handler.execute(api_call)

    Custom configuration:

        >>> handler = create_retry_handler(
        ...     max_retries=5,
        ...     base_delay=0.5,
        ...     strategy=RetryStrategy.FIBONACCI
        ... )

    Quick retries:

        >>> handler = create_retry_handler(
        ...     max_retries=10,
        ...     base_delay=0.1,
        ...     strategy=RetryStrategy.CONSTANT
        ... )
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
    Create a circuit breaker with the specified configuration.

    Factory function that creates a configured circuit breaker.

    Parameters
    ----------
    failure_threshold : int, optional
        Failures before opening circuit. Default is 5.
    recovery_timeout : float, optional
        Seconds before testing recovery. Default is 30.0.

    Returns
    -------
    RateLimitCircuitBreaker
        Configured circuit breaker.

    Examples
    --------
    Default configuration:

        >>> breaker = create_circuit_breaker()
        >>> result = breaker.execute(api_call)

    Sensitive configuration:

        >>> breaker = create_circuit_breaker(
        ...     failure_threshold=3,
        ...     recovery_timeout=60.0
        ... )

    Resilient configuration:

        >>> breaker = create_circuit_breaker(
        ...     failure_threshold=10,
        ...     recovery_timeout=10.0
        ... )
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
    Create a fully configured executor with all protections.

    Factory function that creates a RateLimitedExecutor with rate
    limiting, retry handling, circuit breaker, and concurrency
    limiting all configured.

    Parameters
    ----------
    rate : float, optional
        Rate limit (requests per second). Default is 10.0.
    max_retries : int, optional
        Maximum retry attempts. Default is 3.
    failure_threshold : int, optional
        Circuit breaker failure threshold. Default is 5.
    max_concurrent : int, optional
        Maximum concurrent executions. Default is 10.

    Returns
    -------
    RateLimitedExecutor
        Fully configured executor.

    Examples
    --------
    Default executor:

        >>> executor = create_executor()
        >>> result = executor.execute(api_call)

    Custom configuration:

        >>> executor = create_executor(
        ...     rate=5.0,
        ...     max_retries=5,
        ...     failure_threshold=3,
        ...     max_concurrent=5
        ... )

    High-throughput executor:

        >>> executor = create_executor(
        ...     rate=100.0,
        ...     max_concurrent=50
        ... )
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
    Execute function with exponential backoff retry.

    Convenience function for one-off execution with retry logic.
    For repeated calls, consider using RetryHandler directly.

    Parameters
    ----------
    func : Callable[[], T]
        Function to execute.
    max_retries : int, optional
        Maximum retry attempts. Default is 3.
    base_delay : float, optional
        Base delay in seconds. Default is 1.0.

    Returns
    -------
    T
        Function result.

    Raises
    ------
    Exception
        If all retries fail.

    Examples
    --------
    Simple retry:

        >>> result = execute_with_backoff(api_call)

    Custom retry parameters:

        >>> result = execute_with_backoff(
        ...     lambda: requests.get(url),
        ...     max_retries=5,
        ...     base_delay=0.5
        ... )

    With error handling:

        >>> try:
        ...     result = execute_with_backoff(flaky_operation)
        ... except Exception as e:
        ...     print(f"All retries failed: {e}")
    """
    handler = create_retry_handler(max_retries, base_delay)
    result = handler.execute(func)
    if result.success:
        return result.result
    raise Exception(result.final_error)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import CircuitBreaker. The canonical name is
# RateLimitCircuitBreaker.
CircuitBreaker = RateLimitCircuitBreaker

# Older code and tests may import RetryConfig. The canonical name is
# RateLimitRetryConfig.
RetryConfig = RateLimitRetryConfig

# Older code and tests may import RetryResult. The canonical name is
# RateLimitRetryResult.
RetryResult = RateLimitRetryResult

# Older code and tests may import SlidingWindowRateLimiter. The canonical name is
# ThreadSafeSlidingWindowRateLimiter.
SlidingWindowRateLimiter = ThreadSafeSlidingWindowRateLimiter
