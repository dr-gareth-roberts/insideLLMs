"""Retry utilities with exponential backoff for API calls.

This module provides robust retry mechanisms for handling transient failures
when interacting with LLM APIs, including:
- Configurable exponential backoff
- Jitter to prevent thundering herd
- Per-exception retry policies
- Circuit breaker pattern
- Async support
"""

import asyncio
import functools
import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Generic,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from insideLLMs.exceptions import (
    InsideLLMsError,
    RateLimitError,
    TimeoutError as ModelTimeoutError,
    is_retryable,
    get_retry_delay,
)


logger = logging.getLogger("insideLLMs.retry")


T = TypeVar("T")


class BackoffStrategy(Enum):
    """Backoff strategy options."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        exponential_base: Base for exponential backoff (default 2).
        jitter: Whether to add random jitter to delays.
        jitter_factor: Maximum jitter as fraction of delay (0.0-1.0).
        strategy: Backoff strategy to use.
        retryable_exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback called before each retry.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        RateLimitError,
        ModelTimeoutError,
        ConnectionError,
        TimeoutError,
    )
    on_retry: Optional[Callable[[Exception, int, float], None]] = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: The attempt number (1-indexed).

        Returns:
            Delay in seconds.
        """
        if self.strategy == BackoffStrategy.CONSTANT:
            delay = self.initial_delay
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay * attempt
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        elif self.strategy == BackoffStrategy.FIBONACCI:
            delay = self.initial_delay * self._fibonacci(attempt)
        else:
            delay = self.initial_delay

        # Apply max delay cap
        delay = min(delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b


@dataclass
class RetryResult(Generic[T]):
    """Result of a retry operation.

    Attributes:
        success: Whether the operation succeeded.
        result: The result if successful.
        exception: The last exception if failed.
        attempts: Number of attempts made.
        total_delay: Total delay time spent waiting.
        history: List of (attempt, exception, delay) tuples.
    """

    success: bool
    result: Optional[T] = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    history: list = field(default_factory=list)


class RetryExhaustedError(InsideLLMsError):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Exception,
        history: list,
    ):
        super().__init__(
            message,
            {
                "attempts": attempts,
                "last_exception_type": type(last_exception).__name__,
                "last_exception_message": str(last_exception),
            },
        )
        self.attempts = attempts
        self.last_exception = last_exception
        self.history = history


def retry(
    config: Optional[RetryConfig] = None,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function.

    Can be used with a config object or individual parameters.

    Args:
        config: RetryConfig object (takes precedence).
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        retryable_exceptions: Exceptions to retry on.
        on_retry: Callback before each retry.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry(max_retries=3, initial_delay=1.0)
        def call_api():
            return api.generate("Hello")

        @retry(config=RetryConfig(strategy=BackoffStrategy.FIBONACCI))
        def another_call():
            return api.generate("World")
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries if max_retries is not None else 3,
            initial_delay=initial_delay if initial_delay is not None else 1.0,
            max_delay=max_delay if max_delay is not None else 60.0,
            retryable_exceptions=retryable_exceptions or RetryConfig.retryable_exceptions,
            on_retry=on_retry,
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return execute_with_retry(func, args, kwargs, config)

        return wrapper

    return decorator


def retry_async(
    config: Optional[RetryConfig] = None,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable:
    """Async version of retry decorator.

    Same parameters as retry().
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries if max_retries is not None else 3,
            initial_delay=initial_delay if initial_delay is not None else 1.0,
            max_delay=max_delay if max_delay is not None else 60.0,
            retryable_exceptions=retryable_exceptions or RetryConfig.retryable_exceptions,
            on_retry=on_retry,
        )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await execute_with_retry_async(func, args, kwargs, config)

        return wrapper

    return decorator


def execute_with_retry(
    func: Callable[..., T],
    args: tuple,
    kwargs: dict,
    config: RetryConfig,
) -> T:
    """Execute a function with retry logic.

    Args:
        func: Function to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        config: Retry configuration.

    Returns:
        Result of the function.

    Raises:
        RetryExhaustedError: If all retries are exhausted.
    """
    history = []
    total_delay = 0.0
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 2):  # +2 because first try isn't a retry
        try:
            return func(*args, **kwargs)

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt > config.max_retries:
                # No more retries
                break

            # Check for rate limit specific delay
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = e.retry_after
            else:
                delay = config.calculate_delay(attempt)

            history.append({
                "attempt": attempt,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "delay": delay,
            })

            logger.warning(
                f"Attempt {attempt} failed: {type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            if config.on_retry:
                config.on_retry(e, attempt, delay)

            time.sleep(delay)
            total_delay += delay

        except Exception as e:
            # Non-retryable exception
            raise

    # All retries exhausted
    raise RetryExhaustedError(
        f"All {config.max_retries} retries exhausted",
        attempts=len(history) + 1,
        last_exception=last_exception,
        history=history,
    )


async def execute_with_retry_async(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: RetryConfig,
) -> Any:
    """Async version of execute_with_retry."""
    history = []
    total_delay = 0.0
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 2):
        try:
            return await func(*args, **kwargs)

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt > config.max_retries:
                break

            if isinstance(e, RateLimitError) and e.retry_after:
                delay = e.retry_after
            else:
                delay = config.calculate_delay(attempt)

            history.append({
                "attempt": attempt,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "delay": delay,
            })

            logger.warning(
                f"Attempt {attempt} failed: {type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            if config.on_retry:
                config.on_retry(e, attempt, delay)

            await asyncio.sleep(delay)
            total_delay += delay

        except Exception:
            raise

    raise RetryExhaustedError(
        f"All {config.max_retries} retries exhausted",
        attempts=len(history) + 1,
        last_exception=last_exception,
        history=history,
    )


# Circuit Breaker Pattern

class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures to open circuit.
        success_threshold: Number of successes to close circuit.
        reset_timeout: Seconds before trying again after opening.
        half_open_max_calls: Max concurrent calls in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreakerOpen(InsideLLMsError):
    """Raised when circuit breaker is open."""

    def __init__(self, circuit_name: str, time_until_reset: float):
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open. "
            f"Try again in {time_until_reset:.1f}s",
            {"circuit_name": circuit_name, "time_until_reset": time_until_reset},
        )


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed

    Example:
        circuit = CircuitBreaker("api_calls")

        @circuit
        def call_api():
            return api.generate("Hello")

        # Or use as context manager
        with circuit:
            api.generate("Hello")
    """

    def __init__(
        self,
        name: str = "default",
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_state_transition()
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.reset_timeout:
                    logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                logger.info(f"Circuit '{self.name}' transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit '{self.name}' transitioning to OPEN (half-open failure)")
            self._state = CircuitState.OPEN
            self._success_count = 0
        elif self._failure_count >= self.config.failure_threshold:
            logger.warning(f"Circuit '{self.name}' transitioning to OPEN")
            self._state = CircuitState.OPEN

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use circuit breaker as a decorator."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the function.

        Raises:
            CircuitBreakerOpen: If circuit is open.
        """
        self._check_state_transition()

        if self._state == CircuitState.OPEN:
            time_until_reset = (
                self.config.reset_timeout -
                (time.time() - (self._last_failure_time or 0))
            )
            raise CircuitBreakerOpen(self.name, max(0, time_until_reset))

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(self.name, 0)
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            raise

    @contextmanager
    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        self._check_state_transition()

        if self._state == CircuitState.OPEN:
            time_until_reset = (
                self.config.reset_timeout -
                (time.time() - (self._last_failure_time or 0))
            )
            raise CircuitBreakerOpen(self.name, max(0, time_until_reset))

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit."""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
        return False

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info(f"Circuit '{self.name}' manually reset")


# Retry with fallback

def retry_with_fallback(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    config: Optional[RetryConfig] = None,
) -> Callable[..., T]:
    """Create a function that retries then falls back to alternative.

    Args:
        primary_func: Primary function to try.
        fallback_func: Fallback function if primary fails.
        config: Retry configuration for primary.

    Returns:
        Wrapped function.

    Example:
        def primary_api():
            return expensive_model.generate("Hello")

        def fallback_api():
            return cheap_model.generate("Hello")

        safe_call = retry_with_fallback(primary_api, fallback_api)
        result = safe_call()
    """
    config = config or RetryConfig()

    @functools.wraps(primary_func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return execute_with_retry(primary_func, args, kwargs, config)
        except RetryExhaustedError:
            logger.warning("Primary function exhausted retries, using fallback")
            return fallback_func(*args, **kwargs)

    return wrapper


# Utility functions

def with_timeout(
    timeout: float,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a function with a timeout.

    Note: This uses threading and may not interrupt all blocking operations.

    Args:
        timeout: Timeout in seconds.
        func: Function to execute.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of the function.

    Raises:
        ModelTimeoutError: If execution times out.
    """
    import threading

    result: list = []
    exception: list = []

    def target() -> None:
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise ModelTimeoutError("function", timeout)

    if exception:
        raise exception[0]

    return result[0]


async def with_timeout_async(
    timeout: float,
    coro: Any,
) -> Any:
    """Execute a coroutine with a timeout.

    Args:
        timeout: Timeout in seconds.
        coro: Coroutine to execute.

    Returns:
        Result of the coroutine.

    Raises:
        ModelTimeoutError: If execution times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise ModelTimeoutError("coroutine", timeout)
