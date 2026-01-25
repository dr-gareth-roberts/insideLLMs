"""Retry utilities with exponential backoff for API calls.

This module provides robust retry mechanisms for handling transient failures
when interacting with LLM APIs, including:
- Configurable exponential backoff
- Jitter to prevent thundering herd
- Per-exception retry policies
- Circuit breaker pattern
- Async support

Overview
--------
The retry module offers multiple strategies for handling transient failures:

1. **Simple Retry Decorator**: Use ``@retry()`` to wrap functions with automatic
   retry logic and exponential backoff.

2. **Async Retry Decorator**: Use ``@retry_async()`` for asynchronous functions.

3. **Circuit Breaker**: Protect against cascading failures by temporarily
   blocking requests to a failing service.

4. **Retry with Fallback**: Automatically switch to a fallback function when
   the primary function fails after all retries.

5. **Timeout Utilities**: Execute functions with configurable timeouts.

Examples
--------
Basic retry decorator usage:

>>> from insideLLMs.retry import retry, RetryConfig
>>>
>>> @retry(max_retries=3, initial_delay=1.0)
... def call_api():
...     return client.generate("Hello, world!")
>>>
>>> # Automatically retries up to 3 times with exponential backoff
>>> result = call_api()

Using a custom retry configuration:

>>> config = RetryConfig(
...     max_retries=5,
...     initial_delay=0.5,
...     max_delay=30.0,
...     strategy=BackoffStrategy.FIBONACCI,
...     jitter=True
... )
>>>
>>> @retry(config=config)
... def call_expensive_api():
...     return premium_client.generate("Complex query")

Circuit breaker for protecting against cascading failures:

>>> from insideLLMs.retry import CircuitBreaker
>>>
>>> circuit = CircuitBreaker("api_service", CircuitBreakerConfig(
...     failure_threshold=5,
...     reset_timeout=30.0
... ))
>>>
>>> @circuit
... def protected_call():
...     return api.generate("Hello")
>>>
>>> # After 5 failures, the circuit opens and rejects calls immediately
>>> try:
...     protected_call()
... except CircuitBreakerOpen as e:
...     print(f"Service unavailable, retry in {e.time_until_reset}s")

Retry with automatic fallback:

>>> from insideLLMs.retry import retry_with_fallback
>>>
>>> def premium_model():
...     return gpt4.generate("Hello")
>>>
>>> def fallback_model():
...     return gpt35.generate("Hello")
>>>
>>> safe_call = retry_with_fallback(premium_model, fallback_model)
>>> result = safe_call()  # Falls back to gpt35 if gpt4 fails

See Also
--------
- :class:`RetryConfig` : Configuration options for retry behavior
- :class:`BackoffStrategy` : Available backoff strategies
- :class:`CircuitBreaker` : Circuit breaker pattern implementation
- :func:`retry` : Synchronous retry decorator
- :func:`retry_async` : Asynchronous retry decorator
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

from insideLLMs.exceptions import (
    InsideLLMsError,
    RateLimitError,
)
from insideLLMs.exceptions import (
    TimeoutError as ModelTimeoutError,
)

logger = logging.getLogger("insideLLMs.retry")


T = TypeVar("T")


class BackoffStrategy(Enum):
    """Enumeration of available backoff strategies for retry logic.

    Backoff strategies determine how the delay between retry attempts
    increases over time. Different strategies are suited for different
    scenarios depending on the expected recovery time and load patterns.

    Attributes
    ----------
    CONSTANT : str
        Fixed delay between retries. Use when the expected recovery
        time is predictable and consistent.
    LINEAR : str
        Delay increases linearly with each attempt (delay = initial * attempt).
        Use for gradual backoff when services recover incrementally.
    EXPONENTIAL : str
        Delay doubles with each attempt (delay = initial * base^(attempt-1)).
        The most common strategy, suitable for most API retry scenarios.
    FIBONACCI : str
        Delay follows the Fibonacci sequence. Provides a middle ground
        between linear and exponential growth.

    Examples
    --------
    Using constant backoff for predictable delays:

    >>> config = RetryConfig(
    ...     strategy=BackoffStrategy.CONSTANT,
    ...     initial_delay=2.0,
    ...     max_retries=3
    ... )
    >>> # Delays: 2s, 2s, 2s (constant)

    Using linear backoff for gradual increase:

    >>> config = RetryConfig(
    ...     strategy=BackoffStrategy.LINEAR,
    ...     initial_delay=1.0,
    ...     max_retries=5
    ... )
    >>> # Delays: 1s, 2s, 3s, 4s, 5s (linear growth)

    Using exponential backoff (default and recommended):

    >>> config = RetryConfig(
    ...     strategy=BackoffStrategy.EXPONENTIAL,
    ...     initial_delay=1.0,
    ...     exponential_base=2.0,
    ...     max_retries=4
    ... )
    >>> # Delays: 1s, 2s, 4s, 8s (exponential growth)

    Using Fibonacci backoff for moderate growth:

    >>> config = RetryConfig(
    ...     strategy=BackoffStrategy.FIBONACCI,
    ...     initial_delay=0.5,
    ...     max_retries=6
    ... )
    >>> # Delays: 0.5s, 0.5s, 1s, 1.5s, 2.5s, 4s (Fibonacci sequence)

    See Also
    --------
    RetryConfig : Configuration class that uses BackoffStrategy.
    RetryConfig.calculate_delay : Method that computes delays based on strategy.
    """

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior with customizable backoff strategies.

    This dataclass encapsulates all configuration options for retry logic,
    including delay calculation, jitter, and exception handling. It provides
    sensible defaults while allowing fine-grained control over retry behavior.

    Parameters
    ----------
    max_retries : int, default=3
        Maximum number of retry attempts after the initial call fails.
        Set to 0 to disable retries (only the initial attempt).
    initial_delay : float, default=1.0
        Initial delay in seconds before the first retry attempt.
        Subsequent delays are calculated based on the backoff strategy.
    max_delay : float, default=60.0
        Maximum delay in seconds between retries. Prevents delays from
        growing unbounded in exponential or Fibonacci strategies.
    exponential_base : float, default=2.0
        Base multiplier for exponential backoff. Only used when
        ``strategy=BackoffStrategy.EXPONENTIAL``.
    jitter : bool, default=True
        Whether to add random jitter to delays. Jitter helps prevent
        the "thundering herd" problem when many clients retry simultaneously.
    jitter_factor : float, default=0.1
        Maximum jitter as a fraction of the delay (0.0 to 1.0).
        A value of 0.1 means delays vary by +/- 10%.
    strategy : BackoffStrategy, default=BackoffStrategy.EXPONENTIAL
        The backoff strategy to use for calculating delays between retries.
    retryable_exceptions : tuple[type[Exception], ...], default=(RateLimitError, ...)
        Tuple of exception types that should trigger a retry. Other exceptions
        are raised immediately without retry.
    on_retry : Callable[[Exception, int, float], None], optional
        Optional callback function called before each retry attempt.
        Receives the exception, attempt number, and calculated delay.

    Attributes
    ----------
    max_retries : int
        Maximum number of retry attempts.
    initial_delay : float
        Initial delay in seconds before first retry.
    max_delay : float
        Maximum delay cap in seconds.
    exponential_base : float
        Base for exponential backoff calculation.
    jitter : bool
        Whether jitter is enabled.
    jitter_factor : float
        Maximum jitter as fraction of delay.
    strategy : BackoffStrategy
        The backoff strategy being used.
    retryable_exceptions : tuple[type[Exception], ...]
        Exception types that trigger retries.
    on_retry : Callable or None
        Callback function for retry events.

    Examples
    --------
    Basic configuration with defaults:

    >>> config = RetryConfig()
    >>> config.max_retries
    3
    >>> config.calculate_delay(1)  # First retry delay (approximately 1s with jitter)
    1.05  # varies due to jitter

    Aggressive retry for critical operations:

    >>> config = RetryConfig(
    ...     max_retries=10,
    ...     initial_delay=0.1,
    ...     max_delay=30.0,
    ...     strategy=BackoffStrategy.EXPONENTIAL
    ... )
    >>> # Delays: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s, 3.2s, 6.4s, 12.8s, 25.6s, 30s

    Conservative retry with linear backoff:

    >>> config = RetryConfig(
    ...     max_retries=3,
    ...     initial_delay=5.0,
    ...     strategy=BackoffStrategy.LINEAR,
    ...     jitter=False  # Disable jitter for predictable delays
    ... )
    >>> config.calculate_delay(1)
    5.0
    >>> config.calculate_delay(2)
    10.0
    >>> config.calculate_delay(3)
    15.0

    Custom retry callback for logging:

    >>> def log_retry(exc, attempt, delay):
    ...     print(f"Retry {attempt}: {exc}, waiting {delay:.2f}s")
    ...
    >>> config = RetryConfig(
    ...     max_retries=3,
    ...     on_retry=log_retry
    ... )
    >>> # When a retry occurs: "Retry 1: ConnectionError(...), waiting 1.05s"

    Custom exception handling:

    >>> config = RetryConfig(
    ...     retryable_exceptions=(ConnectionError, TimeoutError, ValueError),
    ...     max_retries=5
    ... )
    >>> # Only retries on ConnectionError, TimeoutError, or ValueError

    See Also
    --------
    BackoffStrategy : Available backoff strategies.
    retry : Decorator that uses RetryConfig.
    execute_with_retry : Function that executes with retry logic.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    retryable_exceptions: tuple[type[Exception], ...] = (
        RateLimitError,
        ModelTimeoutError,
        ConnectionError,
        TimeoutError,
    )
    on_retry: Optional[Callable[[Exception, int, float], None]] = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate the delay before a retry attempt based on the configured strategy.

        This method computes the appropriate wait time before the next retry
        attempt, taking into account the backoff strategy, maximum delay cap,
        and optional jitter.

        Parameters
        ----------
        attempt : int
            The attempt number (1-indexed). The first retry attempt is 1,
            the second is 2, and so on.

        Returns
        -------
        float
            The calculated delay in seconds, guaranteed to be non-negative.
            The delay is capped at ``max_delay`` and includes jitter if enabled.

        Examples
        --------
        Exponential backoff calculation:

        >>> config = RetryConfig(
        ...     initial_delay=1.0,
        ...     exponential_base=2.0,
        ...     strategy=BackoffStrategy.EXPONENTIAL,
        ...     jitter=False
        ... )
        >>> config.calculate_delay(1)
        1.0
        >>> config.calculate_delay(2)
        2.0
        >>> config.calculate_delay(3)
        4.0
        >>> config.calculate_delay(4)
        8.0

        Linear backoff calculation:

        >>> config = RetryConfig(
        ...     initial_delay=2.0,
        ...     strategy=BackoffStrategy.LINEAR,
        ...     jitter=False
        ... )
        >>> config.calculate_delay(1)
        2.0
        >>> config.calculate_delay(2)
        4.0
        >>> config.calculate_delay(3)
        6.0

        Max delay cap prevents unbounded growth:

        >>> config = RetryConfig(
        ...     initial_delay=10.0,
        ...     max_delay=30.0,
        ...     strategy=BackoffStrategy.EXPONENTIAL,
        ...     jitter=False
        ... )
        >>> config.calculate_delay(1)
        10.0
        >>> config.calculate_delay(2)
        20.0
        >>> config.calculate_delay(3)  # Would be 40, but capped at 30
        30.0

        Jitter adds randomness to prevent thundering herd:

        >>> config = RetryConfig(
        ...     initial_delay=10.0,
        ...     jitter=True,
        ...     jitter_factor=0.1
        ... )
        >>> delay = config.calculate_delay(1)
        >>> 9.0 <= delay <= 11.0  # 10 +/- 10%
        True

        Notes
        -----
        The delay calculation follows these steps:
        1. Calculate base delay using the configured strategy
        2. Apply the max_delay cap
        3. Add jitter if enabled (uniform distribution within jitter_factor range)
        4. Ensure the result is non-negative
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
        """Calculate the nth Fibonacci number for Fibonacci backoff strategy.

        Uses an iterative approach for O(n) time complexity and O(1) space.
        The sequence starts with 1, 1, 2, 3, 5, 8, 13, 21, ...

        Parameters
        ----------
        n : int
            The position in the Fibonacci sequence (1-indexed).
            n=1 and n=2 both return 1.

        Returns
        -------
        int
            The nth Fibonacci number.

        Examples
        --------
        >>> RetryConfig._fibonacci(1)
        1
        >>> RetryConfig._fibonacci(2)
        1
        >>> RetryConfig._fibonacci(3)
        2
        >>> RetryConfig._fibonacci(5)
        5
        >>> RetryConfig._fibonacci(10)
        55
        """
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b


@dataclass
class RetryResult(Generic[T]):
    """Result container for a retry operation with detailed execution history.

    This generic dataclass captures the outcome of a retry operation, including
    whether it succeeded, the result or exception, and detailed history of all
    attempts. It is useful for debugging, logging, and analyzing retry behavior.

    Type Parameters
    ---------------
    T
        The type of the successful result value.

    Parameters
    ----------
    success : bool
        Whether the operation ultimately succeeded after all retry attempts.
    result : T, optional
        The result value if the operation succeeded. None if it failed.
    exception : Exception, optional
        The last exception raised if the operation failed. None if it succeeded.
    attempts : int, default=0
        Total number of attempts made (including the initial attempt).
    total_delay : float, default=0.0
        Total time in seconds spent waiting between retry attempts.
    history : list, default=[]
        List of dictionaries recording each failed attempt with keys:
        - ``attempt``: The attempt number
        - ``exception_type``: Name of the exception class
        - ``exception_message``: String representation of the exception
        - ``delay``: Delay in seconds before the next attempt

    Attributes
    ----------
    success : bool
        Whether the operation succeeded.
    result : T or None
        The successful result, if any.
    exception : Exception or None
        The last exception encountered, if the operation failed.
    attempts : int
        Total number of attempts made.
    total_delay : float
        Total delay time spent waiting.
    history : list
        Detailed history of all retry attempts.

    Examples
    --------
    Successful operation result:

    >>> result = RetryResult(
    ...     success=True,
    ...     result="Hello, World!",
    ...     attempts=3,
    ...     total_delay=2.5,
    ...     history=[
    ...         {"attempt": 1, "exception_type": "ConnectionError",
    ...          "exception_message": "Connection refused", "delay": 1.0},
    ...         {"attempt": 2, "exception_type": "TimeoutError",
    ...          "exception_message": "Request timed out", "delay": 1.5}
    ...     ]
    ... )
    >>> result.success
    True
    >>> result.result
    'Hello, World!'
    >>> len(result.history)
    2

    Failed operation result:

    >>> result = RetryResult(
    ...     success=False,
    ...     exception=ConnectionError("Service unavailable"),
    ...     attempts=4,
    ...     total_delay=7.0,
    ...     history=[
    ...         {"attempt": 1, "exception_type": "ConnectionError", "delay": 1.0},
    ...         {"attempt": 2, "exception_type": "ConnectionError", "delay": 2.0},
    ...         {"attempt": 3, "exception_type": "ConnectionError", "delay": 4.0}
    ...     ]
    ... )
    >>> result.success
    False
    >>> result.attempts
    4
    >>> result.total_delay
    7.0

    Analyzing retry history:

    >>> for entry in result.history:
    ...     print(f"Attempt {entry['attempt']}: {entry['exception_type']}, "
    ...           f"waited {entry['delay']}s")
    Attempt 1: ConnectionError, waited 1.0s
    Attempt 2: ConnectionError, waited 2.0s
    Attempt 3: ConnectionError, waited 4.0s

    Using with type hints:

    >>> from typing import Dict
    >>> result: RetryResult[Dict[str, str]] = RetryResult(
    ...     success=True,
    ...     result={"status": "ok"},
    ...     attempts=1
    ... )

    See Also
    --------
    execute_with_retry : Function that produces RetryResult-style outcomes.
    RetryExhaustedError : Exception raised when all retries are exhausted.
    """

    success: bool
    result: Optional[T] = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    history: list = field(default_factory=list)


class RetryExhaustedError(InsideLLMsError):
    """Exception raised when all retry attempts have been exhausted.

    This exception is raised by the retry mechanisms when all configured retry
    attempts have failed. It contains detailed information about the retry
    history, allowing for debugging and error reporting.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    attempts : int
        Total number of attempts made (including the initial attempt).
    last_exception : Exception
        The final exception that caused the last retry to fail.
    history : list
        List of dictionaries recording each failed attempt, with keys:
        - ``attempt``: The attempt number
        - ``exception_type``: Name of the exception class
        - ``exception_message``: String representation of the exception
        - ``delay``: Delay in seconds before the next attempt

    Attributes
    ----------
    attempts : int
        Total number of attempts made.
    last_exception : Exception
        The exception from the final failed attempt.
    history : list
        Complete history of all retry attempts.

    Examples
    --------
    Catching and inspecting retry exhaustion:

    >>> from insideLLMs.retry import retry, RetryExhaustedError
    >>>
    >>> @retry(max_retries=3)
    ... def flaky_operation():
    ...     raise ConnectionError("Always fails")
    ...
    >>> try:
    ...     flaky_operation()
    ... except RetryExhaustedError as e:
    ...     print(f"Failed after {e.attempts} attempts")
    ...     print(f"Last error: {e.last_exception}")
    ...     print(f"Retry history: {len(e.history)} retries")
    Failed after 4 attempts
    Last error: Always fails
    Retry history: 3 retries

    Analyzing the retry history for patterns:

    >>> try:
    ...     flaky_operation()
    ... except RetryExhaustedError as e:
    ...     for entry in e.history:
    ...         print(f"Attempt {entry['attempt']}: "
    ...               f"{entry['exception_type']} - waited {entry['delay']:.2f}s")
    Attempt 1: ConnectionError - waited 1.00s
    Attempt 2: ConnectionError - waited 2.00s
    Attempt 3: ConnectionError - waited 4.00s

    Re-raising the original exception:

    >>> try:
    ...     flaky_operation()
    ... except RetryExhaustedError as e:
    ...     # Re-raise the original exception if needed
    ...     raise e.last_exception from e

    Logging retry failures:

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>>
    >>> try:
    ...     flaky_operation()
    ... except RetryExhaustedError as e:
    ...     logger.error(
    ...         "Operation failed after %d attempts. Last error: %s",
    ...         e.attempts,
    ...         e.last_exception
    ...     )

    See Also
    --------
    execute_with_retry : Function that raises this exception.
    RetryConfig : Configuration that controls retry behavior.
    """

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
    retryable_exceptions: Optional[tuple[type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic with exponential backoff to synchronous functions.

    This decorator wraps a function to automatically retry on specified exceptions,
    with configurable backoff strategies and delay parameters. It can be used with
    either a ``RetryConfig`` object or individual parameters.

    Parameters
    ----------
    config : RetryConfig, optional
        A ``RetryConfig`` object containing all retry settings. If provided,
        this takes precedence over individual parameters.
    max_retries : int, optional
        Maximum number of retry attempts after the initial call fails.
        Defaults to 3 if not specified.
    initial_delay : float, optional
        Initial delay in seconds before the first retry. Defaults to 1.0.
    max_delay : float, optional
        Maximum delay in seconds between retries. Defaults to 60.0.
    retryable_exceptions : tuple[type[Exception], ...], optional
        Tuple of exception types that should trigger a retry. Defaults to
        ``(RateLimitError, TimeoutError, ConnectionError)``.
    on_retry : Callable[[Exception, int, float], None], optional
        Callback function invoked before each retry attempt. Receives the
        exception, attempt number, and calculated delay.

    Returns
    -------
    Callable[[Callable[..., T]], Callable[..., T]]
        A decorator that wraps functions with retry logic.

    Raises
    ------
    RetryExhaustedError
        When all retry attempts have been exhausted.

    Examples
    --------
    Basic usage with default settings:

    >>> @retry()
    ... def fetch_data():
    ...     return api.get("/data")
    ...
    >>> result = fetch_data()  # Retries up to 3 times with exponential backoff

    Custom retry parameters:

    >>> @retry(max_retries=5, initial_delay=0.5, max_delay=30.0)
    ... def call_external_service():
    ...     return service.request()
    ...
    >>> result = call_external_service()

    Using a RetryConfig object for full control:

    >>> config = RetryConfig(
    ...     max_retries=3,
    ...     initial_delay=1.0,
    ...     strategy=BackoffStrategy.FIBONACCI,
    ...     jitter=True,
    ...     jitter_factor=0.2
    ... )
    >>>
    >>> @retry(config=config)
    ... def call_llm():
    ...     return client.generate("Hello")

    Custom exception handling:

    >>> @retry(
    ...     retryable_exceptions=(ConnectionError, TimeoutError, ValueError),
    ...     max_retries=3
    ... )
    ... def parse_response():
    ...     response = api.get()
    ...     return parse(response)  # Retries on ValueError too

    With a retry callback for logging or metrics:

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>>
    >>> def log_retry(exc, attempt, delay):
    ...     logger.warning(
    ...         "Attempt %d failed: %s. Retrying in %.2f seconds...",
    ...         attempt, exc, delay
    ...     )
    ...
    >>> @retry(max_retries=5, on_retry=log_retry)
    ... def important_operation():
    ...     return critical_api.call()

    Handling retry exhaustion:

    >>> @retry(max_retries=2)
    ... def unreliable_call():
    ...     raise ConnectionError("Service unavailable")
    ...
    >>> try:
    ...     unreliable_call()
    ... except RetryExhaustedError as e:
    ...     print(f"All {e.attempts} attempts failed")
    ...     print(f"Last error: {e.last_exception}")
    ...     # Implement fallback logic here

    Notes
    -----
    - The decorator preserves the original function's signature and metadata
      using ``functools.wraps``.
    - Non-retryable exceptions are raised immediately without any retry attempts.
    - For ``RateLimitError`` exceptions with a ``retry_after`` attribute, the
      delay from the exception takes precedence over the calculated delay.

    See Also
    --------
    retry_async : Async version of this decorator.
    execute_with_retry : The underlying function that handles retry logic.
    RetryConfig : Configuration options for retry behavior.
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
    retryable_exceptions: Optional[tuple[type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable:
    """Decorator to add retry logic with exponential backoff to async functions.

    This is the asynchronous version of the ``retry`` decorator. It wraps async
    functions to automatically retry on specified exceptions, using ``asyncio.sleep``
    for non-blocking delays between attempts.

    Parameters
    ----------
    config : RetryConfig, optional
        A ``RetryConfig`` object containing all retry settings. If provided,
        this takes precedence over individual parameters.
    max_retries : int, optional
        Maximum number of retry attempts after the initial call fails.
        Defaults to 3 if not specified.
    initial_delay : float, optional
        Initial delay in seconds before the first retry. Defaults to 1.0.
    max_delay : float, optional
        Maximum delay in seconds between retries. Defaults to 60.0.
    retryable_exceptions : tuple[type[Exception], ...], optional
        Tuple of exception types that should trigger a retry. Defaults to
        ``(RateLimitError, TimeoutError, ConnectionError)``.
    on_retry : Callable[[Exception, int, float], None], optional
        Callback function invoked before each retry attempt. Receives the
        exception, attempt number, and calculated delay.

    Returns
    -------
    Callable
        A decorator that wraps async functions with retry logic.

    Raises
    ------
    RetryExhaustedError
        When all retry attempts have been exhausted.

    Examples
    --------
    Basic async usage:

    >>> @retry_async()
    ... async def fetch_data():
    ...     return await api.get_async("/data")
    ...
    >>> result = await fetch_data()

    Async function with custom configuration:

    >>> @retry_async(max_retries=5, initial_delay=0.5)
    ... async def stream_response():
    ...     async with client.stream("prompt") as stream:
    ...         return await stream.read()

    Using with async context managers:

    >>> @retry_async(max_retries=3)
    ... async def upload_file(file_path):
    ...     async with aiofiles.open(file_path, 'rb') as f:
    ...         content = await f.read()
    ...         return await api.upload(content)

    With a RetryConfig for full control:

    >>> config = RetryConfig(
    ...     max_retries=5,
    ...     initial_delay=0.1,
    ...     max_delay=10.0,
    ...     strategy=BackoffStrategy.EXPONENTIAL
    ... )
    >>>
    >>> @retry_async(config=config)
    ... async def call_llm_async():
    ...     return await async_client.generate("Hello")

    Multiple async operations with retry:

    >>> @retry_async(max_retries=3)
    ... async def batch_process(items):
    ...     results = []
    ...     for item in items:
    ...         result = await process_item(item)
    ...         results.append(result)
    ...     return results

    Notes
    -----
    - Uses ``asyncio.sleep`` for non-blocking delays, allowing other coroutines
      to run during wait periods.
    - The decorator preserves the original function's signature and metadata.
    - The ``on_retry`` callback is called synchronously; for async callbacks,
      consider wrapping in ``asyncio.create_task``.

    See Also
    --------
    retry : Synchronous version of this decorator.
    execute_with_retry_async : The underlying async function that handles retry logic.
    RetryConfig : Configuration options for retry behavior.
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
    """Execute a synchronous function with retry logic and configurable backoff.

    This is the core function that implements retry behavior for synchronous
    functions. It handles exception catching, delay calculation, retry callbacks,
    and maintains a history of all attempts.

    Parameters
    ----------
    func : Callable[..., T]
        The function to execute. Should be a synchronous callable.
    args : tuple
        Positional arguments to pass to the function.
    kwargs : dict
        Keyword arguments to pass to the function.
    config : RetryConfig
        Configuration object specifying retry behavior, including max retries,
        delay strategy, and retryable exceptions.

    Returns
    -------
    T
        The return value of the function if it succeeds.

    Raises
    ------
    RetryExhaustedError
        When all retry attempts have been exhausted without success.
    Exception
        Any non-retryable exception is re-raised immediately.

    Examples
    --------
    Direct usage with a function:

    >>> def fetch_data():
    ...     return api.get("/data")
    ...
    >>> config = RetryConfig(max_retries=3, initial_delay=1.0)
    >>> result = execute_with_retry(fetch_data, (), {}, config)

    With positional and keyword arguments:

    >>> def call_api(endpoint, method="GET", timeout=30):
    ...     return requests.request(method, endpoint, timeout=timeout)
    ...
    >>> config = RetryConfig(max_retries=5)
    >>> result = execute_with_retry(
    ...     call_api,
    ...     ("/users",),
    ...     {"method": "POST", "timeout": 60},
    ...     config
    ... )

    Handling RateLimitError with retry_after:

    >>> # If the API returns a RateLimitError with retry_after=5.0,
    >>> # the retry delay will be 5 seconds instead of the calculated delay
    >>> result = execute_with_retry(rate_limited_func, (), {}, config)

    Using a callback for monitoring:

    >>> retries = []
    >>> def track_retries(exc, attempt, delay):
    ...     retries.append({"attempt": attempt, "error": str(exc)})
    ...
    >>> config = RetryConfig(max_retries=3, on_retry=track_retries)
    >>> try:
    ...     execute_with_retry(failing_func, (), {}, config)
    ... except RetryExhaustedError:
    ...     print(f"Failed after {len(retries)} retries")

    Notes
    -----
    - The function uses ``time.sleep()`` for delays, which blocks the thread.
      For async operations, use ``execute_with_retry_async``.
    - When a ``RateLimitError`` with a ``retry_after`` attribute is caught,
      the delay from the exception takes precedence over the calculated delay.
    - The attempt count in the loop is 1-indexed, matching the ``calculate_delay``
      method's expectations.

    See Also
    --------
    execute_with_retry_async : Async version of this function.
    retry : Decorator that uses this function.
    RetryConfig : Configuration for retry behavior.
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

            history.append(
                {
                    "attempt": attempt,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "delay": delay,
                }
            )

            logger.warning(
                f"Attempt {attempt} failed: {type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
            )

            if config.on_retry:
                config.on_retry(e, attempt, delay)

            time.sleep(delay)
            total_delay += delay

        except Exception:
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
    """Execute an async function with retry logic and configurable backoff.

    This is the core function that implements retry behavior for asynchronous
    functions. It uses ``asyncio.sleep`` for non-blocking delays, allowing other
    coroutines to run during wait periods.

    Parameters
    ----------
    func : Callable
        The async function to execute. Should be a coroutine function.
    args : tuple
        Positional arguments to pass to the function.
    kwargs : dict
        Keyword arguments to pass to the function.
    config : RetryConfig
        Configuration object specifying retry behavior, including max retries,
        delay strategy, and retryable exceptions.

    Returns
    -------
    Any
        The return value of the async function if it succeeds.

    Raises
    ------
    RetryExhaustedError
        When all retry attempts have been exhausted without success.
    Exception
        Any non-retryable exception is re-raised immediately.

    Examples
    --------
    Direct usage with an async function:

    >>> async def fetch_data():
    ...     return await api.get_async("/data")
    ...
    >>> config = RetryConfig(max_retries=3, initial_delay=1.0)
    >>> result = await execute_with_retry_async(fetch_data, (), {}, config)

    With async HTTP client:

    >>> async def call_api(endpoint, headers=None):
    ...     async with aiohttp.ClientSession() as session:
    ...         async with session.get(endpoint, headers=headers) as resp:
    ...             return await resp.json()
    ...
    >>> config = RetryConfig(max_retries=5)
    >>> result = await execute_with_retry_async(
    ...     call_api,
    ...     ("https://api.example.com/users",),
    ...     {"headers": {"Authorization": "Bearer token"}},
    ...     config
    ... )

    Concurrent execution with retry:

    >>> async def process_batch(items):
    ...     tasks = [process_item(item) for item in items]
    ...     return await asyncio.gather(*tasks)
    ...
    >>> config = RetryConfig(max_retries=3)
    >>> results = await execute_with_retry_async(
    ...     process_batch,
    ...     (["a", "b", "c"],),
    ...     {},
    ...     config
    ... )

    Using in an async context manager:

    >>> async def stream_data():
    ...     async with client.stream() as stream:
    ...         return [chunk async for chunk in stream]
    ...
    >>> result = await execute_with_retry_async(stream_data, (), {}, config)

    Notes
    -----
    - Uses ``asyncio.sleep()`` for non-blocking delays, unlike the synchronous
      version which uses ``time.sleep()``.
    - The event loop continues to process other coroutines during delay periods.
    - When a ``RateLimitError`` with a ``retry_after`` attribute is caught,
      the delay from the exception takes precedence over the calculated delay.

    See Also
    --------
    execute_with_retry : Synchronous version of this function.
    retry_async : Decorator that uses this function.
    RetryConfig : Configuration for retry behavior.
    """
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

            history.append(
                {
                    "attempt": attempt,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "delay": delay,
                }
            )

            logger.warning(
                f"Attempt {attempt} failed: {type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
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
    """Enumeration of circuit breaker states.

    The circuit breaker pattern uses three states to manage service
    availability and protect against cascading failures. The state
    transitions occur automatically based on success/failure thresholds.

    Attributes
    ----------
    CLOSED : str
        Normal operation state. Requests pass through to the underlying
        service. Failures are counted, and if the failure threshold is
        reached, the circuit transitions to OPEN.
    OPEN : str
        Failure state. All requests are rejected immediately without
        attempting the underlying operation. After the reset timeout,
        the circuit transitions to HALF_OPEN.
    HALF_OPEN : str
        Recovery testing state. A limited number of requests are allowed
        through to test if the service has recovered. If successful, the
        circuit closes; if a failure occurs, it reopens.

    Examples
    --------
    Checking circuit state:

    >>> circuit = CircuitBreaker("api")
    >>> circuit.state == CircuitState.CLOSED
    True

    State transition flow:

    >>> # CLOSED -> OPEN (after failure_threshold failures)
    >>> # OPEN -> HALF_OPEN (after reset_timeout seconds)
    >>> # HALF_OPEN -> CLOSED (after success_threshold successes)
    >>> # HALF_OPEN -> OPEN (on any failure)

    Using state in conditional logic:

    >>> if circuit.state == CircuitState.OPEN:
    ...     use_cached_response()
    ... else:
    ...     call_service()

    See Also
    --------
    CircuitBreaker : The circuit breaker implementation.
    CircuitBreakerConfig : Configuration for circuit breaker behavior.
    """

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior and thresholds.

    This dataclass defines the parameters that control how the circuit
    breaker transitions between states and protects against cascading
    failures.

    Parameters
    ----------
    failure_threshold : int, default=5
        Number of consecutive failures required to open the circuit.
        Once this threshold is reached, the circuit transitions from
        CLOSED to OPEN state.
    success_threshold : int, default=2
        Number of consecutive successes required to close the circuit
        when in HALF_OPEN state. This ensures the service has truly
        recovered before resuming normal operation.
    reset_timeout : float, default=30.0
        Time in seconds to wait before transitioning from OPEN to
        HALF_OPEN state. This gives the failing service time to recover.
    half_open_max_calls : int, default=1
        Maximum number of concurrent calls allowed in HALF_OPEN state.
        Limits the load on a recovering service during the test phase.

    Attributes
    ----------
    failure_threshold : int
        Failures needed to open the circuit.
    success_threshold : int
        Successes needed to close the circuit.
    reset_timeout : float
        Seconds before testing recovery.
    half_open_max_calls : int
        Max concurrent calls when half-open.

    Examples
    --------
    Default configuration (suitable for most services):

    >>> config = CircuitBreakerConfig()
    >>> config.failure_threshold
    5
    >>> config.reset_timeout
    30.0

    Aggressive failure detection for critical services:

    >>> config = CircuitBreakerConfig(
    ...     failure_threshold=3,
    ...     success_threshold=3,
    ...     reset_timeout=10.0
    ... )
    >>> # Opens after 3 failures, requires 3 successes to close

    Conservative configuration for stable services:

    >>> config = CircuitBreakerConfig(
    ...     failure_threshold=10,
    ...     success_threshold=1,
    ...     reset_timeout=60.0,
    ...     half_open_max_calls=3
    ... )
    >>> # More tolerant of failures, allows more test calls

    High-availability configuration:

    >>> config = CircuitBreakerConfig(
    ...     failure_threshold=2,
    ...     success_threshold=5,
    ...     reset_timeout=5.0,
    ...     half_open_max_calls=1
    ... )
    >>> # Quick to open, careful before closing, fast recovery attempts

    See Also
    --------
    CircuitBreaker : Uses this configuration.
    CircuitState : The states managed by the circuit breaker.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreakerOpen(InsideLLMsError):
    """Exception raised when attempting to execute through an open circuit breaker.

    This exception is raised when a call is made while the circuit breaker
    is in OPEN state (or HALF_OPEN with max calls reached). It indicates
    that the underlying service is considered unhealthy and requests are
    being rejected to prevent cascading failures.

    Parameters
    ----------
    circuit_name : str
        The name of the circuit breaker that is open.
    time_until_reset : float
        Estimated time in seconds until the circuit breaker will
        transition to HALF_OPEN state and allow test requests.

    Attributes
    ----------
    circuit_name : str
        Name of the open circuit.
    time_until_reset : float
        Seconds until the circuit may allow requests again.

    Examples
    --------
    Catching and handling circuit breaker rejection:

    >>> circuit = CircuitBreaker("api_service")
    >>>
    >>> try:
    ...     circuit.execute(lambda: api.call())
    ... except CircuitBreakerOpen as e:
    ...     print(f"Service {e.circuit_name} is unavailable")
    ...     print(f"Retry in {e.time_until_reset:.1f} seconds")
    ...     return cached_response()

    Implementing exponential backoff on circuit open:

    >>> import time
    >>> max_wait = 60
    >>> wait_time = 1
    >>>
    >>> while True:
    ...     try:
    ...         result = circuit.execute(api.call)
    ...         break
    ...     except CircuitBreakerOpen as e:
    ...         actual_wait = min(e.time_until_reset, wait_time, max_wait)
    ...         time.sleep(actual_wait)
    ...         wait_time *= 2

    Using with fallback logic:

    >>> def call_with_fallback():
    ...     try:
    ...         return circuit.execute(primary_service.call)
    ...     except CircuitBreakerOpen:
    ...         return fallback_service.call()

    Logging circuit breaker events:

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>>
    >>> try:
    ...     result = circuit.execute(api.call)
    ... except CircuitBreakerOpen as e:
    ...     logger.warning(
    ...         "Circuit %s is open, reset in %.1fs",
    ...         e.circuit_name,
    ...         e.time_until_reset
    ...     )
    ...     raise

    See Also
    --------
    CircuitBreaker : The circuit breaker that raises this exception.
    CircuitState : The OPEN state that triggers this exception.
    """

    def __init__(self, circuit_name: str, time_until_reset: float):
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open. Try again in {time_until_reset:.1f}s",
            {"circuit_name": circuit_name, "time_until_reset": time_until_reset},
        )


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    The circuit breaker pattern prevents an application from repeatedly
    trying to execute an operation that is likely to fail. It allows
    the system to detect when a service is failing and stop sending
    requests until the service recovers.

    The circuit breaker operates in three states:

    - **CLOSED**: Normal operation. Requests pass through to the underlying
      service. Failures are counted, and if the failure threshold is reached,
      the circuit opens.

    - **OPEN**: Service is failing. All requests are rejected immediately
      with a ``CircuitBreakerOpen`` exception. After the reset timeout,
      the circuit transitions to HALF_OPEN.

    - **HALF_OPEN**: Testing recovery. A limited number of requests are
      allowed through. If they succeed, the circuit closes; if any fail,
      the circuit reopens.

    Parameters
    ----------
    name : str, default="default"
        A name for this circuit breaker, used in logging and exceptions.
        Useful for identifying which service is affected.
    config : CircuitBreakerConfig, optional
        Configuration specifying thresholds and timeouts. If not provided,
        uses default ``CircuitBreakerConfig`` values.

    Attributes
    ----------
    name : str
        The circuit breaker's name.
    config : CircuitBreakerConfig
        The configuration being used.
    state : CircuitState
        Current state of the circuit (read-only property).
    is_closed : bool
        True if circuit is in CLOSED state.
    is_open : bool
        True if circuit is in OPEN state.

    Examples
    --------
    Basic usage as a decorator:

    >>> circuit = CircuitBreaker("api_service")
    >>>
    >>> @circuit
    ... def call_api():
    ...     return api.generate("Hello")
    >>>
    >>> result = call_api()  # Works normally when service is healthy

    Using with custom configuration:

    >>> config = CircuitBreakerConfig(
    ...     failure_threshold=3,
    ...     success_threshold=2,
    ...     reset_timeout=15.0
    ... )
    >>> circuit = CircuitBreaker("critical_service", config)
    >>>
    >>> @circuit
    ... def critical_call():
    ...     return service.execute()

    Using as a context manager:

    >>> circuit = CircuitBreaker("database")
    >>>
    >>> with circuit:
    ...     result = db.query("SELECT * FROM users")
    >>>
    >>> # Failures inside the context are tracked automatically

    Manual execution with error handling:

    >>> circuit = CircuitBreaker("external_api")
    >>>
    >>> try:
    ...     result = circuit.execute(api.fetch_data)
    ... except CircuitBreakerOpen as e:
    ...     print(f"Service down, retry in {e.time_until_reset:.1f}s")
    ...     result = use_cached_data()

    Monitoring circuit state:

    >>> circuit = CircuitBreaker("payment_gateway")
    >>>
    >>> if circuit.is_open:
    ...     logger.warning("Payment gateway circuit is open!")
    ...     return defer_payment()
    >>> else:
    ...     return circuit.execute(process_payment)

    Combining with retry logic:

    >>> circuit = CircuitBreaker("llm_api")
    >>>
    >>> @retry(max_retries=2)
    ... @circuit
    ... def generate_text(prompt):
    ...     return llm.generate(prompt)
    >>>
    >>> # Retries within the circuit, circuit opens if service keeps failing

    Manual reset for testing or recovery:

    >>> circuit = CircuitBreaker("test_service")
    >>> # ... service fails and circuit opens ...
    >>> circuit.reset()  # Manually reset to CLOSED state
    >>> circuit.is_closed
    True

    Notes
    -----
    - The circuit breaker is not thread-safe. For multi-threaded applications,
      use appropriate synchronization.
    - State transitions are logged using the ``insideLLMs.retry`` logger.
    - The ``reset()`` method can be used to manually close the circuit.

    See Also
    --------
    CircuitBreakerConfig : Configuration options.
    CircuitState : The three circuit states.
    CircuitBreakerOpen : Exception raised when circuit is open.
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
        """Get the current circuit breaker state.

        This property checks for potential state transitions before returning
        the current state, ensuring the returned value reflects any time-based
        transitions (e.g., OPEN to HALF_OPEN after reset_timeout).

        Returns
        -------
        CircuitState
            The current state of the circuit breaker (CLOSED, OPEN, or HALF_OPEN).

        Examples
        --------
        >>> circuit = CircuitBreaker("api")
        >>> circuit.state
        <CircuitState.CLOSED: 'closed'>

        >>> # After multiple failures...
        >>> circuit.state
        <CircuitState.OPEN: 'open'>
        """
        self._check_state_transition()
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if the circuit is in CLOSED state (normal operation).

        Returns
        -------
        bool
            True if the circuit is closed and requests can pass through.

        Examples
        --------
        >>> circuit = CircuitBreaker("api")
        >>> circuit.is_closed
        True

        >>> if circuit.is_closed:
        ...     result = circuit.execute(api.call)
        """
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if the circuit is in OPEN state (rejecting requests).

        Returns
        -------
        bool
            True if the circuit is open and requests will be rejected.

        Examples
        --------
        >>> circuit = CircuitBreaker("api")
        >>> circuit.is_open
        False

        >>> if circuit.is_open:
        ...     return use_fallback()
        """
        return self.state == CircuitState.OPEN

    def _check_state_transition(self) -> None:
        """Check if the circuit should transition from OPEN to HALF_OPEN.

        This internal method checks if enough time has elapsed since the
        last failure to allow testing whether the service has recovered.
        If the reset_timeout has passed, transitions to HALF_OPEN state.

        Notes
        -----
        This method is called automatically by the ``state`` property
        and before executing operations to ensure state is current.
        """
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.reset_timeout:
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def _record_success(self) -> None:
        """Record a successful operation and update circuit state.

        In HALF_OPEN state, tracks consecutive successes and transitions
        to CLOSED if the success_threshold is reached. In CLOSED state,
        resets the failure count (consecutive failure semantics).

        Notes
        -----
        Called automatically after successful operations through
        ``execute()`` or context manager exit.
        """
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                logger.info(f"Circuit '{self.name}' transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._last_failure_time = None
                self._half_open_calls = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed operation and update circuit state.

        Increments the failure count and records the failure time. In
        HALF_OPEN state, immediately transitions back to OPEN. In CLOSED
        state, transitions to OPEN if failure_threshold is reached.

        Notes
        -----
        Called automatically after failed operations through
        ``execute()`` or context manager exit with an exception.
        """
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
        """Use the circuit breaker as a decorator.

        This allows the circuit breaker to be used with the ``@`` decorator
        syntax. The decorated function will have circuit breaker protection
        applied automatically.

        Parameters
        ----------
        func : Callable[..., T]
            The function to wrap with circuit breaker protection.

        Returns
        -------
        Callable[..., T]
            The wrapped function that executes through the circuit breaker.

        Examples
        --------
        >>> circuit = CircuitBreaker("api")
        >>>
        >>> @circuit
        ... def call_api():
        ...     return api.generate("Hello")
        >>>
        >>> result = call_api()  # Protected by circuit breaker
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection.

        This method executes the provided function while applying circuit
        breaker logic. It checks the circuit state before execution and
        records success or failure after execution.

        Parameters
        ----------
        func : Callable[..., T]
            The function to execute.
        *args : Any
            Positional arguments to pass to the function.
        **kwargs : Any
            Keyword arguments to pass to the function.

        Returns
        -------
        T
            The return value of the function if successful.

        Raises
        ------
        CircuitBreakerOpen
            If the circuit is in OPEN state or HALF_OPEN with max calls reached.
        Exception
            Any exception raised by the function (after recording the failure).

        Examples
        --------
        Basic usage:

        >>> circuit = CircuitBreaker("api")
        >>> result = circuit.execute(api.get_data)

        With arguments:

        >>> result = circuit.execute(api.post_data, {"key": "value"}, timeout=30)

        With error handling:

        >>> try:
        ...     result = circuit.execute(api.call)
        ... except CircuitBreakerOpen as e:
        ...     result = use_fallback()
        ... except APIError as e:
        ...     # Handle API errors (circuit records the failure)
        ...     log_error(e)
        ...     raise

        With lambda for simple operations:

        >>> result = circuit.execute(lambda: db.query("SELECT * FROM users"))
        """
        self._check_state_transition()

        if self._state == CircuitState.OPEN:
            time_until_reset = self.config.reset_timeout - (
                time.time() - (self._last_failure_time or 0)
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

        except Exception:
            self._record_failure()
            raise

    def __enter__(self) -> "CircuitBreaker":
        """Enter the circuit breaker context manager.

        Checks the circuit state and raises ``CircuitBreakerOpen`` if the
        circuit is open. Otherwise, allows the context block to execute.

        Returns
        -------
        CircuitBreaker
            This circuit breaker instance.

        Raises
        ------
        CircuitBreakerOpen
            If the circuit is in OPEN state.

        Examples
        --------
        >>> circuit = CircuitBreaker("database")
        >>>
        >>> with circuit:
        ...     result = db.execute_query("SELECT * FROM users")
        >>>
        >>> # Exceptions in the block are recorded as failures

        Notes
        -----
        The context manager tracks success/failure based on whether an
        exception is raised in the context block.
        """
        self._check_state_transition()

        if self._state == CircuitState.OPEN:
            time_until_reset = self.config.reset_timeout - (
                time.time() - (self._last_failure_time or 0)
            )
            raise CircuitBreakerOpen(self.name, max(0, time_until_reset))

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(self.name, 0)
            self._half_open_calls += 1

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the circuit breaker context manager.

        Records success or failure based on whether an exception occurred
        in the context block. Does not suppress exceptions.

        Parameters
        ----------
        exc_type : type or None
            The exception type if an exception occurred, None otherwise.
        exc_val : Exception or None
            The exception instance if one occurred.
        exc_tb : traceback or None
            The traceback if an exception occurred.

        Returns
        -------
        bool
            Always returns False (does not suppress exceptions).

        Examples
        --------
        >>> circuit = CircuitBreaker("api")
        >>>
        >>> try:
        ...     with circuit:
        ...         raise ConnectionError("Failed")
        ... except ConnectionError:
        ...     pass  # Circuit records this as a failure
        """
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
        return False

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        Clears all failure and success counts and resets the circuit to
        CLOSED state. Useful for testing or when you know a service has
        recovered and want to bypass the normal timeout.

        Examples
        --------
        Manual reset after service recovery:

        >>> circuit = CircuitBreaker("api")
        >>> # ... circuit opens due to failures ...
        >>> circuit.is_open
        True
        >>> circuit.reset()
        >>> circuit.is_closed
        True

        Using in tests:

        >>> def test_circuit_behavior():
        ...     circuit = CircuitBreaker("test")
        ...     # ... run tests that may trip the circuit ...
        ...     circuit.reset()  # Clean up for next test

        Notes
        -----
        This method logs an info message when called.
        """
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
    """Create a function that retries then falls back to an alternative.

    This function combines retry logic with a fallback strategy. It first
    attempts to execute the primary function with retries. If all retry
    attempts are exhausted, it automatically switches to the fallback
    function.

    Parameters
    ----------
    primary_func : Callable[..., T]
        The primary function to try first. This function will be retried
        according to the retry configuration before falling back.
    fallback_func : Callable[..., T]
        The fallback function to use if the primary function fails after
        all retries. Should accept the same arguments as the primary.
    config : RetryConfig, optional
        Configuration for retry behavior on the primary function.
        Defaults to a standard ``RetryConfig()`` if not provided.

    Returns
    -------
    Callable[..., T]
        A wrapped function that implements retry-with-fallback behavior.

    Examples
    --------
    Basic usage with two API providers:

    >>> def premium_api(prompt):
    ...     return gpt4.generate(prompt)
    ...
    >>> def fallback_api(prompt):
    ...     return gpt35.generate(prompt)
    ...
    >>> safe_generate = retry_with_fallback(premium_api, fallback_api)
    >>> result = safe_generate("Explain quantum computing")

    With custom retry configuration:

    >>> config = RetryConfig(
    ...     max_retries=5,
    ...     initial_delay=0.5,
    ...     strategy=BackoffStrategy.EXPONENTIAL
    ... )
    >>>
    >>> def primary_db():
    ...     return master_db.query("SELECT * FROM users")
    ...
    >>> def fallback_db():
    ...     return replica_db.query("SELECT * FROM users")
    ...
    >>> safe_query = retry_with_fallback(primary_db, fallback_db, config)
    >>> users = safe_query()

    Tiered API fallback pattern:

    >>> def expensive_model(prompt):
    ...     return claude_opus.generate(prompt)
    ...
    >>> def cheap_model(prompt):
    ...     return claude_haiku.generate(prompt)
    ...
    >>> resilient_call = retry_with_fallback(expensive_model, cheap_model)
    >>> # Uses Opus with retries, falls back to Haiku if Opus unavailable

    Using with lambda functions:

    >>> safe_fetch = retry_with_fallback(
    ...     lambda: api.fetch_fresh_data(),
    ...     lambda: cache.get_cached_data()
    ... )
    >>> data = safe_fetch()

    Notes
    -----
    - The fallback function receives the same arguments as the primary function.
    - Only ``RetryExhaustedError`` triggers the fallback; other exceptions
      from the primary function are raised normally.
    - The fallback function is NOT retried - it runs once. Wrap it separately
      if you need retry logic for the fallback too.
    - A warning is logged when the fallback is used.

    See Also
    --------
    retry : Simple retry decorator without fallback.
    execute_with_retry : Low-level retry execution.
    RetryExhaustedError : Exception that triggers fallback.
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
    """Execute a synchronous function with a timeout limit.

    This function runs the provided callable in a separate daemon thread
    and waits for it to complete within the specified timeout. If the
    function doesn't complete in time, a ``ModelTimeoutError`` is raised.

    Parameters
    ----------
    timeout : float
        Maximum time in seconds to wait for the function to complete.
    func : Callable[..., T]
        The function to execute.
    *args : Any
        Positional arguments to pass to the function.
    **kwargs : Any
        Keyword arguments to pass to the function.

    Returns
    -------
    T
        The return value of the function if it completes within the timeout.

    Raises
    ------
    ModelTimeoutError
        If the function does not complete within the specified timeout.
    Exception
        Any exception raised by the function is re-raised.

    Examples
    --------
    Basic usage with a timeout:

    >>> def slow_operation():
    ...     time.sleep(10)
    ...     return "done"
    ...
    >>> try:
    ...     result = with_timeout(5.0, slow_operation)
    ... except TimeoutError as e:
    ...     print("Operation timed out")

    With function arguments:

    >>> def fetch_data(url, timeout=30):
    ...     return requests.get(url, timeout=timeout).json()
    ...
    >>> result = with_timeout(10.0, fetch_data, "https://api.example.com")

    With keyword arguments:

    >>> result = with_timeout(
    ...     5.0,
    ...     api.generate,
    ...     prompt="Hello",
    ...     max_tokens=100
    ... )

    Combining with retry logic:

    >>> @retry(max_retries=3)
    ... def timed_call():
    ...     return with_timeout(30.0, external_api.slow_operation)

    Warnings
    --------
    This function uses threading and has limitations:

    - The thread continues running even after timeout; it cannot be forcibly
      terminated in Python.
    - Blocking operations (like ``socket.recv()``) may not be interruptible.
    - For truly cancellable operations, prefer async with ``with_timeout_async``.
    - The daemon thread may leave resources in an inconsistent state.

    Notes
    -----
    - The function runs in a daemon thread, so it will be terminated when
      the main program exits, regardless of completion status.
    - Results and exceptions are communicated via lists to handle thread
      communication safely.

    See Also
    --------
    with_timeout_async : Async version using asyncio.wait_for.
    ModelTimeoutError : The exception raised on timeout.
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
    """Execute an async coroutine with a timeout limit.

    This function wraps a coroutine with ``asyncio.wait_for`` to enforce
    a maximum execution time. Unlike the synchronous ``with_timeout``,
    this properly cancels the coroutine when the timeout is exceeded.

    Parameters
    ----------
    timeout : float
        Maximum time in seconds to wait for the coroutine to complete.
    coro : Coroutine
        The coroutine to execute. Must be an awaitable object.

    Returns
    -------
    Any
        The return value of the coroutine if it completes within the timeout.

    Raises
    ------
    ModelTimeoutError
        If the coroutine does not complete within the specified timeout.
    Exception
        Any exception raised by the coroutine is re-raised.

    Examples
    --------
    Basic usage with an async function:

    >>> async def slow_fetch():
    ...     await asyncio.sleep(10)
    ...     return "data"
    ...
    >>> try:
    ...     result = await with_timeout_async(5.0, slow_fetch())
    ... except TimeoutError as e:
    ...     print("Fetch timed out")

    With async API calls:

    >>> async def fetch_with_timeout():
    ...     coro = client.generate_async("Hello, world!")
    ...     return await with_timeout_async(30.0, coro)

    In an async context with error handling:

    >>> async def safe_operation():
    ...     try:
    ...         return await with_timeout_async(10.0, external_api.fetch())
    ...     except TimeoutError:
    ...         return await cache.get_fallback()

    Combining with async retry:

    >>> @retry_async(max_retries=3)
    ... async def timed_api_call():
    ...     return await with_timeout_async(15.0, api.slow_endpoint())

    Notes
    -----
    - Uses ``asyncio.wait_for`` internally, which properly cancels the
      coroutine when the timeout is exceeded.
    - The coroutine receives a ``CancelledError`` when timed out, allowing
      it to perform cleanup if it handles cancellation.
    - Prefer this over the synchronous ``with_timeout`` for async code
      as it provides proper cancellation semantics.

    See Also
    --------
    with_timeout : Synchronous version using threading.
    ModelTimeoutError : The exception raised on timeout.
    retry_async : Async retry decorator that can wrap timed operations.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise ModelTimeoutError("coroutine", timeout)
