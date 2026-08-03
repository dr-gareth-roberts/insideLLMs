"""Exception wrapping and retry helpers."""

from typing import Any, Optional

from insideLLMs._exceptions.base import InsideLLMsError
from insideLLMs._exceptions.model import ModelTimeoutError, RateLimitError

TimeoutError = ModelTimeoutError


def wrap_exception(
    error: Exception,
    wrapper_class: type,
    message: Optional[str] = None,
    **details: Any,
) -> InsideLLMsError:
    """Wrap an exception in an InsideLLMs error.

    This utility function converts any exception into an InsideLLMs
    exception type, preserving information about the original error
    for debugging while providing a consistent exception interface.

    Parameters
    ----------
    error : Exception
        The original exception to wrap.
    wrapper_class : type
        The InsideLLMsError subclass to use for wrapping.
    message : str, optional
        Custom message for the wrapped exception. If not provided,
        uses the string representation of the original error.
    **details : Any
        Additional key-value pairs to include in the exception's
        details dictionary.

    Returns
    -------
    InsideLLMsError
        A new exception of the specified wrapper class, containing
        information about the original error.

    Examples
    --------
    Wrapping a standard library exception:

    >>> try:
    ...     result = json.loads(invalid_json)
    ... except json.JSONDecodeError as e:
    ...     raise wrap_exception(
    ...         e,
    ...         ConfigParseError,
    ...         message="Failed to parse JSON configuration",
    ...         path="config.json"
    ...     )

    Converting external library exceptions:

    >>> try:
    ...     response = requests.get(url, timeout=30)
    ... except requests.Timeout as e:
    ...     raise wrap_exception(e, TimeoutError, model_id=model_id)
    ... except requests.RequestException as e:
    ...     raise wrap_exception(e, APIError, model_id=model_id)

    Preserving original error information:

    >>> try:
    ...     data = external_library.process(input)
    ... except Exception as e:
    ...     wrapped = wrap_exception(e, ProbeExecutionError)
    ...     print(f"Original error type: {wrapped.details['original_error_type']}")
    ...     print(f"Original message: {wrapped.details['original_error_message']}")
    ...     raise wrapped

    Notes
    -----
    The wrapped exception always includes 'original_error_type' and
    'original_error_message' in its details dictionary.

    See Also
    --------
    InsideLLMsError : Base exception class for wrapping.
    """
    msg = message or str(error)
    details["original_error_type"] = type(error).__name__
    details["original_error_message"] = str(error)
    return wrapper_class(msg, details)


def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable.

    Determines whether an exception represents a transient failure
    that may succeed if retried. Currently, RateLimitError and
    TimeoutError are considered retryable.

    Parameters
    ----------
    error : Exception
        The exception to check.

    Returns
    -------
    bool
        True if the error is retryable, False otherwise.

    Examples
    --------
    Basic retry loop:

    >>> for attempt in range(max_retries):
    ...     try:
    ...         result = model.generate(prompt)
    ...         break
    ...     except Exception as e:
    ...         if is_retryable(e) and attempt < max_retries - 1:
    ...             time.sleep(get_retry_delay(e, attempt))
    ...         else:
    ...             raise

    Conditional retry handling:

    >>> try:
    ...     result = model.generate(prompt)
    ... except ModelError as e:
    ...     if is_retryable(e):
    ...         print(f"Transient error, will retry: {e}")
    ...         # Schedule retry
    ...     else:
    ...         print(f"Permanent error, failing: {e}")
    ...         raise

    Combining with get_retry_delay:

    >>> def execute_with_retry(fn, max_attempts=3):
    ...     for attempt in range(max_attempts):
    ...         try:
    ...             return fn()
    ...         except Exception as e:
    ...             if not is_retryable(e) or attempt == max_attempts - 1:
    ...                 raise
    ...             delay = get_retry_delay(e, attempt)
    ...             time.sleep(delay)

    Notes
    -----
    Retryable error types:
    - RateLimitError: API rate limit exceeded
    - TimeoutError: Request timed out

    See Also
    --------
    get_retry_delay : Get recommended wait time before retry.
    RateLimitError : Retryable rate limit error.
    TimeoutError : Retryable timeout error.
    """
    retryable_types = (
        RateLimitError,
        TimeoutError,
    )
    return isinstance(error, retryable_types)


def get_retry_delay(
    error: Exception,
    attempt: int = 1,
    seed: int | None = None,
) -> float:
    """Get the recommended retry delay for an error.

    Calculates an appropriate wait time before retrying a failed
    operation. For RateLimitError with a retry_after value, uses
    that value. Otherwise, uses exponential backoff with jitter.

    Parameters
    ----------
    error : Exception
        The exception that triggered the retry.
    attempt : int, default 1
        The current attempt number (1-indexed). Used for calculating
        exponential backoff.
    seed : int or None, default None
        Optional random seed for deterministic jitter.  When provided,
        a dedicated ``random.Random(seed)`` instance is used instead of
        the global ``random`` module, making the delay reproducible.

    Returns
    -------
    float
        Recommended delay in seconds before the next retry attempt.

    Examples
    --------
    Basic usage with retry loop:

    >>> for attempt in range(1, max_retries + 1):
    ...     try:
    ...         result = model.generate(prompt)
    ...         break
    ...     except (RateLimitError, TimeoutError) as e:
    ...         if attempt == max_retries:
    ...             raise
    ...         delay = get_retry_delay(e, attempt)
    ...         print(f"Waiting {delay:.2f}s before retry {attempt + 1}")
    ...         time.sleep(delay)

    Using with async operations:

    >>> async def generate_with_retry(model, prompt, max_attempts=3):
    ...     for attempt in range(1, max_attempts + 1):
    ...         try:
    ...             return await model.agenerate(prompt)
    ...         except (RateLimitError, TimeoutError) as e:
    ...             if attempt == max_attempts:
    ...                 raise
    ...             delay = get_retry_delay(e, attempt)
    ...             await asyncio.sleep(delay)

    Respecting rate limit retry-after:

    >>> try:
    ...     result = model.generate(prompt)
    ... except RateLimitError as e:
    ...     delay = get_retry_delay(e, attempt=1)
    ...     # If e.retry_after was 30, delay will be 30
    ...     # Otherwise, uses exponential backoff
    ...     print(f"Rate limited. Waiting {delay:.1f}s")

    Notes
    -----
    The exponential backoff formula is:
        delay = min(base_delay * 2^attempt, max_delay) + jitter

    Where:
    - base_delay = 1.0 seconds
    - max_delay = 60.0 seconds
    - jitter = random value between 0 and 10% of delay

    Jitter helps prevent thundering herd problems when multiple
    clients retry simultaneously.

    See Also
    --------
    is_retryable : Check if an error should be retried.
    RateLimitError : May include retry_after value.
    """
    if isinstance(error, RateLimitError) and error.retry_after:
        return error.retry_after

    # Exponential backoff with jitter
    import random

    base_delay = 1.0
    max_delay = 60.0
    delay = min(base_delay * (2**attempt), max_delay)
    rng = random.Random(seed) if seed is not None else random  # noqa: S311
    jitter = rng.uniform(0, delay * 0.1)
    return delay + jitter
