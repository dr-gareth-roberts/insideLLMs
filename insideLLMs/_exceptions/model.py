"""Model and provider exception types."""

from typing import Optional

from insideLLMs._exceptions.base import InsideLLMsError


class ModelError(InsideLLMsError):
    """Base exception for model-related errors.

    This is the parent class for all exceptions that occur during model
    operations, including initialization, generation, API communication,
    and rate limiting. Catching this exception will handle any model-related
    failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any model-related error:

    >>> try:
    ...     model = ModelFactory.create("gpt-4")
    ...     result = model.generate("Hello, world!")
    ... except ModelError as e:
    ...     print(f"Model operation failed: {e}")
    ...     # Handle any model error uniformly

    Distinguishing between model error types:

    >>> try:
    ...     result = model.generate(prompt)
    ... except RateLimitError as e:
    ...     # Handle rate limiting specifically
    ...     time.sleep(e.retry_after or 60)
    ... except TimeoutError as e:
    ...     # Handle timeout specifically
    ...     print("Request took too long")
    ... except ModelError as e:
    ...     # Handle all other model errors
    ...     print(f"Unexpected model error: {e}")

    Notes
    -----
    Subclasses include: ModelNotFoundError, ModelInitializationError,
    ModelGenerationError, RateLimitError, APIError, TimeoutError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    ModelNotFoundError : When a requested model doesn't exist.
    ModelGenerationError : When text generation fails.
    """

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found.

    This exception is raised when attempting to load or use a model that
    does not exist in the registry or is not available from the provider.
    The error includes information about which models are available.

    Parameters
    ----------
    model_id : str
        The identifier of the model that was not found.
    available : list, optional
        List of available model identifiers to help users correct their input.

    Attributes
    ----------
    details : dict
        Contains 'model_id' and optionally 'available_models' list.

    Examples
    --------
    Handling model not found with suggestions:

    >>> try:
    ...     model = ModelFactory.create("gpt-5-turbo")
    ... except ModelNotFoundError as e:
    ...     print(f"Model not found: {e.details['model_id']}")
    ...     if 'available_models' in e.details:
    ...         print("Available models:")
    ...         for m in e.details['available_models']:
    ...             print(f"  - {m}")

    Using fuzzy matching to suggest corrections:

    >>> try:
    ...     model = ModelFactory.create("claude-3-opsu")
    ... except ModelNotFoundError as e:
    ...     from difflib import get_close_matches
    ...     available = e.details.get('available_models', [])
    ...     suggestions = get_close_matches(e.details['model_id'], available)
    ...     if suggestions:
    ...         print(f"Did you mean: {suggestions[0]}?")

    Validating model existence before use:

    >>> def get_model_safe(model_id: str):
    ...     try:
    ...         return ModelFactory.create(model_id)
    ...     except ModelNotFoundError:
    ...         return ModelFactory.create("gpt-3.5-turbo")  # fallback

    See Also
    --------
    ModelInitializationError : When model exists but fails to initialize.
    """

    def __init__(self, model_id: str, available: Optional[list] = None):
        details = {"model_id": model_id}
        if available:
            details["available_models"] = available
        super().__init__(f"Model not found: {model_id}", details)


class ModelInitializationError(ModelError):
    """Raised when a model fails to initialize.

    This exception indicates that while the model was found, it could not
    be properly initialized. Common causes include missing API keys,
    network connectivity issues, invalid configuration, or resource
    constraints.

    Parameters
    ----------
    model_id : str
        The identifier of the model that failed to initialize.
    reason : str
        A description of why initialization failed.

    Attributes
    ----------
    details : dict
        Contains 'model_id' and 'reason' keys.

    Examples
    --------
    Handling missing API key:

    >>> try:
    ...     model = ModelFactory.create("gpt-4")
    ... except ModelInitializationError as e:
    ...     if "API key" in e.details.get('reason', ''):
    ...         print("Please set OPENAI_API_KEY environment variable")
    ...     else:
    ...         print(f"Initialization failed: {e}")

    Retry with exponential backoff for transient failures:

    >>> import time
    >>> for attempt in range(3):
    ...     try:
    ...         model = ModelFactory.create("claude-3-opus")
    ...         break
    ...     except ModelInitializationError as e:
    ...         if "network" in str(e).lower():
    ...             time.sleep(2 ** attempt)
    ...         else:
    ...             raise

    Graceful degradation to a simpler model:

    >>> def get_best_available_model():
    ...     for model_id in ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]:
    ...         try:
    ...             return ModelFactory.create(model_id)
    ...         except ModelInitializationError:
    ...             continue
    ...     raise RuntimeError("No models available")

    See Also
    --------
    ModelNotFoundError : When the model doesn't exist at all.
    APIError : When API-level errors occur during operation.
    """

    def __init__(self, model_id: str, reason: str):
        super().__init__(
            f"Failed to initialize model {model_id}: {reason}",
            {"model_id": model_id, "reason": reason},
        )


class ModelGenerationError(ModelError):
    """Raised when model generation fails.

    This exception is raised when a model successfully initializes but
    fails during text generation. This can occur due to content policy
    violations, malformed prompts, context length exceeded, or internal
    model errors.

    Parameters
    ----------
    model_id : str
        The identifier of the model that failed.
    prompt : str
        The prompt that caused the generation failure. Stored truncated
        to 100 characters in details for debugging.
    reason : str
        A description of why generation failed.
    original_error : Exception, optional
        The underlying exception that caused this error.

    Attributes
    ----------
    original_error : Exception or None
        The underlying exception, if any, that caused this error.
    details : dict
        Contains 'model_id', 'prompt_preview', 'reason', and optionally
        'original_error'.

    Examples
    --------
    Handling generation errors with retry logic:

    >>> def generate_with_retry(model, prompt, max_retries=3):
    ...     for attempt in range(max_retries):
    ...         try:
    ...             return model.generate(prompt)
    ...         except ModelGenerationError as e:
    ...             if attempt == max_retries - 1:
    ...                 raise
    ...             print(f"Attempt {attempt + 1} failed: {e.details['reason']}")

    Accessing the original error for debugging:

    >>> try:
    ...     result = model.generate(very_long_prompt)
    ... except ModelGenerationError as e:
    ...     if e.original_error:
    ...         print(f"Caused by: {type(e.original_error).__name__}")
    ...         print(f"Original message: {e.original_error}")
    ...     print(f"Prompt preview: {e.details['prompt_preview']}")

    Handling content policy violations:

    >>> try:
    ...     result = model.generate(prompt)
    ... except ModelGenerationError as e:
    ...     if "content policy" in e.details.get('reason', '').lower():
    ...         print("Prompt violates content policy. Please revise.")
    ...     elif "context length" in e.details.get('reason', '').lower():
    ...         # Truncate prompt and retry
    ...         result = model.generate(prompt[:4000])
    ...     else:
    ...         raise

    See Also
    --------
    RateLimitError : When generation fails due to rate limiting.
    TimeoutError : When generation times out.
    APIError : For other API-related failures.
    """

    def __init__(
        self,
        model_id: str,
        prompt: str,
        reason: str,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "model_id": model_id,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "reason": reason,
        }
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(f"Model generation failed: {reason}", details)
        self.original_error = original_error


class RateLimitError(ModelError):
    """Raised when API rate limit is exceeded.

    This exception indicates that the request was rejected because the
    rate limit for the API has been exceeded. The error may include a
    retry-after value indicating when the next request can be made.

    This is a retryable error - use the ``is_retryable()`` and
    ``get_retry_delay()`` utility functions for automatic handling.

    Parameters
    ----------
    model_id : str
        The identifier of the model being accessed.
    retry_after : float, optional
        Number of seconds to wait before retrying. May be provided by
        the API in the response headers.

    Attributes
    ----------
    retry_after : float or None
        Suggested wait time in seconds before retrying.
    details : dict
        Contains 'model_id' and optionally 'retry_after_seconds'.

    Examples
    --------
    Basic rate limit handling with retry:

    >>> try:
    ...     result = model.generate(prompt)
    ... except RateLimitError as e:
    ...     wait_time = e.retry_after or 60
    ...     print(f"Rate limited. Waiting {wait_time} seconds...")
    ...     time.sleep(wait_time)
    ...     result = model.generate(prompt)  # retry

    Using the retry utilities:

    >>> from insideLLMs.exceptions import is_retryable, get_retry_delay
    >>> try:
    ...     result = model.generate(prompt)
    ... except ModelError as e:
    ...     if is_retryable(e):
    ...         delay = get_retry_delay(e, attempt=1)
    ...         time.sleep(delay)
    ...         result = model.generate(prompt)

    Implementing exponential backoff:

    >>> import time
    >>> max_retries = 5
    >>> for attempt in range(max_retries):
    ...     try:
    ...         result = model.generate(prompt)
    ...         break
    ...     except RateLimitError as e:
    ...         if attempt == max_retries - 1:
    ...             raise
    ...         wait = e.retry_after or (2 ** attempt)
    ...         print(f"Rate limited, attempt {attempt + 1}. Waiting {wait}s")
    ...         time.sleep(wait)

    Notes
    -----
    Rate limits vary by provider and plan. Check your API provider's
    documentation for specific limits and best practices.

    See Also
    --------
    is_retryable : Check if an error can be retried.
    get_retry_delay : Get recommended retry delay.
    TimeoutError : Another retryable error type.
    """

    def __init__(
        self,
        model_id: str,
        retry_after: Optional[float] = None,
    ):
        details = {"model_id": model_id}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            f"Rate limit exceeded for model {model_id}",
            details,
        )
        self.retry_after = retry_after


class APIError(ModelError):
    """Raised for API-specific errors.

    This exception captures errors from the underlying API provider,
    including HTTP errors, authentication failures, server errors, and
    malformed responses. It preserves the HTTP status code and response
    body for debugging.

    Parameters
    ----------
    model_id : str
        The identifier of the model being accessed.
    status_code : int, optional
        The HTTP status code returned by the API.
    message : str, default "API error"
        Human-readable error message.
    response_body : str, optional
        The raw response body from the API (truncated to 500 chars).

    Attributes
    ----------
    status_code : int or None
        The HTTP status code, useful for categorizing errors:
        - 401: Authentication error
        - 403: Permission denied
        - 404: Endpoint not found
        - 429: Rate limited (prefer RateLimitError)
        - 500+: Server errors
    details : dict
        Contains 'model_id', and optionally 'status_code', 'response_body'.

    Examples
    --------
    Handling API errors by status code:

    >>> try:
    ...     result = model.generate(prompt)
    ... except APIError as e:
    ...     if e.status_code == 401:
    ...         print("Invalid API key. Check your credentials.")
    ...     elif e.status_code == 403:
    ...         print("Access denied. Check your permissions.")
    ...     elif e.status_code and e.status_code >= 500:
    ...         print("Server error. Try again later.")
    ...     else:
    ...         print(f"API error: {e}")

    Logging detailed API error information:

    >>> import logging
    >>> try:
    ...     result = model.generate(prompt)
    ... except APIError as e:
    ...     logging.error(
    ...         "API call failed",
    ...         extra={
    ...             "status_code": e.status_code,
    ...             "model": e.details.get('model_id'),
    ...             "response": e.details.get('response_body', 'N/A')
    ...         }
    ...     )

    Implementing fallback behavior:

    >>> def generate_with_fallback(models, prompt):
    ...     for model in models:
    ...         try:
    ...             return model.generate(prompt)
    ...         except APIError as e:
    ...             if e.status_code and e.status_code < 500:
    ...                 raise  # Client error, don't retry
    ...             continue  # Server error, try next model
    ...     raise RuntimeError("All models failed")

    See Also
    --------
    RateLimitError : Specific error for rate limiting (HTTP 429).
    ModelInitializationError : For errors during model setup.
    """

    def __init__(
        self,
        model_id: str,
        status_code: Optional[int] = None,
        message: str = "API error",
        response_body: Optional[str] = None,
    ):
        details = {"model_id": model_id}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]
        super().__init__(message, details)
        self.status_code = status_code


class ModelTimeoutError(ModelError):
    """Raised when a model request times out.

    This exception indicates that a request to the model exceeded the
    configured timeout duration. This is typically a transient error
    that may succeed on retry.

    This is a retryable error - use the ``is_retryable()`` and
    ``get_retry_delay()`` utility functions for automatic handling.

    Parameters
    ----------
    model_id : str
        The identifier of the model that timed out.
    timeout_seconds : float
        The timeout duration that was exceeded.

    Attributes
    ----------
    details : dict
        Contains 'model_id' and 'timeout_seconds'.

    Examples
    --------
    Basic timeout handling with increased timeout:

    >>> try:
    ...     result = model.generate(prompt, timeout=30)
    ... except TimeoutError as e:
    ...     print(f"Request timed out after {e.details['timeout_seconds']}s")
    ...     # Retry with longer timeout
    ...     result = model.generate(prompt, timeout=60)

    Implementing timeout with retry and backoff:

    >>> def generate_with_timeout_retry(model, prompt, initial_timeout=30):
    ...     timeout = initial_timeout
    ...     for attempt in range(3):
    ...         try:
    ...             return model.generate(prompt, timeout=timeout)
    ...         except TimeoutError:
    ...             timeout *= 1.5  # Increase timeout
    ...             if attempt == 2:
    ...                 raise
    ...     return None

    Using is_retryable utility:

    >>> from insideLLMs.exceptions import is_retryable, get_retry_delay
    >>> try:
    ...     result = model.generate(prompt)
    ... except ModelError as e:
    ...     if is_retryable(e):  # True for TimeoutError
    ...         delay = get_retry_delay(e, attempt=1)
    ...         time.sleep(delay)
    ...         result = model.generate(prompt)

    Notes
    -----
    Timeouts may indicate:
    - Network latency issues
    - Model under heavy load
    - Prompt complexity requiring more processing time
    - Insufficient timeout configuration

    Consider increasing timeout for complex prompts or using
    async operations for long-running requests.

    See Also
    --------
    is_retryable : Check if an error can be retried.
    get_retry_delay : Get recommended retry delay.
    RateLimitError : Another retryable error type.
    """

    def __init__(self, model_id: str, timeout_seconds: float):
        super().__init__(
            f"Request to {model_id} timed out after {timeout_seconds}s",
            {"model_id": model_id, "timeout_seconds": timeout_seconds},
        )


# Backward-compatible alias.
TimeoutError = ModelTimeoutError
