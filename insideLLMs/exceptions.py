"""Custom exceptions for insideLLMs.

This module defines the exception hierarchy for the insideLLMs library,
providing clear error types for different failure modes during model
interaction, probe execution, dataset handling, and evaluation.

Exception Hierarchy
-------------------
All exceptions inherit from :class:`InsideLLMsError`, enabling broad
exception catching while still allowing specific error handling:

.. code-block:: none

    InsideLLMsError (base)
    |
    +-- ModelError (model operations)
    |   +-- ModelNotFoundError
    |   +-- ModelInitializationError
    |   +-- ModelGenerationError
    |   +-- RateLimitError
    |   +-- APIError
    |   +-- TimeoutError
    |
    +-- ProbeError (probe operations)
    |   +-- ProbeNotFoundError
    |   +-- ProbeValidationError
    |   +-- ProbeExecutionError
    |   +-- RunnerExecutionError
    |
    +-- DatasetError (dataset operations)
    |   +-- DatasetNotFoundError
    |   +-- DatasetFormatError
    |   +-- DatasetValidationError
    |
    +-- ConfigurationError (configuration)
    |   +-- ConfigValidationError
    |   +-- ConfigNotFoundError
    |   +-- ConfigParseError
    |
    +-- CacheError (caching)
    |   +-- CacheMissError
    |   +-- CacheCorruptionError
    |
    +-- EvaluationError (evaluation)
    |   +-- EvaluatorNotFoundError
    |   +-- EvaluationFailedError
    |
    +-- RegistryError (registry)
        +-- AlreadyRegisteredError
        +-- NotRegisteredError

Examples
--------
Catching all insideLLMs errors:

>>> try:
...     result = model.generate("test prompt")
... except InsideLLMsError as e:
...     print(f"Operation failed: {e}")
...     print(f"Details: {e.details}")

Catching specific error types with retry logic:

>>> from insideLLMs.exceptions import RateLimitError, TimeoutError, is_retryable
>>> try:
...     result = model.generate("test prompt")
... except RateLimitError as e:
...     print(f"Rate limited. Retry after {e.retry_after} seconds")
... except TimeoutError as e:
...     print(f"Request timed out: {e}")

Using the retryable check utility:

>>> if is_retryable(error):
...     delay = get_retry_delay(error, attempt=1)
...     time.sleep(delay)
...     # retry operation

Notes
-----
All exceptions include a ``details`` dictionary containing structured
information about the error, useful for logging and debugging.
"""

from typing import Any, Optional

__all__ = [
    # Base exception
    "InsideLLMsError",
    # Model errors
    "ModelError",
    "ModelNotFoundError",
    "ModelInitializationError",
    "ModelGenerationError",
    "RateLimitError",
    "APIError",
    "TimeoutError",
    # Probe errors
    "ProbeError",
    "ProbeNotFoundError",
    "ProbeValidationError",
    "ProbeExecutionError",
    "RunnerExecutionError",
    # Dataset errors
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetFormatError",
    "DatasetValidationError",
    # Configuration errors
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigNotFoundError",
    "ConfigParseError",
    # Cache errors
    "CacheError",
    "CacheMissError",
    "CacheCorruptionError",
    # Evaluation errors
    "EvaluationError",
    "EvaluatorNotFoundError",
    "EvaluationFailedError",
    # Registry errors
    "RegistryError",
    "AlreadyRegisteredError",
    "NotRegisteredError",
    # Utility functions
    "wrap_exception",
    "is_retryable",
    "get_retry_delay",
]


class InsideLLMsError(Exception):
    """Base exception for all insideLLMs errors.

    This is the root exception class for the insideLLMs library. All other
    exceptions in this module inherit from this class, allowing users to
    catch all library-specific errors with a single except clause.

    Parameters
    ----------
    message : str
        Human-readable error message describing what went wrong.
    details : dict[str, Any], optional
        Structured dictionary containing additional context about the error.
        Useful for logging, debugging, and programmatic error handling.

    Attributes
    ----------
    message : str
        The error message passed during initialization.
    details : dict[str, Any]
        Dictionary of additional error context. Empty dict if not provided.

    Examples
    --------
    Catching all insideLLMs errors:

    >>> try:
    ...     # Any insideLLMs operation
    ...     probe.run(model, dataset)
    ... except InsideLLMsError as e:
    ...     print(f"Error: {e.message}")
    ...     if e.details:
    ...         for key, value in e.details.items():
    ...             print(f"  {key}: {value}")

    Creating a custom error with details:

    >>> error = InsideLLMsError(
    ...     "Operation failed",
    ...     details={"component": "tokenizer", "input_length": 5000}
    ... )
    >>> str(error)
    "Operation failed | Details: {'component': 'tokenizer', 'input_length': 5000}"

    Logging errors with structured data:

    >>> import logging
    >>> try:
    ...     result = model.generate(prompt)
    ... except InsideLLMsError as e:
    ...     logging.error(
    ...         "InsideLLMs error occurred",
    ...         extra={"error_message": e.message, **e.details}
    ...     )

    Notes
    -----
    The string representation includes both the message and details when
    present, making it suitable for direct printing or logging.

    See Also
    --------
    ModelError : Base class for model-related errors.
    ProbeError : Base class for probe-related errors.
    DatasetError : Base class for dataset-related errors.
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Model-related errors


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


class TimeoutError(ModelError):
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


# Probe-related errors


class ProbeError(InsideLLMsError):
    """Base exception for probe-related errors.

    This is the parent class for all exceptions that occur during probe
    operations, including probe lookup, validation, and execution.
    Catching this exception will handle any probe-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any probe-related error:

    >>> try:
    ...     probe = ProbeFactory.create("attention_probe")
    ...     results = probe.run(model, dataset)
    ... except ProbeError as e:
    ...     print(f"Probe operation failed: {e}")
    ...     # Handle any probe error uniformly

    Distinguishing between probe error types:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeValidationError as e:
    ...     print(f"Invalid input: {e.details['reason']}")
    ... except ProbeExecutionError as e:
    ...     print(f"Execution failed at sample {e.details.get('sample_index')}")
    ... except ProbeError as e:
    ...     print(f"Other probe error: {e}")

    Notes
    -----
    Subclasses include: ProbeNotFoundError, ProbeValidationError,
    ProbeExecutionError, RunnerExecutionError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    ProbeExecutionError : When probe execution fails.
    RunnerExecutionError : For detailed runner failure context.
    """

    pass


class ProbeNotFoundError(ProbeError):
    """Raised when a requested probe is not found.

    This exception is raised when attempting to use a probe type that
    does not exist in the registry. The error includes information
    about available probe types to help users correct their input.

    Parameters
    ----------
    probe_type : str
        The identifier of the probe that was not found.
    available : list, optional
        List of available probe type identifiers.

    Attributes
    ----------
    details : dict
        Contains 'probe_type' and optionally 'available_probes' list.

    Examples
    --------
    Handling probe not found with suggestions:

    >>> try:
    ...     probe = ProbeFactory.create("atention_probe")  # typo
    ... except ProbeNotFoundError as e:
    ...     print(f"Unknown probe: {e.details['probe_type']}")
    ...     if 'available_probes' in e.details:
    ...         print("Available probes:")
    ...         for p in e.details['available_probes']:
    ...             print(f"  - {p}")

    Implementing probe selection with fallback:

    >>> def get_probe(probe_type: str, fallback: str = "default_probe"):
    ...     try:
    ...         return ProbeFactory.create(probe_type)
    ...     except ProbeNotFoundError:
    ...         print(f"Probe '{probe_type}' not found, using '{fallback}'")
    ...         return ProbeFactory.create(fallback)

    Validating probe type before use:

    >>> def validate_probe_config(config: dict):
    ...     probe_type = config.get('probe_type')
    ...     try:
    ...         probe = ProbeFactory.create(probe_type)
    ...         return True
    ...     except ProbeNotFoundError as e:
    ...         print(f"Invalid probe in config: {e}")
    ...         return False

    See Also
    --------
    ProbeValidationError : When probe input is invalid.
    ProbeExecutionError : When probe execution fails.
    """

    def __init__(self, probe_type: str, available: Optional[list] = None):
        details = {"probe_type": probe_type}
        if available:
            details["available_probes"] = available
        super().__init__(f"Probe not found: {probe_type}", details)


class ProbeValidationError(ProbeError):
    """Raised when probe input validation fails.

    This exception is raised when input data provided to a probe does
    not meet the required format, type, or constraints. This includes
    invalid prompt formats, missing required fields, or data type
    mismatches.

    Parameters
    ----------
    probe_type : str
        The type of probe that performed validation.
    reason : str
        A description of why validation failed.
    invalid_input : Any, optional
        The input that failed validation (truncated to 100 chars).

    Attributes
    ----------
    details : dict
        Contains 'probe_type', 'reason', and optionally 'invalid_input'.

    Examples
    --------
    Handling validation errors with user feedback:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeValidationError as e:
    ...     print(f"Invalid input for {e.details['probe_type']}:")
    ...     print(f"  Reason: {e.details['reason']}")
    ...     if 'invalid_input' in e.details:
    ...         print(f"  Input: {e.details['invalid_input']}")

    Pre-validating data before running probe:

    >>> def safe_run_probe(probe, model, data):
    ...     try:
    ...         return probe.run(model, data)
    ...     except ProbeValidationError as e:
    ...         # Log validation failure and skip
    ...         logging.warning(f"Skipping invalid sample: {e}")
    ...         return None

    Collecting validation errors for batch processing:

    >>> validation_errors = []
    >>> for sample in samples:
    ...     try:
    ...         probe.validate(sample)
    ...     except ProbeValidationError as e:
    ...         validation_errors.append({
    ...             'sample': sample,
    ...             'error': e.details['reason']
    ...         })
    >>> if validation_errors:
    ...     print(f"Found {len(validation_errors)} invalid samples")

    See Also
    --------
    ProbeExecutionError : When execution fails after validation.
    DatasetValidationError : For dataset-level validation failures.
    """

    def __init__(self, probe_type: str, reason: str, invalid_input: Any = None):
        details = {"probe_type": probe_type, "reason": reason}
        if invalid_input is not None:
            details["invalid_input"] = str(invalid_input)[:100]
        super().__init__(f"Probe validation failed: {reason}", details)


class ProbeExecutionError(ProbeError):
    """Raised when probe execution fails.

    This exception is raised when a probe fails during execution after
    passing validation. This can occur due to model errors, computation
    failures, or unexpected data conditions during runtime.

    Parameters
    ----------
    probe_type : str
        The type of probe that failed.
    reason : str
        A description of why execution failed.
    sample_index : int, optional
        The index of the sample that caused the failure, useful for
        resuming batch processing.
    original_error : Exception, optional
        The underlying exception that caused this error.

    Attributes
    ----------
    original_error : Exception or None
        The underlying exception, if any, for debugging.
    details : dict
        Contains 'probe_type', 'reason', and optionally 'sample_index'
        and 'original_error'.

    Examples
    --------
    Handling execution errors with sample tracking:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeExecutionError as e:
    ...     failed_index = e.details.get('sample_index')
    ...     if failed_index is not None:
    ...         print(f"Failed at sample {failed_index}")
    ...         # Resume from failed sample
    ...         remaining = dataset[failed_index + 1:]
    ...         results = probe.run(model, remaining)

    Accessing the original error for debugging:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeExecutionError as e:
    ...     if e.original_error:
    ...         print(f"Root cause: {type(e.original_error).__name__}")
    ...         import traceback
    ...         traceback.print_exception(type(e.original_error),
    ...                                   e.original_error,
    ...                                   e.original_error.__traceback__)

    Implementing skip-on-failure behavior:

    >>> results = []
    >>> for idx, sample in enumerate(dataset):
    ...     try:
    ...         result = probe.run_single(model, sample)
    ...         results.append(result)
    ...     except ProbeExecutionError as e:
    ...         print(f"Skipping sample {idx}: {e.details['reason']}")
    ...         results.append(None)

    See Also
    --------
    ProbeValidationError : When input fails validation.
    RunnerExecutionError : For detailed runner failure context.
    ModelGenerationError : When the underlying model fails.
    """

    def __init__(
        self,
        probe_type: str,
        reason: str,
        sample_index: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {"probe_type": probe_type, "reason": reason}
        if sample_index is not None:
            details["sample_index"] = sample_index
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(f"Probe execution failed: {reason}", details)
        self.original_error = original_error


class RunnerExecutionError(ProbeError):
    """Raised when runner execution fails with rich context.

    This exception captures the full execution context including model,
    probe, prompt, and timing information for easier debugging. It provides
    detailed information for troubleshooting complex pipeline failures.

    Parameters
    ----------
    reason : str
        A description of why execution failed.
    model_id : str, optional
        The identifier of the model being used.
    probe_id : str, optional
        The identifier of the probe being run.
    prompt : str, optional
        The prompt that caused the error (truncated in output).
    prompt_index : int, optional
        Index of the prompt in the dataset.
    run_id : str, optional
        The unique run identifier for tracking.
    elapsed_seconds : float, optional
        Time elapsed before the error occurred.
    original_error : Exception, optional
        The underlying exception that caused this error.
    suggestions : list[str], optional
        List of suggestions for resolving the error.

    Attributes
    ----------
    model_id : str or None
        The model identifier.
    probe_id : str or None
        The probe identifier.
    prompt : str or None
        The prompt that caused the error.
    prompt_index : int or None
        The index of the problematic prompt.
    run_id : str or None
        The unique run identifier.
    elapsed_seconds : float or None
        Elapsed time before failure.
    original_error : Exception or None
        The underlying exception.
    suggestions : list[str]
        Suggestions for resolving the error.
    details : dict
        Rich context dictionary with all available information.

    Examples
    --------
    Handling runner errors with full context:

    >>> try:
    ...     runner.execute(model, probe, dataset)
    ... except RunnerExecutionError as e:
    ...     print(f"Runner failed: {e.message}")
    ...     if e.model_id:
    ...         print(f"  Model: {e.model_id}")
    ...     if e.probe_id:
    ...         print(f"  Probe: {e.probe_id}")
    ...     if e.prompt_index is not None:
    ...         print(f"  Failed at prompt #{e.prompt_index}")
    ...     if e.elapsed_seconds:
    ...         print(f"  Elapsed time: {e.elapsed_seconds:.2f}s")

    Using suggestions for error resolution:

    >>> try:
    ...     runner.execute(model, probe, dataset)
    ... except RunnerExecutionError as e:
    ...     if e.suggestions:
    ...         print("Suggestions:")
    ...         for suggestion in e.suggestions:
    ...             print(f"  - {suggestion}")

    Logging detailed error information:

    >>> import logging
    >>> try:
    ...     runner.execute(model, probe, dataset)
    ... except RunnerExecutionError as e:
    ...     logging.error(
    ...         "Runner execution failed",
    ...         extra={
    ...             "run_id": e.run_id,
    ...             "model": e.model_id,
    ...             "probe": e.probe_id,
    ...             "prompt_index": e.prompt_index,
    ...             "elapsed": e.elapsed_seconds,
    ...             "error_type": type(e.original_error).__name__ if e.original_error else None,
    ...         }
    ...     )

    Notes
    -----
    The string representation of this exception is specially formatted
    to provide a multi-line detailed error report, including context,
    prompt preview, cause, and suggestions.

    See Also
    --------
    ProbeExecutionError : Simpler execution error without full context.
    ModelGenerationError : When the underlying model fails.
    """

    def __init__(
        self,
        reason: str,
        *,
        model_id: Optional[str] = None,
        probe_id: Optional[str] = None,
        prompt: Optional[str] = None,
        prompt_index: Optional[int] = None,
        run_id: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None,
        suggestions: Optional[list[str]] = None,
    ):
        self.model_id = model_id
        self.probe_id = probe_id
        self.prompt = prompt
        self.prompt_index = prompt_index
        self.run_id = run_id
        self.elapsed_seconds = elapsed_seconds
        self.original_error = original_error
        self.suggestions = suggestions or []

        details: dict[str, Any] = {"reason": reason}
        if model_id:
            details["model_id"] = model_id
        if probe_id:
            details["probe_id"] = probe_id
        if prompt:
            details["prompt_preview"] = prompt[:100] + "..." if len(prompt) > 100 else prompt
        if prompt_index is not None:
            details["prompt_index"] = prompt_index
        if run_id:
            details["run_id"] = run_id
        if elapsed_seconds is not None:
            details["elapsed_seconds"] = round(elapsed_seconds, 3)
        if original_error:
            details["original_error_type"] = type(original_error).__name__
            details["original_error_message"] = str(original_error)

        super().__init__(f"Runner execution failed: {reason}", details)

    def __str__(self) -> str:
        parts = [self.message]

        context_parts = []
        if self.model_id:
            context_parts.append(f"model={self.model_id}")
        if self.probe_id:
            context_parts.append(f"probe={self.probe_id}")
        if self.prompt_index is not None:
            context_parts.append(f"index={self.prompt_index}")
        if self.run_id:
            context_parts.append(f"run_id={self.run_id}")

        if context_parts:
            parts.append(f"Context: [{', '.join(context_parts)}]")

        if self.prompt:
            preview = self.prompt[:80] + "..." if len(self.prompt) > 80 else self.prompt
            parts.append(f"Prompt: {preview!r}")

        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")

        if self.suggestions:
            parts.append("Suggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  - {suggestion}")

        return "\n".join(parts)


# Dataset-related errors


class DatasetError(InsideLLMsError):
    """Base exception for dataset-related errors.

    This is the parent class for all exceptions that occur during dataset
    operations, including loading, parsing, and validation. Catching this
    exception will handle any dataset-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any dataset-related error:

    >>> try:
    ...     dataset = DatasetLoader.load("path/to/data.csv")
    ...     dataset.validate()
    ... except DatasetError as e:
    ...     print(f"Dataset operation failed: {e}")

    Distinguishing between dataset error types:

    >>> try:
    ...     dataset = DatasetLoader.load(path)
    ... except DatasetNotFoundError:
    ...     print("File not found")
    ... except DatasetFormatError as e:
    ...     print(f"Invalid format: {e.details['reason']}")
    ... except DatasetError as e:
    ...     print(f"Other dataset error: {e}")

    Notes
    -----
    Subclasses include: DatasetNotFoundError, DatasetFormatError,
    DatasetValidationError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    DatasetValidationError : For row-level validation failures.
    """

    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset is not found.

    This exception is raised when attempting to load a dataset from a
    path that does not exist or is not accessible.

    Parameters
    ----------
    path : str
        The path to the dataset that was not found.

    Attributes
    ----------
    details : dict
        Contains 'path' key with the attempted path.

    Examples
    --------
    Handling missing dataset with fallback:

    >>> try:
    ...     dataset = DatasetLoader.load("data/custom_dataset.csv")
    ... except DatasetNotFoundError as e:
    ...     print(f"Dataset not found: {e.details['path']}")
    ...     dataset = DatasetLoader.load("data/default_dataset.csv")

    Validating dataset path before loading:

    >>> import os
    >>> def load_dataset_safe(path: str):
    ...     if not os.path.exists(path):
    ...         raise DatasetNotFoundError(path)
    ...     return DatasetLoader.load(path)

    Creating helpful error messages:

    >>> try:
    ...     dataset = DatasetLoader.load(user_provided_path)
    ... except DatasetNotFoundError as e:
    ...     print(f"Could not find dataset at: {e.details['path']}")
    ...     print("Please check the path and try again.")

    See Also
    --------
    DatasetFormatError : When dataset exists but has invalid format.
    ConfigNotFoundError : For missing configuration files.
    """

    def __init__(self, path: str):
        super().__init__(
            f"Dataset not found: {path}",
            {"path": path},
        )


class DatasetFormatError(DatasetError):
    """Raised when dataset format is invalid.

    This exception is raised when a dataset file exists but cannot be
    parsed due to format issues. This includes unsupported file types,
    malformed content, or missing required structure.

    Parameters
    ----------
    reason : str
        A description of why the format is invalid.
    expected_format : str, optional
        Description of the expected format for guidance.

    Attributes
    ----------
    details : dict
        Contains 'reason' and optionally 'expected_format'.

    Examples
    --------
    Handling format errors with expected format info:

    >>> try:
    ...     dataset = DatasetLoader.load("data.json")
    ... except DatasetFormatError as e:
    ...     print(f"Format error: {e.details['reason']}")
    ...     if 'expected_format' in e.details:
    ...         print(f"Expected: {e.details['expected_format']}")

    Attempting multiple formats:

    >>> def load_any_format(base_path: str):
    ...     for ext in ['.csv', '.json', '.parquet']:
    ...         try:
    ...             return DatasetLoader.load(base_path + ext)
    ...         except DatasetNotFoundError:
    ...             continue
    ...         except DatasetFormatError:
    ...             continue
    ...     raise DatasetError("No valid dataset found", {"path": base_path})

    Providing format conversion guidance:

    >>> try:
    ...     dataset = DatasetLoader.load("data.xlsx")
    ... except DatasetFormatError as e:
    ...     print(f"Error: {e}")
    ...     print("Tip: Convert Excel to CSV using pandas:")
    ...     print("  pd.read_excel('data.xlsx').to_csv('data.csv')")

    See Also
    --------
    DatasetNotFoundError : When dataset file doesn't exist.
    DatasetValidationError : When content fails validation.
    """

    def __init__(self, reason: str, expected_format: Optional[str] = None):
        details = {"reason": reason}
        if expected_format:
            details["expected_format"] = expected_format
        super().__init__(f"Invalid dataset format: {reason}", details)


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails.

    This exception is raised when dataset content fails validation checks.
    This includes type mismatches, missing required fields, constraint
    violations, or data quality issues at specific rows.

    Parameters
    ----------
    reason : str
        A description of why validation failed.
    row_index : int, optional
        The index of the row that failed validation (0-based).
    field : str, optional
        The name of the field that failed validation.

    Attributes
    ----------
    details : dict
        Contains 'reason', and optionally 'row_index' and 'field'.

    Examples
    --------
    Handling validation errors with row information:

    >>> try:
    ...     dataset.validate()
    ... except DatasetValidationError as e:
    ...     print(f"Validation failed: {e.details['reason']}")
    ...     if 'row_index' in e.details:
    ...         print(f"  At row: {e.details['row_index']}")
    ...     if 'field' in e.details:
    ...         print(f"  Field: {e.details['field']}")

    Collecting all validation errors:

    >>> errors = []
    >>> for idx, row in enumerate(dataset):
    ...     try:
    ...         validate_row(row)
    ...     except DatasetValidationError as e:
    ...         errors.append({
    ...             'row': idx,
    ...             'reason': e.details['reason'],
    ...             'field': e.details.get('field')
    ...         })
    >>> if errors:
    ...     print(f"Found {len(errors)} validation errors")

    Skipping invalid rows with logging:

    >>> valid_rows = []
    >>> for idx, row in enumerate(dataset):
    ...     try:
    ...         validate_row(row)
    ...         valid_rows.append(row)
    ...     except DatasetValidationError as e:
    ...         logging.warning(f"Skipping row {idx}: {e.details['reason']}")

    See Also
    --------
    DatasetFormatError : When overall format is invalid.
    ProbeValidationError : For probe-specific validation.
    ConfigValidationError : For configuration validation.
    """

    def __init__(
        self,
        reason: str,
        row_index: Optional[int] = None,
        field: Optional[str] = None,
    ):
        details = {"reason": reason}
        if row_index is not None:
            details["row_index"] = row_index
        if field:
            details["field"] = field
        super().__init__(f"Dataset validation failed: {reason}", details)


# Configuration errors


class ConfigurationError(InsideLLMsError):
    """Base exception for configuration errors.

    This is the parent class for all exceptions that occur during
    configuration operations, including loading, parsing, and validation.
    Catching this exception will handle any configuration-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any configuration-related error:

    >>> try:
    ...     config = ConfigLoader.load("config.yaml")
    ...     config.validate()
    ... except ConfigurationError as e:
    ...     print(f"Configuration error: {e}")
    ...     # Fall back to default configuration
    ...     config = Config.default()

    Distinguishing between configuration error types:

    >>> try:
    ...     config = ConfigLoader.load(path)
    ... except ConfigNotFoundError:
    ...     print("Config file not found, creating default...")
    ...     config = Config.default()
    ...     config.save(path)
    ... except ConfigParseError as e:
    ...     print(f"Parse error at line {e.details.get('line')}")
    ... except ConfigValidationError as e:
    ...     print(f"Invalid value for {e.details['field']}")

    Notes
    -----
    Subclasses include: ConfigValidationError, ConfigNotFoundError,
    ConfigParseError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    ConfigValidationError : For field-level validation failures.
    """

    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails.

    This exception is raised when a configuration value does not meet
    the required constraints, type requirements, or business rules.

    Parameters
    ----------
    field : str
        The name of the configuration field that failed validation.
    reason : str
        A description of why validation failed.
    value : Any, optional
        The invalid value (truncated to 100 chars in details).

    Attributes
    ----------
    details : dict
        Contains 'field', 'reason', and optionally 'value'.

    Examples
    --------
    Handling validation errors with field information:

    >>> try:
    ...     config.validate()
    ... except ConfigValidationError as e:
    ...     print(f"Invalid config: {e.details['field']}")
    ...     print(f"  Reason: {e.details['reason']}")
    ...     if 'value' in e.details:
    ...         print(f"  Provided value: {e.details['value']}")

    Providing helpful error messages to users:

    >>> try:
    ...     config = Config(temperature=2.5)
    ... except ConfigValidationError as e:
    ...     if e.details['field'] == 'temperature':
    ...         print("Temperature must be between 0.0 and 2.0")
    ...     else:
    ...         print(f"Configuration error: {e}")

    Validating multiple fields and collecting errors:

    >>> errors = []
    >>> for field, value in user_config.items():
    ...     try:
    ...         validate_field(field, value)
    ...     except ConfigValidationError as e:
    ...         errors.append(e)
    >>> if errors:
    ...     print(f"Found {len(errors)} configuration errors")

    See Also
    --------
    ConfigParseError : When configuration cannot be parsed.
    DatasetValidationError : For dataset validation failures.
    """

    def __init__(self, field: str, reason: str, value: Any = None):
        details = {"field": field, "reason": reason}
        if value is not None:
            details["value"] = str(value)[:100]
        super().__init__(f"Invalid configuration for '{field}': {reason}", details)


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration file is not found.

    This exception is raised when attempting to load a configuration
    file from a path that does not exist or is not accessible.

    Parameters
    ----------
    path : str
        The path to the configuration file that was not found.

    Attributes
    ----------
    details : dict
        Contains 'path' key with the attempted path.

    Examples
    --------
    Handling missing config with defaults:

    >>> try:
    ...     config = ConfigLoader.load("custom_config.yaml")
    ... except ConfigNotFoundError as e:
    ...     print(f"Config not found: {e.details['path']}")
    ...     config = Config.default()

    Creating config file if missing:

    >>> def load_or_create_config(path: str):
    ...     try:
    ...         return ConfigLoader.load(path)
    ...     except ConfigNotFoundError:
    ...         config = Config.default()
    ...         config.save(path)
    ...         print(f"Created default config at {path}")
    ...         return config

    Searching multiple config locations:

    >>> config_paths = ["./config.yaml", "~/.insidellms/config.yaml"]
    >>> for path in config_paths:
    ...     try:
    ...         config = ConfigLoader.load(os.path.expanduser(path))
    ...         break
    ...     except ConfigNotFoundError:
    ...         continue
    ... else:
    ...     raise ConfigNotFoundError("No configuration file found")

    See Also
    --------
    ConfigParseError : When config exists but cannot be parsed.
    DatasetNotFoundError : For missing dataset files.
    """

    def __init__(self, path: str):
        super().__init__(
            f"Configuration file not found: {path}",
            {"path": path},
        )


class ConfigParseError(ConfigurationError):
    """Raised when configuration parsing fails.

    This exception is raised when a configuration file exists but
    cannot be parsed due to syntax errors, encoding issues, or
    invalid structure.

    Parameters
    ----------
    path : str
        The path to the configuration file.
    reason : str
        A description of why parsing failed.
    line : int, optional
        The line number where the error occurred.

    Attributes
    ----------
    details : dict
        Contains 'path', 'reason', and optionally 'line'.

    Examples
    --------
    Handling parse errors with line information:

    >>> try:
    ...     config = ConfigLoader.load("config.yaml")
    ... except ConfigParseError as e:
    ...     print(f"Failed to parse {e.details['path']}")
    ...     print(f"  Error: {e.details['reason']}")
    ...     if 'line' in e.details:
    ...         print(f"  At line: {e.details['line']}")

    Providing helpful debugging information:

    >>> try:
    ...     config = ConfigLoader.load(path)
    ... except ConfigParseError as e:
    ...     if 'line' in e.details:
    ...         with open(path) as f:
    ...             lines = f.readlines()
    ...             error_line = lines[e.details['line'] - 1]
    ...             print(f"Problematic line: {error_line.strip()}")

    Attempting different parsers:

    >>> def load_config_flexible(path: str):
    ...     for loader in [yaml_loader, json_loader, toml_loader]:
    ...         try:
    ...             return loader(path)
    ...         except ConfigParseError:
    ...             continue
    ...     raise ConfigParseError(path, "No compatible parser found")

    See Also
    --------
    ConfigNotFoundError : When config file doesn't exist.
    ConfigValidationError : When config values are invalid.
    DatasetFormatError : For dataset parsing failures.
    """

    def __init__(self, path: str, reason: str, line: Optional[int] = None):
        details = {"path": path, "reason": reason}
        if line is not None:
            details["line"] = line
        super().__init__(f"Failed to parse configuration: {reason}", details)


# Cache errors


class CacheError(InsideLLMsError):
    """Base exception for cache errors.

    This is the parent class for all exceptions that occur during cache
    operations, including lookups, storage, and integrity checks.
    Catching this exception will handle any cache-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any cache-related error:

    >>> try:
    ...     result = cache.get(key)
    ... except CacheError as e:
    ...     print(f"Cache operation failed: {e}")
    ...     # Fall back to computing the result
    ...     result = compute_expensive_result()

    Implementing cache-aside pattern:

    >>> def get_with_cache(key: str, compute_fn):
    ...     try:
    ...         return cache.get(key)
    ...     except CacheMissError:
    ...         result = compute_fn()
    ...         cache.set(key, result)
    ...         return result
    ...     except CacheError as e:
    ...         logging.warning(f"Cache error: {e}")
    ...         return compute_fn()

    Notes
    -----
    Subclasses include: CacheMissError, CacheCorruptionError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    CacheMissError : When a requested key is not in cache.
    CacheCorruptionError : When cached data is invalid.
    """

    pass


class CacheMissError(CacheError):
    """Raised when a cache lookup misses.

    This exception is raised when attempting to retrieve a value from
    the cache that does not exist. This is often expected behavior and
    signals that the value needs to be computed or fetched.

    Parameters
    ----------
    key : str
        The cache key that was not found.

    Attributes
    ----------
    details : dict
        Contains 'key' with the requested cache key.

    Examples
    --------
    Basic cache lookup with miss handling:

    >>> try:
    ...     result = cache.get("model_output_12345")
    ... except CacheMissError as e:
    ...     print(f"Cache miss for key: {e.details['key']}")
    ...     result = model.generate(prompt)
    ...     cache.set(e.details['key'], result)

    Implementing lazy loading:

    >>> def get_cached_result(cache, key, generator_fn):
    ...     try:
    ...         return cache.get(key)
    ...     except CacheMissError:
    ...         result = generator_fn()
    ...         cache.set(key, result)
    ...         return result

    Tracking cache statistics:

    >>> stats = {'hits': 0, 'misses': 0}
    >>> def get_with_stats(cache, key):
    ...     try:
    ...         result = cache.get(key)
    ...         stats['hits'] += 1
    ...         return result
    ...     except CacheMissError:
    ...         stats['misses'] += 1
    ...         raise

    Notes
    -----
    Cache misses are normal and expected. Design your code to handle
    them gracefully by computing or fetching the missing value.

    See Also
    --------
    CacheCorruptionError : When cached data is corrupted.
    """

    def __init__(self, key: str):
        super().__init__(f"Cache miss for key: {key}", {"key": key})


class CacheCorruptionError(CacheError):
    """Raised when cache data is corrupted.

    This exception is raised when cached data cannot be deserialized,
    has an invalid format, or fails integrity checks. The corrupted
    entry should typically be removed and recomputed.

    Parameters
    ----------
    reason : str
        A description of how the corruption was detected.
    key : str, optional
        The cache key of the corrupted entry.

    Attributes
    ----------
    details : dict
        Contains 'reason' and optionally 'key'.

    Examples
    --------
    Handling corruption by clearing the entry:

    >>> try:
    ...     result = cache.get(key)
    ... except CacheCorruptionError as e:
    ...     print(f"Corrupted cache entry: {e.details['reason']}")
    ...     if 'key' in e.details:
    ...         cache.delete(e.details['key'])
    ...     result = compute_fresh_result()

    Implementing self-healing cache:

    >>> def get_or_heal(cache, key, compute_fn):
    ...     try:
    ...         return cache.get(key)
    ...     except CacheCorruptionError:
    ...         cache.delete(key)
    ...         result = compute_fn()
    ...         cache.set(key, result)
    ...         return result
    ...     except CacheMissError:
    ...         result = compute_fn()
    ...         cache.set(key, result)
    ...         return result

    Logging corruption for monitoring:

    >>> try:
    ...     result = cache.get(key)
    ... except CacheCorruptionError as e:
    ...     logging.error(
    ...         "Cache corruption detected",
    ...         extra={
    ...             "key": e.details.get("key"),
    ...             "reason": e.details["reason"]
    ...         }
    ...     )
    ...     # Alert monitoring system
    ...     metrics.increment("cache.corruption")

    See Also
    --------
    CacheMissError : When key simply doesn't exist.
    """

    def __init__(self, reason: str, key: Optional[str] = None):
        details = {"reason": reason}
        if key:
            details["key"] = key
        super().__init__(f"Cache corruption detected: {reason}", details)


# Evaluation errors


class EvaluationError(InsideLLMsError):
    """Base exception for evaluation errors.

    This is the parent class for all exceptions that occur during
    evaluation operations, including evaluator lookup and metric
    computation. Catching this exception will handle any evaluation-related
    failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any evaluation-related error:

    >>> try:
    ...     scores = evaluator.evaluate(predictions, references)
    ... except EvaluationError as e:
    ...     print(f"Evaluation failed: {e}")
    ...     # Skip evaluation and continue
    ...     scores = None

    Distinguishing between evaluation error types:

    >>> try:
    ...     evaluator = EvaluatorFactory.create("bleu")
    ...     scores = evaluator.evaluate(predictions, references)
    ... except EvaluatorNotFoundError:
    ...     print("Evaluator not available")
    ... except EvaluationFailedError as e:
    ...     print(f"Evaluation computation failed: {e.details['reason']}")

    Notes
    -----
    Subclasses include: EvaluatorNotFoundError, EvaluationFailedError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    EvaluationFailedError : When evaluation computation fails.
    """

    pass


class EvaluatorNotFoundError(EvaluationError):
    """Raised when an evaluator type is not found.

    This exception is raised when attempting to use an evaluator that
    does not exist in the registry. The error includes information
    about available evaluators to help users correct their input.

    Parameters
    ----------
    evaluator_type : str
        The identifier of the evaluator that was not found.
    available : list, optional
        List of available evaluator type identifiers.

    Attributes
    ----------
    details : dict
        Contains 'evaluator_type' and optionally 'available_evaluators'.

    Examples
    --------
    Handling evaluator not found with suggestions:

    >>> try:
    ...     evaluator = EvaluatorFactory.create("bleu_score")
    ... except EvaluatorNotFoundError as e:
    ...     print(f"Unknown evaluator: {e.details['evaluator_type']}")
    ...     if 'available_evaluators' in e.details:
    ...         print("Available evaluators:")
    ...         for ev in e.details['available_evaluators']:
    ...             print(f"  - {ev}")

    Implementing evaluator selection with fallback:

    >>> def get_evaluator(eval_type: str, fallback: str = "exact_match"):
    ...     try:
    ...         return EvaluatorFactory.create(eval_type)
    ...     except EvaluatorNotFoundError:
    ...         print(f"'{eval_type}' not found, using '{fallback}'")
    ...         return EvaluatorFactory.create(fallback)

    Validating evaluator configuration:

    >>> def validate_eval_config(config: dict) -> list:
    ...     errors = []
    ...     for metric in config.get('metrics', []):
    ...         try:
    ...             EvaluatorFactory.create(metric)
    ...         except EvaluatorNotFoundError as e:
    ...             errors.append(f"Unknown metric: {metric}")
    ...     return errors

    See Also
    --------
    EvaluationFailedError : When evaluation computation fails.
    ProbeNotFoundError : Similar error for probes.
    """

    def __init__(self, evaluator_type: str, available: Optional[list] = None):
        details = {"evaluator_type": evaluator_type}
        if available:
            details["available_evaluators"] = available
        super().__init__(f"Evaluator not found: {evaluator_type}", details)


class EvaluationFailedError(EvaluationError):
    """Raised when evaluation fails.

    This exception is raised when an evaluation computation fails due
    to incompatible inputs, numerical errors, or other runtime issues
    during metric calculation.

    Parameters
    ----------
    reason : str
        A description of why evaluation failed.
    prediction : str, optional
        The model prediction that caused the failure (truncated to 100 chars).
    reference : str, optional
        The reference/ground truth (truncated to 100 chars).

    Attributes
    ----------
    details : dict
        Contains 'reason', and optionally 'prediction_preview' and
        'reference_preview'.

    Examples
    --------
    Handling evaluation failures with context:

    >>> try:
    ...     score = evaluator.evaluate(prediction, reference)
    ... except EvaluationFailedError as e:
    ...     print(f"Evaluation failed: {e.details['reason']}")
    ...     if 'prediction_preview' in e.details:
    ...         print(f"  Prediction: {e.details['prediction_preview']}")
    ...     if 'reference_preview' in e.details:
    ...         print(f"  Reference: {e.details['reference_preview']}")

    Implementing skip-on-failure for batch evaluation:

    >>> scores = []
    >>> for pred, ref in zip(predictions, references):
    ...     try:
    ...         score = evaluator.evaluate(pred, ref)
    ...         scores.append(score)
    ...     except EvaluationFailedError as e:
    ...         logging.warning(f"Skipping sample: {e.details['reason']}")
    ...         scores.append(None)

    Aggregating evaluation results with error handling:

    >>> def safe_evaluate(evaluator, predictions, references):
    ...     results = {'scores': [], 'errors': []}
    ...     for i, (pred, ref) in enumerate(zip(predictions, references)):
    ...         try:
    ...             results['scores'].append(evaluator.evaluate(pred, ref))
    ...         except EvaluationFailedError as e:
    ...             results['errors'].append({'index': i, 'reason': str(e)})
    ...             results['scores'].append(None)
    ...     return results

    See Also
    --------
    EvaluatorNotFoundError : When evaluator doesn't exist.
    ProbeExecutionError : Similar error for probe execution.
    """

    def __init__(
        self,
        reason: str,
        prediction: Optional[str] = None,
        reference: Optional[str] = None,
    ):
        details = {"reason": reason}
        if prediction:
            details["prediction_preview"] = prediction[:100]
        if reference:
            details["reference_preview"] = reference[:100]
        super().__init__(f"Evaluation failed: {reason}", details)


# Registry errors


class RegistryError(InsideLLMsError):
    """Base exception for registry errors.

    This is the parent class for all exceptions that occur during registry
    operations, including registration and lookup of models, probes, and
    evaluators. Catching this exception will handle any registry-related
    failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any registry-related error:

    >>> try:
    ...     registry.register("my_model", model_class)
    ...     model = registry.get("my_model")
    ... except RegistryError as e:
    ...     print(f"Registry operation failed: {e}")

    Distinguishing between registry error types:

    >>> try:
    ...     registry.register("custom_probe", ProbeClass)
    ... except AlreadyRegisteredError:
    ...     print("Probe already exists, skipping registration")
    ... except RegistryError as e:
    ...     print(f"Registry error: {e}")

    Notes
    -----
    Subclasses include: AlreadyRegisteredError, NotRegisteredError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    AlreadyRegisteredError : When registering duplicate entries.
    NotRegisteredError : When looking up missing entries.
    """

    pass


class AlreadyRegisteredError(RegistryError):
    """Raised when trying to register a duplicate entry.

    This exception is raised when attempting to register an item with
    a name that already exists in the registry. This prevents accidental
    overwrites of existing registrations.

    Parameters
    ----------
    name : str
        The name that was already registered.
    registry_type : str, default "item"
        The type of item being registered (e.g., "model", "probe").

    Attributes
    ----------
    details : dict
        Contains 'name' and 'registry_type'.

    Examples
    --------
    Handling duplicate registration gracefully:

    >>> try:
    ...     registry.register("gpt-4", GPT4Model)
    ... except AlreadyRegisteredError as e:
    ...     print(f"{e.details['registry_type']} '{e.details['name']}' exists")
    ...     # Use existing registration
    ...     pass

    Implementing register-or-update pattern:

    >>> def register_or_update(registry, name, item, item_type="item"):
    ...     try:
    ...         registry.register(name, item)
    ...         print(f"Registered new {item_type}: {name}")
    ...     except AlreadyRegisteredError:
    ...         registry.update(name, item)
    ...         print(f"Updated existing {item_type}: {name}")

    Force-registering with warning:

    >>> def force_register(registry, name, item):
    ...     try:
    ...         registry.register(name, item)
    ...     except AlreadyRegisteredError:
    ...         logging.warning(f"Overwriting existing registration: {name}")
    ...         registry.unregister(name)
    ...         registry.register(name, item)

    See Also
    --------
    NotRegisteredError : When looking up missing entries.
    """

    def __init__(self, name: str, registry_type: str = "item"):
        super().__init__(
            f"{registry_type.capitalize()} already registered: {name}",
            {"name": name, "registry_type": registry_type},
        )


class NotRegisteredError(RegistryError):
    """Raised when looking up an unregistered entry.

    This exception is raised when attempting to retrieve or use an item
    that has not been registered. This helps identify missing dependencies
    or configuration issues.

    Parameters
    ----------
    name : str
        The name that was not found in the registry.
    registry_type : str, default "item"
        The type of item being looked up (e.g., "model", "probe").

    Attributes
    ----------
    details : dict
        Contains 'name' and 'registry_type'.

    Examples
    --------
    Handling missing registration:

    >>> try:
    ...     model_class = registry.get("custom_model")
    ... except NotRegisteredError as e:
    ...     print(f"{e.details['registry_type']} not found: {e.details['name']}")
    ...     print("Available items:", registry.list())

    Implementing lazy registration:

    >>> def get_or_register(registry, name, factory_fn, item_type="item"):
    ...     try:
    ...         return registry.get(name)
    ...     except NotRegisteredError:
    ...         item = factory_fn()
    ...         registry.register(name, item)
    ...         return item

    Providing helpful error messages:

    >>> try:
    ...     probe = registry.get(probe_name)
    ... except NotRegisteredError as e:
    ...     available = registry.list()
    ...     print(f"Probe '{e.details['name']}' not registered.")
    ...     print(f"Available probes: {', '.join(available)}")
    ...     # Suggest similar names
    ...     from difflib import get_close_matches
    ...     suggestions = get_close_matches(e.details['name'], available)
    ...     if suggestions:
    ...         print(f"Did you mean: {suggestions[0]}?")

    See Also
    --------
    AlreadyRegisteredError : When registering duplicate entries.
    ModelNotFoundError : Specific error for models.
    ProbeNotFoundError : Specific error for probes.
    """

    def __init__(self, name: str, registry_type: str = "item"):
        super().__init__(
            f"{registry_type.capitalize()} not registered: {name}",
            {"name": name, "registry_type": registry_type},
        )


# Utility functions


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


def get_retry_delay(error: Exception, attempt: int = 1) -> float:
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
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter
