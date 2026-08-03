"""Base exception type for insideLLMs."""

from typing import Any, Optional


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
