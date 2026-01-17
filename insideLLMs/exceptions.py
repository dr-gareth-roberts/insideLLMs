"""Custom exceptions for insideLLMs.

This module defines the exception hierarchy for the library,
providing clear error types for different failure modes.
"""

from typing import Any, Dict, Optional


class InsideLLMsError(Exception):
    """Base exception for all insideLLMs errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
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
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""

    def __init__(self, model_id: str, available: Optional[list] = None):
        details = {"model_id": model_id}
        if available:
            details["available_models"] = available
        super().__init__(f"Model not found: {model_id}", details)


class ModelInitializationError(ModelError):
    """Raised when a model fails to initialize."""

    def __init__(self, model_id: str, reason: str):
        super().__init__(
            f"Failed to initialize model {model_id}: {reason}",
            {"model_id": model_id, "reason": reason},
        )


class ModelGenerationError(ModelError):
    """Raised when model generation fails."""

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
    """Raised when API rate limit is exceeded."""

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
    """Raised for API-specific errors."""

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
    """Raised when a model request times out."""

    def __init__(self, model_id: str, timeout_seconds: float):
        super().__init__(
            f"Request to {model_id} timed out after {timeout_seconds}s",
            {"model_id": model_id, "timeout_seconds": timeout_seconds},
        )


# Probe-related errors

class ProbeError(InsideLLMsError):
    """Base exception for probe-related errors."""
    pass


class ProbeNotFoundError(ProbeError):
    """Raised when a requested probe is not found."""

    def __init__(self, probe_type: str, available: Optional[list] = None):
        details = {"probe_type": probe_type}
        if available:
            details["available_probes"] = available
        super().__init__(f"Probe not found: {probe_type}", details)


class ProbeValidationError(ProbeError):
    """Raised when probe input validation fails."""

    def __init__(self, probe_type: str, reason: str, invalid_input: Any = None):
        details = {"probe_type": probe_type, "reason": reason}
        if invalid_input is not None:
            details["invalid_input"] = str(invalid_input)[:100]
        super().__init__(f"Probe validation failed: {reason}", details)


class ProbeExecutionError(ProbeError):
    """Raised when probe execution fails."""

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


# Dataset-related errors

class DatasetError(InsideLLMsError):
    """Base exception for dataset-related errors."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset is not found."""

    def __init__(self, path: str):
        super().__init__(
            f"Dataset not found: {path}",
            {"path": path},
        )


class DatasetFormatError(DatasetError):
    """Raised when dataset format is invalid."""

    def __init__(self, reason: str, expected_format: Optional[str] = None):
        details = {"reason": reason}
        if expected_format:
            details["expected_format"] = expected_format
        super().__init__(f"Invalid dataset format: {reason}", details)


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""

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
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, field: str, reason: str, value: Any = None):
        details = {"field": field, "reason": reason}
        if value is not None:
            details["value"] = str(value)[:100]
        super().__init__(f"Invalid configuration for '{field}': {reason}", details)


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""

    def __init__(self, path: str):
        super().__init__(
            f"Configuration file not found: {path}",
            {"path": path},
        )


class ConfigParseError(ConfigurationError):
    """Raised when configuration parsing fails."""

    def __init__(self, path: str, reason: str, line: Optional[int] = None):
        details = {"path": path, "reason": reason}
        if line is not None:
            details["line"] = line
        super().__init__(f"Failed to parse configuration: {reason}", details)


# Cache errors

class CacheError(InsideLLMsError):
    """Base exception for cache errors."""
    pass


class CacheMissError(CacheError):
    """Raised when a cache lookup misses."""

    def __init__(self, key: str):
        super().__init__(f"Cache miss for key: {key}", {"key": key})


class CacheCorruptionError(CacheError):
    """Raised when cache data is corrupted."""

    def __init__(self, reason: str, key: Optional[str] = None):
        details = {"reason": reason}
        if key:
            details["key"] = key
        super().__init__(f"Cache corruption detected: {reason}", details)


# Evaluation errors

class EvaluationError(InsideLLMsError):
    """Base exception for evaluation errors."""
    pass


class EvaluatorNotFoundError(EvaluationError):
    """Raised when an evaluator type is not found."""

    def __init__(self, evaluator_type: str, available: Optional[list] = None):
        details = {"evaluator_type": evaluator_type}
        if available:
            details["available_evaluators"] = available
        super().__init__(f"Evaluator not found: {evaluator_type}", details)


class EvaluationFailedError(EvaluationError):
    """Raised when evaluation fails."""

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
    """Base exception for registry errors."""
    pass


class AlreadyRegisteredError(RegistryError):
    """Raised when trying to register a duplicate entry."""

    def __init__(self, name: str, registry_type: str = "item"):
        super().__init__(
            f"{registry_type.capitalize()} already registered: {name}",
            {"name": name, "registry_type": registry_type},
        )


class NotRegisteredError(RegistryError):
    """Raised when looking up an unregistered entry."""

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

    Args:
        error: The original exception.
        wrapper_class: The wrapper exception class.
        message: Optional message (default: use original).
        **details: Additional details.

    Returns:
        Wrapped exception.
    """
    msg = message or str(error)
    details["original_error_type"] = type(error).__name__
    details["original_error_message"] = str(error)
    return wrapper_class(msg, details)


def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error is retryable.
    """
    retryable_types = (
        RateLimitError,
        TimeoutError,
    )
    return isinstance(error, retryable_types)


def get_retry_delay(error: Exception, attempt: int = 1) -> float:
    """Get the recommended retry delay for an error.

    Args:
        error: The exception.
        attempt: The attempt number (for exponential backoff).

    Returns:
        Recommended delay in seconds.
    """
    if isinstance(error, RateLimitError) and error.retry_after:
        return error.retry_after

    # Exponential backoff with jitter
    import random
    base_delay = 1.0
    max_delay = 60.0
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter
