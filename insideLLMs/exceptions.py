"""Public exception hierarchy and compatibility facade for insideLLMs."""

from insideLLMs._exceptions.base import InsideLLMsError
from insideLLMs._exceptions.cache import CacheCorruptionError, CacheError, CacheMissError
from insideLLMs._exceptions.configuration import (
    ConfigNotFoundError,
    ConfigParseError,
    ConfigurationError,
    ConfigValidationError,
)
from insideLLMs._exceptions.dataset import (
    DatasetError,
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetValidationError,
)
from insideLLMs._exceptions.evaluation import (
    EvaluationError,
    EvaluationFailedError,
    EvaluatorNotFoundError,
)
from insideLLMs._exceptions.model import (
    APIError,
    ModelError,
    ModelGenerationError,
    ModelInitializationError,
    ModelNotFoundError,
    ModelTimeoutError,
    RateLimitError,
    TimeoutError,
)
from insideLLMs._exceptions.probe import (
    ProbeError,
    ProbeExecutionError,
    ProbeNotFoundError,
    ProbeValidationError,
    RunnerExecutionError,
)
from insideLLMs._exceptions.registry import (
    AlreadyRegisteredError,
    NotRegisteredError,
    RegistryError,
)
from insideLLMs._exceptions.utilities import get_retry_delay, is_retryable, wrap_exception

__all__ = [
    "InsideLLMsError",
    "ModelError",
    "ModelNotFoundError",
    "ModelInitializationError",
    "ModelGenerationError",
    "RateLimitError",
    "APIError",
    "ModelTimeoutError",
    "TimeoutError",
    "ProbeError",
    "ProbeNotFoundError",
    "ProbeValidationError",
    "ProbeExecutionError",
    "RunnerExecutionError",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetFormatError",
    "DatasetValidationError",
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigNotFoundError",
    "ConfigParseError",
    "CacheError",
    "CacheMissError",
    "CacheCorruptionError",
    "EvaluationError",
    "EvaluatorNotFoundError",
    "EvaluationFailedError",
    "RegistryError",
    "AlreadyRegisteredError",
    "NotRegisteredError",
    "wrap_exception",
    "is_retryable",
    "get_retry_delay",
]

# Preserve the established public module identity for introspection and pickling.
for _name in __all__:
    _value = globals()[_name]
    if isinstance(_value, type):
        _value.__module__ = __name__
del _name, _value
