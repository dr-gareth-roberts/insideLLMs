"""Tests for custom exceptions."""

import pytest

from insideLLMs.exceptions import (
    APIError,
    AlreadyRegisteredError,
    CacheCorruptionError,
    CacheError,
    CacheMissError,
    ConfigNotFoundError,
    ConfigParseError,
    ConfigurationError,
    ConfigValidationError,
    DatasetError,
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetValidationError,
    EvaluationError,
    EvaluationFailedError,
    EvaluatorNotFoundError,
    InsideLLMsError,
    ModelError,
    ModelGenerationError,
    ModelInitializationError,
    ModelNotFoundError,
    NotRegisteredError,
    ProbeError,
    ProbeExecutionError,
    ProbeNotFoundError,
    ProbeValidationError,
    RateLimitError,
    RegistryError,
    TimeoutError,
    get_retry_delay,
    is_retryable,
    wrap_exception,
)


class TestInsideLLMsError:
    """Tests for base InsideLLMsError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = InsideLLMsError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details."""
        error = InsideLLMsError("Test error", {"key": "value"})
        assert "'key': 'value'" in str(error)
        assert error.details["key"] == "value"


class TestModelErrors:
    """Tests for model-related errors."""

    def test_model_not_found(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("gpt-4", available=["gpt-3.5-turbo"])
        assert "gpt-4" in str(error)
        assert error.details["available_models"] == ["gpt-3.5-turbo"]

    def test_model_initialization_error(self):
        """Test ModelInitializationError."""
        error = ModelInitializationError("gpt-4", "API key missing")
        assert "gpt-4" in str(error)
        assert "API key missing" in str(error)

    def test_model_generation_error(self):
        """Test ModelGenerationError."""
        original = ValueError("Original error")
        error = ModelGenerationError(
            "gpt-4",
            "Hello world",
            "Request failed",
            original,
        )
        assert "Request failed" in str(error)
        assert error.original_error is original

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("gpt-4", retry_after=30.0)
        assert error.retry_after == 30.0
        assert "gpt-4" in str(error)

    def test_api_error(self):
        """Test APIError."""
        error = APIError("gpt-4", status_code=500, message="Server error")
        assert error.status_code == 500
        assert "Server error" in str(error)

    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("gpt-4", 30.0)
        assert "30" in str(error)


class TestProbeErrors:
    """Tests for probe-related errors."""

    def test_probe_not_found(self):
        """Test ProbeNotFoundError."""
        error = ProbeNotFoundError("custom_probe", available=["logic", "bias"])
        assert "custom_probe" in str(error)
        assert error.details["available_probes"] == ["logic", "bias"]

    def test_probe_validation_error(self):
        """Test ProbeValidationError."""
        error = ProbeValidationError("logic", "Invalid input format", invalid_input="bad data")
        assert "Invalid input format" in str(error)

    def test_probe_execution_error(self):
        """Test ProbeExecutionError."""
        original = ValueError("Test error")
        error = ProbeExecutionError(
            "logic",
            "Execution failed",
            sample_index=5,
            original_error=original,
        )
        assert "Execution failed" in str(error)
        assert error.details["sample_index"] == 5


class TestDatasetErrors:
    """Tests for dataset-related errors."""

    def test_dataset_not_found(self):
        """Test DatasetNotFoundError."""
        error = DatasetNotFoundError("/path/to/dataset.json")
        assert "/path/to/dataset.json" in str(error)

    def test_dataset_format_error(self):
        """Test DatasetFormatError."""
        error = DatasetFormatError("Invalid JSON", expected_format="JSON Lines")
        assert "Invalid JSON" in str(error)

    def test_dataset_validation_error(self):
        """Test DatasetValidationError."""
        error = DatasetValidationError(
            "Missing required field",
            row_index=10,
            field="prompt",
        )
        assert "Missing required field" in str(error)


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_config_validation_error(self):
        """Test ConfigValidationError."""
        error = ConfigValidationError("temperature", "Must be between 0 and 2", value=3.0)
        assert "temperature" in str(error)
        assert "Must be between 0 and 2" in str(error)

    def test_config_not_found(self):
        """Test ConfigNotFoundError."""
        error = ConfigNotFoundError("/path/to/config.yaml")
        assert "/path/to/config.yaml" in str(error)

    def test_config_parse_error(self):
        """Test ConfigParseError."""
        error = ConfigParseError("/config.yaml", "Invalid YAML", line=10)
        assert "Invalid YAML" in str(error)


class TestCacheErrors:
    """Tests for cache errors."""

    def test_cache_miss_error(self):
        """Test CacheMissError."""
        error = CacheMissError("abc123")
        assert "abc123" in str(error)

    def test_cache_corruption_error(self):
        """Test CacheCorruptionError."""
        error = CacheCorruptionError("Invalid JSON", key="xyz789")
        assert "Invalid JSON" in str(error)


class TestEvaluationErrors:
    """Tests for evaluation errors."""

    def test_evaluator_not_found(self):
        """Test EvaluatorNotFoundError."""
        error = EvaluatorNotFoundError("custom_eval", available=["exact_match", "fuzzy"])
        assert "custom_eval" in str(error)

    def test_evaluation_failed(self):
        """Test EvaluationFailedError."""
        error = EvaluationFailedError(
            "Metric computation failed",
            prediction="Hello",
            reference="World",
        )
        assert "Metric computation failed" in str(error)


class TestRegistryErrors:
    """Tests for registry errors."""

    def test_already_registered(self):
        """Test AlreadyRegisteredError."""
        error = AlreadyRegisteredError("my_model", registry_type="model")
        assert "my_model" in str(error)
        assert "Model" in str(error)

    def test_not_registered(self):
        """Test NotRegisteredError."""
        error = NotRegisteredError("unknown_probe", registry_type="probe")
        assert "unknown_probe" in str(error)


class TestWrapException:
    """Tests for wrap_exception function."""

    def test_wraps_exception(self):
        """Test wrapping an exception."""
        original = ValueError("Original error")
        wrapped = wrap_exception(original, ModelError, "Wrapped message")

        assert isinstance(wrapped, ModelError)
        assert "Wrapped message" in str(wrapped)
        assert wrapped.details["original_error_type"] == "ValueError"

    def test_preserves_details(self):
        """Test that details are preserved."""
        original = ValueError("Test")
        wrapped = wrap_exception(original, ModelError, key="value")

        assert wrapped.details["key"] == "value"


class TestIsRetryable:
    """Tests for is_retryable function."""

    def test_rate_limit_is_retryable(self):
        """Test that RateLimitError is retryable."""
        error = RateLimitError("gpt-4")
        assert is_retryable(error)

    def test_timeout_is_retryable(self):
        """Test that TimeoutError is retryable."""
        error = TimeoutError("gpt-4", 30.0)
        assert is_retryable(error)

    def test_generic_error_not_retryable(self):
        """Test that generic errors are not retryable."""
        error = ValueError("Test")
        assert not is_retryable(error)

    def test_model_generation_not_retryable(self):
        """Test that ModelGenerationError is not retryable by default."""
        error = ModelGenerationError("gpt-4", "prompt", "failed")
        assert not is_retryable(error)


class TestGetRetryDelay:
    """Tests for get_retry_delay function."""

    def test_rate_limit_with_retry_after(self):
        """Test delay from RateLimitError retry_after."""
        error = RateLimitError("gpt-4", retry_after=30.0)
        delay = get_retry_delay(error)
        assert delay == 30.0

    def test_exponential_backoff(self):
        """Test exponential backoff for attempts."""
        error = ValueError("Test")

        delay1 = get_retry_delay(error, attempt=1)
        delay2 = get_retry_delay(error, attempt=2)
        delay3 = get_retry_delay(error, attempt=3)

        # Each should be approximately double (with jitter)
        assert delay2 > delay1
        assert delay3 > delay2

    def test_max_delay(self):
        """Test maximum delay cap."""
        error = ValueError("Test")
        delay = get_retry_delay(error, attempt=10)

        # Should not exceed max of 60 + jitter
        assert delay <= 66  # 60 + 10% jitter


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_model_error_is_base(self):
        """Test ModelError inherits from InsideLLMsError."""
        error = ModelError("Test")
        assert isinstance(error, InsideLLMsError)

    def test_probe_error_is_base(self):
        """Test ProbeError inherits from InsideLLMsError."""
        error = ProbeError("Test")
        assert isinstance(error, InsideLLMsError)

    def test_config_error_is_base(self):
        """Test ConfigurationError inherits from InsideLLMsError."""
        error = ConfigurationError("Test")
        assert isinstance(error, InsideLLMsError)

    def test_cache_error_is_base(self):
        """Test CacheError inherits from InsideLLMsError."""
        error = CacheError("Test")
        assert isinstance(error, InsideLLMsError)

    def test_evaluation_error_is_base(self):
        """Test EvaluationError inherits from InsideLLMsError."""
        error = EvaluationError("Test")
        assert isinstance(error, InsideLLMsError)

    def test_model_not_found_is_model_error(self):
        """Test ModelNotFoundError inherits from ModelError."""
        error = ModelNotFoundError("gpt-4")
        assert isinstance(error, ModelError)
