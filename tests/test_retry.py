"""Tests for retry utilities."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.retry import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    RetryConfig,
    RetryExhaustedError,
    RetryResult,
    execute_with_retry,
    execute_with_retry_async,
    retry,
    retry_async,
    retry_with_fallback,
    with_timeout,
    with_timeout_async,
)
from insideLLMs.exceptions import RateLimitError, TimeoutError as ModelTimeoutError


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False,
            strategy=BackoffStrategy.EXPONENTIAL,
        )

        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(4) == 8.0

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        config = RetryConfig(
            initial_delay=2.0,
            jitter=False,
            strategy=BackoffStrategy.LINEAR,
        )

        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 6.0

    def test_constant_backoff(self):
        """Test constant backoff calculation."""
        config = RetryConfig(
            initial_delay=5.0,
            jitter=False,
            strategy=BackoffStrategy.CONSTANT,
        )

        assert config.calculate_delay(1) == 5.0
        assert config.calculate_delay(2) == 5.0
        assert config.calculate_delay(3) == 5.0

    def test_fibonacci_backoff(self):
        """Test Fibonacci backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            jitter=False,
            strategy=BackoffStrategy.FIBONACCI,
        )

        assert config.calculate_delay(1) == 1.0  # fib(1) = 1
        assert config.calculate_delay(2) == 1.0  # fib(2) = 1
        assert config.calculate_delay(3) == 2.0  # fib(3) = 2
        assert config.calculate_delay(4) == 3.0  # fib(4) = 3
        assert config.calculate_delay(5) == 5.0  # fib(5) = 5

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=10.0,
            jitter=False,
            strategy=BackoffStrategy.EXPONENTIAL,
        )

        # At attempt 5, delay would be 16, but capped at 10
        assert config.calculate_delay(5) == 10.0
        assert config.calculate_delay(10) == 10.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(
            initial_delay=10.0,
            jitter=True,
            jitter_factor=0.2,
        )

        delays = [config.calculate_delay(1) for _ in range(10)]
        # Should have some variation
        assert len(set(delays)) > 1


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_success_no_retry(self):
        """Test successful call with no retries needed."""
        call_count = 0

        @retry(max_retries=3, initial_delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0

        @retry(max_retries=3, initial_delay=0.01)
        def transient_failure():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = transient_failure()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test that RetryExhaustedError is raised when retries exhausted."""
        call_count = 0

        @retry(max_retries=2, initial_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection failed")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_fails()

        assert exc_info.value.attempts == 3  # Initial + 2 retries
        assert call_count == 3

    def test_no_retry_on_non_retryable(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0

        @retry(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        def value_error_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            value_error_func()

        assert call_count == 1  # No retries

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback_calls = []

        def on_retry_callback(error, attempt, delay):
            callback_calls.append((type(error).__name__, attempt, delay))

        @retry(max_retries=2, initial_delay=0.01, on_retry=on_retry_callback)
        def failing_func():
            raise ConnectionError("Failed")

        with pytest.raises(RetryExhaustedError):
            failing_func()

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == "ConnectionError"
        assert callback_calls[0][1] == 1

    def test_with_config_object(self):
        """Test using RetryConfig object."""
        config = RetryConfig(
            max_retries=1,
            initial_delay=0.01,
            strategy=BackoffStrategy.CONSTANT,
        )
        call_count = 0

        @retry(config=config)
        def configured_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Failed")
            return "success"

        result = configured_func()
        assert result == "success"
        assert call_count == 2


class TestRetryAsync:
    """Tests for async retry."""

    @pytest.mark.asyncio
    async def test_async_success(self):
        """Test async success with no retries."""
        call_count = 0

        @retry_async(max_retries=3, initial_delay=0.01)
        async def async_success():
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await async_success()
        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry on failure."""
        call_count = 0

        @retry_async(max_retries=3, initial_delay=0.01)
        async def async_transient():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Failed")
            return "success"

        result = await async_transient()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_exhausted(self):
        """Test async retry exhaustion."""
        @retry_async(max_retries=2, initial_delay=0.01)
        async def async_always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(RetryExhaustedError):
            await async_always_fails()


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_starts_closed(self):
        """Test circuit starts in closed state."""
        circuit = CircuitBreaker()
        assert circuit.state == CircuitState.CLOSED
        assert circuit.is_closed

    def test_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3, reset_timeout=0.1)
        circuit = CircuitBreaker(config=config)

        for i in range(3):
            with pytest.raises(ValueError):
                circuit.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert circuit.state == CircuitState.OPEN

    def test_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, reset_timeout=1.0)
        circuit = CircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(ValueError):
            circuit.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Should now reject
        with pytest.raises(CircuitBreakerOpen):
            circuit.execute(lambda: "success")

    def test_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, reset_timeout=0.1)
        circuit = CircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(ValueError):
            circuit.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert circuit.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Should be half-open now
        assert circuit.state == CircuitState.HALF_OPEN

    def test_closes_on_success_in_half_open(self):
        """Test circuit closes after success in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            reset_timeout=0.1,
        )
        circuit = CircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(ValueError):
            circuit.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for reset timeout
        time.sleep(0.15)

        # Success should close the circuit
        result = circuit.execute(lambda: "success")
        assert result == "success"
        assert circuit.state == CircuitState.CLOSED

    def test_decorator_usage(self):
        """Test using circuit breaker as decorator."""
        circuit = CircuitBreaker(
            config=CircuitBreakerConfig(failure_threshold=2)
        )
        call_count = 0

        @circuit
        def protected_func():
            nonlocal call_count
            call_count += 1
            return "protected"

        result = protected_func()
        assert result == "protected"
        assert call_count == 1

    def test_manual_reset(self):
        """Test manual reset of circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        circuit = CircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(ValueError):
            circuit.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert circuit.state == CircuitState.OPEN

        # Manual reset
        circuit.reset()
        assert circuit.state == CircuitState.CLOSED

    def test_context_manager(self):
        """Test using circuit breaker as context manager."""
        circuit = CircuitBreaker()

        with circuit:
            pass  # Success

        assert circuit.is_closed


class TestRetryWithFallback:
    """Tests for retry_with_fallback."""

    def test_uses_primary_on_success(self):
        """Test primary function is used when successful."""
        primary_called = False
        fallback_called = False

        def primary():
            nonlocal primary_called
            primary_called = True
            return "primary"

        def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback"

        wrapped = retry_with_fallback(
            primary,
            fallback,
            RetryConfig(max_retries=1, initial_delay=0.01),
        )

        result = wrapped()
        assert result == "primary"
        assert primary_called
        assert not fallback_called

    def test_uses_fallback_on_exhaustion(self):
        """Test fallback is used when primary exhausts retries."""
        primary_calls = 0
        fallback_called = False

        def primary():
            nonlocal primary_calls
            primary_calls += 1
            raise ConnectionError("Always fails")

        def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback"

        wrapped = retry_with_fallback(
            primary,
            fallback,
            RetryConfig(max_retries=2, initial_delay=0.01),
        )

        result = wrapped()
        assert result == "fallback"
        assert primary_calls == 3  # Initial + 2 retries
        assert fallback_called


class TestWithTimeout:
    """Tests for timeout utilities."""

    def test_completes_within_timeout(self):
        """Test function that completes within timeout."""
        def fast_func():
            return "fast"

        result = with_timeout(1.0, fast_func)
        assert result == "fast"

    def test_raises_on_timeout(self):
        """Test timeout raises ModelTimeoutError."""
        def slow_func():
            time.sleep(2)
            return "slow"

        with pytest.raises(ModelTimeoutError):
            with_timeout(0.1, slow_func)

    def test_propagates_exceptions(self):
        """Test exceptions are propagated."""
        def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            with_timeout(1.0, error_func)

    @pytest.mark.asyncio
    async def test_async_timeout_success(self):
        """Test async timeout with successful completion."""
        async def async_fast():
            return "async fast"

        result = await with_timeout_async(1.0, async_fast())
        assert result == "async fast"

    @pytest.mark.asyncio
    async def test_async_timeout_exceeded(self):
        """Test async timeout raises on timeout."""
        async def async_slow():
            await asyncio.sleep(2)
            return "async slow"

        with pytest.raises(ModelTimeoutError):
            await with_timeout_async(0.1, async_slow())


class TestRateLimitHandling:
    """Tests for rate limit specific handling."""

    def test_uses_rate_limit_retry_after(self):
        """Test that RateLimitError retry_after is respected."""
        call_count = 0
        delays = []

        def on_retry(error, attempt, delay):
            delays.append(delay)

        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            on_retry=on_retry,
        )

        def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("gpt-4", retry_after=0.05)
            return "success"

        result = execute_with_retry(rate_limited_func, (), {}, config)
        assert result == "success"
        # Should use retry_after from RateLimitError
        assert all(d == 0.05 for d in delays)


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError."""

    def test_contains_history(self):
        """Test error contains retry history."""
        @retry(max_retries=2, initial_delay=0.01)
        def always_fails():
            raise ConnectionError("Failed")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_fails()

        error = exc_info.value
        assert error.attempts == 3
        assert len(error.history) == 2  # 2 retries
        assert error.last_exception is not None
