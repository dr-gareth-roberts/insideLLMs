"""Tests for rate limiting and retry module."""

import pytest
import time
import asyncio
import threading

from insideLLMs.rate_limiting import (
    # Enums
    RateLimitStrategy,
    RetryStrategy,
    CircuitState,
    RequestPriority,
    # Dataclasses
    RateLimitConfig,
    RetryConfig,
    RateLimitState,
    RetryResult,
    CircuitBreakerState,
    RateLimitStats,
    # Classes
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    RetryHandler,
    CircuitBreaker,
    CircuitOpenError,
    RequestQueue,
    ConcurrencyLimiter,
    RateLimitedExecutor,
    # Decorators
    rate_limited,
    with_retry,
    circuit_protected,
    # Functions
    create_rate_limiter,
    create_retry_handler,
    create_circuit_breaker,
    create_executor,
    execute_with_backoff,
)


class TestEnums:
    """Tests for enum types."""

    def test_rate_limit_strategy_values(self):
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"

    def test_retry_strategy_values(self):
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.CONSTANT.value == "constant"
        assert RetryStrategy.FIBONACCI.value == "fibonacci"

    def test_circuit_state_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_request_priority_ordering(self):
        assert RequestPriority.CRITICAL.value < RequestPriority.HIGH.value
        assert RequestPriority.HIGH.value < RequestPriority.NORMAL.value
        assert RequestPriority.NORMAL.value < RequestPriority.LOW.value


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        config = RateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.requests_per_minute == 600.0
        assert config.burst_size == 20

    def test_custom_config(self):
        config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
        )
        assert config.requests_per_second == 5.0
        assert config.burst_size == 10

    def test_config_to_dict(self):
        config = RateLimitConfig()
        d = config.to_dict()
        assert "requests_per_second" in d
        assert "strategy" in d


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.strategy == RetryStrategy.EXPONENTIAL

    def test_custom_config(self):
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            strategy=RetryStrategy.LINEAR,
        )
        assert config.max_retries == 5
        assert config.strategy == RetryStrategy.LINEAR


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_limiter_creation(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        state = limiter.get_state()
        assert state.available_tokens == 20

    def test_acquire_single(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)
        assert limiter.acquire(tokens=1, block=False)

    def test_acquire_multiple(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)
        for _ in range(5):
            assert limiter.acquire(tokens=1, block=False)

    def test_acquire_exceeds_capacity(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=5)
        # Drain bucket
        for _ in range(5):
            limiter.acquire(tokens=1, block=False)

        # Next should fail without blocking
        assert not limiter.acquire(tokens=1, block=False)

    def test_token_refill(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)

        # Drain bucket
        for _ in range(10):
            limiter.acquire(tokens=1, block=False)

        # Wait for refill
        time.sleep(0.1)

        # Should have tokens now
        state = limiter.get_state()
        assert state.available_tokens > 0

    def test_stats_tracking(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)

        limiter.acquire(tokens=1, block=False)
        limiter.acquire(tokens=1, block=False)

        stats = limiter.get_stats()
        assert stats.total_requests == 2
        assert stats.allowed_requests == 2

    def test_reset(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)

        for _ in range(10):
            limiter.acquire(tokens=1, block=False)

        limiter.reset()

        state = limiter.get_state()
        assert state.available_tokens == 10

    @pytest.mark.asyncio
    async def test_acquire_async(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)
        result = await limiter.acquire_async(tokens=1, block=False)
        assert result


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    def test_limiter_creation(self):
        limiter = SlidingWindowRateLimiter(requests_per_second=10.0)
        state = limiter.get_state()
        assert state.requests_in_window == 0

    def test_acquire_within_limit(self):
        limiter = SlidingWindowRateLimiter(requests_per_second=10.0, window_size_seconds=1.0)

        for _ in range(5):
            assert limiter.acquire(block=False)

    def test_acquire_exceeds_limit(self):
        limiter = SlidingWindowRateLimiter(requests_per_second=5.0, window_size_seconds=1.0)

        # Fill window
        for _ in range(5):
            limiter.acquire(block=False)

        # Next should fail
        assert not limiter.acquire(block=False)

    def test_window_expiration(self):
        limiter = SlidingWindowRateLimiter(requests_per_second=10.0, window_size_seconds=0.1)

        # Fill window
        for _ in range(10):
            limiter.acquire(block=False)

        # Wait for window to expire
        time.sleep(0.15)

        # Should be able to acquire again
        assert limiter.acquire(block=False)

    def test_stats_tracking(self):
        limiter = SlidingWindowRateLimiter(requests_per_second=10.0)

        limiter.acquire(block=False)
        limiter.acquire(block=False)

        stats = limiter.get_stats()
        assert stats.total_requests == 2

    @pytest.mark.asyncio
    async def test_acquire_async(self):
        limiter = SlidingWindowRateLimiter(requests_per_second=10.0)
        result = await limiter.acquire_async(block=False)
        assert result


class TestRetryHandler:
    """Tests for RetryHandler."""

    def test_handler_creation(self):
        handler = RetryHandler()
        assert handler.config.max_retries == 3

    def test_execute_success(self):
        handler = RetryHandler()

        def success_func():
            return "success"

        result = handler.execute(success_func)

        assert result.success
        assert result.result == "success"
        assert result.attempts == 1

    def test_execute_with_retry(self):
        handler = RetryHandler(RetryConfig(max_retries=3, base_delay=0.01))

        attempt_count = 0

        def fail_twice():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Failing")
            return "success"

        result = handler.execute(fail_twice)

        assert result.success
        assert result.attempts == 3

    def test_execute_all_fail(self):
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        def always_fail():
            raise ValueError("Always fails")

        result = handler.execute(always_fail)

        assert not result.success
        assert result.attempts == 3
        assert len(result.errors) == 3

    def test_exponential_delay(self):
        config = RetryConfig(
            max_retries=3,
            base_delay=0.1,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )
        handler = RetryHandler(config)

        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)
        delay3 = handler._calculate_delay(2)

        assert delay2 > delay1
        assert delay3 > delay2

    def test_linear_delay(self):
        config = RetryConfig(
            max_retries=3,
            base_delay=0.1,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
        )
        handler = RetryHandler(config)

        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)
        delay3 = handler._calculate_delay(2)

        assert delay2 - delay1 == pytest.approx(delay3 - delay2, rel=0.1)

    def test_constant_delay(self):
        config = RetryConfig(
            max_retries=3,
            base_delay=0.1,
            strategy=RetryStrategy.CONSTANT,
            jitter=False,
        )
        handler = RetryHandler(config)

        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)

        assert delay1 == delay2

    @pytest.mark.asyncio
    async def test_execute_async(self):
        handler = RetryHandler()

        async def async_success():
            return "async_success"

        result = await handler.execute_async(async_success)

        assert result.success
        assert result.result == "async_success"


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_breaker_creation(self):
        breaker = CircuitBreaker()
        state = breaker.get_state()
        assert state.state == CircuitState.CLOSED

    def test_breaker_allows_execution(self):
        breaker = CircuitBreaker()
        assert breaker.can_execute()

    def test_breaker_opens_after_failures(self):
        breaker = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        state = breaker.get_state()
        assert state.state == CircuitState.OPEN
        assert not breaker.can_execute()

    def test_breaker_recovers_after_timeout(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open the circuit
        for _ in range(2):
            breaker.record_failure()

        assert not breaker.can_execute()

        # Wait for recovery
        time.sleep(0.15)

        # Should be half-open now
        assert breaker.can_execute()
        state = breaker.get_state()
        assert state.state == CircuitState.HALF_OPEN

    def test_breaker_closes_after_success(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_max_calls=2)

        # Open the circuit
        for _ in range(2):
            breaker.record_failure()

        # Wait for recovery
        time.sleep(0.15)
        breaker.can_execute()  # Transition to half-open

        # Record successes
        breaker.record_success()
        breaker.record_success()

        state = breaker.get_state()
        assert state.state == CircuitState.CLOSED

    def test_execute_with_breaker(self):
        breaker = CircuitBreaker()

        def success_func():
            return "success"

        result = breaker.execute(success_func)
        assert result == "success"

    def test_execute_raises_when_open(self):
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()

        with pytest.raises(CircuitOpenError):
            breaker.execute(lambda: "test")

    def test_reset(self):
        breaker = CircuitBreaker(failure_threshold=2)

        for _ in range(2):
            breaker.record_failure()

        breaker.reset()

        state = breaker.get_state()
        assert state.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_execute_async(self):
        breaker = CircuitBreaker()

        async def async_func():
            return "async_result"

        result = await breaker.execute_async(async_func)
        assert result == "async_result"


class TestRequestQueue:
    """Tests for RequestQueue."""

    def test_queue_creation(self):
        queue = RequestQueue()
        assert queue.get_queue_size() == 0

    def test_enqueue(self):
        queue = RequestQueue()
        result = queue.enqueue(lambda: "test")
        assert result
        assert queue.get_queue_size() == 1

    def test_enqueue_with_priority(self):
        queue = RequestQueue()
        queue.enqueue(lambda: "low", RequestPriority.LOW)
        queue.enqueue(lambda: "high", RequestPriority.HIGH)
        queue.enqueue(lambda: "critical", RequestPriority.CRITICAL)

        assert queue.get_queue_size() == 3

    def test_process_one(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=100)
        queue = RequestQueue(rate_limiter=limiter)

        queue.enqueue(lambda: "result")
        result = queue.process_one()

        assert result == "result"
        assert queue.get_queue_size() == 0

    def test_process_respects_priority(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=100)
        queue = RequestQueue(rate_limiter=limiter)

        results = []
        queue.enqueue(lambda: results.append("low") or "low", RequestPriority.LOW)
        queue.enqueue(lambda: results.append("critical") or "critical", RequestPriority.CRITICAL)

        queue.process_one()

        # Critical should be processed first
        assert results[0] == "critical"

    def test_max_queue_size(self):
        queue = RequestQueue(max_queue_size=2)

        assert queue.enqueue(lambda: "1")
        assert queue.enqueue(lambda: "2")
        assert not queue.enqueue(lambda: "3")  # Should be rejected

    def test_stats(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=100)
        queue = RequestQueue(rate_limiter=limiter)

        queue.enqueue(lambda: "test")
        queue.process_one()

        stats = queue.get_stats()
        assert stats["processed_count"] == 1

    def test_clear(self):
        queue = RequestQueue()
        queue.enqueue(lambda: "1")
        queue.enqueue(lambda: "2")

        queue.clear()

        assert queue.get_queue_size() == 0


class TestConcurrencyLimiter:
    """Tests for ConcurrencyLimiter."""

    def test_limiter_creation(self):
        limiter = ConcurrencyLimiter(max_concurrent=5)
        assert limiter.get_available() == 5

    def test_acquire_release(self):
        limiter = ConcurrencyLimiter(max_concurrent=5)

        assert limiter.acquire()
        assert limiter.get_current_count() == 1
        assert limiter.get_available() == 4

        limiter.release()
        assert limiter.get_current_count() == 0
        assert limiter.get_available() == 5

    def test_context_manager(self):
        limiter = ConcurrencyLimiter(max_concurrent=5)

        with limiter:
            assert limiter.get_current_count() == 1

        assert limiter.get_current_count() == 0

    def test_concurrent_access(self):
        limiter = ConcurrencyLimiter(max_concurrent=2)
        results = []

        def worker(id):
            with limiter:
                results.append(f"start_{id}")
                time.sleep(0.05)
                results.append(f"end_{id}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete
        assert len(results) == 6

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        limiter = ConcurrencyLimiter(max_concurrent=5)

        async with limiter:
            assert limiter.get_current_count() == 1

        assert limiter.get_current_count() == 0


class TestRateLimitedExecutor:
    """Tests for RateLimitedExecutor."""

    def test_executor_creation(self):
        executor = RateLimitedExecutor()
        # Should not raise

    def test_execute_with_rate_limiter(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=100)
        executor = RateLimitedExecutor(rate_limiter=limiter)

        result = executor.execute(lambda: "result")
        assert result == "result"

    def test_execute_with_retry(self):
        handler = create_retry_handler(max_retries=3, base_delay=0.01)
        executor = RateLimitedExecutor(retry_handler=handler)

        attempt_count = 0

        def fail_once():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Fail")
            return "success"

        result = executor.execute(fail_once)
        assert result == "success"

    def test_execute_with_circuit_breaker(self):
        breaker = create_circuit_breaker(failure_threshold=3)
        executor = RateLimitedExecutor(circuit_breaker=breaker)

        result = executor.execute(lambda: "success")
        assert result == "success"

    def test_execute_with_all_components(self):
        executor = create_executor(
            rate=100.0,
            max_retries=2,
            failure_threshold=5,
            max_concurrent=10,
        )

        result = executor.execute(lambda: "full_result")
        assert result == "full_result"

    @pytest.mark.asyncio
    async def test_execute_async(self):
        executor = create_executor(rate=100.0)

        async def async_func():
            return "async_result"

        result = await executor.execute_async(async_func)
        assert result == "async_result"


class TestDecorators:
    """Tests for decorators."""

    def test_rate_limited_decorator(self):
        call_count = 0

        @rate_limited(rate=100.0, capacity=100)
        def limited_func():
            nonlocal call_count
            call_count += 1
            return call_count

        result = limited_func()
        assert result == 1

    def test_with_retry_decorator(self):
        attempt_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Fail")
            return "success"

        result = retry_func()
        assert result == "success"
        assert attempt_count == 2

    def test_circuit_protected_decorator(self):
        @circuit_protected(failure_threshold=5)
        def protected_func():
            return "protected"

        result = protected_func()
        assert result == "protected"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_rate_limiter_token_bucket(self):
        limiter = create_rate_limiter(
            rate=10.0,
            capacity=20,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
        )
        assert isinstance(limiter, TokenBucketRateLimiter)

    def test_create_rate_limiter_sliding_window(self):
        limiter = create_rate_limiter(
            rate=10.0,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        assert isinstance(limiter, SlidingWindowRateLimiter)

    def test_create_retry_handler(self):
        handler = create_retry_handler(
            max_retries=5,
            base_delay=2.0,
        )
        assert handler.config.max_retries == 5

    def test_create_circuit_breaker(self):
        breaker = create_circuit_breaker(
            failure_threshold=10,
            recovery_timeout=60.0,
        )
        assert breaker.failure_threshold == 10

    def test_create_executor(self):
        executor = create_executor(
            rate=20.0,
            max_retries=5,
        )
        assert executor.rate_limiter is not None
        assert executor.retry_handler is not None

    def test_execute_with_backoff(self):
        def success_func():
            return "backoff_success"

        result = execute_with_backoff(success_func, max_retries=3)
        assert result == "backoff_success"

    def test_execute_with_backoff_failure(self):
        def fail_func():
            raise ValueError("Always fail")

        with pytest.raises(Exception):
            execute_with_backoff(fail_func, max_retries=2, base_delay=0.01)


class TestDataclasses:
    """Tests for dataclass to_dict methods."""

    def test_rate_limit_state_to_dict(self):
        state = RateLimitState(
            available_tokens=10.0,
            requests_in_window=5,
            tokens_in_window=100,
            last_request_time=None,
            is_limited=False,
            wait_time_ms=0.0,
        )
        d = state.to_dict()
        assert d["available_tokens"] == 10.0

    def test_retry_result_to_dict(self):
        result = RetryResult(
            success=True,
            result="test",
            attempts=1,
            total_time_ms=100.0,
            errors=[],
            final_error=None,
        )
        d = result.to_dict()
        assert d["success"] is True

    def test_circuit_breaker_state_to_dict(self):
        state = CircuitBreakerState(
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=None,
            half_open_successes=0,
        )
        d = state.to_dict()
        assert d["state"] == "closed"

    def test_rate_limit_stats_to_dict(self):
        stats = RateLimitStats(
            total_requests=100,
            allowed_requests=90,
            throttled_requests=10,
        )
        d = stats.to_dict()
        assert d["throttle_rate"] == 0.1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_rate_limiter(self):
        # Rate of 0.1 means very slow
        limiter = TokenBucketRateLimiter(rate=0.1, capacity=1)
        assert limiter.acquire(block=False)
        assert not limiter.acquire(block=False)

    def test_retry_with_no_retries(self):
        handler = RetryHandler(RetryConfig(max_retries=0))

        def fail_func():
            raise ValueError("Fail")

        result = handler.execute(fail_func)
        assert not result.success
        assert result.attempts == 1

    def test_circuit_breaker_immediate_open(self):
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()
        assert not breaker.can_execute()

    def test_concurrent_rate_limiting(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=50)
        results = []

        def worker():
            if limiter.acquire(block=False):
                results.append(True)
            else:
                results.append(False)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least some should succeed
        assert any(results)


class TestIntegration:
    """Integration tests."""

    def test_full_execution_flow(self):
        executor = create_executor(
            rate=100.0,
            max_retries=3,
            failure_threshold=5,
            max_concurrent=10,
        )

        results = []
        for i in range(5):
            result = executor.execute(lambda i=i: f"result_{i}")
            results.append(result)

        assert len(results) == 5

    def test_rate_limit_with_retry(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=100)
        handler = create_retry_handler(max_retries=2, base_delay=0.01)

        attempt_count = 0

        def flaky_func():
            nonlocal attempt_count
            limiter.acquire()
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Flaky")
            return "success"

        result = handler.execute(flaky_func)
        assert result.success

    def test_circuit_breaker_with_recovery(self):
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1,
        )

        # Trip the breaker
        for _ in range(2):
            try:
                breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Fail")))
            except:
                pass

        # Verify it's open
        assert not breaker.can_execute()

        # Wait for recovery
        time.sleep(0.15)

        # Should be half-open
        assert breaker.can_execute()

        # Successful call should close it
        breaker.execute(lambda: "success")

        state = breaker.get_state()
        assert state.state == CircuitState.CLOSED
