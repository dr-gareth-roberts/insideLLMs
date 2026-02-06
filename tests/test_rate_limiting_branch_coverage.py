from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from insideLLMs.rate_limiting import (
    CircuitBreaker,
    CircuitOpenError,
    ConcurrencyLimiter,
    RateLimitedExecutor,
    RateLimitRetryResult,
    RateLimitStrategy,
    RequestQueue,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    circuit_protected,
    create_rate_limiter,
    rate_limited,
    with_retry,
)


def test_token_bucket_invalid_constructor_and_invalid_tokens():
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=0.0, capacity=1)

    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=1.0, capacity=0)

    limiter = TokenBucketRateLimiter(rate=1.0, capacity=1)
    with pytest.raises(ValueError):
        limiter.acquire(tokens=0, block=False)


@pytest.mark.asyncio
async def test_token_bucket_async_invalid_tokens():
    limiter = TokenBucketRateLimiter(rate=1.0, capacity=1)
    with pytest.raises(ValueError):
        await limiter.acquire_async(tokens=0, block=False)


def test_token_bucket_blocking_path_can_return_false_after_wait():
    limiter = TokenBucketRateLimiter(rate=1.0, capacity=1)
    limiter._tokens = 0.0

    with (
        patch.object(limiter, "_refill", return_value=None),
        patch("insideLLMs.rate_limiting.time.sleep", return_value=None) as sleeper,
    ):
        result = limiter.acquire(tokens=1, block=True)

    assert result is False
    assert sleeper.called


@pytest.mark.asyncio
async def test_token_bucket_async_blocking_path_can_return_false_after_wait():
    limiter = TokenBucketRateLimiter(rate=1.0, capacity=1)
    limiter._tokens = 0.0

    with (
        patch.object(limiter, "_refill", return_value=None),
        patch("insideLLMs.rate_limiting.asyncio.sleep", new=AsyncMock()) as sleeper,
    ):
        result = await limiter.acquire_async(tokens=1, block=True)

    assert result is False
    assert sleeper.await_count == 1


def test_token_bucket_get_state_reports_wait_time_when_empty():
    limiter = TokenBucketRateLimiter(rate=2.0, capacity=2)
    limiter._tokens = 0.0
    state = limiter.get_state()
    assert state.wait_time_ms > 0


def test_sliding_window_invalid_configuration_branches():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(requests_per_second=0.0, window_size_seconds=1.0)

    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(requests_per_second=1.0, window_size_seconds=0.0)

    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(requests_per_second=0.5, window_size_seconds=0.5)


def test_sliding_window_blocking_path_can_return_false_after_wait():
    limiter = SlidingWindowRateLimiter(requests_per_second=1.0, window_size_seconds=1.0)
    limiter._requests.append(100.0)

    with (
        patch("insideLLMs.rate_limiting.time.monotonic", return_value=100.0),
        patch("insideLLMs.rate_limiting.time.sleep", return_value=None),
    ):
        result = limiter.acquire(block=True)

    assert result is False


@pytest.mark.asyncio
async def test_sliding_window_async_blocking_path_can_return_false_after_wait():
    limiter = SlidingWindowRateLimiter(requests_per_second=1.0, window_size_seconds=1.0)
    limiter._requests.append(200.0)

    with (
        patch("insideLLMs.rate_limiting.time.monotonic", return_value=200.0),
        patch("insideLLMs.rate_limiting.asyncio.sleep", new=AsyncMock()),
    ):
        result = await limiter.acquire_async(block=True)

    assert result is False


def test_sliding_window_reset_clears_requests():
    limiter = SlidingWindowRateLimiter(requests_per_second=10.0)
    limiter._requests.extend([1.0, 2.0])
    limiter.reset()
    assert len(limiter._requests) == 0


def test_circuit_breaker_open_state_without_last_failure_stays_blocked():
    breaker = CircuitBreaker()
    breaker._state = breaker._state.OPEN
    breaker._last_failure_time = None
    assert breaker.can_execute() is False


def test_circuit_breaker_record_failure_opens_when_threshold_reached():
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    assert breaker.get_state().state == breaker.get_state().state.CLOSED
    breaker.record_failure()
    assert breaker.get_state().state == breaker.get_state().state.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_execute_async_open_raises():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        await breaker.execute_async(lambda: "never")


def test_request_queue_empty_and_exception_paths():
    queue = RequestQueue()
    assert queue.process_one() is None

    queue.enqueue(lambda: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError):
        queue.process_one()


@pytest.mark.asyncio
async def test_request_queue_async_empty_sync_and_exception_paths():
    queue = RequestQueue()
    assert await queue.process_one_async() is None

    queue.enqueue(lambda: "sync-result")
    assert await queue.process_one_async() == "sync-result"

    queue.enqueue(lambda: (_ for _ in ()).throw(RuntimeError("async-boom")))
    with pytest.raises(RuntimeError):
        await queue.process_one_async()


def test_request_queue_process_all_skips_none_results():
    queue = RequestQueue()
    queue.enqueue(lambda: None)
    queue.enqueue(lambda: "ok")
    assert queue.process_all() == ["ok"]


def test_concurrency_limiter_non_blocking_failure_and_async_release_without_semaphore():
    limiter = ConcurrencyLimiter(max_concurrent=1)
    assert limiter.acquire(block=True) is True
    assert limiter.acquire(block=False) is False

    # release_async should be a no-op even if async semaphore was never created.
    import asyncio

    asyncio.run(limiter.release_async())
    limiter.release()


@pytest.mark.asyncio
async def test_concurrency_limiter_async_semaphore_init_and_release():
    limiter = ConcurrencyLimiter(max_concurrent=1)
    assert await limiter.acquire_async() is True
    await limiter.release_async()


def test_rate_limited_executor_retry_and_circuit_failure_paths():
    class _RetryFailure:
        def execute(self, _func):
            return RateLimitRetryResult(
                success=False,
                result=None,
                attempts=1,
                total_time_ms=1.0,
                errors=["err"],
                final_error="final",
            )

    breaker = CircuitBreaker(failure_threshold=10)
    executor = RateLimitedExecutor(retry_handler=_RetryFailure(), circuit_breaker=breaker)

    with pytest.raises(Exception, match="final"):
        executor.execute(lambda: "unused")

    # no-retry branch with exception should record failure and re-raise.
    executor_no_retry = RateLimitedExecutor(circuit_breaker=breaker)
    with pytest.raises(ValueError):
        executor_no_retry.execute(lambda: (_ for _ in ()).throw(ValueError("boom")))


@pytest.mark.asyncio
async def test_rate_limited_executor_async_failure_paths():
    class _AsyncRetryFailure:
        async def execute_async(self, _func):
            return RateLimitRetryResult(
                success=False,
                result=None,
                attempts=1,
                total_time_ms=1.0,
                errors=["err"],
                final_error="final-async",
            )

    breaker = CircuitBreaker(failure_threshold=10)
    executor = RateLimitedExecutor(retry_handler=_AsyncRetryFailure(), circuit_breaker=breaker)

    with pytest.raises(Exception, match="final-async"):
        await executor.execute_async(lambda: "unused")

    # sync callable path inside execute_async.
    executor_no_retry = RateLimitedExecutor(circuit_breaker=breaker)
    assert await executor_no_retry.execute_async(lambda: "sync-result") == "sync-result"

    with pytest.raises(ValueError):
        await executor_no_retry.execute_async(
            lambda: (_ for _ in ()).throw(ValueError("async-branch-fail"))
        )


def test_rate_limited_executor_open_circuit_short_circuits():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()
    executor = RateLimitedExecutor(circuit_breaker=breaker)

    with pytest.raises(CircuitOpenError):
        executor.execute(lambda: "never")


@pytest.mark.asyncio
async def test_rate_limited_executor_async_open_circuit_short_circuits():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()
    executor = RateLimitedExecutor(circuit_breaker=breaker)

    with pytest.raises(CircuitOpenError):
        await executor.execute_async(lambda: "never")


@pytest.mark.asyncio
async def test_decorator_async_branches_for_rate_limit_retry_and_circuit():
    call_counts = {"rate": 0, "retry": 0, "circuit": 0}

    @rate_limited(rate=100.0, capacity=1)
    async def async_rate_fn():
        call_counts["rate"] += 1
        return "rate-ok"

    @with_retry(max_retries=1, base_delay=0.0)
    async def async_retry_fail():
        call_counts["retry"] += 1
        raise RuntimeError("retry-fail")

    @circuit_protected(failure_threshold=2, recovery_timeout=0.01)
    async def async_circuit_fn():
        call_counts["circuit"] += 1
        return "circuit-ok"

    assert await async_rate_fn() == "rate-ok"
    # The decorator currently returns the inner coroutine object from breaker.execute_async.
    assert await (await async_circuit_fn()) == "circuit-ok"
    with pytest.raises(Exception, match="retry-fail"):
        await (await async_retry_fail())

    assert call_counts["rate"] == 1
    assert call_counts["retry"] >= 1
    assert call_counts["circuit"] == 1


def test_with_retry_sync_failure_branch_and_create_rate_limiter_fallback():
    @with_retry(max_retries=1, base_delay=0.0)
    def always_fail():
        raise RuntimeError("sync-retry-fail")

    with pytest.raises(Exception, match="sync-retry-fail"):
        always_fail()

    limiter = create_rate_limiter(rate=5.0, strategy=RateLimitStrategy.FIXED_WINDOW)
    assert isinstance(limiter, TokenBucketRateLimiter)
