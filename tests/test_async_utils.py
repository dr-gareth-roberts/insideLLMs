"""Tests for async utilities."""

import asyncio
import time

import pytest

from insideLLMs.async_utils import (
    AsyncProgress,
    AsyncWorkerPool,
    BatchResult,
    RateLimiter,
    SlidingWindowRateLimiter,
    first_completed,
    for_each_async,
    gather_with_limit,
    map_async,
    map_async_ordered,
    rate_limited,
    retry_until_success,
)


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_success_rate(self):
        """Test success rate calculation."""
        result = BatchResult(
            results=[1, 2, None, 4, None],
            errors=[(2, ValueError("e")), (4, ValueError("e"))],
            total=5,
            succeeded=3,
            failed=2,
            elapsed_time=1.0,
        )
        assert result.success_rate == 0.6

    def test_success_rate_empty(self):
        """Test success rate with empty batch."""
        result = BatchResult(
            results=[],
            errors=[],
            total=0,
            succeeded=0,
            failed=0,
            elapsed_time=0.0,
        )
        assert result.success_rate == 0.0

    def test_get_result(self):
        """Test getting results by index."""
        result = BatchResult(
            results=["a", "b", "c"],
            errors=[],
            total=3,
            succeeded=3,
            failed=0,
            elapsed_time=1.0,
        )
        assert result.get_result(1) == "b"
        assert result.get_result(10) is None

    def test_get_error(self):
        """Test getting errors by index."""
        error = ValueError("test")
        result = BatchResult(
            results=[None, "b"],
            errors=[(0, error)],
            total=2,
            succeeded=1,
            failed=1,
            elapsed_time=1.0,
        )
        assert result.get_error(0) is error
        assert result.get_error(1) is None


class TestMapAsync:
    """Tests for map_async."""

    @pytest.mark.asyncio
    async def test_basic_mapping(self):
        """Test basic async mapping."""
        async def double(x: int) -> int:
            return x * 2

        items = [1, 2, 3, 4, 5]
        result = await map_async(double, items)

        assert result.results == [2, 4, 6, 8, 10]
        assert result.succeeded == 5
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_with_errors(self):
        """Test mapping with some failures."""
        async def maybe_fail(x: int) -> int:
            if x == 3:
                raise ValueError("Three is bad")
            return x

        items = [1, 2, 3, 4, 5]
        result = await map_async(maybe_fail, items)

        assert result.succeeded == 4
        assert result.failed == 1
        assert len(result.errors) == 1
        assert result.errors[0][0] == 2  # Index of failure

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency is limited."""
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency(x: int) -> int:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return x

        items = list(range(10))
        await map_async(track_concurrency, items, max_concurrency=3)

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback is called."""
        progress_calls = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        async def identity(x: int) -> int:
            return x

        items = [1, 2, 3]
        await map_async(identity, items, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


class TestMapAsyncOrdered:
    """Tests for map_async_ordered."""

    @pytest.mark.asyncio
    async def test_ordered_results(self):
        """Test results are yielded in order."""
        async def process(x: int) -> int:
            # Simulate varying processing times
            await asyncio.sleep(0.01 * (5 - x))
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = []

        async for idx, result, error in map_async_ordered(process, items):
            results.append((idx, result))

        # Should be in order despite varying completion times
        assert [r[0] for r in results] == [0, 1, 2, 3, 4]
        assert [r[1] for r in results] == [2, 4, 6, 8, 10]


class TestForEachAsync:
    """Tests for for_each_async."""

    @pytest.mark.asyncio
    async def test_executes_for_all(self):
        """Test function is executed for all items."""
        executed = []

        async def track(x: int) -> None:
            executed.append(x)

        items = [1, 2, 3, 4, 5]
        errors = await for_each_async(track, items)

        assert len(errors) == 0
        assert set(executed) == set(items)

    @pytest.mark.asyncio
    async def test_collects_errors(self):
        """Test errors are collected."""
        async def sometimes_fail(x: int) -> None:
            if x % 2 == 0:
                raise ValueError(f"Even: {x}")

        items = [1, 2, 3, 4, 5]
        errors = await for_each_async(sometimes_fail, items)

        assert len(errors) == 2

    @pytest.mark.asyncio
    async def test_stop_on_error(self):
        """Test stopping on first error."""
        executed = []

        async def track_and_fail(x: int) -> None:
            executed.append(x)
            await asyncio.sleep(0.01 * x)
            if x == 2:
                raise ValueError("Stop here")

        items = [1, 2, 3, 4, 5]
        errors = await for_each_async(track_and_fail, items, stop_on_error=True)

        assert len(errors) >= 1


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting works."""
        limiter = RateLimiter(rate=10, burst=1)

        start = time.perf_counter()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.perf_counter() - start

        # Should take at least 0.4s (4 waits at 0.1s each)
        assert elapsed >= 0.35

    @pytest.mark.asyncio
    async def test_burst(self):
        """Test burst allows immediate requests."""
        limiter = RateLimiter(rate=1, burst=5)

        start = time.perf_counter()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.perf_counter() - start

        # Burst of 5 should be nearly instant
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test using rate limiter as decorator."""
        limiter = RateLimiter(rate=100, burst=10)

        @limiter
        async def limited_func():
            return "done"

        result = await limited_func()
        assert result == "done"


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    @pytest.mark.asyncio
    async def test_sliding_window(self):
        """Test sliding window rate limiting."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=0.5)

        # First 5 should be instant
        start = time.perf_counter()
        for _ in range(5):
            await limiter.acquire()
        first_batch = time.perf_counter() - start

        assert first_batch < 0.1

        # 6th should wait
        start = time.perf_counter()
        await limiter.acquire()
        wait_time = time.perf_counter() - start

        assert wait_time >= 0.3  # Should wait for window to slide


class TestRateLimitedDecorator:
    """Tests for @rate_limited decorator."""

    @pytest.mark.asyncio
    async def test_rate_limited_decorator(self):
        """Test rate_limited decorator."""
        @rate_limited(rate=50, burst=5)
        async def limited_call():
            return "result"

        # Should complete without error
        results = await asyncio.gather(*[limited_call() for _ in range(5)])
        assert all(r == "result" for r in results)


class TestAsyncWorkerPool:
    """Tests for AsyncWorkerPool."""

    @pytest.mark.asyncio
    async def test_worker_pool(self):
        """Test basic worker pool usage."""
        async def double(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        async with AsyncWorkerPool(double, num_workers=3) as pool:
            for i in range(5):
                await pool.submit(i)

        results = pool.results
        assert len(results) == 5

        # Check all values are correct (order maintained)
        for idx, result, error in results:
            assert error is None
            assert result == idx * 2

    @pytest.mark.asyncio
    async def test_worker_pool_with_errors(self):
        """Test worker pool handles errors."""
        async def maybe_fail(x: int) -> int:
            if x == 2:
                raise ValueError("Two is bad")
            return x

        async with AsyncWorkerPool(maybe_fail, num_workers=2) as pool:
            for i in range(5):
                await pool.submit(i)

        results = pool.results
        errors = [r for r in results if r[2] is not None]
        assert len(errors) == 1


class TestAsyncProgress:
    """Tests for AsyncProgress."""

    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        """Test progress tracking."""
        progress = AsyncProgress(total=10)

        await progress.update(3)
        assert progress.completed == 3
        assert progress.percent_complete == 30.0

        await progress.update(2)
        assert progress.completed == 5
        assert progress.percent_complete == 50.0

    @pytest.mark.asyncio
    async def test_items_per_second(self):
        """Test rate calculation."""
        progress = AsyncProgress(total=100)

        await progress.update(10)
        await asyncio.sleep(0.1)
        await progress.update(10)

        # Should have some measurable rate
        assert progress.items_per_second > 0


class TestGatherWithLimit:
    """Tests for gather_with_limit."""

    @pytest.mark.asyncio
    async def test_limited_gather(self):
        """Test gathering with concurrency limit."""
        concurrent_count = 0
        max_concurrent = 0

        async def track(x: int) -> int:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1
            return x

        coros = [track(i) for i in range(10)]
        results = await gather_with_limit(coros, limit=3)

        assert results == list(range(10))
        assert max_concurrent <= 3


class TestFirstCompleted:
    """Tests for first_completed."""

    @pytest.mark.asyncio
    async def test_returns_first(self):
        """Test returning first completed result."""
        async def fast():
            await asyncio.sleep(0.01)
            return "fast"

        async def slow():
            await asyncio.sleep(1.0)
            return "slow"

        result = await first_completed([fast(), slow()])
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_empty_raises(self):
        """Test error on empty list."""
        with pytest.raises(ValueError):
            await first_completed([])


class TestRetryUntilSuccess:
    """Tests for retry_until_success."""

    @pytest.mark.asyncio
    async def test_first_succeeds(self):
        """Test returning first success."""
        async def success():
            return "success"

        async def unused():
            raise ValueError("Should not be called")

        result = await retry_until_success([success, unused])
        assert result == "success"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test fallback to second on first failure."""
        async def fails():
            raise ValueError("First fails")

        async def succeeds():
            return "second"

        result = await retry_until_success([fails, succeeds])
        assert result == "second"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        """Test all failures raises last exception."""
        async def fail1():
            raise ValueError("First")

        async def fail2():
            raise TypeError("Second")

        with pytest.raises(TypeError):
            await retry_until_success([fail1, fail2])
