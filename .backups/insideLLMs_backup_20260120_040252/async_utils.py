"""Async utilities for concurrent LLM execution.

This module provides utilities for efficient concurrent execution of
LLM operations, including:
- Batch processing with concurrency limits
- Rate limiting for API calls
- Progress tracking for async operations
- Async context managers and helpers
"""

import asyncio
import functools
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult(Generic[T]):
    """Result from batch processing.

    Attributes:
        results: List of successful results.
        errors: List of (index, exception) tuples for failures.
        total: Total number of items processed.
        succeeded: Number of successful items.
        failed: Number of failed items.
        elapsed_time: Total processing time in seconds.
    """

    results: list[Optional[T]]
    errors: list[tuple[int, Exception]]
    total: int
    succeeded: int
    failed: int
    elapsed_time: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.succeeded / self.total

    def get_result(self, index: int) -> Optional[T]:
        """Get result at index, or None if failed."""
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def get_error(self, index: int) -> Optional[Exception]:
        """Get error at index, or None if succeeded."""
        for idx, error in self.errors:
            if idx == index:
                return error
        return None


async def map_async(
    func: Callable[[T], Awaitable[R]],
    items: Sequence[T],
    max_concurrency: int = 10,
    return_exceptions: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> BatchResult[R]:
    """Apply an async function to items with controlled concurrency.

    Args:
        func: Async function to apply.
        items: Items to process.
        max_concurrency: Maximum concurrent operations.
        return_exceptions: If True, include exceptions in results.
        on_progress: Optional callback(completed, total) for progress.

    Returns:
        BatchResult with results and errors.

    Example:
        async def process(item):
            return await api.generate(item)

        result = await map_async(process, prompts, max_concurrency=5)
        print(f"Success rate: {result.success_rate:.1%}")
    """
    start_time = time.perf_counter()
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[Optional[R]] = [None] * len(items)
    errors: list[tuple[int, Exception]] = []
    completed = 0

    async def process_item(index: int, item: T) -> None:
        nonlocal completed
        async with semaphore:
            try:
                results[index] = await func(item)
            except Exception as e:
                errors.append((index, e))
                if return_exceptions:
                    results[index] = e  # type: ignore

            completed += 1
            if on_progress:
                on_progress(completed, len(items))

    tasks = [asyncio.create_task(process_item(i, item)) for i, item in enumerate(items)]

    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    succeeded = len(items) - len(errors)

    return BatchResult(
        results=results,
        errors=errors,
        total=len(items),
        succeeded=succeeded,
        failed=len(errors),
        elapsed_time=elapsed,
    )


async def map_async_ordered(
    func: Callable[[T], Awaitable[R]],
    items: Sequence[T],
    max_concurrency: int = 10,
) -> AsyncGenerator[tuple[int, Optional[R], Optional[Exception]], None]:
    """Apply async function and yield results as they complete, with order.

    Args:
        func: Async function to apply.
        items: Items to process.
        max_concurrency: Maximum concurrent operations.

    Yields:
        Tuples of (index, result, error) as each completes.

    Example:
        async for idx, result, error in map_async_ordered(process, items):
            if error:
                print(f"Item {idx} failed: {error}")
            else:
                print(f"Item {idx}: {result}")
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    pending: dict[int, asyncio.Task] = {}

    async def process_item(index: int, item: T) -> tuple[int, Optional[R], Optional[Exception]]:
        async with semaphore:
            try:
                result = await func(item)
                return (index, result, None)
            except Exception as e:
                return (index, None, e)

    # Start all tasks
    for i, item in enumerate(items):
        task = asyncio.create_task(process_item(i, item))
        pending[i] = task

    # Yield results as they complete
    next_to_yield = 0
    completed: dict[int, tuple[int, Optional[R], Optional[Exception]]] = {}

    while pending or completed:
        if next_to_yield in completed:
            yield completed.pop(next_to_yield)
            next_to_yield += 1
        elif pending:
            done, _ = await asyncio.wait(
                pending.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                result = task.result()
                idx = result[0]
                del pending[idx]
                if idx == next_to_yield:
                    yield result
                    next_to_yield += 1
                else:
                    completed[idx] = result


async def for_each_async(
    func: Callable[[T], Awaitable[None]],
    items: Iterable[T],
    max_concurrency: int = 10,
    stop_on_error: bool = False,
) -> list[Exception]:
    """Execute async function for each item, discarding results.

    Args:
        func: Async function to execute.
        items: Items to process.
        max_concurrency: Maximum concurrent operations.
        stop_on_error: Stop processing on first error.

    Returns:
        List of exceptions that occurred.

    Example:
        errors = await for_each_async(save_result, results, max_concurrency=20)
        if errors:
            print(f"{len(errors)} saves failed")
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    errors: list[Exception] = []
    stop_flag = asyncio.Event()

    async def process_item(item: T) -> None:
        if stop_flag.is_set():
            return

        async with semaphore:
            try:
                await func(item)
            except Exception as e:
                errors.append(e)
                if stop_on_error:
                    stop_flag.set()

    tasks = [asyncio.create_task(process_item(item)) for item in items]
    await asyncio.gather(*tasks, return_exceptions=True)

    return errors


# Rate Limiting


@dataclass
class RateLimiter:
    """Token bucket rate limiter for async operations.

    Attributes:
        rate: Maximum operations per second.
        burst: Maximum burst size (bucket capacity).
    """

    rate: float
    burst: int = 1
    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)
        self._last_update = time.monotonic()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(
                    self.burst,
                    self._tokens + elapsed * self.rate,
                )
                self._last_update = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self.rate
                await asyncio.sleep(wait_time)

    def __call__(self, func: Callable) -> Callable:
        """Use rate limiter as a decorator."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await self.acquire()
            return await func(*args, **kwargs)

        return wrapper


@dataclass
class SlidingWindowRateLimiter:
    """Sliding window rate limiter.

    Provides more accurate rate limiting by tracking actual request times.

    Attributes:
        max_requests: Maximum requests in the window.
        window_seconds: Window size in seconds.
    """

    max_requests: int
    window_seconds: float
    _requests: deque = field(default_factory=deque, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            while True:
                now = time.monotonic()

                # Remove expired entries
                while self._requests and now - self._requests[0] >= self.window_seconds:
                    self._requests.popleft()

                if len(self._requests) < self.max_requests:
                    self._requests.append(now)
                    return

                # Wait until oldest request expires
                wait_time = self.window_seconds - (now - self._requests[0])
                await asyncio.sleep(wait_time)


def rate_limited(
    rate: float,
    burst: int = 1,
) -> Callable[[Callable], Callable]:
    """Decorator to rate limit an async function.

    Args:
        rate: Maximum calls per second.
        burst: Maximum burst size.

    Returns:
        Decorated function.

    Example:
        @rate_limited(rate=2, burst=5)
        async def call_api():
            return await api.generate("Hello")
    """
    limiter = RateLimiter(rate=rate, burst=burst)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Async Queue Processing


class AsyncWorkerPool(Generic[T, R]):
    """Pool of async workers for processing a queue.

    Example:
        async def process(item):
            return await api.generate(item)

        pool = AsyncWorkerPool(process, num_workers=5)
        async with pool:
            for prompt in prompts:
                await pool.submit(prompt)

        results = pool.results
    """

    def __init__(
        self,
        worker_func: Callable[[T], Awaitable[R]],
        num_workers: int = 5,
        max_queue_size: int = 0,
    ):
        self.worker_func = worker_func
        self.num_workers = num_workers
        self._queue: asyncio.Queue[Optional[T]] = asyncio.Queue(max_queue_size)
        self._results: list[tuple[int, Optional[R], Optional[Exception]]] = []
        self._workers: list[asyncio.Task] = []
        self._counter = 0
        self._lock = asyncio.Lock()

    @property
    def results(self) -> list[tuple[int, Optional[R], Optional[Exception]]]:
        """Get all results."""
        return sorted(self._results, key=lambda x: x[0])

    async def _worker(self) -> None:
        """Worker coroutine."""
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            async with self._lock:
                index = self._counter
                self._counter += 1

            try:
                result = await self.worker_func(item)
                self._results.append((index, result, None))
            except Exception as e:
                self._results.append((index, None, e))
            finally:
                self._queue.task_done()

    async def submit(self, item: T) -> None:
        """Submit an item for processing."""
        await self._queue.put(item)

    async def __aenter__(self) -> "AsyncWorkerPool[T, R]":
        """Start workers."""
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.num_workers)]
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop workers and wait for completion."""
        # Wait for queue to be processed
        await self._queue.join()

        # Send stop signals
        for _ in self._workers:
            await self._queue.put(None)

        # Wait for workers to finish
        await asyncio.gather(*self._workers)


# Progress Tracking


@dataclass
class AsyncProgress:
    """Progress tracker for async operations.

    Example:
        progress = AsyncProgress(total=100)

        async def process(item):
            result = await api.generate(item)
            await progress.update()
            return result

        await map_async(process, items)
        print(f"Rate: {progress.items_per_second:.1f}/s")
    """

    total: int
    completed: int = 0
    start_time: Optional[float] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed

    @property
    def estimated_remaining(self) -> float:
        """Estimate remaining time in seconds."""
        rate = self.items_per_second
        if rate == 0:
            return float("inf")
        remaining = self.total - self.completed
        return remaining / rate

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 100.0
        return 100.0 * self.completed / self.total

    async def update(self, n: int = 1) -> None:
        """Update progress."""
        async with self._lock:
            if self.start_time is None:
                self.start_time = time.perf_counter()
            self.completed += n

    def __str__(self) -> str:
        return (
            f"{self.completed}/{self.total} ({self.percent_complete:.1f}%) "
            f"| {self.items_per_second:.1f}/s | ETA: {self.estimated_remaining:.0f}s"
        )


# Utility Functions


async def gather_with_limit(
    coros: Sequence[Awaitable[T]],
    limit: int,
) -> list[T]:
    """Gather coroutines with a concurrency limit.

    Args:
        coros: Coroutines to execute.
        limit: Maximum concurrent coroutines.

    Returns:
        List of results in original order.

    Example:
        coros = [api.generate(p) for p in prompts]
        results = await gather_with_limit(coros, limit=5)
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited(c) for c in coros])


async def first_completed(
    coros: Sequence[Awaitable[T]],
    cancel_remaining: bool = True,
) -> T:
    """Return result of first completed coroutine.

    Args:
        coros: Coroutines to race.
        cancel_remaining: Whether to cancel other coroutines.

    Returns:
        Result of first completed coroutine.

    Raises:
        ValueError: If no coroutines provided.

    Example:
        result = await first_completed([
            api1.generate("Hello"),
            api2.generate("Hello"),
        ])
    """
    if not coros:
        raise ValueError("No coroutines provided")

    tasks = [asyncio.ensure_future(c) for c in coros]

    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_remaining:
            for task in pending:
                task.cancel()

        # Return result of first completed
        for task in done:
            return task.result()

        raise RuntimeError("No task completed")  # Should never happen

    except Exception:
        # Cancel all tasks on error
        for task in tasks:
            task.cancel()
        raise


async def retry_until_success(
    coros: Sequence[Callable[[], Awaitable[T]]],
) -> T:
    """Try coroutines in sequence until one succeeds.

    Args:
        coros: Coroutine factories (callables that return coroutines).

    Returns:
        Result of first successful coroutine.

    Raises:
        Exception: Last exception if all fail.

    Example:
        async def try_api1():
            return await api1.generate("Hello")

        async def try_api2():
            return await api2.generate("Hello")

        result = await retry_until_success([try_api1, try_api2])
    """
    last_exception: Optional[Exception] = None

    for coro_factory in coros:
        try:
            return await coro_factory()
        except Exception as e:
            last_exception = e
            continue

    if last_exception:
        raise last_exception
    raise ValueError("No coroutines provided")


@asynccontextmanager
async def async_timeout(seconds: float) -> AsyncGenerator[None, None]:
    """Async context manager with timeout.

    Args:
        seconds: Timeout in seconds.

    Raises:
        asyncio.TimeoutError: If timeout exceeded.

    Example:
        async with async_timeout(5.0):
            await slow_operation()
    """
    task = asyncio.current_task()
    if task is None:
        yield
        return

    loop = asyncio.get_event_loop()
    handle = loop.call_later(seconds, task.cancel)

    try:
        yield
    finally:
        handle.cancel()


def run_async(coro: Awaitable[T]) -> T:
    """Run an async function from sync code.

    Args:
        coro: Coroutine to run.

    Returns:
        Result of the coroutine.

    Example:
        result = run_async(api.generate_async("Hello"))
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, we can't use run_until_complete
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop, create a new one
        return asyncio.run(coro)
