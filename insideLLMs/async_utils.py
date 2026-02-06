"""Async utilities for concurrent LLM execution.

This module provides utilities for efficient concurrent execution of
LLM operations, including batch processing with concurrency limits,
rate limiting for API calls, progress tracking for async operations,
and async context managers and helpers.

Overview
--------
When working with LLM APIs, you often need to process many prompts concurrently
while respecting rate limits and handling failures gracefully. This module
provides a comprehensive toolkit for these scenarios:

**Batch Processing:**
    - `map_async`: Apply an async function to items with controlled concurrency
    - `map_async_ordered`: Process items concurrently but yield results in order
    - `for_each_async`: Execute side-effect operations concurrently
    - `gather_with_limit`: Like asyncio.gather but with concurrency limits

**Rate Limiting:**
    - `AsyncTokenBucketRateLimiter`: Token bucket algorithm for smooth rate limiting
    - `AsyncSlidingWindowRateLimiter`: Sliding window for accurate request tracking
    - `rate_limited`: Decorator to apply rate limiting to any async function

**Worker Pools:**
    - `AsyncWorkerPool`: Pool of async workers processing a shared queue

**Progress Tracking:**
    - `AsyncProgress`: Track progress, rate, and ETA for long-running operations

**Utility Functions:**
    - `first_completed`: Race coroutines and return first result
    - `retry_until_success`: Try fallback coroutines in sequence
    - `async_timeout`: Context manager with timeout
    - `run_async`: Run async code from synchronous context

Examples
--------
Basic batch processing with progress tracking:

>>> import asyncio
>>> from insideLLMs.async_utils import map_async, AsyncProgress
>>>
>>> async def process_prompt(prompt: str) -> str:
...     # Simulate API call
...     await asyncio.sleep(0.1)
...     return f"Response to: {prompt}"
>>>
>>> async def main():
...     prompts = [f"Question {i}" for i in range(100)]
...     progress = AsyncProgress(total=len(prompts))
...
...     async def process_with_progress(prompt):
...         result = await process_prompt(prompt)
...         await progress.update()
...         print(f"\\r{progress}", end="")
...         return result
...
...     result = await map_async(
...         process_with_progress,
...         prompts,
...         max_concurrency=10,
...     )
...     print(f"\\nSuccess rate: {result.success_rate:.1%}")
...     print(f"Total time: {result.elapsed_time:.2f}s")
...     return result
>>>
>>> # asyncio.run(main())

Rate limiting API calls:

>>> from insideLLMs.async_utils import rate_limited, AsyncTokenBucketRateLimiter
>>>
>>> # Using decorator
>>> @rate_limited(rate=2.0, burst=5)
... async def call_api(prompt: str) -> str:
...     # API call here
...     return "response"
>>>
>>> # Using rate limiter directly
>>> limiter = AsyncTokenBucketRateLimiter(rate=10.0, burst=20)
>>>
>>> async def rate_limited_batch(prompts):
...     results = []
...     for prompt in prompts:
...         await limiter.acquire()
...         result = await call_api(prompt)
...         results.append(result)
...     return results

Using worker pools for queue-based processing:

>>> from insideLLMs.async_utils import AsyncWorkerPool
>>>
>>> async def process_item(item: dict) -> dict:
...     # Process the item
...     await asyncio.sleep(0.1)
...     return {"processed": item}
>>>
>>> async def main():
...     items = [{"id": i} for i in range(50)]
...
...     async with AsyncWorkerPool(process_item, num_workers=5) as pool:
...         for item in items:
...             await pool.submit(item)
...
...     for idx, result, error in pool.results:
...         if error:
...             print(f"Item {idx} failed: {error}")
...         else:
...             print(f"Item {idx}: {result}")

Racing multiple providers for fastest response:

>>> from insideLLMs.async_utils import first_completed, retry_until_success
>>>
>>> async def main():
...     # Race multiple providers
...     result = await first_completed([
...         provider1.generate("Hello"),
...         provider2.generate("Hello"),
...         provider3.generate("Hello"),
...     ])
...
...     # Or try providers in sequence until one succeeds
...     result = await retry_until_success([
...         lambda: provider1.generate("Hello"),
...         lambda: provider2.generate("Hello"),
...         lambda: provider3.generate("Hello"),
...     ])

Notes
-----
- All batch processing functions preserve the original order of results
- Rate limiters are thread-safe and can be shared across coroutines
- Progress tracking is designed to work with concurrent operations
- The `run_async` function uses `nest_asyncio` when called from async context

Thread Safety
-------------
All classes in this module use asyncio locks for thread safety within
the async context. However, they are not designed to be shared across
multiple event loops.

See Also
--------
- asyncio : Python's built-in async library
- aiohttp : Async HTTP client for making API requests
- insideLLMs.retry : Retry utilities with exponential backoff
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
class AsyncBatchResult(Generic[T]):
    """Result container from batch processing operations.

    This dataclass encapsulates the results of batch async operations,
    providing access to both successful results and any errors that occurred.
    It includes statistics about the operation and helper methods for
    accessing individual results.

    Parameters
    ----------
    results : list[Optional[T]]
        List of results in the same order as input items. Failed items
        will have None at their index (unless return_exceptions=True was
        used, in which case the exception object is stored).
    errors : list[tuple[int, Exception]]
        List of (index, exception) tuples for items that failed during
        processing. The index corresponds to the position in the original
        input sequence.
    total : int
        Total number of items that were processed.
    succeeded : int
        Number of items that completed successfully without exceptions.
    failed : int
        Number of items that raised exceptions during processing.
    elapsed_time : float
        Total wall-clock time in seconds for the entire batch operation.

    Attributes
    ----------
    success_rate : float
        Ratio of successful items to total items (0.0 to 1.0).

    Examples
    --------
    Basic usage after batch processing:

    >>> import asyncio
    >>> from insideLLMs.async_utils import map_async, AsyncBatchResult
    >>>
    >>> async def process(x: int) -> int:
    ...     if x == 5:
    ...         raise ValueError("Cannot process 5")
    ...     return x * 2
    >>>
    >>> async def main():
    ...     items = list(range(10))
    ...     result: AsyncBatchResult[int] = await map_async(process, items)
    ...
    ...     # Check overall statistics
    ...     print(f"Processed {result.total} items in {result.elapsed_time:.2f}s")
    ...     print(f"Success rate: {result.success_rate:.1%}")
    ...     print(f"Succeeded: {result.succeeded}, Failed: {result.failed}")
    ...
    ...     # Access individual results
    ...     for i in range(result.total):
    ...         if result.get_error(i):
    ...             print(f"Item {i} failed: {result.get_error(i)}")
    ...         else:
    ...             print(f"Item {i} = {result.get_result(i)}")

    Filtering successful results:

    >>> async def main():
    ...     result = await map_async(process, items)
    ...     successful = [r for r in result.results if r is not None]
    ...     print(f"Got {len(successful)} successful results")

    Handling errors:

    >>> async def main():
    ...     result = await map_async(process, items)
    ...     for idx, error in result.errors:
    ...         print(f"Item at index {idx} failed with: {type(error).__name__}: {error}")
    ...     # Retry failed items
    ...     failed_items = [items[idx] for idx, _ in result.errors]
    ...     retry_result = await map_async(process, failed_items)

    See Also
    --------
    map_async : The primary function that produces AsyncBatchResult objects.
    for_each_async : Similar but discards results (for side effects only).
    """

    results: list[Optional[T]]
    errors: list[tuple[int, Exception]]
    total: int
    succeeded: int
    failed: int
    elapsed_time: float

    @property
    def success_rate(self) -> float:
        """Calculate the ratio of successful items to total items.

        Returns
        -------
        float
            Success rate as a value between 0.0 and 1.0. Returns 0.0 if
            no items were processed (total == 0).

        Examples
        --------
        >>> result = AsyncBatchResult(
        ...     results=[1, 2, None, 4],
        ...     errors=[(2, ValueError("error"))],
        ...     total=4, succeeded=3, failed=1, elapsed_time=1.5
        ... )
        >>> result.success_rate
        0.75
        >>> f"{result.success_rate:.0%}"
        '75%'
        """
        if self.total == 0:
            return 0.0
        return self.succeeded / self.total

    def get_result(self, index: int) -> Optional[T]:
        """Get the result at a specific index.

        Safely retrieves the result for the item at the given index.
        Returns None if the index is out of bounds or if the item
        failed during processing.

        Args
        ----
        index : int
            The index of the item in the original input sequence.

        Returns
        -------
        Optional[T]
            The result value if the item succeeded and the index is valid,
            None otherwise.

        Examples
        --------
        >>> result = AsyncBatchResult(
        ...     results=["a", "b", None, "d"],
        ...     errors=[(2, ValueError())],
        ...     total=4, succeeded=3, failed=1, elapsed_time=1.0
        ... )
        >>> result.get_result(0)
        'a'
        >>> result.get_result(2)  # Failed item
        None
        >>> result.get_result(100)  # Out of bounds
        None
        """
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def get_error(self, index: int) -> Optional[Exception]:
        """Get the exception that occurred at a specific index.

        Retrieves the exception for an item that failed during processing.
        Returns None if the item succeeded or the index is not in the
        error list.

        Args
        ----
        index : int
            The index of the item in the original input sequence.

        Returns
        -------
        Optional[Exception]
            The exception if the item failed, None if it succeeded.

        Examples
        --------
        >>> error = ValueError("something went wrong")
        >>> result = AsyncBatchResult(
        ...     results=["a", None],
        ...     errors=[(1, error)],
        ...     total=2, succeeded=1, failed=1, elapsed_time=0.5
        ... )
        >>> result.get_error(0)  # Succeeded
        None
        >>> result.get_error(1)  # Failed
        ValueError('something went wrong')
        >>> isinstance(result.get_error(1), ValueError)
        True
        """
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
) -> AsyncBatchResult[R]:
    """Apply an async function to items with controlled concurrency.

    This is the primary batch processing function for concurrent async operations.
    It processes all items in parallel (up to max_concurrency limit) while
    maintaining result order. Errors are captured rather than raising, allowing
    the batch to complete even if some items fail.

    Args
    ----
    func : Callable[[T], Awaitable[R]]
        An async function that takes a single item of type T and returns
        a result of type R. The function is called once per item in the
        input sequence.
    items : Sequence[T]
        A sequence of items to process. Can be any sequence type (list,
        tuple, etc.) that supports len() and indexing.
    max_concurrency : int, optional
        Maximum number of concurrent operations allowed at any time.
        Higher values increase throughput but may overwhelm APIs or
        consume excessive resources. Default is 10.
    return_exceptions : bool, optional
        If True, exceptions are stored in the results list at their
        corresponding index instead of None. This allows distinguishing
        between a None result and a failed operation. Default is False.
    on_progress : Callable[[int, int], None], optional
        A callback function that receives (completed_count, total_count)
        after each item completes. Useful for progress bars or logging.
        Default is None.

    Returns
    -------
    AsyncBatchResult[R]
        A result container with:
        - results: List of results in original order (None for failures)
        - errors: List of (index, exception) tuples
        - Statistics: total, succeeded, failed, elapsed_time, success_rate

    Raises
    ------
    This function does not raise exceptions from individual item processing.
    All exceptions are captured in the returned AsyncBatchResult.errors list.

    Examples
    --------
    Basic batch processing of LLM prompts:

    >>> import asyncio
    >>> from insideLLMs.async_utils import map_async
    >>>
    >>> async def generate(prompt: str) -> str:
    ...     # Simulate API call
    ...     await asyncio.sleep(0.1)
    ...     return f"Response to: {prompt}"
    >>>
    >>> async def main():
    ...     prompts = ["Hello", "How are you?", "What is Python?"]
    ...     result = await map_async(generate, prompts, max_concurrency=2)
    ...     print(f"Success rate: {result.success_rate:.0%}")
    ...     for i, response in enumerate(result.results):
    ...         print(f"{prompts[i]} -> {response}")
    >>>
    >>> # asyncio.run(main())

    Processing with error handling:

    >>> async def risky_process(item: int) -> int:
    ...     if item % 3 == 0:
    ...         raise ValueError(f"Cannot process {item}")
    ...     return item * 2
    >>>
    >>> async def main():
    ...     items = list(range(10))
    ...     result = await map_async(risky_process, items, max_concurrency=5)
    ...
    ...     print(f"Succeeded: {result.succeeded}/{result.total}")
    ...     for idx, error in result.errors:
    ...         print(f"  Item {items[idx]} failed: {error}")
    ...
    ...     # Get only successful results
    ...     successful = [
    ...         (i, r) for i, r in enumerate(result.results)
    ...         if r is not None
    ...     ]

    Using progress callback for real-time updates:

    >>> async def main():
    ...     def show_progress(completed: int, total: int) -> None:
    ...         pct = 100 * completed / total
    ...         print(f"\\rProgress: {completed}/{total} ({pct:.0f}%)", end="")
    ...
    ...     result = await map_async(
    ...         generate,
    ...         prompts,
    ...         max_concurrency=10,
    ...         on_progress=show_progress,
    ...     )
    ...     print()  # New line after progress

    Storing exceptions in results for detailed analysis:

    >>> async def main():
    ...     result = await map_async(
    ...         risky_process,
    ...         items,
    ...         return_exceptions=True,
    ...     )
    ...     for i, r in enumerate(result.results):
    ...         if isinstance(r, Exception):
    ...             print(f"Item {i} raised {type(r).__name__}: {r}")
    ...         else:
    ...             print(f"Item {i} = {r}")

    See Also
    --------
    map_async_ordered : Yields results as they complete, maintaining order.
    for_each_async : For side-effect-only operations (discards results).
    gather_with_limit : Lower-level alternative using asyncio.gather.
    AsyncBatchResult : The result container class.

    Notes
    -----
    - Results are always returned in the same order as input items
    - The semaphore ensures max_concurrency is never exceeded
    - Failed items have None in results (unless return_exceptions=True)
    - Elapsed time is measured using time.perf_counter() for accuracy
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
                    results[index] = e  # type: ignore[assignment]  # When return_exceptions=True, exceptions stored in results

            completed += 1
            if on_progress:
                on_progress(completed, len(items))

    tasks = [asyncio.create_task(process_item(i, item)) for i, item in enumerate(items)]

    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    succeeded = len(items) - len(errors)

    return AsyncBatchResult(
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
    """Apply async function and yield results in order as they complete.

    Unlike `map_async` which returns all results at once, this function yields
    results as an async generator, maintaining the original order. Results are
    buffered if they complete out of order, ensuring you always receive them
    in sequence (index 0, then 1, then 2, etc.).

    This is useful for streaming results to a UI or processing results
    incrementally without waiting for the entire batch to complete.

    Args
    ----
    func : Callable[[T], Awaitable[R]]
        An async function that takes a single item of type T and returns
        a result of type R. The function is called once per item.
    items : Sequence[T]
        A sequence of items to process. Supports len() and indexing.
    max_concurrency : int, optional
        Maximum number of concurrent operations. Default is 10.

    Yields
    ------
    tuple[int, Optional[R], Optional[Exception]]
        A 3-tuple for each item containing:
        - index: The position in the original input sequence
        - result: The function result if successful, None if failed
        - error: The exception if failed, None if successful

    Examples
    --------
    Basic streaming of results:

    >>> import asyncio
    >>> from insideLLMs.async_utils import map_async_ordered
    >>>
    >>> async def process(item: str) -> str:
    ...     await asyncio.sleep(0.1)
    ...     return f"Processed: {item}"
    >>>
    >>> async def main():
    ...     items = ["a", "b", "c", "d", "e"]
    ...     async for idx, result, error in map_async_ordered(process, items):
    ...         if error:
    ...             print(f"Item {idx} failed: {error}")
    ...         else:
    ...             print(f"Item {idx}: {result}")
    ...         # Results always arrive in order: 0, 1, 2, 3, 4

    Streaming with early termination:

    >>> async def main():
    ...     results = []
    ...     async for idx, result, error in map_async_ordered(process, items):
    ...         if error:
    ...             print(f"Stopping due to error at index {idx}")
    ...             break  # Stop on first error
    ...         results.append(result)
    ...         if len(results) >= 3:
    ...             print("Got enough results")
    ...             break  # Stop after getting 3 results

    Processing results incrementally for memory efficiency:

    >>> async def process_large_batch():
    ...     # Process millions of items without storing all results
    ...     large_items = range(1_000_000)
    ...     total_value = 0
    ...     async for idx, result, error in map_async_ordered(
    ...         process_item, large_items, max_concurrency=100
    ...     ):
    ...         if result is not None:
    ...             total_value += result
    ...             # Each result is processed and can be discarded
    ...     return total_value

    Building a real-time progress display:

    >>> async def main():
    ...     total = len(items)
    ...     async for idx, result, error in map_async_ordered(process, items):
    ...         pct = 100 * (idx + 1) / total
    ...         status = "OK" if error is None else f"FAIL: {error}"
    ...         print(f"[{pct:5.1f}%] Item {idx}: {status}")

    See Also
    --------
    map_async : Returns all results at once (non-streaming).
    for_each_async : For side-effect operations (discards results).

    Notes
    -----
    - Results are yielded in strict order (0, 1, 2, ...) regardless of
      completion order
    - Out-of-order completions are buffered until their turn
    - All items are started immediately (up to max_concurrency limit)
    - Breaking from the async for loop does NOT cancel remaining tasks
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

    This function is optimized for side-effect operations where you don't
    need the return values (e.g., saving to database, sending notifications,
    updating caches). Unlike `map_async`, it doesn't preserve order or
    collect results, making it more memory-efficient for fire-and-forget
    operations.

    Args
    ----
    func : Callable[[T], Awaitable[None]]
        An async function that takes a single item and performs some
        side effect. The return value is ignored.
    items : Iterable[T]
        Items to process. Can be any iterable (list, generator, etc.).
        Note: Will be fully consumed immediately to create tasks.
    max_concurrency : int, optional
        Maximum number of concurrent operations. Default is 10.
    stop_on_error : bool, optional
        If True, stop processing additional items when the first error
        occurs. Already-running tasks will complete, but no new items
        will be started. Default is False.

    Returns
    -------
    list[Exception]
        List of all exceptions that occurred during processing. Empty
        list indicates all items succeeded.

    Examples
    --------
    Saving results to a database:

    >>> import asyncio
    >>> from insideLLMs.async_utils import for_each_async
    >>>
    >>> async def save_to_db(record: dict) -> None:
    ...     # Simulate database write
    ...     await asyncio.sleep(0.01)
    ...     if record.get("invalid"):
    ...         raise ValueError(f"Invalid record: {record}")
    ...     print(f"Saved record {record['id']}")
    >>>
    >>> async def main():
    ...     records = [
    ...         {"id": 1, "data": "a"},
    ...         {"id": 2, "data": "b"},
    ...         {"id": 3, "invalid": True},
    ...         {"id": 4, "data": "d"},
    ...     ]
    ...     errors = await for_each_async(save_to_db, records, max_concurrency=5)
    ...     if errors:
    ...         print(f"{len(errors)} records failed to save")
    ...         for e in errors:
    ...             print(f"  - {e}")
    ...     else:
    ...         print("All records saved successfully")

    Sending notifications with stop-on-error:

    >>> async def send_notification(user_id: int) -> None:
    ...     # If notification service is down, stop trying
    ...     response = await notification_service.send(user_id)
    ...     if not response.ok:
    ...         raise ConnectionError("Notification service unavailable")
    >>>
    >>> async def main():
    ...     user_ids = [1, 2, 3, 4, 5]
    ...     errors = await for_each_async(
    ...         send_notification,
    ...         user_ids,
    ...         max_concurrency=3,
    ...         stop_on_error=True,  # Stop if service is down
    ...     )
    ...     if errors:
    ...         print("Notification service error, stopping")

    Processing a generator lazily (note: tasks are created eagerly):

    >>> def generate_work_items():
    ...     for i in range(1000):
    ...         yield {"id": i, "data": f"item_{i}"}
    >>>
    >>> async def main():
    ...     # Items are consumed immediately, but processed concurrently
    ...     errors = await for_each_async(
    ...         save_to_db,
    ...         generate_work_items(),
    ...         max_concurrency=50,
    ...     )

    Cache warming with high concurrency:

    >>> async def warm_cache(key: str) -> None:
    ...     data = await database.get(key)
    ...     await cache.set(key, data)
    >>>
    >>> async def main():
    ...     keys = ["user:1", "user:2", "config:app", "settings:global"]
    ...     await for_each_async(warm_cache, keys, max_concurrency=100)

    See Also
    --------
    map_async : When you need to collect and use the results.
    map_async_ordered : When you need streaming results in order.

    Notes
    -----
    - All items are consumed immediately to create tasks
    - Results are discarded to save memory
    - Error order in the returned list is non-deterministic
    - With stop_on_error=True, in-flight tasks still complete
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
class AsyncTokenBucketRateLimiter:
    """Token bucket rate limiter for async operations.

    Implements the classic token bucket algorithm for rate limiting. Tokens
    are added to the bucket at a constant rate, and each operation consumes
    tokens. If insufficient tokens are available, the operation waits until
    enough tokens accumulate.

    This algorithm allows for controlled bursting: if the bucket is full,
    multiple operations can proceed immediately up to the burst limit.
    After bursting, operations are limited to the steady-state rate.

    Parameters
    ----------
    rate : float
        Number of tokens added to the bucket per second. This is the
        steady-state rate limit (e.g., rate=10.0 means 10 operations
        per second maximum on average).
    burst : int, optional
        Maximum bucket capacity (number of tokens that can accumulate).
        This determines the maximum burst size. Default is 1 (no bursting).

    Attributes
    ----------
    rate : float
        The configured rate in tokens per second.
    burst : int
        The configured burst capacity.

    Examples
    --------
    Basic rate limiting:

    >>> import asyncio
    >>> from insideLLMs.async_utils import AsyncTokenBucketRateLimiter
    >>>
    >>> # Allow 5 requests per second, burst up to 10
    >>> limiter = AsyncTokenBucketRateLimiter(rate=5.0, burst=10)
    >>>
    >>> async def rate_limited_request(url: str) -> str:
    ...     await limiter.acquire()
    ...     return await http_client.get(url)
    >>>
    >>> async def main():
    ...     urls = [f"https://api.example.com/item/{i}" for i in range(20)]
    ...     # First 10 requests go immediately (burst)
    ...     # Remaining requests throttled to 5/second
    ...     results = await asyncio.gather(*[rate_limited_request(u) for u in urls])

    Using as a decorator:

    >>> limiter = AsyncTokenBucketRateLimiter(rate=2.0, burst=5)
    >>>
    >>> @limiter
    ... async def call_api(prompt: str) -> str:
    ...     return await api.generate(prompt)
    >>>
    >>> async def main():
    ...     # All calls are automatically rate limited
    ...     results = await asyncio.gather(*[
    ...         call_api(f"Question {i}")
    ...         for i in range(10)
    ...     ])

    Acquiring multiple tokens at once:

    >>> limiter = AsyncTokenBucketRateLimiter(rate=100.0, burst=1000)
    >>>
    >>> async def batch_operation(items: list) -> None:
    ...     # Consume tokens proportional to batch size
    ...     await limiter.acquire(tokens=len(items))
    ...     await api.batch_process(items)

    Shared rate limiter across multiple functions:

    >>> api_limiter = AsyncTokenBucketRateLimiter(rate=10.0, burst=20)
    >>>
    >>> async def read_data(id: int) -> dict:
    ...     await api_limiter.acquire()
    ...     return await api.read(id)
    >>>
    >>> async def write_data(id: int, data: dict) -> None:
    ...     await api_limiter.acquire()
    ...     await api.write(id, data)

    See Also
    --------
    AsyncSlidingWindowRateLimiter : Alternative using sliding window algorithm.
    rate_limited : Decorator for simple rate limiting.

    Notes
    -----
    - Thread-safe for use within a single event loop
    - Uses asyncio.Lock to prevent race conditions
    - Time tracked using time.monotonic() for accuracy
    - Bucket starts full (allows immediate burst at startup)
    """

    rate: float
    burst: int = 1
    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize the token bucket with full capacity."""
        self._tokens = float(self.burst)
        self._last_update = time.monotonic()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket, waiting if necessary.

        This method blocks (asynchronously) until the requested number of
        tokens are available. Tokens are added to the bucket based on elapsed
        time since the last update, up to the burst capacity.

        Args
        ----
        tokens : int, optional
            Number of tokens to acquire. Default is 1. For batch operations,
            you can request multiple tokens proportional to work size.

        Examples
        --------
        >>> limiter = AsyncTokenBucketRateLimiter(rate=10.0, burst=20)
        >>>
        >>> async def single_operation():
        ...     await limiter.acquire()  # Acquire 1 token
        ...     await do_work()
        >>>
        >>> async def batch_operation(batch_size: int):
        ...     await limiter.acquire(tokens=batch_size)  # Acquire multiple
        ...     await do_batch_work()

        Notes
        -----
        - Waits asynchronously using asyncio.sleep (non-blocking)
        - Tokens regenerate during the wait period
        - Lock ensures thread-safety within the event loop
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
        """Use rate limiter as a decorator for async functions.

        When the rate limiter is used as a decorator, it automatically
        acquires one token before each function call.

        Args
        ----
        func : Callable
            The async function to wrap with rate limiting.

        Returns
        -------
        Callable
            A wrapped async function that acquires a token before executing.

        Examples
        --------
        >>> limiter = AsyncTokenBucketRateLimiter(rate=5.0, burst=10)
        >>>
        >>> @limiter
        ... async def call_api(prompt: str) -> str:
        ...     return await api.generate(prompt)
        >>>
        >>> # Each call to call_api() now automatically rate limited
        >>> result = await call_api("Hello")
        """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await self.acquire()
            return await func(*args, **kwargs)

        return wrapper


@dataclass
class AsyncSlidingWindowRateLimiter:
    """Sliding window rate limiter for precise request tracking.

    Unlike the token bucket algorithm which allows bursting, the sliding
    window algorithm provides strict rate limiting by tracking the exact
    timestamps of recent requests. This ensures that no more than
    `max_requests` are made within any `window_seconds` period.

    This is useful when APIs have strict rate limits that measure requests
    within a rolling time window (e.g., "100 requests per minute" where
    the minute is measured from each request's timestamp).

    Parameters
    ----------
    max_requests : int
        Maximum number of requests allowed within the time window.
    window_seconds : float
        Size of the sliding window in seconds.

    Attributes
    ----------
    max_requests : int
        The configured maximum requests per window.
    window_seconds : float
        The configured window size in seconds.

    Examples
    --------
    Basic rate limiting (100 requests per minute):

    >>> import asyncio
    >>> from insideLLMs.async_utils import AsyncSlidingWindowRateLimiter
    >>>
    >>> limiter = AsyncSlidingWindowRateLimiter(max_requests=100, window_seconds=60.0)
    >>>
    >>> async def call_api(prompt: str) -> str:
    ...     await limiter.acquire()
    ...     return await api.generate(prompt)
    >>>
    >>> async def main():
    ...     prompts = [f"Question {i}" for i in range(200)]
    ...     # First 100 requests go quickly
    ...     # Then strictly limited to ~1.67 requests/second
    ...     results = await asyncio.gather(*[call_api(p) for p in prompts])

    Strict per-second limit (no bursting):

    >>> limiter = AsyncSlidingWindowRateLimiter(max_requests=5, window_seconds=1.0)
    >>>
    >>> async def main():
    ...     start = time.time()
    ...     for i in range(15):
    ...         await limiter.acquire()
    ...         print(f"Request {i} at t={time.time() - start:.2f}s")
    ...     # Output shows strictly 5 requests per second

    API with tiered rate limits:

    >>> # Different limiters for different API tiers
    >>> free_tier = AsyncSlidingWindowRateLimiter(max_requests=10, window_seconds=60)
    >>> pro_tier = AsyncSlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    >>>
    >>> async def call_api(prompt: str, tier: str) -> str:
    ...     limiter = free_tier if tier == "free" else pro_tier
    ...     await limiter.acquire()
    ...     return await api.generate(prompt)

    Combining with retry logic:

    >>> limiter = AsyncSlidingWindowRateLimiter(max_requests=50, window_seconds=60)
    >>>
    >>> async def safe_api_call(prompt: str) -> str:
    ...     for attempt in range(3):
    ...         await limiter.acquire()
    ...         try:
    ...             return await api.generate(prompt)
    ...         except RateLimitError:
    ...             # API returned 429, wait before retry
    ...             await asyncio.sleep(2 ** attempt)
    ...     raise Exception("All retries failed")

    See Also
    --------
    AsyncTokenBucketRateLimiter : Alternative that allows controlled bursting.
    rate_limited : Decorator for simple rate limiting using token bucket.

    Notes
    -----
    - Tracks actual request timestamps in a deque
    - More memory usage than token bucket (stores N timestamps)
    - Provides stricter guarantees than token bucket
    - Thread-safe within a single event loop
    - Old timestamps are automatically pruned during acquire()
    """

    max_requests: int
    window_seconds: float
    _requests: deque = field(default_factory=deque, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def acquire(self) -> None:
        """Wait until a request slot is available within the window.

        This method blocks (asynchronously) until making a request would
        not exceed the max_requests limit within the current window.
        It tracks the request timestamp when returning.

        Examples
        --------
        >>> limiter = AsyncSlidingWindowRateLimiter(max_requests=10, window_seconds=1.0)
        >>>
        >>> async def make_request():
        ...     await limiter.acquire()
        ...     # Safe to make request now
        ...     return await api.call()

        Notes
        -----
        - Automatically cleans up expired timestamps
        - Calculates optimal wait time based on oldest request
        - Uses asyncio.sleep for non-blocking waits
        """
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
    """Decorator to rate limit an async function using token bucket algorithm.

    A convenience decorator that creates an `AsyncTokenBucketRateLimiter` and
    applies it to the decorated function. Each function gets its own rate
    limiter instance.

    Args
    ----
    rate : float
        Maximum number of calls per second (steady-state rate).
    burst : int, optional
        Maximum burst size (bucket capacity). Allows this many calls
        to proceed immediately before rate limiting kicks in. Default is 1
        (no bursting).

    Returns
    -------
    Callable[[Callable], Callable]
        A decorator that wraps async functions with rate limiting.

    Examples
    --------
    Basic rate limiting:

    >>> from insideLLMs.async_utils import rate_limited
    >>>
    >>> @rate_limited(rate=2.0, burst=5)
    ... async def call_api(prompt: str) -> str:
    ...     return await api.generate(prompt)
    >>>
    >>> async def main():
    ...     # First 5 calls go immediately (burst)
    ...     # Then limited to 2 calls per second
    ...     results = await asyncio.gather(*[
    ...         call_api(f"Question {i}")
    ...         for i in range(20)
    ...     ])

    Rate limiting with no burst:

    >>> @rate_limited(rate=1.0)  # burst=1 by default
    ... async def slow_api(data: dict) -> dict:
    ...     return await external_api.process(data)
    >>>
    >>> async def main():
    ...     # Exactly 1 call per second, no bursting
    ...     for item in items:
    ...         result = await slow_api(item)

    Multiple decorated functions share nothing:

    >>> @rate_limited(rate=10.0, burst=20)
    ... async def api_a(x: int) -> int:
    ...     return await service_a.call(x)
    >>>
    >>> @rate_limited(rate=5.0, burst=10)
    ... async def api_b(x: int) -> int:
    ...     return await service_b.call(x)
    >>>
    >>> # api_a and api_b have independent rate limiters

    Combining with other decorators:

    >>> from functools import lru_cache
    >>>
    >>> @rate_limited(rate=5.0, burst=10)
    ... async def fetch_data(key: str) -> dict:
    ...     # Rate limited first, then fetches
    ...     return await api.get(key)

    See Also
    --------
    AsyncTokenBucketRateLimiter : The underlying rate limiter class.
    AsyncSlidingWindowRateLimiter : For stricter rate limiting without bursting.

    Notes
    -----
    - Each decorated function gets its own rate limiter instance
    - Uses token bucket algorithm (allows controlled bursting)
    - The rate limiter persists for the lifetime of the decorated function
    - For shared rate limiting across functions, create an
      AsyncTokenBucketRateLimiter instance and use it directly
    """
    limiter = AsyncTokenBucketRateLimiter(rate=rate, burst=burst)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Async Queue Processing


class AsyncWorkerPool(Generic[T, R]):
    """Pool of async workers for processing a shared queue.

    A worker pool maintains a fixed number of worker coroutines that pull
    items from a shared queue and process them. This is useful for scenarios
    where work items arrive over time (e.g., from a stream) rather than
    being available upfront.

    The pool is used as an async context manager: workers start when entering
    the context and stop when exiting. Results are collected and available
    after the context exits.

    Parameters
    ----------
    worker_func : Callable[[T], Awaitable[R]]
        An async function that processes a single item. Called by worker
        coroutines as items are pulled from the queue.
    num_workers : int, optional
        Number of concurrent worker coroutines. Default is 5.
    max_queue_size : int, optional
        Maximum items that can be queued. If 0 (default), the queue is
        unbounded. When the queue is full, submit() will block until
        space is available.

    Attributes
    ----------
    worker_func : Callable[[T], Awaitable[R]]
        The processing function.
    num_workers : int
        Number of workers.
    results : list[tuple[int, Optional[R], Optional[Exception]]]
        Collected results after processing, sorted by submission order.

    Examples
    --------
    Basic worker pool usage:

    >>> import asyncio
    >>> from insideLLMs.async_utils import AsyncWorkerPool
    >>>
    >>> async def process_item(item: dict) -> dict:
    ...     await asyncio.sleep(0.1)  # Simulate work
    ...     return {"processed": item["id"]}
    >>>
    >>> async def main():
    ...     items = [{"id": i} for i in range(20)]
    ...
    ...     async with AsyncWorkerPool(process_item, num_workers=5) as pool:
    ...         for item in items:
    ...             await pool.submit(item)
    ...
    ...     # Results are available after context exits
    ...     for idx, result, error in pool.results:
    ...         if error:
    ...             print(f"Item {idx} failed: {error}")
    ...         else:
    ...             print(f"Item {idx}: {result}")

    Processing a stream of items:

    >>> async def main():
    ...     async with AsyncWorkerPool(process_item, num_workers=10) as pool:
    ...         async for item in item_stream():
    ...             await pool.submit(item)
    ...             # Items are processed as they arrive

    Using bounded queue for backpressure:

    >>> async def main():
    ...     # Queue holds max 100 items
    ...     # submit() blocks if queue is full
    ...     async with AsyncWorkerPool(
    ...         process_item,
    ...         num_workers=5,
    ...         max_queue_size=100,
    ...     ) as pool:
    ...         for item in millions_of_items():
    ...             await pool.submit(item)  # Blocks if queue full

    Handling errors in results:

    >>> async def risky_process(x: int) -> int:
    ...     if x % 10 == 0:
    ...         raise ValueError(f"Cannot process {x}")
    ...     return x * 2
    >>>
    >>> async def main():
    ...     async with AsyncWorkerPool(risky_process, num_workers=5) as pool:
    ...         for i in range(100):
    ...             await pool.submit(i)
    ...
    ...     succeeded = [(idx, r) for idx, r, e in pool.results if e is None]
    ...     failed = [(idx, e) for idx, r, e in pool.results if e is not None]
    ...     print(f"Success: {len(succeeded)}, Failed: {len(failed)}")

    Combining with rate limiting:

    >>> limiter = AsyncTokenBucketRateLimiter(rate=10.0, burst=20)
    >>>
    >>> async def rate_limited_process(item: dict) -> dict:
    ...     await limiter.acquire()
    ...     return await api.process(item)
    >>>
    >>> async def main():
    ...     async with AsyncWorkerPool(rate_limited_process, num_workers=20) as pool:
    ...         for item in items:
    ...             await pool.submit(item)

    See Also
    --------
    map_async : For processing a fixed list of items (simpler API).
    for_each_async : For side-effect-only operations.

    Notes
    -----
    - Workers are started when entering the async context
    - Workers are stopped gracefully when exiting the context
    - Results are sorted by submission order, not completion order
    - Each result is a tuple of (index, result_or_None, error_or_None)
    - The queue is drained before workers are stopped
    """

    def __init__(
        self,
        worker_func: Callable[[T], Awaitable[R]],
        num_workers: int = 5,
        max_queue_size: int = 0,
    ):
        """Initialize the worker pool.

        Args
        ----
        worker_func : Callable[[T], Awaitable[R]]
            Async function to process each item.
        num_workers : int, optional
            Number of concurrent workers. Default is 5.
        max_queue_size : int, optional
            Maximum queue size. 0 means unbounded. Default is 0.
        """
        self.worker_func = worker_func
        self.num_workers = num_workers
        self._queue: asyncio.Queue[Optional[T]] = asyncio.Queue(max_queue_size)
        self._results: list[tuple[int, Optional[R], Optional[Exception]]] = []
        self._workers: list[asyncio.Task] = []
        self._counter = 0
        self._lock = asyncio.Lock()

    @property
    def results(self) -> list[tuple[int, Optional[R], Optional[Exception]]]:
        """Get all collected results sorted by submission order.

        Returns
        -------
        list[tuple[int, Optional[R], Optional[Exception]]]
            List of (index, result, error) tuples where:
            - index: The submission order (0-based)
            - result: The return value if successful, None if failed
            - error: The exception if failed, None if successful

        Examples
        --------
        >>> async with AsyncWorkerPool(process, num_workers=5) as pool:
        ...     await pool.submit("a")
        ...     await pool.submit("b")
        ...
        >>> pool.results
        [(0, 'result_a', None), (1, 'result_b', None)]
        """
        return sorted(self._results, key=lambda x: x[0])

    async def _worker(self) -> None:
        """Internal worker coroutine that processes items from the queue."""
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
        """Submit an item for processing by a worker.

        Adds the item to the work queue. If the queue is bounded and full,
        this method will block until a worker consumes an item and makes
        space available.

        Args
        ----
        item : T
            The item to process. Will be passed to worker_func.

        Examples
        --------
        >>> async with AsyncWorkerPool(process, num_workers=5) as pool:
        ...     # Submit items one at a time
        ...     await pool.submit("item1")
        ...     await pool.submit("item2")
        ...
        ...     # Or submit from a loop
        ...     for item in items:
        ...         await pool.submit(item)

        Notes
        -----
        - Must be called within the async context (after __aenter__)
        - Items are processed in FIFO order by available workers
        - Submission order is tracked for result ordering
        """
        await self._queue.put(item)

    async def __aenter__(self) -> "AsyncWorkerPool[T, R]":
        """Start the worker coroutines.

        Returns
        -------
        AsyncWorkerPool[T, R]
            Self, for use with 'async with' statement.

        Examples
        --------
        >>> async with AsyncWorkerPool(process, num_workers=5) as pool:
        ...     # Workers are now running
        ...     await pool.submit(item)
        """
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.num_workers)]
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop workers gracefully and wait for completion.

        This method:
        1. Waits for all queued items to be processed
        2. Sends stop signals to all workers
        3. Waits for all workers to terminate

        After this method returns, all results are available via the
        `results` property.

        Notes
        -----
        - Does not cancel in-progress work
        - Exceptions from workers are captured in results, not raised
        """
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
    """Progress tracker for async operations with rate and ETA calculation.

    A thread-safe progress tracker designed for concurrent async operations.
    Tracks completion count, calculates processing rate, and estimates
    remaining time. Can be displayed directly using str() or formatted
    for custom output.

    Parameters
    ----------
    total : int
        Total number of items to process.
    completed : int, optional
        Initial number of completed items. Default is 0.
    start_time : float, optional
        Start time in seconds (from time.perf_counter()). If None,
        timing starts on first update() call. Default is None.

    Attributes
    ----------
    total : int
        Total number of items.
    completed : int
        Number of items completed so far.
    start_time : float or None
        When processing started (or None if not started).
    elapsed_time : float
        Seconds elapsed since start.
    items_per_second : float
        Current processing rate.
    estimated_remaining : float
        Estimated seconds until completion.
    percent_complete : float
        Completion percentage (0-100).

    Examples
    --------
    Basic progress tracking:

    >>> import asyncio
    >>> from insideLLMs.async_utils import AsyncProgress, map_async
    >>>
    >>> async def main():
    ...     progress = AsyncProgress(total=100)
    ...
    ...     async def process_with_progress(item: int) -> int:
    ...         result = item * 2
    ...         await asyncio.sleep(0.01)
    ...         await progress.update()
    ...         return result
    ...
    ...     items = list(range(100))
    ...     result = await map_async(process_with_progress, items)
    ...     print(f"Final: {progress}")
    ...     print(f"Rate: {progress.items_per_second:.1f} items/sec")

    Real-time progress display:

    >>> async def main():
    ...     progress = AsyncProgress(total=1000)
    ...
    ...     async def process_item(item):
    ...         result = await api.generate(item)
    ...         await progress.update()
    ...         # Overwrite line with current progress
    ...         print(f"\\r{progress}", end="", flush=True)
    ...         return result
    ...
    ...     await map_async(process_item, items, max_concurrency=10)
    ...     print()  # Final newline

    Custom progress formatting:

    >>> async def main():
    ...     progress = AsyncProgress(total=50)
    ...
    ...     async def process(item):
    ...         await asyncio.sleep(0.1)
    ...         await progress.update()
    ...
    ...         # Custom format
    ...         bar_width = 30
    ...         filled = int(bar_width * progress.percent_complete / 100)
    ...         bar = "=" * filled + "-" * (bar_width - filled)
    ...         print(f"\\r[{bar}] {progress.percent_complete:.0f}%", end="")
    ...         return item

    Tracking batches:

    >>> async def main():
    ...     # Each batch processes 10 items
    ...     progress = AsyncProgress(total=1000)
    ...
    ...     async def process_batch(batch: list) -> list:
    ...         results = await api.batch_process(batch)
    ...         await progress.update(n=len(batch))  # Update by batch size
    ...         return results
    ...
    ...     batches = [items[i:i+10] for i in range(0, 1000, 10)]
    ...     await map_async(process_batch, batches)

    Combining with tqdm (if installed):

    >>> async def main():
    ...     from tqdm import tqdm
    ...     progress = AsyncProgress(total=100)
    ...     pbar = tqdm(total=100, desc="Processing")
    ...
    ...     async def process(item):
    ...         result = await api.call(item)
    ...         await progress.update()
    ...         pbar.update(1)
    ...         return result
    ...
    ...     await map_async(process, items)
    ...     pbar.close()

    See Also
    --------
    map_async : Primary batch processing function.
    AsyncWorkerPool : Queue-based processing with workers.

    Notes
    -----
    - Thread-safe for concurrent updates using asyncio.Lock
    - Timer starts lazily on first update() call
    - Rate calculation smooths over entire elapsed time
    - ETA is based on current average rate (no smoothing)
    """

    total: int
    completed: int = 0
    start_time: Optional[float] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since processing started.

        Returns
        -------
        float
            Seconds elapsed since first update() call, or 0.0 if not started.

        Examples
        --------
        >>> progress = AsyncProgress(total=100)
        >>> progress.elapsed_time  # Not started yet
        0.0
        >>> await progress.update()
        >>> await asyncio.sleep(1.0)
        >>> progress.elapsed_time  # Approximately 1.0
        """
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    @property
    def items_per_second(self) -> float:
        """Calculate current processing rate.

        Returns
        -------
        float
            Items processed per second, or 0.0 if no time elapsed.

        Examples
        --------
        >>> progress = AsyncProgress(total=100)
        >>> # After processing 50 items in 10 seconds
        >>> progress.items_per_second
        5.0
        """
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed

    @property
    def estimated_remaining(self) -> float:
        """Estimate remaining time based on current rate.

        Returns
        -------
        float
            Estimated seconds until all items complete. Returns infinity
            if rate is zero (no items processed yet).

        Examples
        --------
        >>> progress = AsyncProgress(total=100)
        >>> # After processing 50 items at 5/second
        >>> progress.estimated_remaining  # 50 remaining / 5 per second
        10.0
        """
        rate = self.items_per_second
        if rate == 0:
            return float("inf")
        remaining = self.total - self.completed
        return remaining / rate

    @property
    def percent_complete(self) -> float:
        """Get completion percentage.

        Returns
        -------
        float
            Percentage complete (0.0 to 100.0). Returns 100.0 if total is 0.

        Examples
        --------
        >>> progress = AsyncProgress(total=100, completed=25)
        >>> progress.percent_complete
        25.0
        """
        if self.total == 0:
            return 100.0
        return 100.0 * self.completed / self.total

    async def update(self, n: int = 1) -> None:
        """Update progress by incrementing the completed count.

        Thread-safe method to record that items have been processed.
        Automatically starts the timer on first call.

        Args
        ----
        n : int, optional
            Number of items completed. Default is 1.

        Examples
        --------
        >>> progress = AsyncProgress(total=100)
        >>>
        >>> async def process(item):
        ...     result = await api.call(item)
        ...     await progress.update()  # Increment by 1
        ...     return result
        >>>
        >>> async def process_batch(batch):
        ...     results = await api.batch_call(batch)
        ...     await progress.update(n=len(batch))  # Increment by batch size
        ...     return results
        """
        async with self._lock:
            if self.start_time is None:
                self.start_time = time.perf_counter()
            self.completed += n

    def __str__(self) -> str:
        """Return a formatted progress string.

        Returns
        -------
        str
            Progress in format: "completed/total (percent%) | rate/s | ETA: Xs"

        Examples
        --------
        >>> progress = AsyncProgress(total=100, completed=50)
        >>> str(progress)
        '50/100 (50.0%) | 10.0/s | ETA: 5s'
        """
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

    A simpler alternative to `map_async` when you already have coroutine
    objects and want behavior similar to asyncio.gather() but with
    concurrency control.

    Unlike `map_async`, this function raises the first exception encountered
    rather than collecting errors. Use `map_async` if you need error handling.

    Args
    ----
    coros : Sequence[Awaitable[T]]
        Sequence of coroutine objects to execute. Note: these should be
        unawaited coroutines, not coroutine functions.
    limit : int
        Maximum number of coroutines running concurrently.

    Returns
    -------
    list[T]
        List of results in the same order as input coroutines.

    Raises
    ------
    Exception
        The first exception raised by any coroutine (via asyncio.gather).

    Examples
    --------
    Basic usage with pre-created coroutines:

    >>> import asyncio
    >>> from insideLLMs.async_utils import gather_with_limit
    >>>
    >>> async def fetch(url: str) -> str:
    ...     await asyncio.sleep(0.1)
    ...     return f"Response from {url}"
    >>>
    >>> async def main():
    ...     urls = [f"https://api.example.com/{i}" for i in range(20)]
    ...     # Create coroutines (not yet started)
    ...     coros = [fetch(url) for url in urls]
    ...     # Run with limit of 5 concurrent
    ...     results = await gather_with_limit(coros, limit=5)
    ...     for url, result in zip(urls, results):
    ...         print(f"{url}: {result}")

    Replacing unbounded asyncio.gather:

    >>> async def main():
    ...     # Instead of:
    ...     # results = await asyncio.gather(*[fetch(u) for u in urls])
    ...
    ...     # Use:
    ...     results = await gather_with_limit(
    ...         [fetch(u) for u in urls],
    ...         limit=10,
    ...     )

    Combined with other async operations:

    >>> async def main():
    ...     # Process in batches with different limits
    ...     batch1 = [process(x) for x in items[:50]]
    ...     batch2 = [process(x) for x in items[50:]]
    ...
    ...     results1, results2 = await asyncio.gather(
    ...         gather_with_limit(batch1, limit=10),
    ...         gather_with_limit(batch2, limit=5),
    ...     )

    See Also
    --------
    map_async : More feature-rich alternative with error handling.
    asyncio.gather : Standard library function (no concurrency limit).

    Notes
    -----
    - Results are returned in input order, not completion order
    - First exception stops execution (asyncio.gather behavior)
    - For error handling, prefer `map_async` instead
    - Coroutines start immediately when gather_with_limit is awaited
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
    """Race coroutines and return the result of the first to complete.

    Starts all coroutines concurrently and returns as soon as any one
    completes successfully. This is useful for racing multiple providers
    or implementing timeout patterns with fallbacks.

    Args
    ----
    coros : Sequence[Awaitable[T]]
        Coroutines to race. All start immediately and run concurrently.
    cancel_remaining : bool, optional
        If True (default), cancel all pending coroutines when one completes.
        Set to False if you want remaining coroutines to continue running
        in the background.

    Returns
    -------
    T
        Result of the first coroutine to complete successfully.

    Raises
    ------
    ValueError
        If no coroutines are provided (empty sequence).
    Exception
        If the first completed coroutine raises an exception, that exception
        is propagated. Note: this means a fast failure beats a slow success.

    Examples
    --------
    Racing multiple API providers for fastest response:

    >>> import asyncio
    >>> from insideLLMs.async_utils import first_completed
    >>>
    >>> async def main():
    ...     result = await first_completed([
    ...         openai_client.generate("Hello"),
    ...         anthropic_client.generate("Hello"),
    ...         google_client.generate("Hello"),
    ...     ])
    ...     # Returns result from whichever API responds first
    ...     print(f"Got response: {result}")

    Implementing timeout with fallback:

    >>> async def with_timeout(coro, timeout_seconds):
    ...     async def timeout():
    ...         await asyncio.sleep(timeout_seconds)
    ...         raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
    ...     return await first_completed([coro, timeout()])
    >>>
    >>> async def main():
    ...     try:
    ...         result = await with_timeout(slow_api.call(), 5.0)
    ...     except TimeoutError:
    ...         result = "default_value"

    Racing with different strategies:

    >>> async def main():
    ...     # Try different models with same prompt
    ...     result = await first_completed([
    ...         fast_model.generate(prompt),     # Quick but less accurate
    ...         slow_model.generate(prompt),     # Slower but better
    ...         cached_lookup(prompt),           # Instant if cached
    ...     ])

    Keeping remaining tasks running (fire-and-forget):

    >>> async def main():
    ...     # Start analytics in background, don't wait for them
    ...     result = await first_completed(
    ...         [
    ...             get_data(),
    ...             log_analytics(),  # Will continue running
    ...             update_cache(),   # Will continue running
    ...         ],
    ...         cancel_remaining=False,
    ...     )

    Handling the "first failure" case:

    >>> async def main():
    ...     # Be aware: fast errors beat slow successes
    ...     async def fast_fail():
    ...         raise ValueError("Quick error")
    ...
    ...     async def slow_success():
    ...         await asyncio.sleep(1)
    ...         return "success"
    ...
    ...     # This raises ValueError, not returns "success"
    ...     try:
    ...         result = await first_completed([fast_fail(), slow_success()])
    ...     except ValueError:
    ...         print("Fast failure won the race")

    See Also
    --------
    retry_until_success : Try coroutines sequentially until one succeeds.
    asyncio.wait : Lower-level API for waiting on tasks.

    Notes
    -----
    - All coroutines start immediately (no lazy evaluation)
    - The "winner" is determined by completion time, not success
    - If the first to complete raises an error, that error is raised
    - Cancelled tasks may still run briefly before stopping
    - With cancel_remaining=False, ensure you handle orphaned tasks
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
    """Try coroutine factories in sequence until one succeeds.

    Executes coroutine factories one at a time, in order, until one
    completes without raising an exception. This is useful for implementing
    fallback chains where you want to try alternatives sequentially
    (not concurrently like `first_completed`).

    Args
    ----
    coros : Sequence[Callable[[], Awaitable[T]]]
        Sequence of coroutine factories (callables that return coroutines).
        Must be factories (functions) rather than coroutine objects because
        coroutines can only be awaited once.

    Returns
    -------
    T
        Result of the first coroutine factory that completes without error.

    Raises
    ------
    ValueError
        If an empty sequence is provided.
    Exception
        The last exception raised if all coroutine factories fail.
        This preserves the exception type and traceback.

    Examples
    --------
    Basic fallback chain for API calls:

    >>> import asyncio
    >>> from insideLLMs.async_utils import retry_until_success
    >>>
    >>> async def try_primary():
    ...     # Primary API might be down
    ...     response = await primary_api.generate("Hello")
    ...     return response
    >>>
    >>> async def try_backup():
    ...     # Backup API is more reliable but slower
    ...     response = await backup_api.generate("Hello")
    ...     return response
    >>>
    >>> async def try_cached():
    ...     # Fall back to cached response
    ...     return cache.get("Hello")
    >>>
    >>> async def main():
    ...     result = await retry_until_success([
    ...         try_primary,
    ...         try_backup,
    ...         try_cached,
    ...     ])

    Using lambda for simple cases:

    >>> async def main():
    ...     result = await retry_until_success([
    ...         lambda: api1.generate("Hello"),
    ...         lambda: api2.generate("Hello"),
    ...         lambda: api3.generate("Hello"),
    ...     ])

    With retry delays between attempts:

    >>> async def main():
    ...     async def try_with_delay(factory, delay):
    ...         if delay > 0:
    ...             await asyncio.sleep(delay)
    ...         return await factory()
    ...
    ...     result = await retry_until_success([
    ...         lambda: api.generate("Hello"),                    # Immediate
    ...         lambda: try_with_delay(api.generate, 1.0),       # 1s delay
    ...         lambda: try_with_delay(api.generate, 5.0),       # 5s delay
    ...     ])

    Implementing exponential backoff:

    >>> async def with_backoff(factory, max_retries=3, base_delay=1.0):
    ...     async def attempt():
    ...         return await factory()
    ...
    ...     factories = []
    ...     for i in range(max_retries):
    ...         delay = base_delay * (2 ** i)
    ...         async def delayed_attempt(d=delay):
    ...             await asyncio.sleep(d)
    ...             return await factory()
    ...         factories.append(delayed_attempt)
    ...
    ...     return await retry_until_success(factories)

    Combining different providers with logging:

    >>> async def main():
    ...     async def try_openai():
    ...         print("Trying OpenAI...")
    ...         return await openai.generate("Hello")
    ...
    ...     async def try_anthropic():
    ...         print("OpenAI failed, trying Anthropic...")
    ...         return await anthropic.generate("Hello")
    ...
    ...     async def try_local():
    ...         print("Both cloud APIs failed, using local model...")
    ...         return await local_model.generate("Hello")
    ...
    ...     result = await retry_until_success([
    ...         try_openai,
    ...         try_anthropic,
    ...         try_local,
    ...     ])

    See Also
    --------
    first_completed : Race coroutines concurrently (parallel, not sequential).
    insideLLMs.retry : More sophisticated retry logic with backoff.

    Notes
    -----
    - Factories are called lazily (only when previous attempts fail)
    - Must pass callables (factories), not coroutine objects
    - The last exception is re-raised if all attempts fail
    - No concurrency: each attempt completes before the next starts
    - Consider `first_completed` if parallel execution is acceptable
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import BatchResult. The canonical name is
# AsyncBatchResult.
BatchResult = AsyncBatchResult

# Older code and tests may import these shorter names.
RateLimiter = AsyncTokenBucketRateLimiter
SlidingWindowRateLimiter = AsyncSlidingWindowRateLimiter
