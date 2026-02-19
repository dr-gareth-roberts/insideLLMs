"""
Cache Utilities
===============

Utility classes for cache warming, memoization, and related functionality.
"""

from __future__ import annotations

import hashlib
from functools import wraps
from typing import Any, Callable, Optional

from insideLLMs.caching.backends.strategy import StrategyCache
from insideLLMs.caching.base import generate_cache_key
from insideLLMs.caching.types import CacheConfig

# Type alias for backward compatibility
BaseCache = StrategyCache


class CacheWarmer:
    """Preloads cache with common prompts for cold start optimization.

    CacheWarmer queues prompts for pre-generation and caching before they
    are actually requested. This reduces latency for common queries and
    improves user experience during application startup.

    Parameters
    ----------
    cache : BaseCache
        The cache to warm with pre-generated responses.
    generator : Optional[Callable[[str], Any]]
        Function that generates responses from prompts. Required for
        the warm() method. Should match your LLM's generate signature.

    Attributes
    ----------
    cache : BaseCache
        The target cache instance.
    generator : Optional[Callable[[str], Any]]
        The response generator function.
    _warming_queue : list[dict]
        Queue of prompts pending warming.
    _warming_results : list[dict]
        Results from completed warming operations.

    Examples
    --------
    Basic cache warming:

        >>> from insideLLMs.caching import CacheWarmer, StrategyCache
        >>> cache = StrategyCache()
        >>> def generate(prompt):
        ...     return f"Response for: {prompt}"
        >>> warmer = CacheWarmer(cache, generate)
        >>> warmer.add_prompt("Hello")
        >>> warmer.add_prompt("Goodbye")
        >>> results = warmer.warm()
        >>> for r in results:
        ...     print(f"{r['prompt']}: {r['status']}")
        Hello: success
        Goodbye: success

    Priority-based warming:

        >>> warmer = CacheWarmer(cache, generate)
        >>> warmer.add_prompt("Low priority", priority=1)
        >>> warmer.add_prompt("High priority", priority=10)
        >>> warmer.add_prompt("Medium priority", priority=5)
        >>> results = warmer.warm(batch_size=2)
        >>> # High priority is warmed first
        >>> print([r['prompt'] for r in results])
        ['High priority', 'Medium priority']

    Batch warming with skip existing:

        >>> cache.set(generate_cache_key("Already cached"), "value")
        >>> warmer.add_prompt("Already cached")
        >>> warmer.add_prompt("New prompt")
        >>> results = warmer.warm(skip_existing=True)
        >>> for r in results:
        ...     print(f"{r['prompt']}: {r['status']}")

    Warming with model parameters:

        >>> warmer = CacheWarmer(cache, generate)
        >>> warmer.add_prompt(
        ...     prompt="What is AI?",
        ...     model="gpt-4",
        ...     params={"temperature": 0.7},
        ...     priority=5
        ... )
        >>> results = warmer.warm()

    Managing the warming queue:

        >>> warmer = CacheWarmer(cache, generate)
        >>> for i in range(100):
        ...     warmer.add_prompt(f"Prompt {i}")
        >>> print(f"Queue size: {warmer.get_queue_size()}")
        Queue size: 100
        >>> warmer.warm(batch_size=50)
        >>> print(f"Remaining: {warmer.get_queue_size()}")
        Remaining: 50
        >>> warmer.clear_queue()
        >>> print(f"After clear: {warmer.get_queue_size()}")
        After clear: 0

    See Also
    --------
    create_cache_warmer : Convenience function for creating warmers.
    PromptCache.cache_response : Manual response caching.
    """

    def __init__(
        self,
        cache: BaseCache,
        generator: Optional[Callable[[str], Any]] = None,
    ):
        self.cache = cache
        self.generator = generator
        self._warming_queue: list[dict] = []
        self._warming_results: list[dict] = []

    def add_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[dict] = None,
        priority: int = 0,
    ):
        """Add prompt to warming queue."""
        self._warming_queue.append(
            {
                "prompt": prompt,
                "model": model,
                "params": params,
                "priority": priority,
            }
        )

    def warm(
        self,
        batch_size: int = 10,
        skip_existing: bool = True,
    ) -> list[dict]:
        """Execute cache warming."""
        if not self.generator:
            raise ValueError("Generator function required for warming")

        self._warming_queue.sort(key=lambda x: x["priority"], reverse=True)

        results = []
        batch = self._warming_queue[:batch_size]

        for item in batch:
            key = generate_cache_key(
                item["prompt"],
                item["model"],
                item["params"],
            )

            if skip_existing and self.cache.contains(key):
                results.append(
                    {
                        "prompt": item["prompt"],
                        "status": "skipped",
                        "reason": "already_cached",
                    }
                )
                continue

            try:
                value = self.generator(item["prompt"])
                self.cache.set(
                    key,
                    value,
                    metadata={
                        "prompt": item["prompt"],
                        "model": item["model"],
                        "params": item["params"],
                        "warmed": True,
                    },
                )
                results.append(
                    {
                        "prompt": item["prompt"],
                        "status": "success",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "prompt": item["prompt"],
                        "status": "error",
                        "error": str(e),
                    }
                )

        self._warming_queue = self._warming_queue[batch_size:]
        self._warming_results.extend(results)

        return results

    def get_queue_size(self) -> int:
        """Get number of prompts in warming queue."""
        return len(self._warming_queue)

    def get_results(self) -> list[dict]:
        """Get warming results."""
        return self._warming_results.copy()

    def clear_queue(self):
        """Clear warming queue."""
        self._warming_queue.clear()


class MemoizedFunction:
    """Wrapper for memoizing function calls with cache-backed storage.

    MemoizedFunction wraps any callable to automatically cache its return
    values based on the input arguments. Subsequent calls with the same
    arguments return the cached value without re-executing the function.

    Parameters
    ----------
    func : Callable
        The function to memoize.
    cache : Optional[BaseCache]
        Cache instance for storing memoized values. If None, creates a
        new StrategyCache with default settings.
    key_generator : Optional[Callable[..., str]]
        Custom function for generating cache keys from arguments. If None,
        uses a default implementation that hashes the stringified arguments.

    Attributes
    ----------
    func : Callable
        The wrapped function.
    cache : BaseCache
        The backing cache instance.
    key_generator : Callable[..., str]
        The key generation function.
    _call_count : int
        Total number of times the function has been called.
    _cache_calls : int
        Number of calls that returned cached values.

    Examples
    --------
    Basic memoization:

        >>> from insideLLMs.caching import MemoizedFunction
        >>> def expensive_fn(x, y):
        ...     print("Computing...")
        ...     return x + y
        >>> memo_fn = MemoizedFunction(expensive_fn)
        >>> print(memo_fn(2, 3))  # Computes
        Computing...
        5
        >>> print(memo_fn(2, 3))  # Returns cached
        5
        >>> print(memo_fn(3, 4))  # Different args, computes
        Computing...
        7

    With custom cache:

        >>> from insideLLMs.caching import StrategyCache, CacheConfig
        >>> config = CacheConfig(max_size=100, ttl_seconds=300)
        >>> cache = StrategyCache(config)
        >>> memo_fn = MemoizedFunction(expensive_fn, cache=cache)

    Custom key generation:

        >>> def custom_key(*args, **kwargs):
        ...     return f"custom:{args[0]}:{args[1]}"
        >>> memo_fn = MemoizedFunction(expensive_fn, key_generator=custom_key)

    Checking statistics:

        >>> memo_fn = MemoizedFunction(expensive_fn)
        >>> _ = memo_fn(1, 2)  # Compute
        >>> _ = memo_fn(1, 2)  # Cache hit
        >>> _ = memo_fn(1, 2)  # Cache hit
        >>> stats = memo_fn.get_stats()
        >>> print(f"Cache rate: {stats['cache_rate']:.1%}")
        Cache rate: 66.7%

    Invalidating cached values:

        >>> memo_fn = MemoizedFunction(expensive_fn)
        >>> _ = memo_fn(5, 6)  # Cache result
        >>> memo_fn.invalidate(5, 6)  # Remove from cache
        >>> _ = memo_fn(5, 6)  # Re-compute

    Using with keyword arguments:

        >>> def greet(name, greeting="Hello"):
        ...     return f"{greeting}, {name}!"
        >>> memo_fn = MemoizedFunction(greet)
        >>> print(memo_fn("Alice", greeting="Hi"))
        Hi, Alice!
        >>> print(memo_fn("Alice", greeting="Hi"))  # Cached
        Hi, Alice!

    Notes
    -----
    - The function's __doc__ and __name__ are preserved via functools.wraps.
    - Arguments must be representable as strings for default key generation.
    - Use custom key_generator for complex argument types.

    See Also
    --------
    memoize : Decorator interface for MemoizedFunction.
    StrategyCache : Default cache implementation used.
    """

    def __init__(
        self,
        func: Callable,
        cache: Optional[BaseCache] = None,
        key_generator: Optional[Callable[..., str]] = None,
    ):
        self.func = func
        self.cache = cache or StrategyCache()
        self.key_generator = key_generator or self._default_key_generator
        self._call_count = 0
        self._cache_calls = 0
        wraps(func)(self)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute memoized function."""
        self._call_count += 1

        key = self.key_generator(*args, **kwargs)
        result = self.cache.get(key)

        if result.hit:
            self._cache_calls += 1
            return result.value

        value = self.func(*args, **kwargs)
        self.cache.set(key, value)
        return value

    def _default_key_generator(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def invalidate(self, *args, **kwargs):
        """Invalidate cached result for given arguments."""
        key = self.key_generator(*args, **kwargs)
        self.cache.delete(key)

    def get_stats(self) -> dict[str, Any]:
        """Get memoization statistics."""
        return {
            "call_count": self._call_count,
            "cache_calls": self._cache_calls,
            "cache_rate": self._cache_calls / self._call_count if self._call_count > 0 else 0,
            "cache_stats": self.cache.get_stats().to_dict(),
        }


def memoize(
    func: Optional[Callable] = None,
    cache: Optional[BaseCache] = None,
    max_size: int = 1000,
    ttl_seconds: Optional[int] = 3600,
) -> Callable:
    """Decorator for memoizing functions with automatic caching.

    Provides a convenient decorator interface for adding memoization to
    functions. Can be used with or without parentheses.

    Parameters
    ----------
    func : Optional[Callable]
        The function to memoize (for @memoize without parentheses).
    cache : Optional[BaseCache]
        Custom cache instance. If None, creates a new StrategyCache
        with the specified max_size and ttl_seconds.
    max_size : int
        Maximum cache entries (only used if cache is None). Default 1000.
    ttl_seconds : Optional[int]
        TTL for cache entries in seconds (only used if cache is None).
        Default 3600 (1 hour).

    Returns
    -------
    Callable
        The memoized function (or decorator if called with parentheses).

    Examples
    --------
    Simple usage (without parentheses):

        >>> from insideLLMs.caching import memoize
        >>> @memoize
        ... def fibonacci(n):
        ...     if n < 2:
        ...         return n
        ...     return fibonacci(n-1) + fibonacci(n-2)
        >>> print(fibonacci(30))  # Fast due to memoization

    With configuration:

        >>> @memoize(max_size=100, ttl_seconds=600)
        ... def expensive_lookup(key):
        ...     # Simulates slow operation
        ...     return database.query(key)

    With custom cache:

        >>> from insideLLMs.caching import StrategyCache, CacheConfig, CacheStrategy
        >>> config = CacheConfig(strategy=CacheStrategy.LFU)
        >>> custom_cache = StrategyCache(config)
        >>> @memoize(cache=custom_cache)
        ... def compute(x, y):
        ...     return x * y

    Accessing memoization stats:

        >>> @memoize
        ... def add(a, b):
        ...     return a + b
        >>> _ = add(1, 2)
        >>> _ = add(1, 2)  # Cached
        >>> stats = add.get_stats()
        >>> print(f"Cache hits: {stats['cache_calls']}")

    Invalidating specific entries:

        >>> @memoize
        ... def multiply(x, y):
        ...     return x * y
        >>> _ = multiply(2, 3)  # Cached
        >>> multiply.invalidate(2, 3)  # Remove from cache
        >>> _ = multiply(2, 3)  # Re-computed

    Notes
    -----
    - Returns a MemoizedFunction instance with additional methods.
    - Thread-safe due to underlying StrategyCache implementation.
    - Arguments must be representable as strings for key generation.

    See Also
    --------
    MemoizedFunction : The underlying wrapper class.
    StrategyCache : Default cache implementation.
    """

    def decorator(fn: Callable) -> MemoizedFunction:
        nonlocal cache
        if cache is None:
            cache = StrategyCache(CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds))
        return MemoizedFunction(fn, cache)

    if func is not None:
        return decorator(func)
    return decorator


__all__ = [
    "CacheWarmer",
    "MemoizedFunction",
    "memoize",
]
