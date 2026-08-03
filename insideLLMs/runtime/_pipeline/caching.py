"""Response caching middleware."""

import asyncio
import threading
import time
from typing import Any, Optional

from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import AsyncModelProtocol
from insideLLMs.runtime._pipeline.middleware import Middleware


class CacheMiddleware(Middleware):
    """Middleware for caching model responses to avoid redundant API calls.

    Implements an in-memory cache with LRU (Least Recently Used) eviction and
    optional TTL (Time-To-Live) expiration. Cache keys are deterministic SHA-256
    hashes of the prompt and all parameters, ensuring identical requests always
    hit the cache.

    Caching is particularly useful for:
    - Evaluation runs where the same prompts may be processed multiple times
    - Development and testing to avoid repeated API calls
    - Cost reduction in scenarios with repeated queries
    - Improving latency for frequently requested responses

    Parameters
    ----------
    cache_size : int, default=1000
        Maximum number of entries to store in the cache. When this limit is
        reached, the oldest entry (by insertion order) is evicted to make
        room for new entries.
    ttl_seconds : float, optional
        Time-to-live for cache entries in seconds. Entries older than this
        are considered expired and will not be returned (though they remain
        in storage until evicted or replaced). If None (default), entries
        never expire based on time.

    Attributes
    ----------
    cache : dict[str, tuple[str, float]]
        The cache storage mapping cache keys to (response, timestamp) tuples.
    cache_size : int
        The configured maximum cache size.
    ttl_seconds : float or None
        The configured TTL, or None if no expiration.
    hits : int
        Counter for cache hits (requests served from cache).
    misses : int
        Counter for cache misses (requests forwarded to model).
    hit_rate : float
        Property that calculates the cache hit rate as hits / (hits + misses).

    Examples
    --------
    Basic caching:

        >>> from insideLLMs.runtime.pipeline import CacheMiddleware, ModelPipeline
        >>>
        >>> # Create cache with 500 entries max
        >>> cache_mw = CacheMiddleware(cache_size=500)
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> # First call: cache miss, calls model
        >>> response1 = pipeline.generate("What is Python?")
        >>> print(f"Hits: {cache_mw.hits}, Misses: {cache_mw.misses}")
        Hits: 0, Misses: 1
        >>>
        >>> # Second identical call: cache hit, returns cached response
        >>> response2 = pipeline.generate("What is Python?")
        >>> print(f"Hits: {cache_mw.hits}, Misses: {cache_mw.misses}")
        Hits: 1, Misses: 1
        >>> assert response1 == response2  # Same response

    Caching with TTL expiration:

        >>> # Cache entries expire after 1 hour
        >>> cache_mw = CacheMiddleware(cache_size=1000, ttl_seconds=3600)
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> response = pipeline.generate("Explain AI")
        >>> # After 1 hour, this entry will be considered expired

    Cache key includes parameters:

        >>> cache_mw = CacheMiddleware()
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> # Different parameters = different cache keys
        >>> r1 = pipeline.generate("Hello", temperature=0.7)
        >>> r2 = pipeline.generate("Hello", temperature=0.9)
        >>> print(f"Misses: {cache_mw.misses}")
        Misses: 2  # Both are cache misses (different params)

    Monitoring cache performance:

        >>> cache_mw = CacheMiddleware(cache_size=100)
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> # Run evaluation
        >>> for prompt in test_prompts:
        ...     pipeline.generate(prompt)
        >>>
        >>> # Check cache statistics
        >>> print(f"Hit rate: {cache_mw.hit_rate:.2%}")
        >>> print(f"Cache size: {len(cache_mw.cache)}")
        >>> info = pipeline.info()
        >>> print(f"Stats from pipeline: {info['cache_hit_rate']:.2%}")

    Async caching:

        >>> async def cached_batch():
        ...     cache_mw = CacheMiddleware()
        ...     pipeline = ModelPipeline(model, middlewares=[cache_mw])
        ...
        ...     # Run same prompts twice
        ...     prompts = ["Q1", "Q2", "Q3"]
        ...     await pipeline.abatch_generate(prompts)
        ...     await pipeline.abatch_generate(prompts)  # All cache hits
        ...
        ...     print(f"Final hit rate: {cache_mw.hit_rate:.2%}")
        ...     return cache_mw.hits, cache_mw.misses

    Notes
    -----
    - Cache keys are SHA-256 hashes of JSON-serialized prompt and parameters,
      ensuring deterministic and collision-resistant key generation.
    - The cache uses insertion-order dict semantics for LRU eviction (oldest
      entries are evicted first).
    - Expired entries are only removed when accessed; there is no background
      cleanup process.
    - The cache is not shared across pipeline instances; each CacheMiddleware
      has its own independent cache.
    - Async operations use an asyncio.Lock for thread-safe cache access.

    See Also
    --------
    Middleware : The base middleware class
    ModelPipeline : The pipeline that uses this middleware
    """

    def __init__(self, cache_size: int = 1000, ttl_seconds: Optional[float] = None) -> None:
        """Initialize the cache middleware with size and TTL configuration.

        Args
        ----
        cache_size : int, default=1000
            Maximum number of entries to store. When exceeded, the oldest
            entry is evicted using LRU policy.
        ttl_seconds : float, optional
            Time-to-live for entries in seconds. If None, entries never
            expire based on time.

        Examples
        --------
        Default configuration (1000 entries, no expiration):

            >>> cache_mw = CacheMiddleware()

        Limited cache with 1-hour TTL:

            >>> cache_mw = CacheMiddleware(cache_size=100, ttl_seconds=3600)

        Large cache for batch processing:

            >>> cache_mw = CacheMiddleware(cache_size=10000)
        """
        super().__init__()
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self.cache: dict[str, tuple[str, float]] = {}
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self._sync_lock = threading.Lock()

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        """Generate a deterministic cache key from prompt and parameters.

        Creates a SHA-256 hash of the JSON-serialized prompt and kwargs,
        ensuring that identical inputs always produce the same cache key.

        Args
        ----
        prompt : str
            The input prompt.
        **kwargs : Any
            Additional generation parameters to include in the key.

        Returns
        -------
        str
            A 64-character hexadecimal SHA-256 hash string.

        Examples
        --------
        Key generation:

            >>> key1 = cache_mw._cache_key("Hello", temperature=0.7)
            >>> key2 = cache_mw._cache_key("Hello", temperature=0.7)
            >>> assert key1 == key2  # Same inputs = same key

            >>> key3 = cache_mw._cache_key("Hello", temperature=0.9)
            >>> assert key1 != key3  # Different params = different key
        """
        import hashlib
        import json

        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry has expired based on its timestamp.

        Args
        ----
        timestamp : float
            The time.time() value when the entry was cached.

        Returns
        -------
        bool
            True if the entry is expired, False otherwise. Always returns
            False if ttl_seconds is None.

        Examples
        --------
        Checking expiration:

            >>> cache_mw = CacheMiddleware(ttl_seconds=60)
            >>> old_timestamp = time.time() - 120  # 2 minutes ago
            >>> cache_mw._is_expired(old_timestamp)
            True

            >>> recent_timestamp = time.time() - 30  # 30 seconds ago
            >>> cache_mw._is_expired(recent_timestamp)
            False
        """
        if self.ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self.ttl_seconds

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a generate request with cache lookup and storage.

        First checks if a valid (non-expired) cached response exists for
        the given prompt and parameters. If found, returns the cached
        response immediately (cache hit). Otherwise, delegates to the
        next middleware or model, stores the response in cache, and
        returns it (cache miss).

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. These are included in the
            cache key calculation.

        Returns
        -------
        str
            The generated (or cached) response.

        Raises
        ------
        ModelError
            If generation fails and no cached response is available.

        Examples
        --------
        Cache behavior:

            >>> cache_mw = CacheMiddleware()
            >>> cache_mw.model = model
            >>>
            >>> # First call: cache miss
            >>> r1 = cache_mw.process_generate("What is AI?")
            >>> print(cache_mw.hits, cache_mw.misses)
            0 1
            >>>
            >>> # Second call: cache hit
            >>> r2 = cache_mw.process_generate("What is AI?")
            >>> print(cache_mw.hits, cache_mw.misses)
            1 1
            >>> assert r1 == r2

        With different parameters:

            >>> r3 = cache_mw.process_generate("What is AI?", temperature=0.5)
            >>> print(cache_mw.misses)
            2  # Different params = new cache entry
        """
        key = self._cache_key(prompt, **kwargs)

        with self._sync_lock:
            cached = self.cache.pop(key, None)
            if cached is not None:
                response, timestamp = cached
                if not self._is_expired(timestamp):
                    self.hits += 1
                    self.cache[key] = (response, timestamp)
                    return response

            self.misses += 1

        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")

        with self._sync_lock:
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = (response, time.time())
        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a generate request with cache support.

        The async counterpart to process_generate. Uses an asyncio.Lock
        for thread-safe access to the cache when checking and storing
        entries.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated (or cached) response.

        Raises
        ------
        ModelError
            If generation fails and no cached response is available.

        Examples
        --------
        Async caching:

            >>> async def example():
            ...     cache_mw = CacheMiddleware()
            ...     cache_mw.model = async_model
            ...
            ...     # Concurrent requests for same prompt
            ...     tasks = [
            ...         cache_mw.aprocess_generate("Question?")
            ...         for _ in range(5)
            ...     ]
            ...     results = await asyncio.gather(*tasks)
            ...
            ...     # First one was a miss, rest were hits
            ...     print(f"Hits: {cache_mw.hits}, Misses: {cache_mw.misses}")
            ...     return results
        """
        key = self._cache_key(prompt, **kwargs)

        # Check cache (async-safe since cache is just a dict read)
        async with self._get_lock():
            cached = self.cache.pop(key, None)
            if cached is not None:
                response, timestamp = cached
                if not self._is_expired(timestamp):
                    self.hits += 1
                    self.cache[key] = (response, timestamp)
                    return response

            self.misses += 1

        # Cache miss - generate and cache
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if isinstance(self.model, AsyncModelProtocol):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")

        # Store in cache with LRU eviction
        async with self._get_lock():
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = (response, time.time())

        return response

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an asyncio.Lock for thread-safe cache access.

        Lazily creates the lock on first access. This is necessary because
        asyncio.Lock cannot be created outside of an async context in some
        Python versions.

        Returns
        -------
        asyncio.Lock
            The lock instance for synchronizing cache access.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> async def safe_operation():
            ...     cache_mw = CacheMiddleware()
            ...     async with cache_mw._get_lock():
            ...         # Thread-safe cache operations
            ...         pass
        """
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Computes the ratio of cache hits to total requests (hits + misses).

        Returns
        -------
        float
            The hit rate as a value between 0.0 and 1.0. Returns 0.0 if
            no requests have been processed yet.

        Examples
        --------
        Monitoring hit rate:

            >>> cache_mw = CacheMiddleware()
            >>> # After processing requests...
            >>> print(f"Hit rate: {cache_mw.hit_rate:.2%}")
            Hit rate: 75.00%

            >>> # With no requests yet
            >>> empty_cache = CacheMiddleware()
            >>> print(empty_cache.hit_rate)
            0.0
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
