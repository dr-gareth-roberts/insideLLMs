"""
Prompt Caching and Memoization Module
======================================

.. deprecated:: 1.0.0
    This module is deprecated and will be removed in a future version.
    Import directly from ``insideLLMs.caching_unified`` instead::

        # Old (deprecated):
        from insideLLMs.caching import create_cache, CacheStrategy

        # New (recommended):
        from insideLLMs.caching_unified import create_cache, CacheStrategy

This module provides intelligent caching for LLM operations, enabling efficient
response caching, memoization, and cache management. It supports multiple eviction
strategies, semantic similarity-based lookups, and comprehensive cache analytics.

This module serves as the public API for the caching subsystem, re-exporting
components from the unified caching implementation (``caching_unified.py``).
All imports should be made from this module for stability and backward compatibility.

Overview
--------
The caching module is designed for high-performance LLM applications that benefit from:

- **Response Caching**: Store and retrieve LLM responses based on prompts, models,
  and generation parameters. Reduces API costs and latency for repeated queries.
- **Semantic Similarity Matching**: Find cached responses for similar prompts using
  configurable similarity thresholds. Useful for handling paraphrased queries.
- **Multiple Eviction Strategies**: LRU (Least Recently Used), LFU (Least Frequently
  Used), FIFO (First In First Out), TTL (Time To Live), and SIZE-based eviction.
- **Cache Warming**: Preload caches with common prompts for faster cold starts,
  reducing latency for frequently-used queries.
- **Async Support**: Use caching in async contexts with the AsyncCacheAdapter,
  enabling integration with asyncio-based applications.
- **Cache Namespaces**: Organize multiple caches with named access for multi-tenant
  or multi-model applications.

Architecture
------------
The caching module implements a layered architecture:

1. **Enums and Configuration**: ``CacheStrategy``, ``CacheStatus``, ``CacheScope``,
   and ``CacheConfig`` provide type-safe configuration options.
2. **Data Classes**: ``CacheEntry``, ``CacheStats``, and ``CacheLookupResult``
   encapsulate cache data and operation results.
3. **Base Cache**: ``BaseCache`` (alias for ``StrategyCache``) provides a full-featured
   cache with configurable eviction strategies.
4. **Prompt Cache**: ``PromptCache`` extends ``BaseCache`` with LLM-specific features
   like semantic similarity matching.
5. **Utilities**: ``CacheWarmer``, ``MemoizedFunction``, ``CacheNamespace``,
   ``ResponseDeduplicator``, and ``AsyncCacheAdapter`` provide additional capabilities.
6. **Factory Functions**: ``create_cache``, ``create_prompt_cache``, ``create_cache_warmer``,
   and ``create_namespace`` provide convenient cache instantiation.

Key Components
--------------
CacheStrategy : Enum
    Eviction strategy selection (LRU, LFU, FIFO, TTL, SIZE). Determines which
    entries are removed when the cache reaches capacity.

CacheStatus : Enum
    Entry status tracking (ACTIVE, EXPIRED, EVICTED, WARMING). Indicates the
    current lifecycle state of a cache entry.

CacheScope : Enum
    Cache scope contexts (GLOBAL, SESSION, REQUEST, USER, MODEL). Enables
    logical partitioning of cache entries by context.

CacheConfig : dataclass
    Configuration for cache behavior including max_size, ttl_seconds, strategy,
    enable_stats, enable_compression, hash_algorithm, and scope.

CacheEntry : dataclass
    A single cache entry with value, timestamps, access tracking, and metadata.
    Includes methods for checking expiration and updating access counts.

CacheStats : dataclass
    Cache performance statistics including hits, misses, evictions, expirations,
    size metrics, and hit rate calculations.

CacheLookupResult : dataclass
    Result of cache lookup operations containing hit status, value, key, entry
    details, and lookup timing information.

BaseCache : class
    Strategy-based cache with configurable eviction. Supports LRU, LFU, FIFO,
    TTL, and SIZE strategies. Thread-safe implementation using locks.

PromptCache : class
    LLM-specific cache with semantic similarity support. Extends BaseCache
    with prompt-specific features like find_similar() for finding similar
    cached prompts.

CacheWarmer : class
    Utility for preloading cache entries with common prompts. Supports priority
    queuing and batch warming operations.

MemoizedFunction : class
    Wrapper for memoizing function calls with cache-backed storage. Provides
    statistics tracking and selective invalidation.

CacheNamespace : class
    Manager for multiple named caches. Enables organized multi-cache management
    with shared configuration defaults.

ResponseDeduplicator : class
    Utility for identifying duplicate responses using exact or similarity-based
    matching. Useful for detecting redundant API calls.

AsyncCacheAdapter : class
    Adapter for async cache operations. Wraps synchronous cache implementations
    for use in asyncio contexts.

Examples
--------
Basic cache usage with default LRU strategy:

    >>> from insideLLMs.caching import create_cache, CacheStrategy
    >>> cache = create_cache(max_size=100, ttl_seconds=3600)
    >>> # Store a value
    >>> entry = cache.set("greeting", "Hello, world!")
    >>> print(f"Stored: {entry.key} -> {entry.value}")
    Stored: greeting -> Hello, world!
    >>> # Retrieve the value
    >>> result = cache.get("greeting")
    >>> print(f"Hit: {result.hit}, Value: {result.value}")
    Hit: True, Value: Hello, world!
    >>> # Check for missing key
    >>> result = cache.get("nonexistent")
    >>> print(f"Hit: {result.hit}, Value: {result.value}")
    Hit: False, Value: None

Creating a cache with LFU (Least Frequently Used) eviction:

    >>> from insideLLMs.caching import create_cache, CacheStrategy
    >>> cache = create_cache(
    ...     max_size=50,
    ...     ttl_seconds=1800,
    ...     strategy=CacheStrategy.LFU
    ... )
    >>> # Frequently accessed items are kept longer
    >>> cache.set("popular", "accessed often")
    >>> cache.set("rare", "accessed once")
    >>> for _ in range(10):
    ...     _ = cache.get("popular")  # Increase access count
    >>> # When eviction occurs, "rare" is evicted before "popular"

Creating a prompt cache for LLM responses:

    >>> from insideLLMs.caching import create_prompt_cache
    >>> prompt_cache = create_prompt_cache(
    ...     max_size=500,
    ...     similarity_threshold=0.9
    ... )
    >>> # Cache an LLM response
    >>> prompt_cache.cache_response(
    ...     prompt="What is Python?",
    ...     response="Python is a high-level programming language known for "
    ...              "its readability and versatility.",
    ...     model="gpt-4",
    ...     params={"temperature": 0.7}
    ... )
    >>> # Retrieve cached response
    >>> result = prompt_cache.get_response("What is Python?", model="gpt-4")
    >>> print(f"Cache hit: {result.hit}")
    Cache hit: True
    >>> # Find similar prompts
    >>> similar = prompt_cache.find_similar("Explain Python programming", limit=3)
    >>> for prompt, entry, score in similar:
    ...     print(f"Similarity {score:.2f}: {prompt[:50]}...")

Using the memoize decorator for function caching:

    >>> from insideLLMs.caching import memoize
    >>> @memoize(max_size=100, ttl_seconds=600)
    ... def expensive_computation(x, y):
    ...     # Simulates expensive operation
    ...     import time
    ...     time.sleep(0.1)
    ...     return x ** y
    >>> # First call computes the result
    >>> result1 = expensive_computation(2, 10)  # Takes ~100ms
    >>> print(result1)
    1024
    >>> # Second call returns cached result
    >>> result2 = expensive_computation(2, 10)  # Returns immediately
    >>> print(result2)
    1024
    >>> # Check memoization statistics
    >>> stats = expensive_computation.get_stats()
    >>> print(f"Cache rate: {stats['cache_rate']:.1%}")
    Cache rate: 50.0%

Cache warming for cold start optimization:

    >>> from insideLLMs.caching import create_cache, create_cache_warmer
    >>> cache = create_cache(max_size=1000)
    >>> # Define a response generator
    >>> def generate_response(prompt):
    ...     # In production, this would call an LLM API
    ...     return f"Response for: {prompt}"
    >>> # Create and configure cache warmer
    >>> warmer = create_cache_warmer(cache, generate_response)
    >>> warmer.add_prompt("Hello, how can I help?", priority=10)
    >>> warmer.add_prompt("What is your name?", priority=8)
    >>> warmer.add_prompt("Tell me about the weather", priority=5)
    >>> # Execute warming
    >>> results = warmer.warm(batch_size=10)
    >>> for r in results:
    ...     print(f"{r['prompt'][:30]}... -> {r['status']}")
    Hello, how can I help?... -> success
    What is your name?... -> success
    Tell me about the weather... -> success

Using cache namespaces for organized multi-cache management:

    >>> from insideLLMs.caching import create_namespace
    >>> namespace = create_namespace(default_max_size=100)
    >>> # Create separate caches for different purposes
    >>> user_cache = namespace.get_cache("users")
    >>> prompt_cache = namespace.get_prompt_cache("prompts")
    >>> session_cache = namespace.get_cache("sessions")
    >>> # Use each cache independently
    >>> user_cache.set("user_123", {"name": "Alice", "role": "admin"})
    >>> prompt_cache.cache_response(
    ...     prompt="Hello",
    ...     response="Hi there!",
    ...     model="gpt-4"
    ... )
    >>> # Get statistics for all caches
    >>> stats = namespace.get_all_stats()
    >>> for name, cache_stats in stats.items():
    ...     print(f"{name}: {cache_stats.entry_count} entries")

Async cache usage with AsyncCacheAdapter:

    >>> from insideLLMs.caching import create_cache, AsyncCacheAdapter
    >>> import asyncio
    >>> cache = create_cache(max_size=100)
    >>> async_cache = AsyncCacheAdapter(cache)
    >>> async def cache_operation():
    ...     # Async set operation
    ...     await async_cache.set("async_key", {"data": "value"})
    ...     # Async get operation
    ...     result = await async_cache.get("async_key")
    ...     return result.value
    >>> # Run the async operation
    >>> value = asyncio.run(cache_operation())
    >>> print(value)
    {'data': 'value'}

Generating cache keys for custom caching:

    >>> from insideLLMs.caching import generate_cache_key, get_cache_key
    >>> # Basic key generation
    >>> key1 = generate_cache_key(prompt="What is AI?")
    >>> print(f"Key length: {len(key1)}")  # SHA-256 hash
    Key length: 64
    >>> # Key with model and parameters
    >>> key2 = generate_cache_key(
    ...     prompt="What is AI?",
    ...     model="gpt-4",
    ...     params={"temperature": 0.7, "max_tokens": 100}
    ... )
    >>> # Same inputs produce same key
    >>> key3 = generate_cache_key(
    ...     prompt="What is AI?",
    ...     model="gpt-4",
    ...     params={"temperature": 0.7, "max_tokens": 100}
    ... )
    >>> print(f"Keys match: {key2 == key3}")
    Keys match: True
    >>> # Different parameters produce different keys
    >>> key4 = generate_cache_key(
    ...     prompt="What is AI?",
    ...     model="gpt-4",
    ...     params={"temperature": 1.0}
    ... )
    >>> print(f"Keys differ: {key2 != key4}")
    Keys differ: True

Response deduplication for identifying duplicate responses:

    >>> from insideLLMs.caching import ResponseDeduplicator
    >>> dedup = ResponseDeduplicator(similarity_threshold=0.9)
    >>> # Add first unique response
    >>> is_dup, idx = dedup.add("prompt1", "This is a unique response.")
    >>> print(f"Is duplicate: {is_dup}")
    Is duplicate: False
    >>> # Add similar response
    >>> is_dup, idx = dedup.add("prompt2", "This is a unique response.")
    >>> print(f"Is duplicate: {is_dup}, matches index: {idx}")
    Is duplicate: True, matches index: 0
    >>> # Add different response
    >>> is_dup, idx = dedup.add("prompt3", "A completely different answer.")
    >>> print(f"Is duplicate: {is_dup}")
    Is duplicate: False
    >>> # Get all unique responses
    >>> unique = dedup.get_unique_responses()
    >>> print(f"Unique responses: {len(unique)}")
    Unique responses: 2

Using cached_response for one-shot caching:

    >>> from insideLLMs.caching import cached_response, create_prompt_cache
    >>> def generate_llm_response(prompt):
    ...     # Simulates LLM API call
    ...     return f"Generated response for: {prompt}"
    >>> cache = create_prompt_cache()
    >>> # First call generates the response
    >>> response, was_cached = cached_response(
    ...     "Hello",
    ...     generate_llm_response,
    ...     cache
    ... )
    >>> print(f"Response: {response}, Cached: {was_cached}")
    Response: Generated response for: Hello, Cached: False
    >>> # Second call returns cached response
    >>> response, was_cached = cached_response(
    ...     "Hello",
    ...     generate_llm_response,
    ...     cache
    ... )
    >>> print(f"Response: {response}, Cached: {was_cached}")
    Response: Generated response for: Hello, Cached: True

Cache statistics monitoring and optimization:

    >>> from insideLLMs.caching import create_cache
    >>> cache = create_cache(max_size=100)
    >>> # Perform some operations
    >>> cache.set("key1", "value1")
    >>> cache.set("key2", "value2")
    >>> _ = cache.get("key1")  # Hit
    >>> _ = cache.get("key1")  # Hit
    >>> _ = cache.get("key3")  # Miss
    >>> # Get statistics
    >>> stats = cache.get_stats()
    >>> print(f"Hits: {stats.hits}, Misses: {stats.misses}")
    Hits: 2, Misses: 1
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    Hit rate: 66.7%
    >>> print(f"Entry count: {stats.entry_count}")
    Entry count: 2
    >>> # Monitor for optimization
    >>> if stats.hit_rate < 0.5:
    ...     print("Consider increasing cache size or adjusting TTL")
    >>> if stats.evictions > stats.entry_count * 2:
    ...     print("High eviction rate - cache may be undersized")

Working with cache entries and metadata:

    >>> from insideLLMs.caching import create_cache
    >>> from datetime import datetime, timedelta
    >>> cache = create_cache()
    >>> # Store with custom metadata
    >>> entry = cache.set(
    ...     key="user_query",
    ...     value="The answer is 42",
    ...     metadata={
    ...         "user_id": "user123",
    ...         "model": "gpt-4",
    ...         "tokens_used": 150
    ...     }
    ... )
    >>> print(f"Entry created at: {entry.created_at}")
    >>> print(f"Entry size: {entry.size_bytes} bytes")
    >>> # Check entry status
    >>> print(f"Is expired: {entry.is_expired()}")
    Is expired: False
    >>> # Retrieve with entry details
    >>> result = cache.get("user_query")
    >>> if result.entry:
    ...     print(f"Access count: {result.entry.access_count}")
    ...     print(f"Metadata: {result.entry.metadata}")

Thread-safe cache usage in concurrent applications:

    >>> from insideLLMs.caching import create_cache
    >>> import threading
    >>> cache = create_cache(max_size=1000)
    >>> def worker(worker_id):
    ...     for i in range(100):
    ...         key = f"worker_{worker_id}_item_{i}"
    ...         cache.set(key, f"value_{i}")
    ...         _ = cache.get(key)
    >>> # Create and run multiple threads
    >>> threads = [
    ...     threading.Thread(target=worker, args=(i,))
    ...     for i in range(4)
    ... ]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()
    >>> # Cache remains consistent
    >>> stats = cache.get_stats()
    >>> print(f"Total entries: {stats.entry_count}")

Notes
-----
- All cache implementations are thread-safe, using locks to protect internal state.
- TTL (Time To Live) is specified in seconds. A TTL of None means entries never expire.
- The default cache strategy is LRU (Least Recently Used), which is generally suitable
  for most workloads with temporal locality.
- Semantic similarity matching in PromptCache uses word overlap (Jaccard index).
  For production use with complex prompts, consider using embedding-based similarity.
- Cache entries can include arbitrary metadata for tracking custom information like
  user IDs, model versions, or token counts.
- The default hash algorithm is SHA-256, which provides good collision resistance.
  Use MD5 or SHA-1 only when performance is critical and security is not a concern.
- Cache warming is most effective when you know the common prompts ahead of time.
  Consider analyzing usage patterns to identify candidates for warming.

Performance Considerations
--------------------------
- In-memory caches (StrategyCache) are fastest but limited by available RAM.
- For large-scale caching, consider using DiskCache (from caching_unified) or
  external caching systems like Redis.
- The SIZE eviction strategy is most efficient for memory-constrained environments
  when entries have varying sizes.
- LFU strategy has slightly higher overhead than LRU due to access count tracking.
- Consider using cache namespaces to isolate different workloads and prevent
  cache pollution between unrelated operations.

See Also
--------
caching_unified : Full implementation of caching components with additional
    backends like InMemoryCache and DiskCache.
types.ModelResponse : Response type used with CachedModel wrapper.

References
----------
.. [1] LRU Cache implementation: https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU
.. [2] Jaccard similarity: https://en.wikipedia.org/wiki/Jaccard_index
"""

import warnings as _warnings

_warnings.warn(
    "The 'insideLLMs.caching' module is deprecated. "
    "Import from 'insideLLMs.caching_unified' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from unified module for backward compatibility
from insideLLMs.caching_unified import (
    AsyncCacheAdapter,
    # Configuration
    CacheConfig,
    # Data Classes
    CacheEntry,
    CacheLookupResult,
    CacheNamespace,
    CacheScope,
    CacheStats,
    CacheStatus,
    # Enums
    CacheStrategy,
    # Utilities
    CacheWarmer,
    MemoizedFunction,
    PromptCache,
    ResponseDeduplicator,
    cached_response,
    # Convenience Functions
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    # Key Generation
    generate_cache_key,
    get_cache_key,
    # Decorator
    memoize,
)
from insideLLMs.caching_unified import (
    # Cache Implementations - BaseCache is StrategyCache in unified
    StrategyCache as BaseCache,
)

# For backward compatibility with old module structure
__all__ = [
    # Enums
    "CacheStrategy",
    "CacheStatus",
    "CacheScope",
    # Configuration
    "CacheConfig",
    # Data Classes
    "CacheEntry",
    "CacheStats",
    "CacheLookupResult",
    # Key Generation
    "generate_cache_key",
    # Cache Implementations
    "BaseCache",
    "PromptCache",
    # Utilities
    "CacheWarmer",
    "MemoizedFunction",
    "CacheNamespace",
    "ResponseDeduplicator",
    "AsyncCacheAdapter",
    # Convenience Functions
    "create_cache",
    "create_prompt_cache",
    "create_cache_warmer",
    "create_namespace",
    "get_cache_key",
    "cached_response",
    # Decorator
    "memoize",
]
