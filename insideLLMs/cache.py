"""
Caching Layer for Model Responses
=================================

This module provides a backward-compatible interface to the insideLLMs caching
infrastructure. It re-exports essential caching components from the unified
caching module (``caching_unified.py``), making it easy to add caching to LLM
applications with minimal code changes.

Overview
--------
The caching layer is designed to reduce redundant LLM API calls by storing
and retrieving responses based on deterministic cache keys. This can
significantly reduce latency and API costs for applications with repeated
or similar queries.

Key components exported from this module:

- **BaseCache**: Abstract base class defining the cache interface contract.
- **InMemoryCache**: Fast, thread-safe in-memory cache with LRU eviction.
- **DiskCache**: SQLite-based persistent cache for durability across restarts.
- **CachedModel**: Transparent wrapper that adds caching to any model.
- **CacheEntry**: Data class representing a single cached item with metadata.
- **CacheStats**: Statistics for monitoring cache performance.
- **cached**: Decorator for caching function results.
- **generate_cache_key**: Creates deterministic keys from prompts and parameters.

Architecture
------------
This module follows the facade pattern, providing a simplified interface to
the more comprehensive ``caching_unified`` module. For advanced features like
multiple eviction strategies (LFU, FIFO, TTL, SIZE), prompt caching with
semantic similarity, cache warming, and memoization, import directly from
``caching_unified``.

Thread Safety
-------------
All cache implementations use appropriate locking mechanisms (``threading.RLock``
for in-memory, SQLite's built-in thread safety for disk) to ensure safe
concurrent access from multiple threads.

Examples
--------
Basic in-memory caching for function results:

    >>> from insideLLMs.cache import InMemoryCache, cached
    >>> cache = InMemoryCache(max_size=100, default_ttl=3600)
    >>> cache.set("user_123_profile", {"name": "Alice", "role": "admin"})
    >>> profile = cache.get("user_123_profile")
    >>> print(profile["name"])
    Alice

Using the ``cached`` decorator for automatic caching:

    >>> from insideLLMs.cache import cached, InMemoryCache
    >>> @cached(cache=InMemoryCache(), ttl=300)
    ... def fetch_user_data(user_id: str) -> dict:
    ...     # Expensive operation (API call, database query, etc.)
    ...     return {"id": user_id, "data": "..."}
    >>> # First call computes the result
    >>> data1 = fetch_user_data("user_123")
    >>> # Second call returns cached result
    >>> data2 = fetch_user_data("user_123")

Persistent caching with DiskCache:

    >>> from insideLLMs.cache import DiskCache
    >>> from pathlib import Path
    >>> cache = DiskCache(
    ...     path=Path("/tmp/my_app_cache.db"),
    ...     max_size_mb=50,
    ...     default_ttl=86400  # 24 hours
    ... )
    >>> cache.set("model_response", {"text": "Hello, World!", "tokens": 5})
    >>> # Response persists across application restarts

Wrapping a model with automatic caching:

    >>> from insideLLMs.cache import CachedModel, InMemoryCache
    >>> class MyLLM:
    ...     def generate(self, prompt, temperature=0.0, max_tokens=None):
    ...         # Simulated API call
    ...         return ModelResponse(content=f"Response: {prompt}")
    >>> llm = MyLLM()
    >>> cached_llm = CachedModel(llm, cache=InMemoryCache(max_size=500))
    >>> # First call hits the API
    >>> response1 = cached_llm.generate("Explain caching", temperature=0)
    >>> # Second identical call returns cached response
    >>> response2 = cached_llm.generate("Explain caching", temperature=0)

Generating cache keys for custom caching logic:

    >>> from insideLLMs.cache import generate_cache_key
    >>> key = generate_cache_key(
    ...     prompt="What is machine learning?",
    ...     model="gpt-4",
    ...     params={"temperature": 0.0, "max_tokens": 100}
    ... )
    >>> len(key)  # SHA-256 hash
    64
    >>> # Same inputs always produce the same key
    >>> key2 = generate_cache_key(
    ...     prompt="What is machine learning?",
    ...     model="gpt-4",
    ...     params={"temperature": 0.0, "max_tokens": 100}
    ... )
    >>> key == key2
    True

Using global default cache for application-wide caching:

    >>> from insideLLMs.cache import (
    ...     get_default_cache,
    ...     set_default_cache,
    ...     clear_default_cache,
    ...     InMemoryCache,
    ... )
    >>> # Configure application-wide cache
    >>> set_default_cache(InMemoryCache(max_size=10000, default_ttl=7200))
    >>> # Access from anywhere in the application
    >>> cache = get_default_cache()
    >>> cache.set("shared_key", "shared_value")
    >>> # Clear when needed (e.g., on config change)
    >>> clear_default_cache()

Monitoring cache performance:

    >>> from insideLLMs.cache import InMemoryCache
    >>> cache = InMemoryCache(max_size=100)
    >>> cache.set("key1", "value1")
    >>> cache.set("key2", "value2")
    >>> _ = cache.get("key1")  # Hit
    >>> _ = cache.get("key3")  # Miss
    >>> stats = cache.stats()
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    Hit rate: 50.0%
    >>> print(f"Entries: {stats.entry_count}")
    Entries: 2

Notes
-----
- **Deterministic caching**: For reliable cache hits with LLMs, use
  ``temperature=0`` to ensure deterministic outputs. Non-deterministic
  parameters may produce different responses for the same prompt.

- **Key generation**: Cache keys are generated using SHA-256 by default,
  combining the prompt, model, and parameters into a unique identifier.

- **Memory management**: InMemoryCache uses LRU (Least Recently Used)
  eviction when the cache reaches max_size. DiskCache manages size by
  evicting expired entries first, then LRU entries.

- **Serialization**: Values are stored as JSON. Ensure cached values are
  JSON-serializable (strings, numbers, lists, dicts, None, bools).

- **TTL (Time To Live)**: Expired entries are removed lazily on access
  or during eviction. Set ``default_ttl=None`` to disable expiration.

See Also
--------
insideLLMs.caching_unified : Full caching module with advanced features.
    Includes StrategyCache (multiple eviction strategies), PromptCache
    (semantic similarity matching), CacheWarmer (preloading), CacheNamespace
    (managing multiple caches), and MemoizedFunction (function memoization).

insideLLMs.types.ModelResponse : Response type used by CachedModel.

References
----------
.. [1] LRU Cache: https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU
.. [2] SQLite: https://www.sqlite.org/docs.html
"""

# Re-export from unified module for backward compatibility
from insideLLMs.caching_unified import (
    BaseCacheABC as BaseCache,
)
from insideLLMs.caching_unified import (
    CachedModel,
    CacheEntry,
    CacheStats,
    DiskCache,
    InMemoryCache,
    cached,
    clear_default_cache,
    generate_cache_key,
    get_default_cache,
    set_default_cache,
)

# =============================================================================
# Class Docstrings (Attached to Re-exported Classes)
# =============================================================================

# Note: The classes below are re-exported from caching_unified.py.
# Their docstrings are defined in that module. This section provides
# additional documentation for users importing from this module.

BaseCache.__doc__ = """
Abstract base class defining the cache interface contract.

BaseCache (alias for BaseCacheABC) provides the standard interface that all
cache implementations must follow. It defines methods for getting, setting,
deleting, and clearing cache entries, as well as checking existence and
retrieving statistics.

This class uses Python's ABC (Abstract Base Class) mechanism to enforce
that subclasses implement all required methods.

Parameters
----------
None (abstract class, not instantiated directly).

Attributes
----------
None (interface only).

Methods
-------
get(key: str) -> Optional[T]
    Retrieve a value from the cache. Returns None if not found or expired.

set(key: str, value: T, ttl: Optional[int] = None, metadata: Optional[dict] = None) -> None
    Store a value in the cache with optional TTL and metadata.

delete(key: str) -> bool
    Remove an entry from the cache. Returns True if entry existed.

clear() -> None
    Remove all entries from the cache.

stats() -> CacheStats
    Get current cache statistics.

has(key: str) -> bool
    Check if a key exists in the cache (convenience method).

Examples
--------
Creating a custom cache implementation:

    >>> from insideLLMs.cache import BaseCache, CacheStats
    >>> from typing import Optional, Any
    >>> class MyCustomCache(BaseCache):
    ...     def __init__(self):
    ...         self._data = {}
    ...
    ...     def get(self, key: str) -> Optional[Any]:
    ...         return self._data.get(key)
    ...
    ...     def set(self, key: str, value: Any, ttl: Optional[int] = None,
    ...             metadata: Optional[dict] = None) -> None:
    ...         self._data[key] = value
    ...
    ...     def delete(self, key: str) -> bool:
    ...         if key in self._data:
    ...             del self._data[key]
    ...             return True
    ...         return False
    ...
    ...     def clear(self) -> None:
    ...         self._data.clear()
    ...
    ...     def stats(self) -> CacheStats:
    ...         return CacheStats(entry_count=len(self._data))

Using built-in implementations (recommended):

    >>> from insideLLMs.cache import InMemoryCache, DiskCache
    >>> # In-memory cache for single-process applications
    >>> mem_cache = InMemoryCache(max_size=1000)
    >>> # Disk cache for persistence
    >>> disk_cache = DiskCache(max_size_mb=100)

Type parameter T for generic caching:

    >>> from typing import TypeVar
    >>> T = TypeVar('T')
    >>> # BaseCache[str] would indicate string values
    >>> # BaseCache[dict] would indicate dictionary values

See Also
--------
InMemoryCache : In-memory implementation with LRU eviction.
DiskCache : SQLite-based persistent implementation.
caching_unified.StrategyCache : Full-featured cache with multiple strategies.
"""

InMemoryCache.__doc__ = """
Fast, thread-safe in-memory cache with LRU eviction.

InMemoryCache stores entries in a Python dictionary with automatic
eviction of least-recently-used entries when the cache reaches its
maximum size. All operations are protected by a reentrant lock for
thread safety.

This is the recommended cache for single-process applications where
persistence is not required and cache size fits in memory.

Parameters
----------
max_size : int, default=1000
    Maximum number of entries the cache can hold. When this limit is
    reached, the least recently used entry is evicted to make room.

default_ttl : Optional[int], default=None
    Default time-to-live in seconds for cache entries. If None,
    entries never expire unless explicitly set. Can be overridden
    per-entry when calling set().

Attributes
----------
_cache : dict[str, dict]
    Internal dictionary storing cache entries.

_max_size : int
    Maximum number of entries allowed.

_default_ttl : Optional[int]
    Default TTL for entries.

_stats : CacheStats
    Cache statistics tracker.

_lock : threading.RLock
    Reentrant lock for thread safety.

Examples
--------
Basic usage:

    >>> from insideLLMs.cache import InMemoryCache
    >>> cache = InMemoryCache(max_size=100, default_ttl=3600)
    >>> # Store a value
    >>> cache.set("user:123", {"name": "Alice", "age": 30})
    >>> # Retrieve the value
    >>> user = cache.get("user:123")
    >>> print(user["name"])
    Alice

With TTL expiration:

    >>> import time
    >>> cache = InMemoryCache(default_ttl=2)  # 2 second TTL
    >>> cache.set("temp_key", "temp_value")
    >>> cache.get("temp_key")  # Returns value
    'temp_value'
    >>> time.sleep(3)  # Wait for expiration
    >>> cache.get("temp_key")  # Returns None (expired)

LRU eviction behavior:

    >>> cache = InMemoryCache(max_size=3)
    >>> cache.set("a", 1)
    >>> cache.set("b", 2)
    >>> cache.set("c", 3)
    >>> # Cache is full, adding new entry evicts oldest
    >>> _ = cache.get("a")  # Access 'a' to make it recently used
    >>> cache.set("d", 4)  # Evicts 'b' (least recently used)
    >>> cache.get("b")  # Returns None (evicted)
    >>> cache.get("a")  # Returns 1 (still present)
    1

Thread-safe access:

    >>> import threading
    >>> cache = InMemoryCache(max_size=1000)
    >>> def writer(key_prefix, count):
    ...     for i in range(count):
    ...         cache.set(f"{key_prefix}:{i}", i)
    >>> threads = [
    ...     threading.Thread(target=writer, args=("thread1", 100)),
    ...     threading.Thread(target=writer, args=("thread2", 100)),
    ... ]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()
    >>> # All entries safely stored

Checking statistics:

    >>> cache = InMemoryCache(max_size=100)
    >>> cache.set("key1", "value1")
    >>> _ = cache.get("key1")  # Hit
    >>> _ = cache.get("key2")  # Miss
    >>> stats = cache.stats()
    >>> print(f"Hits: {stats.hits}, Misses: {stats.misses}")
    Hits: 1, Misses: 1

Per-entry TTL override:

    >>> cache = InMemoryCache(default_ttl=3600)  # 1 hour default
    >>> # This entry expires in 60 seconds, not 1 hour
    >>> cache.set("short_lived", "data", ttl=60)
    >>> # This entry uses the default 1 hour TTL
    >>> cache.set("normal", "data")

With metadata:

    >>> cache = InMemoryCache()
    >>> cache.set(
    ...     "model_response",
    ...     "The answer is 42.",
    ...     metadata={"model": "gpt-4", "tokens": 5}
    ... )

See Also
--------
DiskCache : Persistent cache using SQLite.
caching_unified.StrategyCache : Cache with configurable eviction strategies.
"""

DiskCache.__doc__ = """
SQLite-based persistent disk cache with size-based management.

DiskCache stores cache entries in a SQLite database, providing durability
across application restarts. It manages cache size by file size (megabytes)
rather than entry count, making it suitable for caches with varying entry
sizes.

The cache uses thread-local database connections for safe concurrent access
from multiple threads, and SQLite's built-in transaction support for data
integrity.

Parameters
----------
path : Optional[Union[str, Path]], default=None
    Path to the SQLite database file. If None, uses the default location
    at ``~/.cache/insideLLMs/response_cache.db``.

max_size_mb : int, default=100
    Maximum cache size in megabytes. When the database file exceeds this
    size, entries are evicted (expired first, then LRU).

default_ttl : Optional[int], default=None
    Default time-to-live in seconds for cache entries. If None,
    entries never expire unless explicitly set.

Attributes
----------
_path : Path
    Path to the SQLite database file.

_max_size_bytes : int
    Maximum cache size in bytes.

_default_ttl : Optional[int]
    Default TTL for entries.

_local : threading.local
    Thread-local storage for database connections.

_stats : CacheStats
    Cache statistics tracker.

Examples
--------
Basic usage with default path:

    >>> from insideLLMs.cache import DiskCache
    >>> cache = DiskCache(max_size_mb=50, default_ttl=86400)  # 24h TTL
    >>> cache.set("user:123", {"name": "Alice", "preferences": {...}})
    >>> # Value persists across restarts
    >>> user = cache.get("user:123")

Custom database location:

    >>> from pathlib import Path
    >>> cache = DiskCache(
    ...     path=Path("/var/cache/myapp/llm_cache.db"),
    ...     max_size_mb=500,
    ...     default_ttl=604800  # 1 week
    ... )

Export and import for backup/migration:

    >>> cache = DiskCache(path="/tmp/cache.db")
    >>> cache.set("key1", {"data": "value1"})
    >>> cache.set("key2", {"data": "value2"})
    >>> # Export to JSON
    >>> count = cache.export_to_file("/tmp/cache_backup.json")
    >>> print(f"Exported {count} entries")
    Exported 2 entries
    >>> # Import on another system
    >>> new_cache = DiskCache(path="/tmp/new_cache.db")
    >>> imported = new_cache.import_from_file("/tmp/cache_backup.json")
    >>> print(f"Imported {imported} entries")
    Imported 2 entries

Handling large responses:

    >>> cache = DiskCache(max_size_mb=1000)  # 1 GB cache
    >>> # Store large model outputs
    >>> cache.set(
    ...     "document_summary",
    ...     {
    ...         "summary": "..." * 10000,  # Large text
    ...         "chapters": [...],
    ...         "metadata": {...}
    ...     },
    ...     ttl=86400 * 7  # Keep for 1 week
    ... )

Automatic eviction:

    >>> cache = DiskCache(max_size_mb=10)  # Small cache for demo
    >>> # When cache exceeds 10 MB:
    >>> # 1. Expired entries are removed first
    >>> # 2. Then LRU entries are removed until size < 8 MB (80%)
    >>> # 3. VACUUM is run to reclaim space

Concurrent access from multiple threads:

    >>> import threading
    >>> cache = DiskCache()
    >>> def store_data(thread_id):
    ...     for i in range(100):
    ...         cache.set(f"thread_{thread_id}_key_{i}", f"value_{i}")
    >>> threads = [
    ...     threading.Thread(target=store_data, args=(i,))
    ...     for i in range(4)
    ... ]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()
    >>> # All data safely stored

Checking cache statistics:

    >>> cache = DiskCache()
    >>> cache.set("a", 1)
    >>> cache.set("b", 2)
    >>> _ = cache.get("a")  # Hit
    >>> _ = cache.get("c")  # Miss
    >>> stats = cache.stats()
    >>> print(f"Entries: {stats.entry_count}")
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")

Notes
-----
- Database indexes are created on ``expires_at`` and ``last_accessed``
  columns for efficient expiration and LRU queries.

- The cache runs ``VACUUM`` after eviction to reclaim disk space.

- For very high-throughput applications, consider using InMemoryCache
  with periodic persistence instead.

See Also
--------
InMemoryCache : In-memory cache for non-persistent use cases.
caching_unified.StrategyCache : More eviction strategy options.
"""

CacheEntry.__doc__ = """
A single cache entry with metadata and access tracking.

CacheEntry is a dataclass that represents a cached value along with all
associated metadata needed for cache management, including timestamps,
access counts, and size information. It is used internally by cache
implementations and can be accessed for debugging or monitoring.

Parameters
----------
key : str
    Unique identifier for this cache entry. Typically a hash of the
    request parameters generated by ``generate_cache_key()``.

value : Any
    The cached value. Can be any JSON-serializable type.

created_at : datetime, default=datetime.now()
    Timestamp when this entry was created. Auto-set if not provided.

expires_at : Optional[datetime], default=None
    Timestamp when this entry expires. None means no expiration.

access_count : int, default=0
    Number of times this entry has been accessed. Used by LFU strategy.

last_accessed : datetime, default=datetime.now()
    Timestamp of most recent access. Used by LRU strategy.

size_bytes : int, default=0
    Estimated size of the cached value in bytes. Auto-calculated if 0.

metadata : dict[str, Any], default={}
    Additional metadata associated with this entry.

Attributes
----------
All parameters are also available as attributes.

Examples
--------
Creating a cache entry manually:

    >>> from insideLLMs.cache import CacheEntry
    >>> from datetime import datetime, timedelta
    >>> entry = CacheEntry(
    ...     key="abc123def456",
    ...     value={"response": "Hello, World!", "tokens": 5},
    ...     expires_at=datetime.now() + timedelta(hours=1),
    ...     metadata={"model": "gpt-4", "prompt": "Say hello"}
    ... )
    >>> print(entry.key[:8])
    abc123de

Checking expiration:

    >>> entry = CacheEntry(
    ...     key="test",
    ...     value="data",
    ...     expires_at=datetime.now() + timedelta(seconds=1)
    ... )
    >>> entry.is_expired()
    False
    >>> import time; time.sleep(2)
    >>> entry.is_expired()
    True

Entry with no expiration:

    >>> entry = CacheEntry(key="permanent", value="forever")
    >>> entry.expires_at is None
    True
    >>> entry.is_expired()  # Never expires
    False

Tracking access patterns:

    >>> entry = CacheEntry(key="popular", value="frequently accessed")
    >>> print(f"Initial access count: {entry.access_count}")
    Initial access count: 0
    >>> entry.touch()  # Simulate access
    >>> entry.touch()
    >>> entry.touch()
    >>> print(f"After accesses: {entry.access_count}")
    After accesses: 3

Size estimation:

    >>> entry = CacheEntry(key="data", value={"large": "x" * 1000})
    >>> print(f"Estimated size: {entry.size_bytes} bytes")
    Estimated size: 1017 bytes

Converting to dictionary:

    >>> entry = CacheEntry(key="k1", value="v1", metadata={"source": "api"})
    >>> d = entry.to_dict()
    >>> print(d["key"])
    k1
    >>> print(d["metadata"])
    {'source': 'api'}

Accessing entry from cache lookup:

    >>> from insideLLMs.cache import InMemoryCache
    >>> cache = InMemoryCache()
    >>> cache.set("my_key", "my_value", metadata={"info": "test"})
    >>> # Note: InMemoryCache.get() returns the value directly
    >>> # For CacheEntry access, use StrategyCache from caching_unified

See Also
--------
CacheStats : Aggregate statistics for the entire cache.
caching_unified.CacheLookupResult : Full lookup result including entry.
"""

CacheStats.__doc__ = """
Cache statistics for monitoring and optimization.

CacheStats is a dataclass that provides comprehensive metrics about
cache performance, including hit rates, eviction counts, and size
information. Use these statistics to tune cache configuration and
identify performance issues.

Parameters
----------
hits : int, default=0
    Number of successful cache lookups (key found and not expired).

misses : int, default=0
    Number of cache lookups that did not find a valid entry.

evictions : int, default=0
    Number of entries removed due to capacity constraints.

expirations : int, default=0
    Number of entries that expired (TTL exceeded).

total_size_bytes : int, default=0
    Total estimated size of all cached values in bytes.

entry_count : int, default=0
    Current number of entries in the cache.

oldest_entry : Optional[datetime], default=None
    Timestamp of the oldest cache entry, or None if cache is empty.

newest_entry : Optional[datetime], default=None
    Timestamp of the newest cache entry, or None if cache is empty.

avg_access_count : float, default=0.0
    Average number of accesses per entry.

hit_rate : float, default=0.0
    Ratio of hits to total lookups. Range 0.0 to 1.0.

Examples
--------
Getting cache statistics:

    >>> from insideLLMs.cache import InMemoryCache
    >>> cache = InMemoryCache(max_size=100)
    >>> # Simulate some cache operations
    >>> cache.set("key1", "value1")
    >>> cache.set("key2", "value2")
    >>> _ = cache.get("key1")  # Hit
    >>> _ = cache.get("key1")  # Hit
    >>> _ = cache.get("key3")  # Miss
    >>> stats = cache.stats()
    >>> print(f"Hits: {stats.hits}, Misses: {stats.misses}")
    Hits: 2, Misses: 1
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    Hit rate: 66.7%

Monitoring cache health:

    >>> stats = cache.stats()
    >>> if stats.hit_rate < 0.5:
    ...     print("Warning: Low hit rate - consider reviewing cache keys")
    >>> if stats.evictions > stats.entry_count * 2:
    ...     print("Warning: High eviction rate - consider increasing max_size")

Exporting statistics for logging:

    >>> import json
    >>> stats = cache.stats()
    >>> stats_dict = stats.to_dict()
    >>> # Safe for JSON serialization (datetimes converted to ISO strings)
    >>> print(json.dumps(stats_dict, indent=2))

Comparing statistics over time:

    >>> # Take baseline measurement
    >>> baseline = cache.stats()
    >>> baseline_hits = baseline.hits
    >>> baseline_misses = baseline.misses
    >>> # ... perform operations ...
    >>> # Compare after operations
    >>> current = cache.stats()
    >>> new_hits = current.hits - baseline_hits
    >>> new_misses = current.misses - baseline_misses
    >>> period_hit_rate = new_hits / (new_hits + new_misses) if (new_hits + new_misses) > 0 else 0
    >>> print(f"Period hit rate: {period_hit_rate:.1%}")

Creating manual statistics (for testing):

    >>> from insideLLMs.cache import CacheStats
    >>> from datetime import datetime
    >>> stats = CacheStats(
    ...     hits=1000,
    ...     misses=200,
    ...     evictions=50,
    ...     entry_count=500,
    ...     hit_rate=0.833
    ... )
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    Hit rate: 83.3%

See Also
--------
CacheEntry : Individual cache entry with its own metadata.
caching_unified.StrategyCache.get_stats : Get stats from strategy cache.
"""

CachedModel.__doc__ = """
Wrapper that adds transparent caching to any model.

CachedModel wraps an existing model object and intercepts calls to its
``generate()`` method, checking a cache before calling the underlying
model and storing responses in the cache for future use. This is the
recommended way to add caching to LLM applications.

By default, only deterministic requests (temperature=0) are cached,
since non-deterministic outputs would return potentially unexpected
cached responses.

Parameters
----------
model : Any
    The underlying model to wrap. Must have a ``generate()`` method with
    signature: ``generate(prompt, temperature=0.0, max_tokens=None, **kwargs)``.

cache : Optional[BaseCacheABC], default=None
    Cache backend to use. If None, a new InMemoryCache is created.

cache_only_deterministic : bool, default=True
    If True, only cache requests with temperature=0. Set to False to
    cache all requests (use with caution).

Attributes
----------
model : Any
    The underlying model (read-only property).

cache : BaseCacheABC
    The cache backend (read-only property).

Examples
--------
Basic usage with a custom model:

    >>> from insideLLMs.cache import CachedModel, InMemoryCache
    >>> from insideLLMs.types import ModelResponse
    >>> class SimpleLLM:
    ...     model_id = "simple-llm-v1"
    ...     def generate(self, prompt, temperature=0.0, max_tokens=None):
    ...         # Simulate expensive API call
    ...         return ModelResponse(content=f"Response to: {prompt[:20]}...")
    >>> llm = SimpleLLM()
    >>> cached_llm = CachedModel(llm, cache=InMemoryCache(max_size=1000))
    >>> # First call - hits the model
    >>> resp1 = cached_llm.generate("What is Python?", temperature=0)
    >>> # Second identical call - returns cached response
    >>> resp2 = cached_llm.generate("What is Python?", temperature=0)

With DiskCache for persistence:

    >>> from insideLLMs.cache import CachedModel, DiskCache
    >>> from pathlib import Path
    >>> cache = DiskCache(
    ...     path=Path("/tmp/model_cache.db"),
    ...     max_size_mb=500,
    ...     default_ttl=86400  # 24 hours
    ... )
    >>> cached_llm = CachedModel(model, cache=cache)
    >>> # Cached responses persist across restarts

Temperature handling:

    >>> cached_llm = CachedModel(model, cache_only_deterministic=True)
    >>> # This is cached (temperature=0)
    >>> resp1 = cached_llm.generate("Hello", temperature=0)
    >>> # This is NOT cached (temperature>0, non-deterministic)
    >>> resp2 = cached_llm.generate("Hello", temperature=0.7)
    >>> # Same prompt, different temperature - different (uncached) response

Caching non-deterministic requests (use with caution):

    >>> cached_llm = CachedModel(model, cache_only_deterministic=False)
    >>> # Now ALL requests are cached, even non-deterministic ones
    >>> resp = cached_llm.generate("Write a poem", temperature=1.0)
    >>> # Warning: Second call returns same cached response!

Accessing the underlying model:

    >>> cached_llm = CachedModel(model)
    >>> # Access model attributes directly
    >>> print(cached_llm.model.model_id)
    >>> # Or through delegation
    >>> print(cached_llm.model_id)  # Delegated to underlying model

Monitoring cache effectiveness:

    >>> cached_llm = CachedModel(model, cache=InMemoryCache())
    >>> # Make some requests...
    >>> for prompt in ["Hello", "Hello", "World", "Hello"]:
    ...     _ = cached_llm.generate(prompt, temperature=0)
    >>> stats = cached_llm.cache.stats()
    >>> print(f"Cache hit rate: {stats.hit_rate:.1%}")
    Cache hit rate: 50.0%  # 2 unique prompts, 4 total calls

With custom generation parameters:

    >>> resp = cached_llm.generate(
    ...     "Explain quantum computing",
    ...     temperature=0,
    ...     max_tokens=500,
    ...     top_p=0.9,  # Additional kwargs passed to model
    ...     stop=["\\n\\n"]
    ... )

Notes
-----
- The cache key includes: model_id, prompt, temperature, max_tokens,
  and any additional kwargs. Different parameters = different cache key.

- Attribute access is delegated to the underlying model via __getattr__,
  so ``cached_llm.some_attr`` returns ``model.some_attr``.

- The ModelResponse is serialized/deserialized using dataclasses.asdict()
  and the ModelResponse constructor.

See Also
--------
InMemoryCache : Recommended cache for single-process applications.
DiskCache : Recommended cache for persistent caching.
generate_cache_key : Function used internally for key generation.
caching_unified.generate_model_cache_key : Specialized key generation.
"""

cached.__doc__ = """
Decorator to cache function results using a simple cache backend.

The ``cached`` decorator wraps a function and caches its return values
based on the function arguments. Subsequent calls with the same arguments
return the cached result instead of re-executing the function.

This is useful for expensive computations, API calls, or any function
where the same inputs should produce the same outputs.

Parameters
----------
cache : Optional[BaseCacheABC], default=None
    Cache backend to use. If None, a new InMemoryCache is created.

ttl : Optional[int], default=None
    Time-to-live in seconds for cached entries. If None, uses the
    cache's default TTL.

key_fn : Optional[Callable[..., str]], default=None
    Custom function to generate cache keys from arguments. If None,
    a default key generator using SHA-256 is used.

Returns
-------
Callable
    Decorated function with caching behavior.

Examples
--------
Basic usage:

    >>> from insideLLMs.cache import cached
    >>> @cached()
    ... def expensive_computation(x: int, y: int) -> int:
    ...     print("Computing...")
    ...     return x ** y
    >>> expensive_computation(2, 10)  # Prints "Computing..."
    Computing...
    1024
    >>> expensive_computation(2, 10)  # Returns cached result (no print)
    1024

With custom cache and TTL:

    >>> from insideLLMs.cache import cached, InMemoryCache
    >>> shared_cache = InMemoryCache(max_size=500)
    >>> @cached(cache=shared_cache, ttl=60)  # 60 second TTL
    ... def fetch_user_profile(user_id: str) -> dict:
    ...     # Simulate API call
    ...     return {"id": user_id, "name": f"User {user_id}"}
    >>> profile = fetch_user_profile("123")
    >>> # Cached for 60 seconds

With custom key function:

    >>> def my_key_fn(user_id: str, include_details: bool = True) -> str:
    ...     # Only cache based on user_id, ignore include_details
    ...     return f"user_profile:{user_id}"
    >>> @cached(key_fn=my_key_fn)
    ... def get_user(user_id: str, include_details: bool = True) -> dict:
    ...     return {"id": user_id, "details": include_details}
    >>> get_user("123", include_details=True)
    >>> get_user("123", include_details=False)  # Returns same cached result!

Caching with different argument types:

    >>> @cached()
    ... def process_data(items: tuple, config: dict) -> str:
    ...     return f"Processed {len(items)} items"
    >>> # Note: Arguments must be JSON-serializable for default key generation
    >>> result = process_data((1, 2, 3), {"mode": "fast"})

Accessing the original function:

    >>> @cached()
    ... def my_func(x):
    ...     return x * 2
    >>> # The original unwrapped function
    >>> my_func.__wrapped__(5)  # Bypasses cache
    10

Sharing cache across functions:

    >>> from insideLLMs.cache import InMemoryCache
    >>> shared_cache = InMemoryCache(max_size=1000)
    >>> @cached(cache=shared_cache)
    ... def func_a(x):
    ...     return x + 1
    >>> @cached(cache=shared_cache)
    ... def func_b(x):
    ...     return x * 2
    >>> # Both functions share the same cache
    >>> # Keys include function name, so no collisions

Notes
-----
- The default key generator creates a SHA-256 hash from the function
  name, positional args, and keyword args (sorted).

- Arguments must be JSON-serializable. For non-serializable types,
  provide a custom key_fn.

- Unlike ``functools.lru_cache``, this decorator supports TTL expiration
  and custom cache backends.

- The decorator preserves the original function's metadata via
  ``functools.wraps``.

See Also
--------
caching_unified.memoize : Alternative decorator with statistics tracking.
caching_unified.MemoizedFunction : Class-based memoization with invalidation.
InMemoryCache : Default cache backend used by this decorator.
"""

generate_cache_key.__doc__ = """
Generate a deterministic cache key from prompt and parameters.

Creates a unique hash key by combining the prompt, model identifier,
generation parameters, and any additional keyword arguments. The same
inputs will always produce the same key, enabling reliable cache lookups.

This function is used internally by cache implementations but can also
be called directly for custom caching logic.

Parameters
----------
prompt : str
    The prompt text to include in the key. This is the primary component.

model : Optional[str], default=None
    Model identifier (e.g., "gpt-4", "claude-3"). If provided, requests
    to different models will have different cache keys.

params : Optional[dict], default=None
    Generation parameters like temperature, max_tokens, etc. Parameters
    are sorted for deterministic key generation.

algorithm : str, default="sha256"
    Hash algorithm to use. Options:
    - "sha256": 64-character hash (default, most secure)
    - "sha1": 40-character hash (faster, less secure)
    - "md5": 32-character hash (fastest, not secure for crypto)

**kwargs : Any
    Additional key-value pairs to include in the cache key. Useful for
    custom parameters or context identifiers.

Returns
-------
str
    Hexadecimal hash string. Length depends on algorithm:
    - sha256: 64 characters
    - sha1: 40 characters
    - md5: 32 characters

Examples
--------
Basic key generation:

    >>> from insideLLMs.cache import generate_cache_key
    >>> key = generate_cache_key("What is Python?")
    >>> print(f"Key length: {len(key)}")
    Key length: 64
    >>> print(f"Key: {key[:16]}...")
    Key: a1b2c3d4e5f6g7h8...

Deterministic keys (same inputs = same key):

    >>> key1 = generate_cache_key("Hello, world!")
    >>> key2 = generate_cache_key("Hello, world!")
    >>> key1 == key2
    True

Different inputs produce different keys:

    >>> key1 = generate_cache_key("Hello")
    >>> key2 = generate_cache_key("World")
    >>> key1 != key2
    True

With model identifier:

    >>> key_gpt = generate_cache_key("Explain AI", model="gpt-4")
    >>> key_claude = generate_cache_key("Explain AI", model="claude-3")
    >>> key_gpt != key_claude
    True

With parameters:

    >>> key1 = generate_cache_key(
    ...     "Write a poem",
    ...     params={"temperature": 0.0, "max_tokens": 100}
    ... )
    >>> key2 = generate_cache_key(
    ...     "Write a poem",
    ...     params={"temperature": 0.7, "max_tokens": 100}
    ... )
    >>> key1 != key2  # Different temperature = different key
    True

Parameter order doesn't matter:

    >>> key1 = generate_cache_key(
    ...     "Test",
    ...     params={"a": 1, "b": 2}
    ... )
    >>> key2 = generate_cache_key(
    ...     "Test",
    ...     params={"b": 2, "a": 1}  # Different order
    ... )
    >>> key1 == key2  # Same key (params are sorted)
    True

Using different hash algorithms:

    >>> key_sha256 = generate_cache_key("Test", algorithm="sha256")
    >>> key_sha1 = generate_cache_key("Test", algorithm="sha1")
    >>> key_md5 = generate_cache_key("Test", algorithm="md5")
    >>> print(f"SHA256: {len(key_sha256)} chars")
    SHA256: 64 chars
    >>> print(f"SHA1: {len(key_sha1)} chars")
    SHA1: 40 chars
    >>> print(f"MD5: {len(key_md5)} chars")
    MD5: 32 chars

With custom kwargs:

    >>> key = generate_cache_key(
    ...     "Translate to French",
    ...     model="gpt-4",
    ...     user_id="user_123",
    ...     session_id="session_456"
    ... )

Notes
-----
- The key string format is: ``prompt|model:X|params:JSON|key1:val1|...``
  This string is then hashed using the specified algorithm.

- Parameters dict is serialized with sorted keys for determinism.

- Use SHA-256 (default) for production. SHA-1 and MD5 are provided for
  compatibility but are not cryptographically secure.

See Also
--------
caching_unified.generate_model_cache_key : Specialized for model requests.
caching_unified.get_cache_key : Simplified wrapper for common use cases.
"""

get_default_cache.__doc__ = """
Get the global default cache instance.

Returns the application-wide default cache, creating a new InMemoryCache
if one hasn't been set. This provides a convenient way to share a cache
across modules without passing cache instances explicitly.

Returns
-------
BaseCacheABC
    The global default cache instance. If not previously set, returns
    a new InMemoryCache with default settings.

Examples
--------
Basic usage:

    >>> from insideLLMs.cache import get_default_cache
    >>> cache = get_default_cache()
    >>> cache.set("shared_key", "shared_value")
    >>> # From another module
    >>> cache2 = get_default_cache()
    >>> cache2.get("shared_key")
    'shared_value'

Using with the cached decorator:

    >>> from insideLLMs.cache import get_default_cache, cached
    >>> @cached(cache=get_default_cache())
    ... def expensive_function(x):
    ...     return x ** 2

Checking if default cache is set:

    >>> cache = get_default_cache()
    >>> # First call creates a new InMemoryCache
    >>> type(cache).__name__
    'InMemoryCache'

See Also
--------
set_default_cache : Set a custom default cache.
clear_default_cache : Clear all entries from the default cache.
"""

set_default_cache.__doc__ = """
Set the global default cache instance.

Configures the application-wide default cache. Use this to replace the
default InMemoryCache with a custom cache implementation (e.g., DiskCache)
or a cache with specific configuration.

Parameters
----------
cache : BaseCacheABC
    The cache instance to use as the global default.

Returns
-------
None

Examples
--------
Setting a custom InMemoryCache:

    >>> from insideLLMs.cache import set_default_cache, get_default_cache, InMemoryCache
    >>> custom_cache = InMemoryCache(max_size=5000, default_ttl=7200)
    >>> set_default_cache(custom_cache)
    >>> cache = get_default_cache()
    >>> cache._max_size
    5000

Setting a DiskCache for persistence:

    >>> from insideLLMs.cache import set_default_cache, DiskCache
    >>> from pathlib import Path
    >>> disk_cache = DiskCache(
    ...     path=Path("/var/cache/myapp/default.db"),
    ...     max_size_mb=500
    ... )
    >>> set_default_cache(disk_cache)

Application initialization pattern:

    >>> def configure_caching():
    ...     from insideLLMs.cache import set_default_cache, InMemoryCache
    ...     cache = InMemoryCache(
    ...         max_size=10000,
    ...         default_ttl=3600
    ...     )
    ...     set_default_cache(cache)
    >>> # Call during application startup
    >>> configure_caching()

See Also
--------
get_default_cache : Get the current default cache.
clear_default_cache : Clear the default cache contents.
"""

clear_default_cache.__doc__ = """
Clear all entries from the global default cache.

Removes all cached entries from the default cache without changing the
cache instance itself. This is useful for invalidating all cached data
(e.g., after configuration changes, model updates, or during testing).

Returns
-------
None

Examples
--------
Basic cache clearing:

    >>> from insideLLMs.cache import get_default_cache, clear_default_cache
    >>> cache = get_default_cache()
    >>> cache.set("key1", "value1")
    >>> cache.set("key2", "value2")
    >>> stats = cache.stats()
    >>> print(f"Before clear: {stats.entry_count} entries")
    Before clear: 2 entries
    >>> clear_default_cache()
    >>> stats = cache.stats()
    >>> print(f"After clear: {stats.entry_count} entries")
    After clear: 0 entries

After configuration change:

    >>> def on_config_change():
    ...     # Configuration changed, cached responses may be invalid
    ...     clear_default_cache()
    ...     print("Cache cleared due to configuration change")

In testing:

    >>> def setup_test():
    ...     clear_default_cache()  # Start with empty cache
    >>> def teardown_test():
    ...     clear_default_cache()  # Clean up after test

Safe when no cache is set:

    >>> # Even if default cache hasn't been accessed yet
    >>> clear_default_cache()  # No error, does nothing

See Also
--------
get_default_cache : Get the default cache instance.
set_default_cache : Replace the default cache entirely.
"""


# For backward compatibility with old module structure
__all__ = [
    "BaseCache",
    "CacheEntry",
    "CacheStats",
    "CachedModel",
    "DiskCache",
    "InMemoryCache",
    "cached",
    "clear_default_cache",
    "generate_cache_key",
    "get_default_cache",
    "set_default_cache",
]
