"""
In-Memory Cache Backend
=======================

Simple in-memory cache with LRU eviction and optional TTL.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional, TypeVar

from insideLLMs.caching.base import BaseCacheABC
from insideLLMs.caching.types import CacheStats

T = TypeVar("T")


class InMemoryCache(BaseCacheABC[T]):
    """Simple in-memory cache with LRU eviction and optional TTL.

    A thread-safe, in-memory cache implementation that stores JSON-serializable
    values. Uses LRU (Least Recently Used) eviction when the cache reaches
    its maximum size.

    Parameters
    ----------
    max_size : int
        Maximum number of entries the cache can hold. When exceeded, the
        least recently used entry is evicted. Default is 1000.
    default_ttl : Optional[int]
        Default time-to-live in seconds for new entries. None means no
        expiration. Default is None.

    Attributes
    ----------
    _cache : dict[str, dict]
        Internal storage mapping keys to entry dictionaries.
    _max_size : int
        Maximum cache capacity.
    _default_ttl : Optional[int]
        Default TTL for new entries.
    _stats : CacheStats
        Statistics tracking hits, misses, and evictions.
    _lock : threading.RLock
        Reentrant lock for thread safety.

    Examples
    --------
    Basic usage:

        >>> from insideLLMs.caching import InMemoryCache
        >>> cache = InMemoryCache(max_size=100)
        >>> cache.set("key1", {"data": "value"})
        >>> value = cache.get("key1")
        >>> print(value)
        {'data': 'value'}

    With TTL expiration:

        >>> cache = InMemoryCache(default_ttl=60)  # 60 seconds
        >>> cache.set("temp", "data")
        >>> # After 60 seconds, get() returns None

    Override TTL per entry:

        >>> cache = InMemoryCache(default_ttl=3600)  # 1 hour default
        >>> cache.set("short_lived", "data", ttl=60)  # 1 minute
        >>> cache.set("long_lived", "data", ttl=86400)  # 1 day

    Checking cache statistics:

        >>> cache = InMemoryCache()
        >>> cache.set("a", 1)
        >>> cache.get("a")  # Hit
        >>> cache.get("b")  # Miss
        >>> stats = cache.stats()
        >>> print(f"Hits: {stats.hits}, Misses: {stats.misses}")
        Hits: 1, Misses: 1

    Thread-safe access:

        >>> import threading
        >>> cache = InMemoryCache()
        >>> def worker(key, value):
        ...     cache.set(key, value)
        >>> threads = [threading.Thread(target=worker, args=(f"k{i}", i))
        ...            for i in range(10)]
        >>> for t in threads: t.start()
        >>> for t in threads: t.join()

    See Also
    --------
    DiskCache : Persistent SQLite-based cache.
    StrategyCache : Cache with configurable eviction strategies.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
    ):
        self._cache: dict[str, dict] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check expiration
            if entry.get("expires_at") and time.time() > entry["expires_at"]:
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Update access tracking
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            entry["last_accessed"] = time.time()
            self._stats.hits += 1

            return json.loads(entry["value"])

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set a value in the cache."""
        with self._lock:
            # Evict if necessary
            while len(self._cache) >= self._max_size:
                self._evict_lru()

            ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + ttl if ttl is not None else None

            self._cache[key] = {
                "key": key,
                "value": json.dumps(value),
                "created_at": time.time(),
                "expires_at": expires_at,
                "last_accessed": time.time(),
                "hit_count": 0,
                "metadata": metadata or {},
            }
            self._stats.entry_count = len(self._cache)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.entry_count = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            total = self._stats.hits + self._stats.misses
            self._stats.hit_rate = self._stats.hits / total if total > 0 else 0.0
            self._stats.entry_count = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                entry_count=self._stats.entry_count,
                evictions=self._stats.evictions,
                hit_rate=self._stats.hit_rate,
            )

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].get("last_accessed", self._cache[k].get("created_at", 0)),
        )
        del self._cache[lru_key]
        self._stats.evictions += 1

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())


__all__ = ["InMemoryCache"]
