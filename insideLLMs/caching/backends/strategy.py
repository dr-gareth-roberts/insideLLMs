"""
Strategy-Based Cache Backend
============================

Full-featured cache with configurable eviction strategies.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional

from insideLLMs.caching.types import (
    CacheConfig,
    CacheEntry,
    CacheLookupResult,
    CacheStats,
    CacheStrategy,
)


class StrategyCache:
    """Cache with configurable eviction strategies and rich features.

    StrategyCache is the primary cache implementation, supporting multiple
    eviction strategies (LRU, LFU, FIFO, TTL, SIZE) and providing comprehensive
    statistics tracking. It returns CacheLookupResult objects with metadata
    for cache hits.

    Parameters
    ----------
    config : Optional[CacheConfig]
        Configuration for cache behavior. If None, uses default configuration
        (1000 entries, 1 hour TTL, LRU strategy).

    Attributes
    ----------
    config : CacheConfig
        The active cache configuration.
    _entries : OrderedDict[str, CacheEntry]
        Ordered dictionary storing cache entries.
    _stats : CacheStats
        Statistics tracking.
    _lock : threading.RLock
        Reentrant lock for thread safety.

    Examples
    --------
    Basic usage with defaults:

        >>> from insideLLMs.caching import StrategyCache
        >>> cache = StrategyCache()
        >>> entry = cache.set("greeting", "Hello!")
        >>> result = cache.get("greeting")
        >>> print(result.hit, result.value)
        True Hello!

    LRU cache (Least Recently Used):

        >>> from insideLLMs.caching import StrategyCache, CacheConfig, CacheStrategy
        >>> config = CacheConfig(
        ...     max_size=100,
        ...     strategy=CacheStrategy.LRU
        ... )
        >>> cache = StrategyCache(config)
        >>> cache.set("a", 1)
        >>> cache.set("b", 2)
        >>> cache.get("a")  # "a" is now most recently used
        >>> # If cache is full, "b" would be evicted first

    LFU cache (Least Frequently Used):

        >>> config = CacheConfig(strategy=CacheStrategy.LFU)
        >>> cache = StrategyCache(config)
        >>> cache.set("popular", "data")
        >>> for _ in range(10):
        ...     cache.get("popular")  # Increase access count
        >>> # "popular" is less likely to be evicted

    TTL-based eviction:

        >>> config = CacheConfig(
        ...     strategy=CacheStrategy.TTL,
        ...     ttl_seconds=60
        ... )
        >>> cache = StrategyCache(config)
        >>> cache.set("short", "data", ttl_seconds=10)  # 10 seconds
        >>> cache.set("long", "data", ttl_seconds=3600)  # 1 hour
        >>> # When eviction needed, "short" is evicted first

    SIZE-based eviction:

        >>> config = CacheConfig(strategy=CacheStrategy.SIZE)
        >>> cache = StrategyCache(config)
        >>> cache.set("small", "x")
        >>> cache.set("large", "x" * 10000)
        >>> # "large" would be evicted first to free more space

    Checking cache statistics:

        >>> cache = StrategyCache()
        >>> cache.set("key", "value")
        >>> cache.get("key")
        >>> cache.get("missing")
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")

    Iterating cache contents:

        >>> cache = StrategyCache()
        >>> cache.set("a", 1)
        >>> cache.set("b", 2)
        >>> print(cache.keys())
        ['a', 'b']
        >>> print(cache.items())
        [('a', 1), ('b', 2)]

    Notes
    -----
    - Thread-safe: All operations are protected by a reentrant lock.
    - OrderedDict maintains insertion order for FIFO and LRU strategies.
    - Statistics are updated in real-time for monitoring.
    - Expired entries are lazily removed on access.

    See Also
    --------
    InMemoryCache : Simpler cache with LRU only.
    DiskCache : Persistent SQLite-based cache.
    PromptCache : LLM-specialized cache extending StrategyCache.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> CacheLookupResult:
        """Get value from cache."""
        start_time = time.time()

        with self._lock:
            if key not in self._entries:
                self._stats.misses += 1
                return CacheLookupResult(
                    hit=False,
                    value=None,
                    key=key,
                    lookup_time_ms=(time.time() - start_time) * 1000,
                )

            entry = self._entries[key]

            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.expirations += 1
                return CacheLookupResult(
                    hit=False,
                    value=None,
                    key=key,
                    lookup_time_ms=(time.time() - start_time) * 1000,
                )

            entry.touch()
            self._stats.hits += 1

            # Move to end for LRU
            if self.config.strategy == CacheStrategy.LRU:
                self._entries.move_to_end(key)

            return CacheLookupResult(
                hit=True,
                value=entry.value,
                key=key,
                entry=entry,
                lookup_time_ms=(time.time() - start_time) * 1000,
            )

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Set value in cache."""
        with self._lock:
            # Evict if needed
            while len(self._entries) >= self.config.max_size:
                self._evict_one()

            ttl = ttl_seconds if ttl_seconds is not None else self.config.ttl_seconds
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._entries[key] = entry
            self._update_stats()

            return entry

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._update_stats()

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._entries:
                return False
            entry = self._entries[key]
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return self._stats

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._entries.keys())

    def values(self) -> list[Any]:
        """Get all cached values."""
        with self._lock:
            return [e.value for e in self._entries.values() if not e.is_expired()]

    def items(self) -> list[tuple[str, Any]]:
        """Get all key-value pairs."""
        with self._lock:
            return [(k, e.value) for k, e in self._entries.items() if not e.is_expired()]

    def _evict_one(self):
        """Evict one entry based on strategy."""
        if not self._entries:
            return

        if self.config.strategy == CacheStrategy.LRU:
            key = next(iter(self._entries))
            self._remove_entry(key)

        elif self.config.strategy == CacheStrategy.LFU:
            min_count = min(e.access_count for e in self._entries.values())
            for key, entry in self._entries.items():
                if entry.access_count == min_count:
                    self._remove_entry(key)
                    break

        elif self.config.strategy == CacheStrategy.FIFO:
            key = next(iter(self._entries))
            self._remove_entry(key)

        elif self.config.strategy == CacheStrategy.TTL:
            soonest_key = None
            soonest_time = None
            for key, entry in self._entries.items():
                if entry.expires_at:
                    if soonest_time is None or entry.expires_at < soonest_time:
                        soonest_time = entry.expires_at
                        soonest_key = key
            if soonest_key:
                self._remove_entry(soonest_key)
            else:
                key = next(iter(self._entries))
                self._remove_entry(key)

        elif self.config.strategy == CacheStrategy.SIZE:
            max_size = max(e.size_bytes for e in self._entries.values())
            for key, entry in self._entries.items():
                if entry.size_bytes == max_size:
                    self._remove_entry(key)
                    break

        self._stats.evictions += 1

    def _remove_entry(self, key: str):
        """Remove entry by key."""
        if key in self._entries:
            del self._entries[key]

    def _update_stats(self):
        """Update statistics."""
        entries = list(self._entries.values())
        valid_entries = [e for e in entries if not e.is_expired()]

        self._stats.entry_count = len(valid_entries)
        self._stats.total_size_bytes = sum(e.size_bytes for e in valid_entries)

        if valid_entries:
            self._stats.oldest_entry = min(e.created_at for e in valid_entries)
            self._stats.newest_entry = max(e.created_at for e in valid_entries)
            self._stats.avg_access_count = sum(e.access_count for e in valid_entries) / len(
                valid_entries
            )

        total = self._stats.hits + self._stats.misses
        self._stats.hit_rate = self._stats.hits / total if total > 0 else 0.0


# Alias for backward compatibility
BaseCache = StrategyCache


__all__ = ["StrategyCache", "BaseCache"]
