"""
Cache Types and Data Structures
===============================

This module contains all the type definitions, enums, and dataclasses used
throughout the caching system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class CacheEntryMixin:
    """Mixin providing common cache entry methods.

    Requires the class to have:
    - expires_at: Optional[datetime]
    - access_count: int
    - last_accessed: datetime
    """

    expires_at: Optional[datetime]
    access_count: int
    last_accessed: datetime

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self):
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheStrategy(Enum):
    """Cache eviction strategies for controlling which entries are removed when cache is full.

    Each strategy defines a different policy for selecting which cache entry to evict
    when the cache reaches its maximum size.

    Attributes
    ----------
    LRU : str
        Least Recently Used. Evicts entries that have not been accessed for the longest
        time. Best for workloads where recently accessed items are likely to be accessed
        again (temporal locality).
    LFU : str
        Least Frequently Used. Evicts entries with the lowest access count. Best for
        workloads where frequently accessed items should be kept regardless of recency.
    FIFO : str
        First In First Out. Evicts the oldest entries first, regardless of access
        patterns. Simple and predictable, best for streaming workloads.
    TTL : str
        Time To Live based. Evicts entries closest to expiration first. Best when
        entries have varying TTLs and expired data should be removed promptly.
    SIZE : str
        Size-based eviction. Evicts the largest entries first to reclaim maximum space
        quickly. Best when cache entries have varying sizes and memory is constrained.

    Examples
    --------
    Selecting LRU strategy for a web application cache:

        >>> from insideLLMs.caching import CacheStrategy, CacheConfig, create_cache
        >>> config = CacheConfig(strategy=CacheStrategy.LRU, max_size=1000)
        >>> cache = create_cache(strategy=CacheStrategy.LRU)

    Using LFU for a frequently-accessed reference data cache:

        >>> config = CacheConfig(strategy=CacheStrategy.LFU)

    SIZE strategy for caching large model responses:

        >>> config = CacheConfig(strategy=CacheStrategy.SIZE, max_size=100)

    Comparing strategy values:

        >>> CacheStrategy.LRU.value
        'lru'
        >>> CacheStrategy.LFU == CacheStrategy.LFU
        True
    """

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based
    SIZE = "size"  # Size-based eviction


class CacheStatus(Enum):
    """Cache entry status indicating the current state of a cached item.

    Used to track the lifecycle of cache entries from creation through
    expiration or eviction.

    Attributes
    ----------
    ACTIVE : str
        Entry is valid and available for retrieval. This is the normal state
        for cached items that have not expired.
    EXPIRED : str
        Entry has exceeded its TTL (Time To Live) and is no longer valid.
        Expired entries are typically removed on next access or during cleanup.
    EVICTED : str
        Entry was removed from the cache due to capacity constraints. The
        eviction strategy determines which entries are evicted first.
    WARMING : str
        Entry is being preloaded by a CacheWarmer. This transient state
        indicates the entry is being prepared but not yet fully available.

    Examples
    --------
    Checking entry status in cache metadata:

        >>> from insideLLMs.caching import CacheStatus
        >>> status = CacheStatus.ACTIVE
        >>> if status == CacheStatus.ACTIVE:
        ...     print("Entry is valid")
        Entry is valid

    Using status for logging:

        >>> entry_status = CacheStatus.EXPIRED
        >>> print(f"Cache entry is {entry_status.value}")
        Cache entry is expired

    Status comparison:

        >>> CacheStatus.ACTIVE != CacheStatus.EXPIRED
        True

    Iterating over all statuses:

        >>> for status in CacheStatus:
        ...     print(status.value)
        active
        expired
        evicted
        warming
    """

    # Backward-compatible lookup statuses (used by older tests/consumers).
    HIT = "hit"
    MISS = "miss"

    ACTIVE = "active"
    EXPIRED = "expired"
    EVICTED = "evicted"
    WARMING = "warming"


class CacheScope(Enum):
    """Cache scope for partitioning cache entries by context.

    Scopes allow logical separation of cached data based on different contexts,
    enabling isolation between users, sessions, requests, or models.

    Attributes
    ----------
    GLOBAL : str
        Cache entries are shared across all contexts. Use for data that is
        the same for all users and sessions (e.g., system prompts, reference data).
    SESSION : str
        Cache entries are scoped to a specific user session. Use for data that
        should persist within a session but not across sessions.
    REQUEST : str
        Cache entries are scoped to a single request. Use for short-lived data
        that should not persist beyond the current operation.
    USER : str
        Cache entries are scoped to a specific user. Use for user-specific
        data that should persist across sessions for the same user.
    MODEL : str
        Cache entries are scoped to a specific model. Use when caching responses
        from multiple models that should be kept separate.

    Examples
    --------
    Configuring global scope for shared data:

        >>> from insideLLMs.caching import CacheScope, CacheConfig
        >>> config = CacheConfig(scope=CacheScope.GLOBAL)

    User-scoped caching for personalization:

        >>> config = CacheConfig(scope=CacheScope.USER)

    Model-scoped caching for multi-model systems:

        >>> from insideLLMs.caching import CacheScope, CacheConfig, create_cache
        >>> # Create separate caches per model
        >>> gpt4_cache = create_cache()
        >>> claude_cache = create_cache()

    Using scope values for namespacing:

        >>> scope = CacheScope.SESSION
        >>> cache_key = f"{scope.value}:user123:prompt_hash"
        >>> print(cache_key)
        session:user123:prompt_hash
    """

    GLOBAL = "global"
    SESSION = "session"
    REQUEST = "request"
    USER = "user"
    MODEL = "model"


@dataclass
class CacheConfig:
    """Configuration for cache behavior and policies.

    This dataclass encapsulates all configuration options for cache instances,
    including capacity limits, expiration policies, eviction strategies, and
    optional features like compression and statistics tracking.

    Attributes
    ----------
    max_size : int
        Maximum number of entries the cache can hold. When this limit is reached,
        entries are evicted according to the selected strategy. Default is 1000.
    ttl_seconds : Optional[int]
        Default Time To Live for cache entries in seconds. Entries expire after
        this duration. None means entries never expire. Default is 3600 (1 hour).
    strategy : CacheStrategy
        Eviction strategy to use when cache is full. Options include LRU, LFU,
        FIFO, TTL, and SIZE. Default is CacheStrategy.LRU.
    enable_stats : bool
        Whether to track cache statistics (hits, misses, evictions). Enabling
        stats adds minimal overhead but provides useful observability. Default True.
    enable_compression : bool
        Whether to compress cached values. Useful for large values but adds
        CPU overhead. Currently not implemented. Default False.
    hash_algorithm : str
        Algorithm for generating cache keys from prompts. Options include "sha256",
        "sha1", and "md5". Default is "sha256" for security.
    scope : CacheScope
        Scope for cache entries, enabling logical partitioning. Default is GLOBAL.

    Examples
    --------
    Creating a default configuration:

        >>> from insideLLMs.caching import CacheConfig
        >>> config = CacheConfig()
        >>> print(config.max_size, config.ttl_seconds)
        1000 3600

    High-capacity cache with no expiration:

        >>> config = CacheConfig(
        ...     max_size=10000,
        ...     ttl_seconds=None,
        ...     strategy=CacheStrategy.LRU
        ... )

    Short-lived request cache:

        >>> from insideLLMs.caching import CacheConfig, CacheStrategy, CacheScope
        >>> config = CacheConfig(
        ...     max_size=100,
        ...     ttl_seconds=60,
        ...     strategy=CacheStrategy.FIFO,
        ...     scope=CacheScope.REQUEST
        ... )

    Converting to dictionary for serialization:

        >>> config = CacheConfig(max_size=500)
        >>> config_dict = config.to_dict()
        >>> print(config_dict["max_size"])
        500

    See Also
    --------
    CacheStrategy : Available eviction strategies.
    CacheScope : Available cache scopes.
    """

    max_size: int = 1000
    ttl_seconds: Optional[int] = 3600  # 1 hour default
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_stats: bool = True
    enable_compression: bool = False
    hash_algorithm: str = "sha256"
    scope: CacheScope = CacheScope.GLOBAL

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all configuration values with enum values
            converted to their string representations.

        Examples
        --------
        >>> config = CacheConfig(max_size=500, strategy=CacheStrategy.LFU)
        >>> d = config.to_dict()
        >>> d["strategy"]
        'lfu'
        """
        return {
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "strategy": self.strategy.value,
            "enable_stats": self.enable_stats,
            "enable_compression": self.enable_compression,
            "hash_algorithm": self.hash_algorithm,
            "scope": self.scope.value,
        }


@dataclass
class CacheEntry(CacheEntryMixin):
    """A single cache entry with metadata and access tracking.

    CacheEntry represents a cached value along with all associated metadata
    needed for cache management, including timestamps, access counts, and
    size information.

    Attributes
    ----------
    key : str
        Unique identifier for this cache entry. Typically a hash of the
        request parameters.
    value : Any
        The cached value. Can be any JSON-serializable type.
    created_at : datetime
        Timestamp when this entry was created. Auto-set to current time.
    expires_at : Optional[datetime]
        Timestamp when this entry expires. None means no expiration.
    access_count : int
        Number of times this entry has been accessed. Used by LFU strategy.
    last_accessed : datetime
        Timestamp of most recent access. Used by LRU strategy.
    size_bytes : int
        Estimated size of the cached value in bytes. Auto-calculated if not set.
    metadata : dict[str, Any]
        Additional metadata associated with this entry (e.g., prompt, model).

    Methods
    -------
    is_expired()
        Check if entry has exceeded its TTL.
    touch()
        Update access tracking (count and timestamp).
    to_dict()
        Convert entry to dictionary representation.

    Examples
    --------
    Creating a basic cache entry:

        >>> from insideLLMs.caching import CacheEntry
        >>> from datetime import datetime, timedelta
        >>> entry = CacheEntry(
        ...     key="abc123",
        ...     value="Hello, World!",
        ...     expires_at=datetime.now() + timedelta(hours=1)
        ... )
        >>> print(entry.is_expired())
        False

    Entry with metadata:

        >>> entry = CacheEntry(
        ...     key="prompt_hash",
        ...     value="LLM response here",
        ...     metadata={"prompt": "What is AI?", "model": "gpt-4"}
        ... )
        >>> entry.metadata["prompt"]
        'What is AI?'

    Tracking access patterns:

        >>> entry = CacheEntry(key="test", value="data")
        >>> entry.touch()  # Simulate access
        >>> entry.touch()
        >>> print(entry.access_count)
        2

    Converting to dictionary for serialization:

        >>> entry = CacheEntry(key="k1", value={"a": 1})
        >>> d = entry.to_dict()
        >>> d["key"]
        'k1'

    See Also
    --------
    CacheEntryMixin : Provides is_expired() and touch() methods.
    CacheLookupResult : Returned when looking up entries.
    """

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._estimate_size()

    def _estimate_size(self) -> int:
        """Estimate memory size of cached value in bytes.

        Attempts JSON serialization first for accurate sizing, falls back
        to string representation for non-serializable objects.

        Returns
        -------
        int
            Estimated size in bytes.

        Examples
        --------
        >>> entry = CacheEntry(key="test", value="Hello")
        >>> entry.size_bytes > 0
        True
        """
        try:
            return len(json.dumps(self.value))
        except (TypeError, ValueError):
            return len(str(self.value))

    # is_expired() and touch() inherited from CacheEntryMixin

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary representation.

        Useful for serialization, logging, and debugging. Datetime fields
        are converted to ISO format strings.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all entry fields.

        Examples
        --------
        >>> entry = CacheEntry(key="k", value="v")
        >>> d = entry.to_dict()
        >>> "created_at" in d
        True
        """
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


@dataclass
class CacheStats:
    """Cache statistics for monitoring and optimization.

    CacheStats provides comprehensive metrics about cache performance, including
    hit rates, eviction counts, and size information. Use these statistics to
    tune cache configuration and identify performance issues.

    Attributes
    ----------
    hits : int
        Number of successful cache lookups (key found and not expired).
    misses : int
        Number of cache lookups that did not find a valid entry.
    evictions : int
        Number of entries removed due to capacity constraints.
    expirations : int
        Number of entries that expired (TTL exceeded).
    total_size_bytes : int
        Total estimated size of all cached values in bytes.
    entry_count : int
        Current number of entries in the cache.
    oldest_entry : Optional[datetime]
        Timestamp of the oldest cache entry, or None if cache is empty.
    newest_entry : Optional[datetime]
        Timestamp of the newest cache entry, or None if cache is empty.
    avg_access_count : float
        Average number of accesses per entry. High values indicate good cache efficiency.
    hit_rate : float
        Ratio of hits to total lookups (hits / (hits + misses)). Range 0.0 to 1.0.

    Examples
    --------
    Getting cache statistics:

        >>> from insideLLMs.caching import create_cache
        >>> cache = create_cache()
        >>> cache.set("key1", "value1")
        >>> cache.get("key1")  # Hit
        >>> cache.get("key2")  # Miss
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")
        Hit rate: 50.00%

    Monitoring cache health:

        >>> stats = cache.get_stats()
        >>> if stats.hit_rate < 0.5:
        ...     print("Consider increasing cache size")
        >>> if stats.evictions > stats.entry_count:
        ...     print("High eviction rate - cache may be too small")

    Exporting statistics for logging:

        >>> stats = cache.get_stats()
        >>> stats_dict = stats.to_dict()
        >>> import json
        >>> print(json.dumps(stats_dict, indent=2, default=str))

    Comparing statistics over time:

        >>> initial_hits = cache.get_stats().hits
        >>> # ... perform operations ...
        >>> final_stats = cache.get_stats()
        >>> new_hits = final_stats.hits - initial_hits
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None
    avg_access_count: float = 0.0
    hit_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary representation.

        Datetime fields are converted to ISO format strings for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all statistics.

        Examples
        --------
        >>> stats = CacheStats(hits=100, misses=20, hit_rate=0.833)
        >>> d = stats.to_dict()
        >>> d["hit_rate"]
        0.833
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "oldest_entry": self.oldest_entry.isoformat() if self.oldest_entry else None,
            "newest_entry": self.newest_entry.isoformat() if self.newest_entry else None,
            "avg_access_count": self.avg_access_count,
            "hit_rate": self.hit_rate,
        }


@dataclass(init=False)
class CacheLookupResult:
    """Result of a cache lookup operation.

    CacheLookupResult encapsulates the outcome of a cache.get() operation,
    providing both the cached value (if found) and metadata about the lookup
    including timing information.

    Attributes
    ----------
    hit : bool
        True if the key was found and the entry was valid (not expired).
    value : Optional[Any]
        The cached value if hit is True, None otherwise.
    key : str
        The key that was looked up.
    entry : Optional[CacheEntry]
        The full cache entry if hit is True, None otherwise. Contains
        metadata like access count and expiration time.
    lookup_time_ms : float
        Time taken for the lookup operation in milliseconds. Useful for
        performance monitoring.

    Examples
    --------
    Checking for cache hit:

        >>> from insideLLMs.caching import create_cache
        >>> cache = create_cache()
        >>> cache.set("greeting", "Hello!")
        >>> result = cache.get("greeting")
        >>> if result.hit:
        ...     print(result.value)
        Hello!

    Handling cache miss:

        >>> result = cache.get("nonexistent")
        >>> if not result.hit:
        ...     print("Cache miss - computing value...")
        ...     value = expensive_computation()
        ...     cache.set("nonexistent", value)

    Accessing entry metadata:

        >>> cache.set("data", {"key": "value"})
        >>> result = cache.get("data")
        >>> if result.entry:
        ...     print(f"Accessed {result.entry.access_count} times")
        ...     print(f"Size: {result.entry.size_bytes} bytes")

    Performance monitoring:

        >>> result = cache.get("key")
        >>> if result.lookup_time_ms > 10:
        ...     print("Slow lookup detected")
    """

    hit: bool
    status: CacheStatus
    value: Optional[Any]
    key: str
    entry: Optional[CacheEntry] = None
    lookup_time_ms: float = 0.0

    def __init__(
        self,
        hit: Optional[bool] = None,
        value: Optional[Any] = None,
        key: str = "",
        entry: Optional[CacheEntry] = None,
        lookup_time_ms: float = 0.0,
        *,
        status: Optional[CacheStatus] = None,
    ) -> None:
        if status is None and hit is None:
            raise TypeError("CacheLookupResult requires either 'hit' or 'status'")

        if status is None:
            status = CacheStatus.HIT if hit else CacheStatus.MISS

        status_implies_hit = status in {CacheStatus.HIT, CacheStatus.ACTIVE, CacheStatus.WARMING}
        if hit is None:
            hit = status_implies_hit

        if bool(hit) != status_implies_hit:
            raise ValueError(
                f"CacheLookupResult hit/status mismatch: hit={hit!r} status={status.value!r}"
            )

        self.hit = bool(hit)
        self.status = status
        self.value = value
        self.key = key
        self.entry = entry
        self.lookup_time_ms = float(lookup_time_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all result fields. The entry field is
            recursively converted if present.

        Examples
        --------
        >>> result = CacheLookupResult(hit=True, value="data", key="k1")
        >>> d = result.to_dict()
        >>> d["hit"]
        True
        """
        return {
            "hit": self.hit,
            "status": self.status.value,
            "value": self.value,
            "key": self.key,
            "entry": self.entry.to_dict() if self.entry else None,
            "lookup_time_ms": self.lookup_time_ms,
        }


__all__ = [
    "CacheEntryMixin",
    "CacheStrategy",
    "CacheStatus",
    "CacheScope",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheLookupResult",
]
