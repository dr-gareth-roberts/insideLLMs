"""
Unified Caching Module for insideLLMs
=====================================

This module provides comprehensive caching capabilities for LLM operations and general
purpose caching needs. It offers multiple cache backends, eviction strategies, and
specialized utilities for prompt caching and model response management.

Overview
--------
The unified caching module implements a layered architecture:

1. **Abstract Base (BaseCacheABC)**: Defines the cache interface contract.
2. **Simple Backends (InMemoryCache, DiskCache)**: Basic cache implementations.
3. **Strategy Cache (StrategyCache)**: Full-featured cache with configurable eviction.
4. **Prompt Cache (PromptCache)**: LLM-specialized cache with semantic similarity.
5. **Model Wrapper (CachedModel)**: Adds caching to any model transparently.

Design Principles
-----------------
- **Thread Safety**: All cache implementations use locks for thread-safe operations.
- **Composability**: Caches can be wrapped, adapted, and combined as needed.
- **Flexibility**: Multiple eviction strategies and configuration options.
- **Performance**: Efficient data structures (OrderedDict, SQLite indexes).
- **Observability**: Built-in statistics tracking for monitoring and debugging.

Cache Backends
--------------
- **InMemoryCache**: Fast, thread-safe in-memory cache with LRU eviction.
  Best for: Single-process applications with moderate cache sizes.

- **DiskCache**: SQLite-based persistent cache with size-based management.
  Best for: Large caches, persistence across restarts, shared access.

- **StrategyCache**: Full-featured cache supporting multiple eviction strategies.
  Best for: Applications needing fine-grained control over cache behavior.

Eviction Strategies
-------------------
- **LRU (Least Recently Used)**: Evicts entries not accessed recently.
- **LFU (Least Frequently Used)**: Evicts entries with lowest access count.
- **FIFO (First In First Out)**: Evicts oldest entries first.
- **TTL (Time To Live)**: Evicts entries closest to expiration.
- **SIZE**: Evicts largest entries first to reclaim space.

Examples
--------
Basic in-memory caching:

    >>> from insideLLMs.caching_unified import InMemoryCache
    >>> cache = InMemoryCache(max_size=100, default_ttl=3600)
    >>> cache.set("greeting", "Hello, World!")
    >>> value = cache.get("greeting")
    >>> print(value)
    Hello, World!

Strategy-based cache with LFU eviction:

    >>> from insideLLMs.caching_unified import StrategyCache, CacheConfig, CacheStrategy
    >>> config = CacheConfig(
    ...     max_size=500,
    ...     ttl_seconds=1800,
    ...     strategy=CacheStrategy.LFU
    ... )
    >>> cache = StrategyCache(config)
    >>> entry = cache.set("key", {"data": "value"})
    >>> result = cache.get("key")
    >>> print(result.hit, result.value)
    True {'data': 'value'}

Persistent disk caching:

    >>> from insideLLMs.caching_unified import DiskCache
    >>> from pathlib import Path
    >>> cache = DiskCache(
    ...     path=Path("/tmp/my_cache.db"),
    ...     max_size_mb=50,
    ...     default_ttl=86400
    ... )
    >>> cache.set("persistent_key", {"complex": "data"})
    >>> # Cache persists across restarts

LLM prompt caching with semantic matching:

    >>> from insideLLMs.caching_unified import PromptCache, CacheConfig
    >>> cache = PromptCache(similarity_threshold=0.9)
    >>> cache.cache_response(
    ...     prompt="Explain machine learning",
    ...     response="Machine learning is...",
    ...     model="gpt-4",
    ...     params={"temperature": 0.7}
    ... )
    >>> # Find similar cached prompts
    >>> similar = cache.find_similar("What is ML?", limit=3)
    >>> for prompt, entry, score in similar:
    ...     print(f"Similarity: {score:.2f}")

Wrapping a model with caching:

    >>> from insideLLMs.caching_unified import CachedModel, InMemoryCache
    >>> class MyModel:
    ...     def generate(self, prompt, temperature=0.0, max_tokens=None):
    ...         return ModelResponse(content=f"Response to: {prompt}")
    >>> model = MyModel()
    >>> cached_model = CachedModel(model, cache=InMemoryCache())
    >>> response = cached_model.generate("Hello", temperature=0)  # Cached
    >>> response = cached_model.generate("Hello", temperature=0)  # From cache

See Also
--------
- caching : Re-export module for backward compatibility.
- types.ModelResponse : Response type used with CachedModel.
"""

import hashlib
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path

from insideLLMs.nlp.similarity import word_overlap_similarity
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from insideLLMs.types import ModelResponse

T = TypeVar("T")


# =============================================================================
# Cache Entry Mixin
# =============================================================================


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


# =============================================================================
# Enums and Configuration
# =============================================================================


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


# =============================================================================
# Data Classes
# =============================================================================


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

        >>> from insideLLMs.caching_unified import CacheEntry
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


@dataclass
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
    value: Optional[Any]
    key: str
    entry: Optional[CacheEntry] = None
    lookup_time_ms: float = 0.0

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
            "value": self.value,
            "key": self.key,
            "entry": self.entry.to_dict() if self.entry else None,
            "lookup_time_ms": self.lookup_time_ms,
        }


# =============================================================================
# Key Generation
# =============================================================================


def generate_cache_key(
    prompt: str,
    model: Optional[str] = None,
    params: Optional[dict] = None,
    algorithm: str = "sha256",
    **kwargs: Any,
) -> str:
    """Generate a deterministic cache key from prompt and parameters.

    Creates a unique hash key by combining the prompt, model identifier,
    generation parameters, and any additional keyword arguments. The same
    inputs will always produce the same key, enabling reliable cache lookups.

    Parameters
    ----------
    prompt : str
        The prompt text to include in the key. This is the primary component.
    model : Optional[str]
        Model identifier (e.g., "gpt-4", "claude-3"). If provided, requests
        to different models will have different cache keys.
    params : Optional[dict]
        Generation parameters like temperature, max_tokens, etc. Parameters
        are sorted for deterministic key generation.
    algorithm : str
        Hash algorithm to use. Options: "sha256" (default, most secure),
        "sha1" (faster, less secure), "md5" (fastest, not secure).
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

        >>> from insideLLMs.caching import generate_cache_key
        >>> key = generate_cache_key("What is Python?")
        >>> len(key)
        64

    Key with model and parameters:

        >>> key = generate_cache_key(
        ...     prompt="Explain AI",
        ...     model="gpt-4",
        ...     params={"temperature": 0.7, "max_tokens": 100}
        ... )

    Same inputs produce same key (deterministic):

        >>> key1 = generate_cache_key("Hello", model="gpt-4")
        >>> key2 = generate_cache_key("Hello", model="gpt-4")
        >>> key1 == key2
        True

    Different parameters produce different keys:

        >>> key1 = generate_cache_key("Hello", params={"temp": 0.0})
        >>> key2 = generate_cache_key("Hello", params={"temp": 1.0})
        >>> key1 != key2
        True

    Using custom kwargs:

        >>> key = generate_cache_key(
        ...     "Prompt",
        ...     user_id="user123",
        ...     session="abc"
        ... )

    See Also
    --------
    get_cache_key : Simplified wrapper for common use cases.
    generate_model_cache_key : Specialized for model request caching.
    """
    key_parts = [prompt]

    if model:
        key_parts.append(f"model:{model}")

    if params:
        sorted_params = json.dumps(params, sort_keys=True)
        key_parts.append(f"params:{sorted_params}")

    # Add any additional kwargs (sorted for determinism)
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")

    key_string = "|".join(key_parts)

    if algorithm == "md5":
        return hashlib.md5(key_string.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(key_string.encode()).hexdigest()
    else:
        return hashlib.sha256(key_string.encode()).hexdigest()


def generate_model_cache_key(
    model_id: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Generate cache key specifically for model generation requests.

    Specialized key generation for caching model responses. Includes all
    parameters that affect model output, ensuring that different generation
    configurations produce different cache keys.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g., "gpt-4", "claude-3-opus").
    prompt : str
        The prompt text sent to the model.
    temperature : float
        Sampling temperature. Default 0.0 (deterministic).
    max_tokens : Optional[int]
        Maximum number of tokens to generate.
    **kwargs : Any
        Additional generation parameters (e.g., top_p, frequency_penalty).

    Returns
    -------
    str
        64-character SHA-256 hash key.

    Examples
    --------
    Basic model cache key:

        >>> from insideLLMs.caching_unified import generate_model_cache_key
        >>> key = generate_model_cache_key(
        ...     model_id="gpt-4",
        ...     prompt="Hello, world!"
        ... )
        >>> len(key)
        64

    Key with generation parameters:

        >>> key = generate_model_cache_key(
        ...     model_id="claude-3",
        ...     prompt="Explain quantum computing",
        ...     temperature=0.7,
        ...     max_tokens=500
        ... )

    Temperature 0 caching (deterministic outputs):

        >>> key = generate_model_cache_key(
        ...     model_id="gpt-4",
        ...     prompt="2+2=",
        ...     temperature=0.0
        ... )
        >>> # Safe to cache - output will be identical

    Note: Temperature > 0 produces non-deterministic outputs:

        >>> key = generate_model_cache_key(
        ...     model_id="gpt-4",
        ...     prompt="Write a poem",
        ...     temperature=1.0
        ... )
        >>> # Caching may return stale/unexpected responses

    See Also
    --------
    generate_cache_key : General-purpose key generation.
    CachedModel : Wrapper that uses this function internally.
    """
    key_parts = {
        "model_id": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for k, v in sorted(kwargs.items()):
        key_parts[k] = v

    key_str = json.dumps(key_parts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


# =============================================================================
# Abstract Base Cache
# =============================================================================


class BaseCacheABC(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set a value in the cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self.get(key) is not None


# =============================================================================
# In-Memory Cache (Simple)
# =============================================================================


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
        Default time-to-live in seconds for cache entries. None means
        entries never expire. Can be overridden per-entry in set().

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

        >>> from insideLLMs.caching_unified import InMemoryCache
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


# =============================================================================
# Disk Cache (SQLite-based)
# =============================================================================


class DiskCache(BaseCacheABC[T]):
    """SQLite-based persistent disk cache with size management.

    A persistent cache implementation that stores entries in a SQLite database.
    Supports size-based eviction, TTL expiration, and data export/import.
    Ideal for caches that need to survive process restarts or are too large
    for memory.

    Parameters
    ----------
    path : Optional[Union[str, Path]]
        Path to the SQLite database file. If None, uses the default location
        at ~/.cache/insideLLMs/response_cache.db.
    max_size_mb : int
        Maximum cache size in megabytes. When exceeded, LRU entries are
        evicted until size is below 80% of the limit. Default is 100 MB.
    default_ttl : Optional[int]
        Default time-to-live in seconds. None means no expiration.

    Attributes
    ----------
    _path : Path
        Path to the SQLite database file.
    _max_size_bytes : int
        Maximum cache size in bytes.
    _default_ttl : Optional[int]
        Default TTL for new entries.
    _local : threading.local
        Thread-local storage for database connections.
    _stats : CacheStats
        Statistics tracking.

    Examples
    --------
    Basic disk cache:

        >>> from insideLLMs.caching_unified import DiskCache
        >>> cache = DiskCache(path="/tmp/my_cache.db")
        >>> cache.set("key1", {"large": "data"})
        >>> value = cache.get("key1")

    Default location:

        >>> cache = DiskCache()  # Uses ~/.cache/insideLLMs/response_cache.db
        >>> cache.set("persistent", "data")
        >>> # Data persists across restarts

    Size-limited cache:

        >>> cache = DiskCache(max_size_mb=50)  # 50 MB limit
        >>> # Automatically evicts when size exceeded

    Export and import:

        >>> cache = DiskCache()
        >>> cache.set("data1", "value1")
        >>> cache.set("data2", "value2")
        >>> count = cache.export_to_file("/tmp/backup.json")
        >>> print(f"Exported {count} entries")
        >>> # Later, import to another cache
        >>> new_cache = DiskCache(path="/tmp/new_cache.db")
        >>> imported = new_cache.import_from_file("/tmp/backup.json")

    With TTL:

        >>> cache = DiskCache(default_ttl=86400)  # 24 hours
        >>> cache.set("daily_data", "value")

    Notes
    -----
    - Each thread gets its own database connection for thread safety.
    - The database is automatically created if it doesn't exist.
    - VACUUM is run after evictions to reclaim disk space.
    - Values must be JSON-serializable.

    See Also
    --------
    InMemoryCache : Faster but non-persistent alternative.
    StrategyCache : In-memory cache with more eviction options.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        max_size_mb: int = 100,
        default_ttl: Optional[int] = None,
    ):
        if path is None:
            cache_dir = Path.home() / ".cache" / "insideLLMs"
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / "response_cache.db"

        self._path = Path(path)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl
        self._local = threading.local()
        self._stats = CacheStats()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                hit_count INTEGER DEFAULT 0,
                last_accessed REAL,
                metadata TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)")
        conn.commit()

    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row is None:
            self._stats.misses += 1
            return None

        if row["expires_at"] is not None and time.time() > row["expires_at"]:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            self._stats.misses += 1
            return None

        conn.execute(
            "UPDATE cache SET hit_count = hit_count + 1, last_accessed = ? WHERE key = ?",
            (time.time(), key),
        )
        conn.commit()
        self._stats.hits += 1

        return json.loads(row["value"])

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set a value in the cache."""
        conn = self._get_conn()
        self._evict_if_needed()

        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl if ttl is not None else None
        metadata_json = json.dumps(metadata) if metadata else None

        conn.execute(
            """
            INSERT OR REPLACE INTO cache
            (key, value, created_at, expires_at, hit_count, last_accessed, metadata)
            VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (key, json.dumps(value), time.time(), expires_at, time.time(), metadata_json),
        )
        conn.commit()

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all entries."""
        conn = self._get_conn()
        conn.execute("DELETE FROM cache")
        conn.commit()
        self._stats = CacheStats()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) as count FROM cache")
        row = cursor.fetchone()
        self._stats.entry_count = row["count"] if row else 0
        total = self._stats.hits + self._stats.misses
        self._stats.hit_rate = self._stats.hits / total if total > 0 else 0.0
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            entry_count=self._stats.entry_count,
            evictions=self._stats.evictions,
            hit_rate=self._stats.hit_rate,
        )

    def _evict_if_needed(self) -> None:
        """Evict entries if cache exceeds size limit."""
        conn = self._get_conn()
        db_size = self._path.stat().st_size if self._path.exists() else 0

        if db_size < self._max_size_bytes:
            return

        # Evict expired entries first
        conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )

        # If still too big, evict LRU entries
        while self._path.stat().st_size > self._max_size_bytes * 0.8:
            cursor = conn.execute("""
                DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache ORDER BY last_accessed ASC LIMIT 100
                )
            """)
            if cursor.rowcount == 0:
                break
            self._stats.evictions += cursor.rowcount

        conn.commit()
        conn.execute("VACUUM")
        conn.commit()

    def export_to_file(self, path: Union[str, Path]) -> int:
        """Export cache to a JSON file."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT key, value, created_at, metadata FROM cache")

        entries = []
        for row in cursor:
            entries.append(
                {
                    "key": row["key"],
                    "value": json.loads(row["value"]),
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                }
            )

        with open(path, "w") as f:
            json.dump(entries, f, indent=2)

        return len(entries)

    def import_from_file(self, path: Union[str, Path]) -> int:
        """Import cache from a JSON file."""
        with open(path) as f:
            entries = json.load(f)

        count = 0
        for entry in entries:
            self.set(entry["key"], entry["value"], metadata=entry.get("metadata"))
            count += 1

        return count


# =============================================================================
# Strategy-Based Cache (with multiple eviction strategies)
# =============================================================================


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

        >>> from insideLLMs.caching_unified import StrategyCache
        >>> cache = StrategyCache()
        >>> entry = cache.set("greeting", "Hello!")
        >>> result = cache.get("greeting")
        >>> print(result.hit, result.value)
        True Hello!

    LRU cache (Least Recently Used):

        >>> from insideLLMs.caching_unified import StrategyCache, CacheConfig, CacheStrategy
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
    - All operations are thread-safe via RLock.
    - Expired entries are lazily removed on access.
    - Returns CacheLookupResult with timing information.
    - Aliased as BaseCache for backward compatibility.

    See Also
    --------
    CacheConfig : Configuration options.
    CacheStrategy : Available eviction strategies.
    PromptCache : Specialized cache for LLM prompts.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache with configuration.

        Parameters
        ----------
        config : Optional[CacheConfig]
            Cache configuration. Uses defaults if None.
        """
        self.config = config or CacheConfig()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> CacheLookupResult:
        """Get value from cache."""
        start_time = time.time()

        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._stats.misses += 1
                return CacheLookupResult(
                    hit=False,
                    value=None,
                    key=key,
                    lookup_time_ms=(time.time() - start_time) * 1000,
                )

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

            ttl = ttl_seconds or self.config.ttl_seconds
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


# =============================================================================
# Prompt Cache (LLM-specific)
# =============================================================================


class PromptCache(StrategyCache):
    """Cache specialized for LLM prompts and responses with semantic matching.

    PromptCache extends StrategyCache with LLM-specific features including:
    - Semantic similarity matching for finding similar cached prompts
    - Prompt-specific caching methods (cache_response, get_response)
    - Prompt-to-key mapping for efficient lookups

    Parameters
    ----------
    config : Optional[CacheConfig]
        Cache configuration. Uses defaults if None.
    similarity_threshold : float
        Minimum similarity score (0.0 to 1.0) for find_similar() to return
        matches. Higher values require closer matches. Default is 0.95.

    Attributes
    ----------
    similarity_threshold : float
        Configured similarity threshold for matching.
    _prompt_keys : dict[str, str]
        Mapping from prompt hashes to cache keys for fast lookup.

    Examples
    --------
    Basic prompt caching:

        >>> from insideLLMs.caching_unified import PromptCache
        >>> cache = PromptCache()
        >>> entry = cache.cache_response(
        ...     prompt="What is Python?",
        ...     response="Python is a programming language.",
        ...     model="gpt-4"
        ... )
        >>> result = cache.get_response("What is Python?", model="gpt-4")
        >>> print(result.hit, result.value)
        True Python is a programming language.

    Caching with parameters:

        >>> cache = PromptCache()
        >>> cache.cache_response(
        ...     prompt="Explain AI",
        ...     response="AI is...",
        ...     model="claude-3",
        ...     params={"temperature": 0.7, "max_tokens": 100}
        ... )
        >>> # Same prompt with different params = different cache entry
        >>> cache.cache_response(
        ...     prompt="Explain AI",
        ...     response="AI is... (different)",
        ...     model="claude-3",
        ...     params={"temperature": 0.0, "max_tokens": 50}
        ... )

    Finding similar cached prompts:

        >>> cache = PromptCache(similarity_threshold=0.8)
        >>> cache.cache_response("What is machine learning?", "ML is...")
        >>> cache.cache_response("Explain deep learning", "DL is...")
        >>> similar = cache.find_similar("What is ML?", limit=3)
        >>> for prompt, entry, score in similar:
        ...     print(f"{prompt[:30]}... (similarity: {score:.2f})")

    Lookup by prompt only (ignoring model/params):

        >>> cache = PromptCache()
        >>> cache.cache_response("Hello", "Hi there!", model="gpt-4")
        >>> result = cache.get_by_prompt("Hello")
        >>> print(result.hit)
        True

    With metadata:

        >>> cache = PromptCache()
        >>> cache.cache_response(
        ...     prompt="Translate: Hello",
        ...     response="Hola",
        ...     metadata={"language": "Spanish", "tokens_used": 10}
        ... )
        >>> result = cache.get_response("Translate: Hello")
        >>> if result.entry:
        ...     print(result.entry.metadata["language"])
        Spanish

    Notes
    -----
    - Similarity is calculated using word overlap (Jaccard index).
    - Prompts are hashed using MD5 for the prompt-to-key mapping.
    - Inherits all features from StrategyCache.

    See Also
    --------
    StrategyCache : Parent class with eviction strategies.
    generate_cache_key : Used to create cache keys.
    CacheWarmer : For preloading prompts.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        similarity_threshold: float = 0.95,
    ):
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self._prompt_keys: dict[str, str] = {}  # prompt hash -> cache key

    def cache_response(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Cache an LLM response."""
        key = generate_cache_key(prompt, model, params, self.config.hash_algorithm)

        entry_metadata = {
            "prompt": prompt,
            "model": model,
            "params": params,
            **(metadata or {}),
        }

        entry = self.set(key, response, metadata=entry_metadata)

        # Track prompt -> key mapping
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self._prompt_keys[prompt_hash] = key

        return entry

    def get_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> CacheLookupResult:
        """Get cached response for a prompt."""
        key = generate_cache_key(prompt, model, params, self.config.hash_algorithm)
        return self.get(key)

    def get_by_prompt(self, prompt: str) -> CacheLookupResult:
        """Get cached response by prompt alone (ignoring model/params)."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        key = self._prompt_keys.get(prompt_hash)

        if key:
            return self.get(key)

        return CacheLookupResult(hit=False, value=None, key="")

    def find_similar(
        self,
        prompt: str,
        limit: int = 5,
    ) -> list[tuple[str, CacheEntry, float]]:
        """Find similar cached prompts."""
        results = []

        with self._lock:
            for entry in self._entries.values():
                if entry.is_expired():
                    continue

                cached_prompt = entry.metadata.get("prompt", "")
                if cached_prompt:
                    similarity = self._calculate_similarity(prompt, cached_prompt)
                    if similarity >= self.similarity_threshold:
                        results.append((cached_prompt, entry, similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using Jaccard index."""
        return word_overlap_similarity(text1, text2)


# =============================================================================
# Cached Model Wrapper
# =============================================================================


class CachedModel:
    """Wrapper that adds caching to any model.

    Args:
        model: The underlying model to wrap.
        cache: Cache backend to use.
        cache_only_deterministic: Only cache requests with temperature=0.
    """

    def __init__(
        self,
        model: Any,
        cache: Optional[BaseCacheABC] = None,
        cache_only_deterministic: bool = True,
    ):
        self._model = model
        self._cache = cache or InMemoryCache()
        self._cache_only_deterministic = cache_only_deterministic

    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self._model

    @property
    def cache(self) -> BaseCacheABC:
        """Get the cache backend."""
        return self._cache

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response, using cache if available."""
        should_cache = not self._cache_only_deterministic or temperature == 0

        if should_cache:
            model_id = getattr(self._model, "model_id", str(type(self._model).__name__))
            cache_key = generate_model_cache_key(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._deserialize_response(cached)

        response = self._model.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if should_cache:
            self._cache.set(
                cache_key,
                self._serialize_response(response),
                metadata={"model_id": model_id, "prompt_preview": prompt[:100]},
            )

        return response

    def _serialize_response(self, response: ModelResponse) -> dict[str, Any]:
        """Serialize a ModelResponse for caching."""
        return asdict(response)

    def _deserialize_response(self, data: dict[str, Any]) -> ModelResponse:
        """Deserialize a cached response."""
        return ModelResponse(**data)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model."""
        return getattr(self._model, name)


# =============================================================================
# Cache Warmer
# =============================================================================


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

        >>> from insideLLMs.caching_unified import CacheWarmer, StrategyCache
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


# =============================================================================
# Memoization
# =============================================================================


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

        >>> from insideLLMs.caching_unified import MemoizedFunction
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

        >>> from insideLLMs.caching_unified import StrategyCache, CacheConfig
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

        >>> from insideLLMs.caching_unified import memoize
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

        >>> from insideLLMs.caching_unified import StrategyCache, CacheConfig, CacheStrategy
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


# =============================================================================
# Cache Namespace
# =============================================================================


class CacheNamespace:
    """Manages multiple named caches with shared configuration.

    CacheNamespace provides organized management of multiple cache instances
    with logical naming. Useful for multi-tenant applications, separating
    caches by model type, or organizing different cache categories.

    Parameters
    ----------
    default_config : Optional[CacheConfig]
        Default configuration for new caches. If None, uses CacheConfig
        defaults.

    Attributes
    ----------
    default_config : CacheConfig
        Default configuration for new caches.
    _caches : dict[str, BaseCache]
        Dictionary mapping names to cache instances.
    _lock : threading.RLock
        Lock for thread-safe cache management.

    Examples
    --------
    Basic namespace usage:

        >>> from insideLLMs.caching_unified import CacheNamespace
        >>> namespace = CacheNamespace()
        >>> user_cache = namespace.get_cache("users")
        >>> user_cache.set("user_1", {"name": "Alice"})
        >>> product_cache = namespace.get_cache("products")
        >>> product_cache.set("prod_1", {"name": "Widget"})

    With default configuration:

        >>> from insideLLMs.caching_unified import CacheNamespace, CacheConfig
        >>> config = CacheConfig(max_size=500, ttl_seconds=1800)
        >>> namespace = CacheNamespace(config)
        >>> cache = namespace.get_cache("my_cache")  # Uses config

    Creating prompt caches:

        >>> namespace = CacheNamespace()
        >>> gpt4_cache = namespace.get_prompt_cache("gpt4")
        >>> claude_cache = namespace.get_prompt_cache("claude")
        >>> gpt4_cache.cache_response("Hello", "Hi from GPT-4!")
        >>> claude_cache.cache_response("Hello", "Hi from Claude!")

    Custom configuration per cache:

        >>> from insideLLMs.caching_unified import CacheNamespace, CacheConfig, CacheStrategy
        >>> namespace = CacheNamespace()
        >>> hot_config = CacheConfig(max_size=100, strategy=CacheStrategy.LFU)
        >>> hot_cache = namespace.get_cache("hot_data", config=hot_config)
        >>> cold_config = CacheConfig(max_size=10000, ttl_seconds=86400)
        >>> cold_cache = namespace.get_cache("cold_data", config=cold_config)

    Managing caches:

        >>> namespace = CacheNamespace()
        >>> _ = namespace.get_cache("temp")
        >>> _ = namespace.get_cache("permanent")
        >>> print(namespace.list_caches())
        ['temp', 'permanent']
        >>> namespace.delete_cache("temp")
        >>> print(namespace.list_caches())
        ['permanent']

    Getting aggregated statistics:

        >>> namespace = CacheNamespace()
        >>> cache1 = namespace.get_cache("cache1")
        >>> cache2 = namespace.get_cache("cache2")
        >>> cache1.set("a", 1)
        >>> cache2.set("b", 2)
        >>> all_stats = namespace.get_all_stats()
        >>> for name, stats in all_stats.items():
        ...     print(f"{name}: {stats.entry_count} entries")

    Clearing all caches:

        >>> namespace = CacheNamespace()
        >>> for i in range(5):
        ...     cache = namespace.get_cache(f"cache_{i}")
        ...     cache.set("key", "value")
        >>> namespace.clear_all()  # Clears all cache contents

    Notes
    -----
    - Caches are created lazily on first access.
    - The same name always returns the same cache instance.
    - Thread-safe for concurrent access.

    See Also
    --------
    create_namespace : Convenience function for creating namespaces.
    StrategyCache : Type of cache created by get_cache().
    PromptCache : Type of cache created by get_prompt_cache().
    """

    def __init__(self, default_config: Optional[CacheConfig] = None):
        self.default_config = default_config or CacheConfig()
        self._caches: dict[str, BaseCache] = {}
        self._lock = threading.RLock()

    def get_cache(
        self,
        name: str,
        config: Optional[CacheConfig] = None,
    ) -> BaseCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = StrategyCache(config or self.default_config)
            return self._caches[name]

    def get_prompt_cache(
        self,
        name: str,
        config: Optional[CacheConfig] = None,
    ) -> PromptCache:
        """Get or create a named prompt cache."""
        with self._lock:
            if name not in self._caches or not isinstance(self._caches[name], PromptCache):
                self._caches[name] = PromptCache(config or self.default_config)
            return self._caches[name]

    def delete_cache(self, name: str) -> bool:
        """Delete a named cache."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                return True
            return False

    def list_caches(self) -> list[str]:
        """List all cache names."""
        with self._lock:
            return list(self._caches.keys())

    def get_all_stats(self) -> dict[str, CacheStats]:
        """Get statistics for all caches."""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}

    def clear_all(self):
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()


# =============================================================================
# Response Deduplicator
# =============================================================================


class ResponseDeduplicator:
    """Identifies and deduplicates identical or similar responses.

    ResponseDeduplicator tracks responses and detects duplicates based on
    exact matching or configurable similarity thresholds. Useful for
    identifying redundant API calls or response patterns.

    Parameters
    ----------
    similarity_threshold : float
        Threshold for considering responses as duplicates. Range 0.0 to 1.0.
        1.0 means exact match only, lower values allow fuzzy matching.
        Default is 1.0 (exact match).

    Attributes
    ----------
    similarity_threshold : float
        The configured similarity threshold.
    _responses : list[tuple[str, str, Any]]
        List of (prompt, response, metadata) tuples for unique responses.

    Examples
    --------
    Exact match deduplication:

        >>> from insideLLMs.caching_unified import ResponseDeduplicator
        >>> dedup = ResponseDeduplicator()  # threshold=1.0 (exact)
        >>> is_dup, idx = dedup.add("p1", "Hello, World!")
        >>> print(f"Duplicate: {is_dup}")
        Duplicate: False
        >>> is_dup, idx = dedup.add("p2", "Hello, World!")
        >>> print(f"Duplicate: {is_dup}, matches index: {idx}")
        Duplicate: True, matches index: 0

    Similarity-based deduplication:

        >>> dedup = ResponseDeduplicator(similarity_threshold=0.8)
        >>> dedup.add("p1", "The quick brown fox jumps over the lazy dog")
        >>> is_dup, idx = dedup.add("p2", "The quick brown fox jumps over a lazy dog")
        >>> print(f"Similar enough: {is_dup}")  # True - high word overlap
        Similar enough: True

    Adding metadata:

        >>> dedup = ResponseDeduplicator()
        >>> dedup.add("prompt", "response", metadata={"model": "gpt-4", "tokens": 50})
        >>> unique = dedup.get_unique_responses()
        >>> for prompt, response, meta in unique:
        ...     print(f"Model: {meta['model']}")

    Getting unique responses:

        >>> dedup = ResponseDeduplicator()
        >>> dedup.add("p1", "Response A")
        >>> dedup.add("p2", "Response B")
        >>> dedup.add("p3", "Response A")  # Duplicate
        >>> unique = dedup.get_unique_responses()
        >>> print(f"Unique count: {len(unique)}")
        Unique count: 2

    Clearing stored responses:

        >>> dedup = ResponseDeduplicator()
        >>> for i in range(100):
        ...     dedup.add(f"p{i}", f"Response {i}")
        >>> print(len(dedup.get_unique_responses()))
        100
        >>> dedup.clear()
        >>> print(len(dedup.get_unique_responses()))
        0

    Use case - detecting redundant API calls:

        >>> dedup = ResponseDeduplicator(similarity_threshold=0.95)
        >>> responses = ["Answer A", "Answer A", "Answer B", "Answer A again"]
        >>> for i, resp in enumerate(responses):
        ...     is_dup, _ = dedup.add(f"prompt_{i}", resp)
        ...     if is_dup:
        ...         print(f"Response {i} is redundant")

    Notes
    -----
    - Similarity is calculated using Jaccard index (word overlap).
    - For exact matching (threshold=1.0), uses simple string comparison.
    - Memory usage grows linearly with unique responses.

    See Also
    --------
    PromptCache.find_similar : Similar functionality in cache context.
    """

    def __init__(self, similarity_threshold: float = 1.0):
        self.similarity_threshold = similarity_threshold
        self._responses: list[tuple[str, str, Any]] = []

    def add(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Any] = None,
    ) -> tuple[bool, Optional[int]]:
        """Add response, returning whether it's a duplicate."""
        for i, (_, existing_response, _) in enumerate(self._responses):
            if self._is_duplicate(response, existing_response):
                return True, i

        self._responses.append((prompt, response, metadata))
        return False, None

    def _is_duplicate(self, response1: str, response2: str) -> bool:
        """Check if two responses are duplicates."""
        if self.similarity_threshold == 1.0:
            return response1 == response2

        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0.0

        return similarity >= self.similarity_threshold

    def get_unique_responses(self) -> list[tuple[str, str, Any]]:
        """Get all unique responses."""
        return self._responses.copy()

    def get_duplicate_count(self) -> int:
        """Get count of duplicates found."""
        return 0

    def clear(self):
        """Clear stored responses."""
        self._responses.clear()


# =============================================================================
# Async Adapter
# =============================================================================


class AsyncCacheAdapter:
    """Adapter for using synchronous caches in async contexts.

    AsyncCacheAdapter wraps a synchronous cache implementation to provide
    async/await compatible methods. This enables integration of the caching
    module with asyncio-based applications.

    Parameters
    ----------
    cache : BaseCache
        The synchronous cache instance to wrap.

    Attributes
    ----------
    cache : BaseCache
        The wrapped synchronous cache.

    Examples
    --------
    Basic async usage:

        >>> from insideLLMs.caching_unified import AsyncCacheAdapter, StrategyCache
        >>> import asyncio
        >>> cache = StrategyCache()
        >>> async_cache = AsyncCacheAdapter(cache)
        >>> async def main():
        ...     await async_cache.set("key", "value")
        ...     result = await async_cache.get("key")
        ...     print(f"Hit: {result.hit}, Value: {result.value}")
        >>> asyncio.run(main())
        Hit: True, Value: value

    In an async web application:

        >>> async def handle_request(prompt):
        ...     result = await async_cache.get(generate_cache_key(prompt))
        ...     if result.hit:
        ...         return result.value
        ...     # Generate response...
        ...     response = await generate_llm_response(prompt)
        ...     await async_cache.set(generate_cache_key(prompt), response)
        ...     return response

    With TTL and metadata:

        >>> async def cache_with_meta():
        ...     entry = await async_cache.set(
        ...         key="data",
        ...         value={"result": 42},
        ...         ttl_seconds=300,
        ...         metadata={"source": "api"}
        ...     )
        ...     print(f"Created: {entry.created_at}")

    Async delete and clear:

        >>> async def cleanup():
        ...     deleted = await async_cache.delete("old_key")
        ...     print(f"Deleted: {deleted}")
        ...     await async_cache.clear()
        ...     print("Cache cleared")

    Concurrent cache operations:

        >>> async def concurrent_ops():
        ...     tasks = [
        ...         async_cache.set(f"key_{i}", f"value_{i}")
        ...         for i in range(10)
        ...     ]
        ...     await asyncio.gather(*tasks)
        ...     # All operations complete concurrently

    Notes
    -----
    - The underlying operations are synchronous; this adapter provides
      async syntax compatibility but does not add true async I/O.
    - For true async caching with network backends (Redis, etc.),
      use dedicated async libraries.
    - Thread-safety depends on the wrapped cache implementation.

    See Also
    --------
    StrategyCache : Common cache type to wrap.
    PromptCache : LLM-specific cache to wrap.
    """

    def __init__(self, cache: BaseCache):
        self.cache = cache

    async def get(self, key: str) -> CacheLookupResult:
        """Async get from cache."""
        return self.cache.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Async set in cache."""
        return self.cache.set(key, value, ttl_seconds, metadata)

    async def delete(self, key: str) -> bool:
        """Async delete from cache."""
        return self.cache.delete(key)

    async def clear(self):
        """Async clear cache."""
        self.cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_cache(
    max_size: int = 1000,
    ttl_seconds: Optional[int] = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU,
) -> BaseCache:
    """Create a basic StrategyCache with common settings.

    Convenience function for creating a cache with the most commonly
    used configuration options. For more control, create CacheConfig
    and StrategyCache directly.

    Parameters
    ----------
    max_size : int
        Maximum number of cache entries. Default 1000.
    ttl_seconds : Optional[int]
        Time-to-live for entries in seconds. None for no expiration.
        Default 3600 (1 hour).
    strategy : CacheStrategy
        Eviction strategy. Default LRU.

    Returns
    -------
    BaseCache
        A new StrategyCache instance.

    Examples
    --------
    Default cache:

        >>> from insideLLMs.caching import create_cache
        >>> cache = create_cache()
        >>> cache.set("key", "value")

    High-capacity cache:

        >>> cache = create_cache(max_size=10000, ttl_seconds=86400)

    LFU eviction strategy:

        >>> from insideLLMs.caching import create_cache, CacheStrategy
        >>> cache = create_cache(strategy=CacheStrategy.LFU)

    No expiration:

        >>> cache = create_cache(ttl_seconds=None)

    See Also
    --------
    create_prompt_cache : For LLM-specific caching.
    StrategyCache : The underlying cache class.
    """
    config = CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds, strategy=strategy)
    return StrategyCache(config)


def create_prompt_cache(
    max_size: int = 1000,
    ttl_seconds: Optional[int] = 3600,
    similarity_threshold: float = 0.95,
) -> PromptCache:
    """Create a PromptCache specialized for LLM responses.

    Convenience function for creating a prompt cache with semantic
    similarity matching. Ideal for caching LLM responses.

    Parameters
    ----------
    max_size : int
        Maximum number of cache entries. Default 1000.
    ttl_seconds : Optional[int]
        Time-to-live for entries in seconds. Default 3600 (1 hour).
    similarity_threshold : float
        Minimum similarity for find_similar() matches. Range 0.0-1.0.
        Default 0.95 (95% similar).

    Returns
    -------
    PromptCache
        A new PromptCache instance.

    Examples
    --------
    Default prompt cache:

        >>> from insideLLMs.caching import create_prompt_cache
        >>> cache = create_prompt_cache()
        >>> cache.cache_response("Hello", "Hi there!", model="gpt-4")

    Lower similarity threshold:

        >>> cache = create_prompt_cache(similarity_threshold=0.8)
        >>> # Will match more prompts as "similar"

    Long TTL for stable responses:

        >>> cache = create_prompt_cache(ttl_seconds=86400)  # 24 hours

    High-capacity for production:

        >>> cache = create_prompt_cache(max_size=10000)

    See Also
    --------
    create_cache : For general-purpose caching.
    PromptCache : The underlying cache class.
    """
    config = CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds)
    return PromptCache(config, similarity_threshold)


def create_cache_warmer(
    cache: BaseCache,
    generator: Callable[[str], Any],
) -> CacheWarmer:
    """Create a CacheWarmer for preloading cache entries.

    Convenience function for creating a cache warmer that pre-populates
    a cache with generated responses.

    Parameters
    ----------
    cache : BaseCache
        The cache to warm with pre-generated values.
    generator : Callable[[str], Any]
        Function that generates values from prompts. Typically an LLM
        generate function or wrapper.

    Returns
    -------
    CacheWarmer
        A new CacheWarmer instance.

    Examples
    --------
    Basic usage:

        >>> from insideLLMs.caching import create_cache, create_cache_warmer
        >>> cache = create_cache()
        >>> def gen(prompt):
        ...     return f"Response for: {prompt}"
        >>> warmer = create_cache_warmer(cache, gen)
        >>> warmer.add_prompt("Hello")
        >>> warmer.warm()

    With an LLM:

        >>> from insideLLMs.caching import create_cache_warmer
        >>> warmer = create_cache_warmer(cache, llm.generate)
        >>> warmer.add_prompt("Common question 1", priority=10)
        >>> warmer.add_prompt("Common question 2", priority=5)
        >>> warmer.warm(batch_size=10)

    See Also
    --------
    CacheWarmer : The underlying warmer class.
    """
    return CacheWarmer(cache, generator)


def create_namespace(
    default_max_size: int = 1000,
    default_ttl_seconds: Optional[int] = 3600,
) -> CacheNamespace:
    """Create a CacheNamespace for managing multiple named caches.

    Convenience function for creating a namespace with default configuration
    for all caches created within it.

    Parameters
    ----------
    default_max_size : int
        Default max_size for new caches. Default 1000.
    default_ttl_seconds : Optional[int]
        Default TTL for new caches in seconds. Default 3600 (1 hour).

    Returns
    -------
    CacheNamespace
        A new CacheNamespace instance.

    Examples
    --------
    Basic namespace:

        >>> from insideLLMs.caching import create_namespace
        >>> namespace = create_namespace()
        >>> users = namespace.get_cache("users")
        >>> products = namespace.get_cache("products")

    Custom defaults:

        >>> namespace = create_namespace(
        ...     default_max_size=500,
        ...     default_ttl_seconds=1800
        ... )

    Prompt caches within namespace:

        >>> namespace = create_namespace()
        >>> gpt4 = namespace.get_prompt_cache("gpt4")
        >>> claude = namespace.get_prompt_cache("claude")

    See Also
    --------
    CacheNamespace : The underlying namespace class.
    """
    config = CacheConfig(max_size=default_max_size, ttl_seconds=default_ttl_seconds)
    return CacheNamespace(config)


def get_cache_key(
    prompt: str,
    model: Optional[str] = None,
    params: Optional[dict] = None,
) -> str:
    """Generate a cache key for a prompt (convenience wrapper).

    Simplified wrapper around generate_cache_key() for common use cases.
    Uses SHA-256 hashing by default.

    Parameters
    ----------
    prompt : str
        The prompt text.
    model : Optional[str]
        Model identifier (e.g., "gpt-4").
    params : Optional[dict]
        Generation parameters.

    Returns
    -------
    str
        64-character hexadecimal hash key.

    Examples
    --------
    Basic key:

        >>> from insideLLMs.caching import get_cache_key
        >>> key = get_cache_key("Hello")
        >>> len(key)
        64

    With model and params:

        >>> key = get_cache_key(
        ...     "Explain AI",
        ...     model="gpt-4",
        ...     params={"temperature": 0.7}
        ... )

    See Also
    --------
    generate_cache_key : Full-featured key generation.
    """
    return generate_cache_key(prompt, model, params)


def cached_response(
    prompt: str,
    generator: Callable[[str], str],
    cache: Optional[PromptCache] = None,
    model: Optional[str] = None,
    params: Optional[dict] = None,
) -> tuple[str, bool]:
    """Get a cached response or generate and cache a new one.

    One-shot convenience function for cache-or-compute pattern. Checks
    the cache first, returns cached value if found, otherwise generates
    and caches a new response.

    Parameters
    ----------
    prompt : str
        The prompt to look up or generate a response for.
    generator : Callable[[str], str]
        Function to generate a response if not cached. Takes prompt,
        returns response string.
    cache : Optional[PromptCache]
        Cache to use. If None, creates a new PromptCache.
    model : Optional[str]
        Model identifier for cache key.
    params : Optional[dict]
        Generation parameters for cache key.

    Returns
    -------
    tuple[str, bool]
        Tuple of (response, was_cached). was_cached is True if the
        response came from cache, False if newly generated.

    Examples
    --------
    Basic usage:

        >>> from insideLLMs.caching import cached_response
        >>> def generate(prompt):
        ...     return f"Response for: {prompt}"
        >>> response, cached = cached_response("Hello", generate)
        >>> print(f"Cached: {cached}")
        Cached: False
        >>> response, cached = cached_response("Hello", generate)
        >>> print(f"Cached: {cached}")
        Cached: True

    With persistent cache:

        >>> from insideLLMs.caching import cached_response, create_prompt_cache
        >>> cache = create_prompt_cache()
        >>> response, _ = cached_response("Query", generate, cache=cache)
        >>> # Cache persists for subsequent calls
        >>> response, cached = cached_response("Query", generate, cache=cache)
        >>> print(f"Cached: {cached}")
        Cached: True

    With model and params:

        >>> response, cached = cached_response(
        ...     prompt="Explain AI",
        ...     generator=llm.generate,
        ...     cache=cache,
        ...     model="gpt-4",
        ...     params={"temperature": 0.7}
        ... )

    See Also
    --------
    PromptCache.get_response : Direct cache lookup.
    PromptCache.cache_response : Direct cache storage.
    """
    if cache is None:
        cache = create_prompt_cache()

    result = cache.get_response(prompt, model, params)

    if result.hit:
        return result.value, True

    response = generator(prompt)
    cache.cache_response(prompt, response, model, params)

    return response, False


# =============================================================================
# Global Default Cache
# =============================================================================


_default_cache: Optional[BaseCacheABC] = None


def get_default_cache() -> BaseCacheABC:
    """Get the global default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = InMemoryCache()
    return _default_cache


def set_default_cache(cache: BaseCacheABC) -> None:
    """Set the global default cache instance."""
    global _default_cache
    _default_cache = cache


def clear_default_cache() -> None:
    """Clear the global default cache."""
    global _default_cache
    if _default_cache is not None:
        _default_cache.clear()


def cached(
    cache: Optional[BaseCacheABC] = None,
    ttl: Optional[int] = None,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable:
    """Decorator to cache function results using simple cache."""
    _cache = cache or InMemoryCache()

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key_parts = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                }
                key = hashlib.sha256(
                    json.dumps(key_parts, sort_keys=True, default=str).encode()
                ).hexdigest()

            cached_result = _cache.get(key)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            _cache.set(key, result, ttl=ttl)

            return result

        wrapper.__wrapped__ = func
        return wrapper

    return decorator


# =============================================================================
# Exports
# =============================================================================


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
    "generate_model_cache_key",
    # Cache Implementations
    "BaseCacheABC",
    "InMemoryCache",
    "DiskCache",
    "StrategyCache",
    "BaseCache",  # Alias for StrategyCache
    "PromptCache",
    # Model Wrapper
    "CachedModel",
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
    # Global Cache
    "get_default_cache",
    "set_default_cache",
    "clear_default_cache",
    # Decorators
    "cached",
    "memoize",
]
