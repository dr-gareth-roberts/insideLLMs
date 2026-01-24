"""
Unified Caching Module for insideLLMs

This module provides comprehensive caching capabilities:
- Generic caching with multiple backends (InMemory, Disk)
- LLM-specific prompt caching with semantic similarity
- Multiple eviction strategies (LRU, LFU, FIFO, TTL, SIZE)
- Cache warming and memoization
- Model response caching wrapper

Design:
- BaseCache (ABC): Abstract interface for cache implementations
- InMemoryCache: Thread-safe in-memory LRU cache
- DiskCache: SQLite-based persistent cache
- PromptCache: LLM-specific cache with semantic matching
- CachedModel: Wrapper to add caching to any model
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
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based
    SIZE = "size"  # Size-based eviction


class CacheStatus(Enum):
    """Cache entry status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    EVICTED = "evicted"
    WARMING = "warming"


class CacheScope(Enum):
    """Cache scope for different contexts."""

    GLOBAL = "global"
    SESSION = "session"
    REQUEST = "request"
    USER = "user"
    MODEL = "model"


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    max_size: int = 1000
    ttl_seconds: Optional[int] = 3600  # 1 hour default
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_stats: bool = True
    enable_compression: bool = False
    hash_algorithm: str = "sha256"
    scope: CacheScope = CacheScope.GLOBAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """A single cache entry with metadata."""

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
        """Estimate memory size of cached value."""
        try:
            return len(json.dumps(self.value))
        except (TypeError, ValueError):
            return len(str(self.value))

    # is_expired() and touch() inherited from CacheEntryMixin

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Cache statistics."""

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
        """Convert to dictionary."""
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
    """Result of a cache lookup."""

    hit: bool
    value: Optional[Any]
    key: str
    entry: Optional[CacheEntry] = None
    lookup_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """
    Generate a deterministic cache key from prompt and parameters.

    Args:
        prompt: The prompt text.
        model: Optional model identifier.
        params: Optional generation parameters.
        algorithm: Hash algorithm to use (sha256, md5, sha1).
        **kwargs: Additional parameters to include in key.

    Returns:
        A unique hash key for this request.
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
    """Generate cache key specifically for model requests.

    Args:
        model_id: Model identifier.
        prompt: The prompt text.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.
        **kwargs: Additional parameters.

    Returns:
        A unique hash key for this request.
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
    """Simple in-memory cache with LRU eviction.

    Args:
        max_size: Maximum number of entries.
        default_ttl: Default time-to-live in seconds (None = no expiry).
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
    """SQLite-based persistent disk cache.

    Args:
        path: Path to the cache database file.
        max_size_mb: Maximum cache size in megabytes.
        default_ttl: Default time-to-live in seconds.
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
    """Cache with configurable eviction strategies.

    This is a more feature-rich cache supporting LRU, LFU, FIFO, TTL, and SIZE
    eviction strategies.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache with configuration."""
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
    """Cache specialized for LLM prompts and responses.

    Provides additional features like semantic similarity matching.
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
    """Preloads cache with common prompts."""

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
    """Wrapper for memoizing function calls with caching."""

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
    """Decorator for memoizing functions."""

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
    """Manages multiple named caches."""

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
    """Deduplicates identical or similar responses."""

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
    """Adapter for using cache in async contexts."""

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
    """Create a basic cache."""
    config = CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds, strategy=strategy)
    return StrategyCache(config)


def create_prompt_cache(
    max_size: int = 1000,
    ttl_seconds: Optional[int] = 3600,
    similarity_threshold: float = 0.95,
) -> PromptCache:
    """Create a prompt-specialized cache."""
    config = CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds)
    return PromptCache(config, similarity_threshold)


def create_cache_warmer(
    cache: BaseCache,
    generator: Callable[[str], Any],
) -> CacheWarmer:
    """Create a cache warmer."""
    return CacheWarmer(cache, generator)


def create_namespace(
    default_max_size: int = 1000,
    default_ttl_seconds: Optional[int] = 3600,
) -> CacheNamespace:
    """Create a cache namespace."""
    config = CacheConfig(max_size=default_max_size, ttl_seconds=default_ttl_seconds)
    return CacheNamespace(config)


def get_cache_key(
    prompt: str,
    model: Optional[str] = None,
    params: Optional[dict] = None,
) -> str:
    """Generate cache key for prompt."""
    return generate_cache_key(prompt, model, params)


def cached_response(
    prompt: str,
    generator: Callable[[str], str],
    cache: Optional[PromptCache] = None,
    model: Optional[str] = None,
    params: Optional[dict] = None,
) -> tuple[str, bool]:
    """Get cached response or generate new one."""
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
