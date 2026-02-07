"""
Disk Cache Backend (SQLite-based)
=================================

Persistent disk cache with size management using SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from insideLLMs.caching.base import BaseCacheABC
from insideLLMs.caching.types import CacheStats

T = TypeVar("T")


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

        >>> from insideLLMs.caching import DiskCache
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


__all__ = ["DiskCache"]
