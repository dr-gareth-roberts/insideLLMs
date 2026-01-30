"""
Async Cache Adapter
===================

Provides async interface to synchronous cache backends.
"""

from __future__ import annotations

from typing import Any, Optional

from insideLLMs.caching.backends.strategy import StrategyCache
from insideLLMs.caching.types import CacheEntry, CacheLookupResult

# Type alias for backward compatibility
BaseCache = StrategyCache


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

        >>> from insideLLMs.caching import AsyncCacheAdapter, StrategyCache
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

    async def clear(self) -> None:
        """Async clear cache."""
        self.cache.clear()


__all__ = ["AsyncCacheAdapter"]
