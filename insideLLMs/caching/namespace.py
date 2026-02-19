"""
Cache Namespace
===============

Manages multiple named caches with shared configuration.
"""

from __future__ import annotations

import threading
from typing import Optional

from insideLLMs.caching.backends.strategy import StrategyCache
from insideLLMs.caching.prompt_cache import PromptCache
from insideLLMs.caching.types import CacheConfig, CacheStats

# Type alias for backward compatibility
BaseCache = StrategyCache


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

        >>> from insideLLMs.caching import CacheNamespace
        >>> namespace = CacheNamespace()
        >>> user_cache = namespace.get_cache("users")
        >>> user_cache.set("user_1", {"name": "Alice"})
        >>> product_cache = namespace.get_cache("products")
        >>> product_cache.set("prod_1", {"name": "Widget"})

    With default configuration:

        >>> from insideLLMs.caching import CacheNamespace, CacheConfig
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

        >>> from insideLLMs.caching import CacheNamespace, CacheConfig, CacheStrategy
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


__all__ = ["CacheNamespace"]
