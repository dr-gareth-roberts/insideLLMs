"""
insideLLMs Caching System
=========================

Comprehensive caching infrastructure for LLM applications with multiple
backends, eviction strategies, and specialized prompt caching.

This package provides:

- **Multiple Cache Backends**: In-memory, disk (SQLite), and strategy-based caches
- **Eviction Strategies**: LRU, LFU, FIFO, TTL, and SIZE-based eviction
- **Prompt Caching**: Specialized cache for LLM prompts with semantic similarity
- **Async Support**: Async adapter for non-blocking cache operations
- **Utilities**: Cache warming, memoization, namespaces, and deduplication

Quick Start
-----------
>>> from insideLLMs.caching import create_cache, create_prompt_cache
>>> cache = create_cache()
>>> cache.set("key", "value")
>>> result = cache.get("key")
>>> print(result.hit, result.value)
True value

>>> prompt_cache = create_prompt_cache()
>>> prompt_cache.cache_response("Hello", "Hi there!", model="gpt-4")
>>> result = prompt_cache.get_response("Hello", model="gpt-4")
>>> print(result.value)
Hi there!

See Also
--------
insideLLMs.caching_unified : Legacy module (re-exports from here)
"""

# Types and Enums
from insideLLMs.caching.async_adapter import AsyncCacheAdapter

# Cache Backends
from insideLLMs.caching.backends import (
    BaseCache,
    DiskCache,
    InMemoryCache,
    StrategyCache,
)

# Base Classes and Key Generation
from insideLLMs.caching.base import (
    BaseCacheABC,
    generate_cache_key,
    generate_model_cache_key,
)
from insideLLMs.caching.deduplicator import ResponseDeduplicator

# Factory Functions
from insideLLMs.caching.factory import (
    cached,
    cached_response,
    clear_default_cache,
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    get_cache_key,
    get_default_cache,
    set_default_cache,
)
from insideLLMs.caching.model_wrapper import CachedModel
from insideLLMs.caching.namespace import CacheNamespace

# Specialized Caches
from insideLLMs.caching.prompt_cache import PromptCache
from insideLLMs.caching.types import (
    CacheConfig,
    CacheEntry,
    CacheEntryMixin,
    CacheLookupResult,
    CacheScope,
    CacheStats,
    CacheStatus,
    CacheStrategy,
)

# Utilities
from insideLLMs.caching.utilities import (
    CacheWarmer,
    MemoizedFunction,
    memoize,
)

__all__ = [
    # Enums
    "CacheStrategy",
    "CacheStatus",
    "CacheScope",
    # Configuration
    "CacheConfig",
    # Data Classes
    "CacheEntry",
    "CacheEntryMixin",
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
    "memoize",
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
    # Decorator
    "cached",
]
