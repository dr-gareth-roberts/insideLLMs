"""
Unified Caching Module (Re-export Shim)
=======================================

This module re-exports all caching functionality from the new
insideLLMs.caching package for backward compatibility.

All imports from this module continue to work as before:

    >>> from insideLLMs.caching_unified import create_cache, PromptCache
    >>> cache = create_cache()

For new code, prefer importing from insideLLMs.caching directly:

    >>> from insideLLMs.caching import create_cache, PromptCache

See Also
--------
insideLLMs.caching : The new modular caching package
"""

# Re-export everything from the new caching package
from insideLLMs.caching import (
    AsyncCacheAdapter,
    BaseCache,
    # Cache Implementations
    BaseCacheABC,
    # Configuration
    CacheConfig,
    # Model Wrapper
    CachedModel,
    # Data Classes
    CacheEntry,
    CacheEntryMixin,
    CacheLookupResult,
    CacheNamespace,
    CacheScope,
    CacheStats,
    CacheStatus,
    # Enums
    CacheStrategy,
    # Utilities
    CacheWarmer,
    DiskCache,
    InMemoryCache,
    MemoizedFunction,
    PromptCache,
    ResponseDeduplicator,
    StrategyCache,
    # Decorator
    cached,
    cached_response,
    clear_default_cache,
    # Convenience Functions
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    # Key Generation
    generate_cache_key,
    generate_model_cache_key,
    get_cache_key,
    # Global Cache
    get_default_cache,
    memoize,
    set_default_cache,
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
