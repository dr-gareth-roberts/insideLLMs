"""
Prompt Caching and Memoization Module

Intelligent caching for LLM operations including:
- Response caching with configurable strategies
- Semantic similarity-based cache lookup
- TTL and LRU cache management
- Cache warming and preloading
- Distributed cache support
- Cache analytics and statistics

This module provides backward-compatible imports from the unified caching module.
See caching_unified.py for the full implementation.
"""

# Re-export from unified module for backward compatibility
from insideLLMs.caching_unified import (
    AsyncCacheAdapter,
    # Configuration
    CacheConfig,
    # Data Classes
    CacheEntry,
    CacheLookupResult,
    CacheNamespace,
    CacheScope,
    CacheStats,
    CacheStatus,
    # Enums
    CacheStrategy,
    # Utilities
    CacheWarmer,
    MemoizedFunction,
    PromptCache,
    ResponseDeduplicator,
    cached_response,
    # Convenience Functions
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    # Key Generation
    generate_cache_key,
    get_cache_key,
    # Decorator
    memoize,
)
from insideLLMs.caching_unified import (
    # Cache Implementations - BaseCache is StrategyCache in unified
    StrategyCache as BaseCache,
)

# For backward compatibility with old module structure
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
    # Cache Implementations
    "BaseCache",
    "PromptCache",
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
    # Decorator
    "memoize",
]
