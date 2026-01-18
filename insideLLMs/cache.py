"""
Caching layer for model responses.

This module provides backward-compatible imports from the unified caching module.
See caching_unified.py for the full implementation.
"""

# Re-export from unified module for backward compatibility
from insideLLMs.caching_unified import (
    BaseCacheABC as BaseCache,
)
from insideLLMs.caching_unified import (
    # Model wrapper
    CachedModel,
    # Core types
    CacheEntry,
    CacheStats,
    DiskCache,
    # Implementations
    InMemoryCache,
    # Decorator
    cached,
    clear_default_cache,
    # Key generation
    generate_cache_key,
    # Global cache functions
    get_default_cache,
    set_default_cache,
)

# For backward compatibility with old module structure
__all__ = [
    "BaseCache",
    "CacheEntry",
    "CacheStats",
    "CachedModel",
    "DiskCache",
    "InMemoryCache",
    "cached",
    "clear_default_cache",
    "generate_cache_key",
    "get_default_cache",
    "set_default_cache",
]
