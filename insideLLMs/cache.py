"""
Caching layer for model responses.

This module provides backward-compatible imports from the unified caching module.
See caching_unified.py for the full implementation.
"""

# Re-export from unified module for backward compatibility
from insideLLMs.caching_unified import (
    # Core types
    CacheEntry,
    CacheStats,
    # Key generation
    generate_cache_key,
    generate_model_cache_key,
    # Abstract base
    BaseCacheABC as BaseCache,
    # Implementations
    InMemoryCache,
    DiskCache,
    # Model wrapper
    CachedModel,
    # Global cache functions
    get_default_cache,
    set_default_cache,
    clear_default_cache,
    # Decorator
    cached,
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
