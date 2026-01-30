"""
Cache Factory Functions
=======================

Convenience functions for creating cache instances and global cache management.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from insideLLMs.caching.backends.memory import InMemoryCache
from insideLLMs.caching.backends.strategy import StrategyCache
from insideLLMs.caching.base import BaseCacheABC, generate_cache_key
from insideLLMs.caching.namespace import CacheNamespace
from insideLLMs.caching.prompt_cache import PromptCache
from insideLLMs.caching.types import CacheConfig, CacheStrategy
from insideLLMs.caching.utilities import CacheWarmer

# Type alias for backward compatibility
BaseCache = StrategyCache


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

        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper

    return decorator


__all__ = [
    # Factory Functions
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
