"""Cache exception types."""

from typing import Optional

from insideLLMs._exceptions.base import InsideLLMsError


class CacheError(InsideLLMsError):
    """Base exception for cache errors.

    This is the parent class for all exceptions that occur during cache
    operations, including lookups, storage, and integrity checks.
    Catching this exception will handle any cache-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any cache-related error:

    >>> try:
    ...     result = cache.get(key)
    ... except CacheError as e:
    ...     print(f"Cache operation failed: {e}")
    ...     # Fall back to computing the result
    ...     result = compute_expensive_result()

    Implementing cache-aside pattern:

    >>> def get_with_cache(key: str, compute_fn):
    ...     try:
    ...         return cache.get(key)
    ...     except CacheMissError:
    ...         result = compute_fn()
    ...         cache.set(key, result)
    ...         return result
    ...     except CacheError as e:
    ...         logging.warning(f"Cache error: {e}")
    ...         return compute_fn()

    Notes
    -----
    Subclasses include: CacheMissError, CacheCorruptionError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    CacheMissError : When a requested key is not in cache.
    CacheCorruptionError : When cached data is invalid.
    """

    pass


class CacheMissError(CacheError):
    """Raised when a cache lookup misses.

    This exception is raised when attempting to retrieve a value from
    the cache that does not exist. This is often expected behavior and
    signals that the value needs to be computed or fetched.

    Parameters
    ----------
    key : str
        The cache key that was not found.

    Attributes
    ----------
    details : dict
        Contains 'key' with the requested cache key.

    Examples
    --------
    Basic cache lookup with miss handling:

    >>> try:
    ...     result = cache.get("model_output_12345")
    ... except CacheMissError as e:
    ...     print(f"Cache miss for key: {e.details['key']}")
    ...     result = model.generate(prompt)
    ...     cache.set(e.details['key'], result)

    Implementing lazy loading:

    >>> def get_cached_result(cache, key, generator_fn):
    ...     try:
    ...         return cache.get(key)
    ...     except CacheMissError:
    ...         result = generator_fn()
    ...         cache.set(key, result)
    ...         return result

    Tracking cache statistics:

    >>> stats = {'hits': 0, 'misses': 0}
    >>> def get_with_stats(cache, key):
    ...     try:
    ...         result = cache.get(key)
    ...         stats['hits'] += 1
    ...         return result
    ...     except CacheMissError:
    ...         stats['misses'] += 1
    ...         raise

    Notes
    -----
    Cache misses are normal and expected. Design your code to handle
    them gracefully by computing or fetching the missing value.

    See Also
    --------
    CacheCorruptionError : When cached data is corrupted.
    """

    def __init__(self, key: str):
        super().__init__(f"Cache miss for key: {key}", {"key": key})


class CacheCorruptionError(CacheError):
    """Raised when cache data is corrupted.

    This exception is raised when cached data cannot be deserialized,
    has an invalid format, or fails integrity checks. The corrupted
    entry should typically be removed and recomputed.

    Parameters
    ----------
    reason : str
        A description of how the corruption was detected.
    key : str, optional
        The cache key of the corrupted entry.

    Attributes
    ----------
    details : dict
        Contains 'reason' and optionally 'key'.

    Examples
    --------
    Handling corruption by clearing the entry:

    >>> try:
    ...     result = cache.get(key)
    ... except CacheCorruptionError as e:
    ...     print(f"Corrupted cache entry: {e.details['reason']}")
    ...     if 'key' in e.details:
    ...         cache.delete(e.details['key'])
    ...     result = compute_fresh_result()

    Implementing self-healing cache:

    >>> def get_or_heal(cache, key, compute_fn):
    ...     try:
    ...         return cache.get(key)
    ...     except CacheCorruptionError:
    ...         cache.delete(key)
    ...         result = compute_fn()
    ...         cache.set(key, result)
    ...         return result
    ...     except CacheMissError:
    ...         result = compute_fn()
    ...         cache.set(key, result)
    ...         return result

    Logging corruption for monitoring:

    >>> try:
    ...     result = cache.get(key)
    ... except CacheCorruptionError as e:
    ...     logging.error(
    ...         "Cache corruption detected",
    ...         extra={
    ...             "key": e.details.get("key"),
    ...             "reason": e.details["reason"]
    ...         }
    ...     )
    ...     # Alert monitoring system
    ...     metrics.increment("cache.corruption")

    See Also
    --------
    CacheMissError : When key simply doesn't exist.
    """

    def __init__(self, reason: str, key: Optional[str] = None):
        details = {"reason": reason}
        if key:
            details["key"] = key
        super().__init__(f"Cache corruption detected: {reason}", details)
