"""
Cache Base Classes and Key Generation
======================================

This module contains the abstract base class for cache implementations
and key generation utilities.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from insideLLMs.caching.types import CacheStats

T = TypeVar("T")


class BaseCacheABC(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set a value in the cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self.get(key) is not None


def generate_cache_key(
    prompt: str,
    model: Optional[str] = None,
    params: Optional[dict] = None,
    algorithm: str = "sha256",
    **kwargs: Any,
) -> str:
    """Generate a deterministic cache key from prompt and parameters.

    Creates a unique hash key by combining the prompt, model identifier,
    generation parameters, and any additional keyword arguments. The same
    inputs will always produce the same key, enabling reliable cache lookups.

    Parameters
    ----------
    prompt : str
        The prompt text to include in the key. This is the primary component.
    model : Optional[str]
        Model identifier (e.g., "gpt-4", "claude-3"). If provided, requests
        to different models will have different cache keys.
    params : Optional[dict]
        Generation parameters like temperature, max_tokens, etc. Parameters
        are sorted for deterministic key generation.
    algorithm : str
        Hash algorithm to use. Options: "sha256" (default, most secure),
        "sha1" (faster, less secure), "md5" (fastest, not secure).
    **kwargs : Any
        Additional key-value pairs to include in the cache key. Useful for
        custom parameters or context identifiers.

    Returns
    -------
    str
        Hexadecimal hash string. Length depends on algorithm:
        - sha256: 64 characters
        - sha1: 40 characters
        - md5: 32 characters

    Examples
    --------
    Basic key generation:

        >>> from insideLLMs.caching import generate_cache_key
        >>> key = generate_cache_key("What is Python?")
        >>> len(key)
        64

    Key with model and parameters:

        >>> key = generate_cache_key(
        ...     prompt="Explain AI",
        ...     model="gpt-4",
        ...     params={"temperature": 0.7, "max_tokens": 100}
        ... )

    Same inputs produce same key (deterministic):

        >>> key1 = generate_cache_key("Hello", model="gpt-4")
        >>> key2 = generate_cache_key("Hello", model="gpt-4")
        >>> key1 == key2
        True

    Different parameters produce different keys:

        >>> key1 = generate_cache_key("Hello", params={"temp": 0.0})
        >>> key2 = generate_cache_key("Hello", params={"temp": 1.0})
        >>> key1 != key2
        True

    Using custom kwargs:

        >>> key = generate_cache_key(
        ...     "Prompt",
        ...     user_id="user123",
        ...     session="abc"
        ... )

    See Also
    --------
    get_cache_key : Simplified wrapper for common use cases.
    generate_model_cache_key : Specialized for model request caching.
    """
    key_parts = [prompt]

    if model:
        key_parts.append(f"model:{model}")

    if params:
        sorted_params = json.dumps(params, sort_keys=True)
        key_parts.append(f"params:{sorted_params}")

    # Add any additional kwargs (sorted for determinism)
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")

    key_string = "|".join(key_parts)

    if algorithm == "md5":
        return hashlib.md5(key_string.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(key_string.encode()).hexdigest()
    else:
        return hashlib.sha256(key_string.encode()).hexdigest()


def generate_model_cache_key(
    model_id: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Generate cache key specifically for model generation requests.

    Specialized key generation for caching model responses. Includes all
    parameters that affect model output, ensuring that different generation
    configurations produce different cache keys.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g., "gpt-4", "claude-3-opus").
    prompt : str
        The prompt text sent to the model.
    temperature : float
        Sampling temperature. Default 0.0 (deterministic).
    max_tokens : Optional[int]
        Maximum number of tokens to generate.
    **kwargs : Any
        Additional generation parameters (e.g., top_p, frequency_penalty).

    Returns
    -------
    str
        64-character SHA-256 hash key.

    Examples
    --------
    Basic model cache key:

        >>> from insideLLMs.caching import generate_model_cache_key
        >>> key = generate_model_cache_key(
        ...     model_id="gpt-4",
        ...     prompt="Hello, world!"
        ... )
        >>> len(key)
        64

    Key with generation parameters:

        >>> key = generate_model_cache_key(
        ...     model_id="claude-3",
        ...     prompt="Explain quantum computing",
        ...     temperature=0.7,
        ...     max_tokens=500
        ... )

    Temperature 0 caching (deterministic outputs):

        >>> key = generate_model_cache_key(
        ...     model_id="gpt-4",
        ...     prompt="2+2=",
        ...     temperature=0.0
        ... )
        >>> # Safe to cache - output will be identical

    Note: Temperature > 0 produces non-deterministic outputs:

        >>> key = generate_model_cache_key(
        ...     model_id="gpt-4",
        ...     prompt="Write a poem",
        ...     temperature=1.0
        ... )
        >>> # Caching may return stale/unexpected responses

    See Also
    --------
    generate_cache_key : General-purpose key generation.
    CachedModel : Wrapper that uses this function internally.
    """
    key_parts = {
        "model_id": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for k, v in sorted(kwargs.items()):
        key_parts[k] = v

    key_str = json.dumps(key_parts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


__all__ = [
    "BaseCacheABC",
    "generate_cache_key",
    "generate_model_cache_key",
]
