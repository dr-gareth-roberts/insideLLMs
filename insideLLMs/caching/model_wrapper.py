"""
Cached Model Wrapper
====================

Wrapper that adds caching to any model.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from insideLLMs.caching.backends.memory import InMemoryCache
from insideLLMs.caching.base import BaseCacheABC, generate_model_cache_key
from insideLLMs.types import ModelResponse


class CachedModel:
    """Wrapper that adds caching to any model.

    Args:
        model: The underlying model to wrap.
        cache: Cache backend to use.
        cache_only_deterministic: Only cache requests with temperature=0.
    """

    def __init__(
        self,
        model: Any,
        cache: Optional[BaseCacheABC] = None,
        cache_only_deterministic: bool = True,
    ):
        self._model = model
        self._cache = cache or InMemoryCache()
        self._cache_only_deterministic = cache_only_deterministic

    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self._model

    @property
    def cache(self) -> BaseCacheABC:
        """Get the cache backend."""
        return self._cache

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response, using cache if available."""
        should_cache = not self._cache_only_deterministic or temperature == 0

        if should_cache:
            model_id = getattr(self._model, "model_id", str(type(self._model).__name__))
            cache_key = generate_model_cache_key(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._deserialize_response(cached)

        response = self._model.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if should_cache:
            self._cache.set(
                cache_key,
                self._serialize_response(response),
                metadata={"model_id": model_id, "prompt_preview": prompt[:100]},
            )

        return response

    def _serialize_response(self, response: ModelResponse) -> dict[str, Any]:
        """Serialize a ModelResponse for caching."""
        return asdict(response)

    def _deserialize_response(self, data: dict[str, Any]) -> ModelResponse:
        """Deserialize a cached response."""
        return ModelResponse(**data)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model."""
        return getattr(self._model, name)


__all__ = ["CachedModel"]
