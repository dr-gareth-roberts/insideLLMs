"""
Prompt Cache (LLM-specific)
===========================

Cache specialized for LLM prompts and responses with semantic matching.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from insideLLMs.caching.backends.strategy import StrategyCache
from insideLLMs.caching.base import generate_cache_key
from insideLLMs.caching.types import CacheConfig, CacheEntry, CacheLookupResult
from insideLLMs.nlp.similarity import word_overlap_similarity


class PromptCache(StrategyCache):
    """Cache specialized for LLM prompts and responses with semantic matching.

    PromptCache extends StrategyCache with LLM-specific features including:
    - Semantic similarity matching for finding similar cached prompts
    - Prompt-specific caching methods (cache_response, get_response)
    - Prompt-to-key mapping for efficient lookups

    Parameters
    ----------
    config : Optional[CacheConfig]
        Cache configuration. Uses defaults if None.
    similarity_threshold : float
        Minimum similarity score (0.0 to 1.0) for find_similar() to return
        matches. Higher values require closer matches. Default is 0.95.

    Attributes
    ----------
    similarity_threshold : float
        Configured similarity threshold for matching.
    _prompt_keys : dict[str, str]
        Mapping from prompt hashes to cache keys for fast lookup.

    Examples
    --------
    Basic prompt caching:

        >>> from insideLLMs.caching import PromptCache
        >>> cache = PromptCache()
        >>> entry = cache.cache_response(
        ...     prompt="What is Python?",
        ...     response="Python is a programming language.",
        ...     model="gpt-4"
        ... )
        >>> result = cache.get_response("What is Python?", model="gpt-4")
        >>> print(result.hit, result.value)
        True Python is a programming language.

    Caching with parameters:

        >>> cache = PromptCache()
        >>> cache.cache_response(
        ...     prompt="Explain AI",
        ...     response="AI is...",
        ...     model="claude-3",
        ...     params={"temperature": 0.7, "max_tokens": 100}
        ... )
        >>> # Same prompt with different params = different cache entry
        >>> cache.cache_response(
        ...     prompt="Explain AI",
        ...     response="AI is... (different)",
        ...     model="claude-3",
        ...     params={"temperature": 0.0, "max_tokens": 50}
        ... )

    Finding similar cached prompts:

        >>> cache = PromptCache(similarity_threshold=0.8)
        >>> cache.cache_response("What is machine learning?", "ML is...")
        >>> cache.cache_response("Explain deep learning", "DL is...")
        >>> similar = cache.find_similar("What is ML?", limit=3)
        >>> for prompt, entry, score in similar:
        ...     print(f"{prompt[:30]}... (similarity: {score:.2f})")

    Lookup by prompt only (ignoring model/params):

        >>> cache = PromptCache()
        >>> cache.cache_response("Hello", "Hi there!", model="gpt-4")
        >>> result = cache.get_by_prompt("Hello")
        >>> print(result.hit)
        True

    With metadata:

        >>> cache = PromptCache()
        >>> cache.cache_response(
        ...     prompt="Translate: Hello",
        ...     response="Hola",
        ...     metadata={"language": "Spanish", "tokens_used": 10}
        ... )
        >>> result = cache.get_response("Translate: Hello")
        >>> if result.entry:
        ...     print(result.entry.metadata["language"])
        Spanish

    Notes
    -----
    - Similarity is calculated using word overlap (Jaccard index).
    - Prompts are hashed using MD5 for the prompt-to-key mapping.
    - Inherits all features from StrategyCache.

    See Also
    --------
    StrategyCache : Parent class with eviction strategies.
    generate_cache_key : Used to create cache keys.
    CacheWarmer : For preloading prompts.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        similarity_threshold: float = 0.95,
    ):
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self._prompt_keys: dict[str, str] = {}  # prompt hash -> cache key

    def cache_response(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Cache an LLM response."""
        key = generate_cache_key(prompt, model, params, self.config.hash_algorithm)

        entry_metadata = {
            "prompt": prompt,
            "model": model,
            "params": params,
            **(metadata or {}),
        }

        entry = self.set(key, response, metadata=entry_metadata)

        # Track prompt -> key mapping
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self._prompt_keys[prompt_hash] = key

        return entry

    def get_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> CacheLookupResult:
        """Get cached response for a prompt."""
        key = generate_cache_key(prompt, model, params, self.config.hash_algorithm)
        return self.get(key)

    def get_by_prompt(self, prompt: str) -> CacheLookupResult:
        """Get cached response by prompt alone (ignoring model/params)."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        key = self._prompt_keys.get(prompt_hash)

        if key:
            return self.get(key)

        return CacheLookupResult(hit=False, value=None, key="")

    def find_similar(
        self,
        prompt: str,
        limit: int = 5,
    ) -> list[tuple[str, CacheEntry, float]]:
        """Find similar cached prompts."""
        results = []

        with self._lock:
            for entry in self._entries.values():
                if entry.is_expired():
                    continue

                cached_prompt = entry.metadata.get("prompt", "")
                if cached_prompt:
                    similarity = self._calculate_similarity(prompt, cached_prompt)
                    if similarity >= self.similarity_threshold:
                        results.append((cached_prompt, entry, similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using Jaccard index."""
        return word_overlap_similarity(text1, text2)


__all__ = ["PromptCache"]
