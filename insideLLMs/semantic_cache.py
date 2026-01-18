"""Semantic Caching Module for insideLLMs.

This module provides advanced caching capabilities:
- Redis-based distributed caching
- Vector-based semantic similarity caching
- Embedding-powered cache lookups

Features:
- RedisCache: Fast distributed caching with Redis
- VectorCache: Semantic similarity-based cache using embeddings
- SemanticCacheModel: Model wrapper with semantic caching
- Hybrid caching strategies

Example:
    >>> from insideLLMs.semantic_cache import SemanticCache, quick_semantic_cache
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Simple semantic caching
    >>> cache = SemanticCache(similarity_threshold=0.85)
    >>> cache.set("What is AI?", "AI is artificial intelligence...")
    >>> result = cache.get_similar("What's artificial intelligence?")
    >>>
    >>> # With model wrapper
    >>> model = DummyModel()
    >>> cached_model = SemanticCacheModel(model, cache)
    >>> response = cached_model.generate("Explain machine learning")
"""

import hashlib
import json
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union
import math

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Optional numpy import for vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


T = TypeVar("T")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic caching."""

    # Basic settings
    max_size: int = 1000
    ttl_seconds: Optional[int] = 3600

    # Semantic settings
    similarity_threshold: float = 0.85
    embedding_dimension: int = 384
    use_embeddings: bool = True

    # Redis settings (if using Redis backend)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_prefix: str = "insideLLMs:"

    # Performance settings
    max_candidates: int = 100  # Max candidates for similarity search
    index_batch_size: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
            "embedding_dimension": self.embedding_dimension,
            "use_embeddings": self.use_embeddings,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_prefix": self.redis_prefix,
            "max_candidates": self.max_candidates,
        }


@dataclass
class SemanticCacheEntry:
    """A cache entry with semantic metadata."""

    key: str
    value: Any
    prompt: str
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    similarity_score: float = 1.0  # How similar this entry was to the query
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self):
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "prompt": self.prompt,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
        }


@dataclass
class SemanticLookupResult:
    """Result of a semantic cache lookup."""

    hit: bool
    value: Optional[Any]
    key: str
    similarity: float = 0.0
    entry: Optional[SemanticCacheEntry] = None
    lookup_time_ms: float = 0.0
    candidates_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hit": self.hit,
            "value": self.value,
            "key": self.key,
            "similarity": self.similarity,
            "entry": self.entry.to_dict() if self.entry else None,
            "lookup_time_ms": self.lookup_time_ms,
            "candidates_checked": self.candidates_checked,
        }


@dataclass
class SemanticCacheStats:
    """Statistics for semantic cache."""

    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0  # Hits via semantic similarity
    exact_hits: int = 0  # Hits via exact match
    entry_count: int = 0
    avg_similarity: float = 0.0
    total_lookups: int = 0
    avg_lookup_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        """Calculate semantic hit rate."""
        return self.semantic_hits / self.hits if self.hits > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "exact_hits": self.exact_hits,
            "entry_count": self.entry_count,
            "hit_rate": self.hit_rate,
            "semantic_hit_rate": self.semantic_hit_rate,
            "avg_similarity": self.avg_similarity,
            "total_lookups": self.total_lookups,
            "avg_lookup_time_ms": self.avg_lookup_time_ms,
        }


# =============================================================================
# Embedding Utilities
# =============================================================================


class SimpleEmbedder:
    """Simple text embedding using character/word frequency.

    This is a fallback embedder when no external embedding model is available.
    It uses TF-IDF-like features for basic semantic similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Tokenize
        words = self._tokenize(text)
        if not words:
            return [0.0] * self.dimension

        # Build word frequency
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Create embedding using hash-based projection
        embedding = [0.0] * self.dimension

        for word, freq in word_freq.items():
            # Hash word to get indices
            h = hashlib.md5(word.encode()).hexdigest()
            idx1 = int(h[:8], 16) % self.dimension
            idx2 = int(h[8:16], 16) % self.dimension

            # Weight by term frequency
            weight = math.log(1 + freq)

            embedding[idx1] += weight
            embedding[idx2] -= weight * 0.5

        # Add character-level features
        for i, char in enumerate(text[:100]):
            idx = (ord(char) + i) % self.dimension
            embedding[idx] += 0.1

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) > 1]

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    if NUMPY_AVAILABLE:
        a = np.array(vec1)
        b = np.array(vec2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    else:
        # Pure Python fallback
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


# =============================================================================
# Redis Cache Backend
# =============================================================================


class RedisCache:
    """Redis-based distributed cache.

    Provides fast, distributed caching using Redis. Requires redis-py.

    Args:
        config: Cache configuration
        client: Optional existing Redis client
    """

    def __init__(
        self,
        config: Optional[SemanticCacheConfig] = None,
        client: Optional["redis.Redis"] = None,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support requires redis-py. Install with: pip install redis"
            )

        self.config = config or SemanticCacheConfig()
        self._prefix = self.config.redis_prefix
        self._stats = SemanticCacheStats()

        if client is not None:
            self._client = client
        else:
            self._client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
            )

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start = time.time()
        redis_key = self._make_key(key)

        try:
            data = self._client.get(redis_key)
            if data is None:
                self._stats.misses += 1
                return None

            # Update access tracking
            self._client.hincrby(f"{redis_key}:meta", "access_count", 1)
            self._client.hset(f"{redis_key}:meta", "last_accessed", time.time())

            self._stats.hits += 1
            self._stats.exact_hits += 1
            self._stats.total_lookups += 1
            self._stats.avg_lookup_time_ms = (
                (self._stats.avg_lookup_time_ms * (self._stats.total_lookups - 1)
                 + (time.time() - start) * 1000) / self._stats.total_lookups
            )

            return json.loads(data)
        except Exception:
            self._stats.misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in cache."""
        redis_key = self._make_key(key)
        ttl = ttl or self.config.ttl_seconds

        try:
            # Store value
            if ttl:
                self._client.setex(redis_key, ttl, json.dumps(value))
            else:
                self._client.set(redis_key, json.dumps(value))

            # Store metadata
            meta = {
                "created_at": time.time(),
                "access_count": 0,
                "last_accessed": time.time(),
                **(metadata or {}),
            }
            self._client.hset(f"{redis_key}:meta", mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in meta.items()
            })
            if ttl:
                self._client.expire(f"{redis_key}:meta", ttl)

            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        redis_key = self._make_key(key)
        try:
            self._client.delete(redis_key, f"{redis_key}:meta")
            return True
        except Exception:
            return False

    def clear(self) -> int:
        """Clear all cache entries."""
        try:
            keys = self._client.keys(f"{self._prefix}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception:
            return 0

    def stats(self) -> SemanticCacheStats:
        """Get cache statistics."""
        try:
            keys = self._client.keys(f"{self._prefix}*")
            # Count only value keys, not metadata keys
            self._stats.entry_count = len([k for k in keys if not k.endswith(":meta")])
        except Exception:
            pass
        return self._stats

    def keys(self) -> List[str]:
        """Get all cache keys."""
        try:
            all_keys = self._client.keys(f"{self._prefix}*")
            # Return keys without prefix and without :meta suffix
            result = []
            for k in all_keys:
                if not k.endswith(":meta"):
                    result.append(k.replace(self._prefix, "", 1))
            return result
        except Exception:
            return []

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(self._client.exists(self._make_key(key)))

    def ttl(self, key: str) -> int:
        """Get TTL for a key."""
        return self._client.ttl(self._make_key(key))


# =============================================================================
# Vector Cache (Semantic Similarity)
# =============================================================================


class VectorCache:
    """Cache with vector-based semantic similarity lookup.

    Uses embeddings to find semantically similar cached entries.

    Args:
        config: Cache configuration
        embedder: Optional embedding function/object
    """

    def __init__(
        self,
        config: Optional[SemanticCacheConfig] = None,
        embedder: Optional[Union[Callable[[str], List[float]], SimpleEmbedder]] = None,
    ):
        self.config = config or SemanticCacheConfig()
        self._embedder = embedder or SimpleEmbedder(self.config.embedding_dimension)
        self._entries: Dict[str, SemanticCacheEntry] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._stats = SemanticCacheStats()
        self._lock = threading.RLock()
        self._similarity_sums = 0.0
        self._similarity_count = 0

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if callable(self._embedder):
            return self._embedder(text)
        return self._embedder.embed(text)

    def set(
        self,
        prompt: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticCacheEntry:
        """Set value in cache with semantic embedding.

        Args:
            prompt: The prompt text (used for similarity matching)
            value: The value to cache
            ttl: Time-to-live in seconds
            metadata: Optional metadata

        Returns:
            The created cache entry
        """
        with self._lock:
            # Evict if needed
            while len(self._entries) >= self.config.max_size:
                self._evict_lru()

            # Generate key and embedding
            key = hashlib.sha256(prompt.encode()).hexdigest()
            embedding = self._get_embedding(prompt) if self.config.use_embeddings else None

            # Calculate expiry
            expires_at = None
            if ttl or self.config.ttl_seconds:
                ttl_val = ttl or self.config.ttl_seconds
                expires_at = datetime.now() + timedelta(seconds=ttl_val)

            entry = SemanticCacheEntry(
                key=key,
                value=value,
                prompt=prompt,
                embedding=embedding,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._entries[key] = entry
            if embedding:
                self._embeddings[key] = embedding

            self._stats.entry_count = len(self._entries)
            return entry

    def get(self, prompt: str) -> SemanticLookupResult:
        """Get exact match from cache.

        Args:
            prompt: The prompt to look up

        Returns:
            Lookup result
        """
        start = time.time()
        key = hashlib.sha256(prompt.encode()).hexdigest()

        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._stats.misses += 1
                self._stats.total_lookups += 1
                return SemanticLookupResult(
                    hit=False,
                    value=None,
                    key=key,
                    lookup_time_ms=(time.time() - start) * 1000,
                )

            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.total_lookups += 1
                return SemanticLookupResult(
                    hit=False,
                    value=None,
                    key=key,
                    lookup_time_ms=(time.time() - start) * 1000,
                )

            entry.touch()
            self._stats.hits += 1
            self._stats.exact_hits += 1
            self._stats.total_lookups += 1

            return SemanticLookupResult(
                hit=True,
                value=entry.value,
                key=key,
                similarity=1.0,
                entry=entry,
                lookup_time_ms=(time.time() - start) * 1000,
            )

    def get_similar(
        self,
        prompt: str,
        threshold: Optional[float] = None,
    ) -> SemanticLookupResult:
        """Get semantically similar cached entry.

        Args:
            prompt: The prompt to find similar entries for
            threshold: Similarity threshold (default from config)

        Returns:
            Lookup result with similarity score
        """
        start = time.time()
        threshold = threshold or self.config.similarity_threshold

        # First try exact match
        exact_result = self.get(prompt)
        if exact_result.hit:
            return exact_result

        if not self.config.use_embeddings:
            return SemanticLookupResult(
                hit=False,
                value=None,
                key="",
                lookup_time_ms=(time.time() - start) * 1000,
            )

        # Get query embedding
        query_embedding = self._get_embedding(prompt)

        with self._lock:
            best_entry = None
            best_similarity = 0.0
            candidates_checked = 0

            for key, embedding in list(self._embeddings.items())[:self.config.max_candidates]:
                entry = self._entries.get(key)
                if entry is None or entry.is_expired():
                    continue

                candidates_checked += 1
                similarity = cosine_similarity(query_embedding, embedding)

                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_entry = entry

            if best_entry is not None:
                best_entry.touch()
                best_entry.similarity_score = best_similarity
                self._stats.hits += 1
                self._stats.semantic_hits += 1
                self._similarity_sums += best_similarity
                self._similarity_count += 1
                self._stats.avg_similarity = self._similarity_sums / self._similarity_count

                return SemanticLookupResult(
                    hit=True,
                    value=best_entry.value,
                    key=best_entry.key,
                    similarity=best_similarity,
                    entry=best_entry,
                    lookup_time_ms=(time.time() - start) * 1000,
                    candidates_checked=candidates_checked,
                )

            self._stats.misses += 1
            return SemanticLookupResult(
                hit=False,
                value=None,
                key="",
                similarity=best_similarity,
                lookup_time_ms=(time.time() - start) * 1000,
                candidates_checked=candidates_checked,
            )

    def find_similar(
        self,
        prompt: str,
        limit: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[SemanticCacheEntry, float]]:
        """Find all similar cached entries above threshold.

        Args:
            prompt: The prompt to find similar entries for
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of (entry, similarity) tuples sorted by similarity
        """
        if not self.config.use_embeddings:
            return []

        query_embedding = self._get_embedding(prompt)
        results = []

        with self._lock:
            for key, embedding in self._embeddings.items():
                entry = self._entries.get(key)
                if entry is None or entry.is_expired():
                    continue

                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    results.append((entry, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def delete(self, prompt: str) -> bool:
        """Delete entry by prompt."""
        key = hashlib.sha256(prompt.encode()).hexdigest()
        with self._lock:
            return self._remove_entry(key)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._embeddings.clear()
            self._stats = SemanticCacheStats()

    def stats(self) -> SemanticCacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.entry_count = len(self._entries)
            self._update_stats()
            return self._stats

    def _remove_entry(self, key: str) -> bool:
        """Remove entry by key."""
        if key in self._entries:
            del self._entries[key]
            if key in self._embeddings:
                del self._embeddings[key]
            return True
        return False

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._entries:
            return

        lru_key = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].last_accessed,
        )
        self._remove_entry(lru_key)

    def _update_stats(self) -> None:
        """Update statistics."""
        self._stats.total_lookups = self._stats.hits + self._stats.misses
        if self._stats.total_lookups > 0:
            elapsed = self._stats.avg_lookup_time_ms * self._stats.total_lookups
            self._stats.avg_lookup_time_ms = elapsed / self._stats.total_lookups


# =============================================================================
# Semantic Cache (Combined)
# =============================================================================


class SemanticCache:
    """High-level semantic cache combining exact and similarity matching.

    This cache first tries exact match, then falls back to semantic similarity.

    Args:
        config: Cache configuration
        embedder: Optional embedding function
        backend: Optional backend ('memory' or 'redis')
    """

    def __init__(
        self,
        config: Optional[SemanticCacheConfig] = None,
        embedder: Optional[Union[Callable[[str], List[float]], SimpleEmbedder]] = None,
        backend: str = "memory",
        redis_client: Optional["redis.Redis"] = None,
    ):
        self.config = config or SemanticCacheConfig()

        # Set up backend
        if backend == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis support requires redis-py")
            self._redis = RedisCache(self.config, redis_client)
        else:
            self._redis = None

        # Set up vector cache for semantic lookup
        self._vector_cache = VectorCache(self.config, embedder)

    def set(
        self,
        prompt: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticCacheEntry:
        """Cache a value with semantic embedding.

        Args:
            prompt: The prompt text
            value: The value to cache
            ttl: Time-to-live in seconds
            metadata: Optional metadata

        Returns:
            The created cache entry
        """
        entry = self._vector_cache.set(prompt, value, ttl, metadata)

        # Also store in Redis if available
        if self._redis:
            self._redis.set(
                entry.key,
                {
                    "value": value,
                    "prompt": prompt,
                    "embedding": entry.embedding,
                },
                ttl,
                metadata,
            )

        return entry

    def get(
        self,
        prompt: str,
        use_semantic: bool = True,
        threshold: Optional[float] = None,
    ) -> SemanticLookupResult:
        """Get cached value, optionally using semantic similarity.

        Args:
            prompt: The prompt to look up
            use_semantic: Whether to use semantic similarity
            threshold: Similarity threshold for semantic lookup

        Returns:
            Lookup result
        """
        if use_semantic:
            return self._vector_cache.get_similar(prompt, threshold)
        return self._vector_cache.get(prompt)

    def get_exact(self, prompt: str) -> SemanticLookupResult:
        """Get exact match only (no semantic similarity)."""
        return self._vector_cache.get(prompt)

    def get_similar(
        self,
        prompt: str,
        threshold: Optional[float] = None,
    ) -> SemanticLookupResult:
        """Get semantically similar cached value."""
        return self._vector_cache.get_similar(prompt, threshold)

    def find_similar(
        self,
        prompt: str,
        limit: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[SemanticCacheEntry, float]]:
        """Find all similar cached entries."""
        return self._vector_cache.find_similar(prompt, limit, threshold)

    def delete(self, prompt: str) -> bool:
        """Delete cached entry."""
        success = self._vector_cache.delete(prompt)
        if self._redis:
            key = hashlib.sha256(prompt.encode()).hexdigest()
            self._redis.delete(key)
        return success

    def clear(self) -> None:
        """Clear all cached entries."""
        self._vector_cache.clear()
        if self._redis:
            self._redis.clear()

    def stats(self) -> SemanticCacheStats:
        """Get cache statistics."""
        return self._vector_cache.stats()


# =============================================================================
# Cached Model Wrapper
# =============================================================================


class SemanticCacheModel:
    """Model wrapper with semantic caching.

    Wraps any model to add semantic caching capabilities.

    Args:
        model: The underlying model
        cache: Semantic cache instance
        cache_temperature_zero_only: Only cache deterministic (temp=0) responses
    """

    def __init__(
        self,
        model: Any,
        cache: Optional[SemanticCache] = None,
        cache_temperature_zero_only: bool = True,
    ):
        self._model = model
        self._cache = cache or SemanticCache()
        self._cache_temp_zero_only = cache_temperature_zero_only

    @property
    def model(self) -> Any:
        """Get underlying model."""
        return self._model

    @property
    def cache(self) -> SemanticCache:
        """Get cache instance."""
        return self._cache

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        use_cache: bool = True,
        cache_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response with semantic caching.

        Args:
            prompt: The prompt text
            temperature: Sampling temperature
            use_cache: Whether to use cache
            cache_threshold: Similarity threshold for cache lookup
            **kwargs: Additional model arguments

        Returns:
            Model response (from cache or fresh)
        """
        should_cache = use_cache and (
            not self._cache_temp_zero_only or temperature == 0
        )

        if should_cache:
            # Try to get from cache
            result = self._cache.get(prompt, threshold=cache_threshold)
            if result.hit:
                return result.value

        # Generate fresh response
        response = self._model.generate(prompt, temperature=temperature, **kwargs)

        if should_cache:
            # Cache the response
            self._cache.set(
                prompt,
                response,
                metadata={
                    "model": getattr(self._model, "model_id", str(type(self._model).__name__)),
                    "temperature": temperature,
                },
            )

        return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Chat completion with semantic caching.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            use_cache: Whether to use cache
            **kwargs: Additional model arguments

        Returns:
            Chat response
        """
        # Create cache key from messages
        prompt = json.dumps(messages, sort_keys=True)

        should_cache = use_cache and (
            not self._cache_temp_zero_only or temperature == 0
        )

        if should_cache:
            result = self._cache.get_exact(prompt)
            if result.hit:
                return result.value

        response = self._model.chat(messages, temperature=temperature, **kwargs)

        if should_cache:
            self._cache.set(prompt, response)

        return response

    def __getattr__(self, name: str) -> Any:
        """Delegate to underlying model."""
        return getattr(self._model, name)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_semantic_cache(
    similarity_threshold: float = 0.85,
    max_size: int = 1000,
    ttl_seconds: Optional[int] = 3600,
    use_redis: bool = False,
    **redis_kwargs: Any,
) -> SemanticCache:
    """Create a semantic cache.

    Args:
        similarity_threshold: Minimum similarity for cache hits
        max_size: Maximum cache size
        ttl_seconds: Default TTL
        use_redis: Whether to use Redis backend
        **redis_kwargs: Redis configuration

    Returns:
        Configured semantic cache
    """
    config = SemanticCacheConfig(
        similarity_threshold=similarity_threshold,
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        **{k: v for k, v in redis_kwargs.items()
           if k in SemanticCacheConfig.__dataclass_fields__},
    )

    backend = "redis" if use_redis else "memory"
    return SemanticCache(config, backend=backend)


def quick_semantic_cache(
    prompt: str,
    generator: Callable[[str], Any],
    cache: Optional[SemanticCache] = None,
    threshold: float = 0.85,
) -> Tuple[Any, bool, float]:
    """Quick helper for semantic caching.

    Args:
        prompt: The prompt text
        generator: Function to generate response if not cached
        cache: Optional existing cache
        threshold: Similarity threshold

    Returns:
        Tuple of (response, was_cached, similarity_score)
    """
    if cache is None:
        cache = SemanticCache()

    result = cache.get(prompt, threshold=threshold)

    if result.hit:
        return result.value, True, result.similarity

    response = generator(prompt)
    cache.set(prompt, response)

    return response, False, 1.0


def wrap_model_with_semantic_cache(
    model: Any,
    similarity_threshold: float = 0.85,
    **cache_kwargs: Any,
) -> SemanticCacheModel:
    """Wrap a model with semantic caching.

    Args:
        model: The model to wrap
        similarity_threshold: Cache similarity threshold
        **cache_kwargs: Additional cache configuration

    Returns:
        Wrapped model with semantic caching
    """
    cache = create_semantic_cache(similarity_threshold, **cache_kwargs)
    return SemanticCacheModel(model, cache)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    "SemanticCacheConfig",
    "SemanticCacheEntry",
    "SemanticLookupResult",
    "SemanticCacheStats",
    # Embedding utilities
    "SimpleEmbedder",
    "cosine_similarity",
    # Cache implementations
    "RedisCache",
    "VectorCache",
    "SemanticCache",
    # Model wrapper
    "SemanticCacheModel",
    # Convenience functions
    "create_semantic_cache",
    "quick_semantic_cache",
    "wrap_model_with_semantic_cache",
    # Availability flags
    "REDIS_AVAILABLE",
    "NUMPY_AVAILABLE",
]
