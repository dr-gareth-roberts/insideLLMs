"""Semantic Caching Module for insideLLMs.

This module provides advanced caching capabilities that go beyond simple key-value
storage by enabling semantic similarity matching. When a query is semantically
similar to a previously cached query (even if not identical), the cached response
can be returned, significantly reducing API costs and latency.

Overview
--------
The semantic caching system supports multiple caching backends and strategies:

- **In-Memory Cache**: Fast, single-process caching using Python dictionaries
  with optional LRU eviction and TTL support.

- **Redis Cache**: Distributed caching using Redis for multi-process or
  multi-server deployments.

- **Vector Cache**: Embedding-based semantic similarity lookup that finds
  cached entries based on meaning rather than exact string matching.

- **Semantic Cache**: High-level interface combining exact and semantic matching
  with configurable similarity thresholds.

Architecture
------------
The caching system is organized in layers:

1. **Configuration Layer**: ``SemanticCacheConfig`` provides centralized settings
   for all cache components including TTL, size limits, similarity thresholds,
   and Redis connection parameters.

2. **Entry Layer**: ``SemanticCacheEntry`` stores cached values along with
   metadata including embeddings, access statistics, and expiration times.

3. **Backend Layer**: ``RedisCache`` and ``VectorCache`` provide the actual
   storage and retrieval mechanisms.

4. **Interface Layer**: ``SemanticCache`` and ``SemanticCacheModel`` provide
   high-level APIs for common use cases.

Key Features
------------
- **Semantic Similarity**: Uses embeddings to find semantically similar queries
- **Configurable Thresholds**: Fine-tune the similarity threshold for cache hits
- **TTL Support**: Automatic expiration of cached entries
- **LRU Eviction**: Automatic eviction when cache reaches maximum size
- **Access Tracking**: Statistics on cache hits, misses, and similarity scores
- **Thread Safety**: All caches are thread-safe with proper locking
- **Redis Integration**: Optional distributed caching with Redis
- **Model Wrapper**: Easy integration with any LLM through SemanticCacheModel

Dependencies
------------
Required:
    - Python 3.9+

Optional:
    - redis: For Redis-based distributed caching
    - numpy: For optimized vector similarity calculations

Examples
--------
Basic semantic caching with similarity lookup:

    >>> from insideLLMs.semantic_cache import SemanticCache
    >>>
    >>> # Create a cache with 85% similarity threshold
    >>> cache = SemanticCache()
    >>> cache.config.similarity_threshold = 0.85
    >>>
    >>> # Cache a response for a query
    >>> cache.set("What is artificial intelligence?",
    ...           "AI is the simulation of human intelligence by machines.")
    >>>
    >>> # Query with semantically similar text - returns cached response
    >>> result = cache.get_similar("What's AI and how does it work?")
    >>> if result.hit:
    ...     print(f"Cache hit! Similarity: {result.similarity:.2f}")
    ...     print(f"Response: {result.value}")
    Cache hit! Similarity: 0.92
    Response: AI is the simulation of human intelligence by machines.

Using the quick helper function:

    >>> from insideLLMs.semantic_cache import quick_semantic_cache
    >>>
    >>> def expensive_api_call(prompt):
    ...     # Simulates an expensive LLM API call
    ...     return f"Generated response for: {prompt}"
    >>>
    >>> # First call - generates and caches
    >>> response, was_cached, similarity = quick_semantic_cache(
    ...     "Explain quantum computing",
    ...     expensive_api_call
    ... )
    >>> print(f"Cached: {was_cached}, Similarity: {similarity}")
    Cached: False, Similarity: 1.0
    >>>
    >>> # Second call with similar query - returns cached
    >>> response, was_cached, similarity = quick_semantic_cache(
    ...     "What is quantum computing?",
    ...     expensive_api_call
    ... )
    >>> print(f"Cached: {was_cached}")
    Cached: True

Wrapping a model with semantic caching:

    >>> from insideLLMs.semantic_cache import wrap_model_with_semantic_cache
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Wrap any model with semantic caching
    >>> model = DummyModel()
    >>> cached_model = wrap_model_with_semantic_cache(
    ...     model,
    ...     similarity_threshold=0.90,
    ...     max_size=500,
    ...     ttl_seconds=7200  # 2 hours
    ... )
    >>>
    >>> # Generate responses - similar prompts will use cache
    >>> response1 = cached_model.generate("What is machine learning?")
    >>> response2 = cached_model.generate("Explain ML to me")  # May hit cache
    >>>
    >>> # Check cache statistics
    >>> stats = cached_model.cache.stats()
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")

Using Redis as a distributed backend:

    >>> from insideLLMs.semantic_cache import create_semantic_cache
    >>>
    >>> # Create cache with Redis backend for distributed caching
    >>> cache = create_semantic_cache(
    ...     use_redis=True,
    ...     redis_host="localhost",
    ...     redis_port=6379,
    ...     redis_db=1,
    ...     redis_prefix="myapp:"
    ... )
    >>>
    >>> # Use like any other cache - now distributed across processes
    >>> cache.set("query", "response")
    >>> result = cache.get("query")

Notes
-----
- The ``SimpleEmbedder`` provides a basic fallback embedding when no external
  embedding model is available. For production use, consider using a proper
  embedding model (e.g., sentence-transformers) for better semantic matching.

- When using Redis, ensure the Redis server is running and accessible. The
  cache will raise an ImportError if redis-py is not installed.

- The similarity threshold significantly affects cache behavior:
  - Higher threshold (0.95+): Only very similar queries match
  - Lower threshold (0.70-0.85): More queries match, but may be less relevant
  - Recommended starting point: 0.85

- For deterministic results with LLMs, set temperature=0 when using
  ``SemanticCacheModel.generate()``. By default, only temperature=0 responses
  are cached to ensure consistency.

See Also
--------
insideLLMs.caching_unified : Unified caching system with multiple backends
insideLLMs.tokens.EmbeddingUtils : Utilities for working with embeddings
"""

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, TypeVar, Union

from insideLLMs.caching import CacheEntryMixin

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
    """Configuration for semantic caching behavior.

    This dataclass centralizes all configuration options for the semantic caching
    system, including cache size limits, TTL settings, similarity thresholds,
    embedding parameters, and Redis connection details.

    Parameters
    ----------
    max_size : int, default=1000
        Maximum number of entries the cache can hold. When this limit is reached,
        the least recently used (LRU) entry is evicted to make room for new entries.
    ttl_seconds : int or None, default=3600
        Default time-to-live for cache entries in seconds. After this duration,
        entries are considered expired and will not be returned. Set to None for
        no expiration.
    similarity_threshold : float, default=0.85
        Minimum cosine similarity score (0.0 to 1.0) required for a semantic match.
        Higher values require closer matches; lower values allow more distant matches.
    embedding_dimension : int, default=384
        Dimensionality of embedding vectors used for similarity calculations.
        Should match the output dimension of your embedding model.
    use_embeddings : bool, default=True
        Whether to compute and store embeddings for semantic similarity lookup.
        Set to False to disable semantic matching and use only exact key matching.
    redis_host : str, default="localhost"
        Hostname or IP address of the Redis server for distributed caching.
    redis_port : int, default=6379
        Port number of the Redis server.
    redis_db : int, default=0
        Redis database number to use (0-15 on most Redis configurations).
    redis_password : str or None, default=None
        Password for Redis authentication. Set to None if no authentication is required.
    redis_prefix : str, default="insideLLMs:"
        Prefix for all Redis keys. Useful for namespacing when multiple applications
        share the same Redis instance.
    max_candidates : int, default=100
        Maximum number of cache entries to consider during similarity search.
        Higher values improve accuracy but increase search time.
    index_batch_size : int, default=50
        Batch size for indexing operations when adding multiple entries.

    Attributes
    ----------
    max_size : int
        Maximum cache size.
    ttl_seconds : int or None
        Default TTL in seconds.
    similarity_threshold : float
        Minimum similarity for cache hits.
    embedding_dimension : int
        Vector dimension for embeddings.
    use_embeddings : bool
        Whether semantic matching is enabled.
    redis_host : str
        Redis server hostname.
    redis_port : int
        Redis server port.
    redis_db : int
        Redis database number.
    redis_password : str or None
        Redis authentication password.
    redis_prefix : str
        Prefix for Redis keys.
    max_candidates : int
        Max entries to search for similarity.
    index_batch_size : int
        Batch size for indexing.

    Examples
    --------
    Creating a basic in-memory cache configuration:

        >>> config = SemanticCacheConfig(
        ...     max_size=500,
        ...     ttl_seconds=1800,  # 30 minutes
        ...     similarity_threshold=0.90
        ... )
        >>> cache = SemanticCache(config)

    Configuration for a high-precision semantic cache:

        >>> config = SemanticCacheConfig(
        ...     similarity_threshold=0.95,  # Require very close matches
        ...     max_candidates=200,  # Search more entries for better matches
        ...     embedding_dimension=768  # Use larger embeddings
        ... )

    Configuration for Redis-backed distributed cache:

        >>> config = SemanticCacheConfig(
        ...     redis_host="redis.mycompany.com",
        ...     redis_port=6380,
        ...     redis_db=2,
        ...     redis_password="secret123",
        ...     redis_prefix="chatbot_cache:",
        ...     ttl_seconds=7200  # 2 hours
        ... )

    See Also
    --------
    SemanticCache : Main cache class that uses this configuration
    VectorCache : Vector-based cache implementation
    RedisCache : Redis-backed cache implementation
    """

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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns a dictionary representation of all configuration parameters,
        useful for serialization, logging, or passing to other components.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all configuration parameters with their
            current values.

        Examples
        --------
        Serializing configuration for logging:

            >>> config = SemanticCacheConfig(max_size=500, ttl_seconds=1800)
            >>> config_dict = config.to_dict()
            >>> print(f"Cache size limit: {config_dict['max_size']}")
            Cache size limit: 500

        Saving configuration to JSON:

            >>> import json
            >>> config = SemanticCacheConfig()
            >>> with open("cache_config.json", "w") as f:
            ...     json.dump(config.to_dict(), f, indent=2)
        """
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
class SemanticCacheEntry(CacheEntryMixin):
    """A cache entry with semantic metadata and embedding information.

    This dataclass represents a single entry in the semantic cache, storing
    the cached value along with its embedding vector, access statistics,
    and expiration information. It inherits from ``CacheEntryMixin`` which
    provides ``is_expired()`` and ``touch()`` methods.

    Parameters
    ----------
    key : str
        Unique identifier for this cache entry, typically a hash of the prompt.
    value : Any
        The cached value (e.g., model response, computation result).
    prompt : str
        The original prompt text that was used to generate the cached value.
    embedding : list[float] or None, default=None
        Vector embedding of the prompt for semantic similarity matching.
        None if embeddings are disabled.
    created_at : datetime, default=datetime.now()
        Timestamp when the entry was created.
    expires_at : datetime or None, default=None
        Timestamp when the entry expires. None means no expiration.
    access_count : int, default=0
        Number of times this entry has been accessed (cache hits).
    last_accessed : datetime, default=datetime.now()
        Timestamp of the most recent access to this entry.
    similarity_score : float, default=1.0
        Similarity score when this entry was matched. 1.0 indicates exact match,
        lower values indicate semantic matches.
    metadata : dict[str, Any], default={}
        Additional metadata associated with the entry (e.g., model info, temperature).

    Attributes
    ----------
    key : str
        Cache entry key (hash of prompt).
    value : Any
        Cached response value.
    prompt : str
        Original prompt text.
    embedding : list[float] or None
        Prompt embedding vector.
    created_at : datetime
        Creation timestamp.
    expires_at : datetime or None
        Expiration timestamp.
    access_count : int
        Number of accesses.
    last_accessed : datetime
        Last access timestamp.
    similarity_score : float
        Match similarity score.
    metadata : dict[str, Any]
        Additional metadata.

    Examples
    --------
    Creating a cache entry manually:

        >>> from datetime import datetime, timedelta
        >>> entry = SemanticCacheEntry(
        ...     key="abc123",
        ...     value="The capital of France is Paris.",
        ...     prompt="What is the capital of France?",
        ...     embedding=[0.1, 0.2, 0.3],  # Simplified embedding
        ...     expires_at=datetime.now() + timedelta(hours=1)
        ... )
        >>> print(f"Prompt: {entry.prompt}")
        Prompt: What is the capital of France?

    Checking if an entry is expired:

        >>> from datetime import datetime, timedelta
        >>> # Create an already-expired entry
        >>> entry = SemanticCacheEntry(
        ...     key="expired",
        ...     value="old data",
        ...     prompt="old query",
        ...     expires_at=datetime.now() - timedelta(hours=1)
        ... )
        >>> entry.is_expired()
        True

    Updating access statistics:

        >>> entry = SemanticCacheEntry(
        ...     key="test",
        ...     value="response",
        ...     prompt="query"
        ... )
        >>> entry.access_count
        0
        >>> entry.touch()  # Simulates a cache hit
        >>> entry.access_count
        1

    See Also
    --------
    SemanticLookupResult : Result returned from cache lookups
    CacheEntryMixin : Mixin providing is_expired() and touch() methods
    """

    key: str
    value: Any
    prompt: str
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    similarity_score: float = 1.0  # How similar this entry was to the query
    metadata: dict[str, Any] = field(default_factory=dict)

    # is_expired() and touch() inherited from CacheEntryMixin

    def to_dict(self) -> dict[str, Any]:
        """Convert cache entry to a dictionary.

        Serializes the cache entry to a dictionary format suitable for
        JSON serialization, logging, or storage. Datetime fields are
        converted to ISO format strings.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the cache entry with all fields.

        Examples
        --------
        Serializing an entry for logging:

            >>> entry = SemanticCacheEntry(
            ...     key="test123",
            ...     value="Hello, World!",
            ...     prompt="Say hello"
            ... )
            >>> d = entry.to_dict()
            >>> print(d["prompt"])
            Say hello

        Saving entry to JSON file:

            >>> import json
            >>> entry = SemanticCacheEntry(
            ...     key="json_test",
            ...     value={"response": "data"},
            ...     prompt="Get data",
            ...     metadata={"model": "gpt-4"}
            ... )
            >>> json_str = json.dumps(entry.to_dict(), indent=2)
        """
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
    """Result of a semantic cache lookup operation.

    This dataclass encapsulates the result of a cache lookup, including
    whether a matching entry was found (hit), the cached value, similarity
    score, and performance metrics. It provides all information needed to
    determine how to proceed after a cache lookup.

    Parameters
    ----------
    hit : bool
        Whether the lookup found a matching cache entry. True indicates
        a cache hit (exact or semantic match found), False indicates a miss.
    value : Any or None
        The cached value if hit is True, None otherwise.
    key : str
        The cache key that was looked up or matched.
    similarity : float, default=0.0
        Cosine similarity score between the query and matched entry.
        1.0 for exact matches, lower values for semantic matches,
        0.0 for cache misses.
    entry : SemanticCacheEntry or None, default=None
        The full cache entry if hit is True, None otherwise.
        Provides access to metadata, access statistics, and embedding.
    lookup_time_ms : float, default=0.0
        Time taken for the lookup operation in milliseconds.
        Useful for performance monitoring and optimization.
    candidates_checked : int, default=0
        Number of cache entries examined during semantic similarity search.
        Higher values indicate more thorough but slower searches.

    Attributes
    ----------
    hit : bool
        Whether lookup was successful.
    value : Any or None
        Cached value or None.
    key : str
        Cache key looked up.
    similarity : float
        Match similarity score.
    entry : SemanticCacheEntry or None
        Full entry details.
    lookup_time_ms : float
        Lookup duration in ms.
    candidates_checked : int
        Entries examined in search.

    Examples
    --------
    Handling a cache lookup result:

        >>> cache = SemanticCache()
        >>> cache.set("What is Python?", "Python is a programming language.")
        >>> result = cache.get_similar("Tell me about Python")
        >>> if result.hit:
        ...     print(f"Cache hit! Similarity: {result.similarity:.2f}")
        ...     print(f"Cached response: {result.value}")
        ... else:
        ...     print("Cache miss - need to generate response")
        Cache hit! Similarity: 0.89
        Cached response: Python is a programming language.

    Monitoring cache performance:

        >>> result = cache.get_similar("Some query")
        >>> print(f"Lookup took {result.lookup_time_ms:.2f}ms")
        >>> print(f"Checked {result.candidates_checked} candidates")
        Lookup took 1.23ms
        Checked 50 candidates

    Accessing full entry metadata on a hit:

        >>> result = cache.get("Cached query")
        >>> if result.hit and result.entry:
        ...     print(f"Entry accessed {result.entry.access_count} times")
        ...     print(f"Created at: {result.entry.created_at}")

    See Also
    --------
    SemanticCacheEntry : The cache entry returned in the entry field
    SemanticCache.get : Method that returns this result type
    SemanticCache.get_similar : Semantic lookup returning this result
    """

    hit: bool
    value: Optional[Any]
    key: str
    similarity: float = 0.0
    entry: Optional[SemanticCacheEntry] = None
    lookup_time_ms: float = 0.0
    candidates_checked: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert lookup result to a dictionary.

        Serializes the lookup result to a dictionary format for logging,
        analysis, or API responses. Nested entry is also converted.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the lookup result.

        Examples
        --------
        Logging cache lookup results:

            >>> result = cache.get_similar("test query")
            >>> result_dict = result.to_dict()
            >>> print(f"Hit: {result_dict['hit']}, Similarity: {result_dict['similarity']}")

        Sending result as JSON API response:

            >>> import json
            >>> result = cache.get("query")
            >>> json_response = json.dumps(result.to_dict())
        """
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
    """Statistics and metrics for semantic cache performance monitoring.

    This dataclass tracks cache performance metrics including hit/miss counts,
    the breakdown between exact and semantic matches, average similarity scores,
    and lookup timing. Use these statistics to monitor cache effectiveness and
    tune configuration parameters.

    Parameters
    ----------
    hits : int, default=0
        Total number of cache hits (both exact and semantic matches).
    misses : int, default=0
        Total number of cache misses.
    semantic_hits : int, default=0
        Number of hits that were semantic matches (similarity < 1.0).
    exact_hits : int, default=0
        Number of hits that were exact key matches (similarity = 1.0).
    entry_count : int, default=0
        Current number of entries in the cache.
    avg_similarity : float, default=0.0
        Average similarity score across all semantic hits.
    total_lookups : int, default=0
        Total number of lookup operations performed.
    avg_lookup_time_ms : float, default=0.0
        Average time per lookup in milliseconds.

    Attributes
    ----------
    hits : int
        Total cache hits.
    misses : int
        Total cache misses.
    semantic_hits : int
        Hits via semantic similarity.
    exact_hits : int
        Hits via exact match.
    entry_count : int
        Current cache size.
    avg_similarity : float
        Mean similarity of semantic hits.
    total_lookups : int
        Total lookups performed.
    avg_lookup_time_ms : float
        Mean lookup time in ms.
    hit_rate : float
        Ratio of hits to total lookups (property).
    semantic_hit_rate : float
        Ratio of semantic hits to total hits (property).

    Examples
    --------
    Monitoring cache effectiveness:

        >>> cache = SemanticCache()
        >>> # ... perform many cache operations ...
        >>> stats = cache.stats()
        >>> print(f"Cache hit rate: {stats.hit_rate:.1%}")
        >>> print(f"Total entries: {stats.entry_count}")
        >>> print(f"Semantic match rate: {stats.semantic_hit_rate:.1%}")
        Cache hit rate: 75.2%
        Total entries: 247
        Semantic match rate: 45.3%

    Analyzing performance metrics:

        >>> stats = cache.stats()
        >>> print(f"Avg lookup time: {stats.avg_lookup_time_ms:.2f}ms")
        >>> print(f"Avg semantic similarity: {stats.avg_similarity:.3f}")
        Avg lookup time: 0.85ms
        Avg semantic similarity: 0.912

    Exporting stats for dashboards:

        >>> stats = cache.stats()
        >>> stats_dict = stats.to_dict()
        >>> # Send to monitoring system
        >>> metrics.record(stats_dict)

    See Also
    --------
    SemanticCache.stats : Method that returns these statistics
    VectorCache.stats : Vector cache statistics method
    """

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
        """Calculate the cache hit rate.

        Computes the ratio of cache hits to total lookups. This is a key
        metric for evaluating cache effectiveness - higher values indicate
        better cache utilization and cost savings.

        Returns
        -------
        float
            Hit rate as a decimal between 0.0 and 1.0.
            Returns 0.0 if no lookups have been performed.

        Examples
        --------
        Checking hit rate:

            >>> stats = SemanticCacheStats(hits=75, misses=25)
            >>> stats.hit_rate
            0.75

        Formatting as percentage:

            >>> stats = SemanticCacheStats(hits=85, misses=15)
            >>> print(f"Hit rate: {stats.hit_rate:.1%}")
            Hit rate: 85.0%
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        """Calculate the semantic hit rate among all hits.

        Computes the ratio of semantic matches to total hits. This metric
        shows how often semantic similarity matching is providing value
        beyond exact key matching.

        Returns
        -------
        float
            Semantic hit rate as a decimal between 0.0 and 1.0.
            Returns 0.0 if no hits have occurred.

        Examples
        --------
        Understanding semantic match contribution:

            >>> stats = SemanticCacheStats(hits=100, semantic_hits=40, exact_hits=60)
            >>> stats.semantic_hit_rate
            0.4

        Evaluating if semantic matching is valuable:

            >>> stats = cache.stats()
            >>> if stats.semantic_hit_rate > 0.3:
            ...     print("Semantic matching is providing significant value")
            ... else:
            ...     print("Consider adjusting similarity threshold")
        """
        return self.semantic_hits / self.hits if self.hits > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary.

        Serializes all statistics including computed properties (hit_rate,
        semantic_hit_rate) to a dictionary format suitable for JSON
        serialization or monitoring systems.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all statistics with descriptive keys.

        Examples
        --------
        Exporting to JSON for logging:

            >>> import json
            >>> stats = cache.stats()
            >>> print(json.dumps(stats.to_dict(), indent=2))

        Sending to monitoring system:

            >>> stats = cache.stats()
            >>> datadog.gauge("cache.hit_rate", stats.to_dict()["hit_rate"])
            >>> datadog.gauge("cache.entries", stats.to_dict()["entry_count"])
        """
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
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # Tokenize
        words = self._tokenize(text)
        if not words:
            return [0.0] * self.dimension

        # Build word frequency
        word_freq: dict[str, int] = {}
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

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re

        text = text.lower()
        words = re.findall(r"\b\w+\b", text)
        return [w for w in words if len(w) > 1]

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Note: This is a numpy-optimized version for performance.
    See also: insideLLMs.tokens.EmbeddingUtils.cosine_similarity for pure Python.
    """
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
            raise ImportError("Redis support requires redis-py. Install with: pip install redis")

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
                self._stats.avg_lookup_time_ms * (self._stats.total_lookups - 1)
                + (time.time() - start) * 1000
            ) / self._stats.total_lookups

            return json.loads(data)
        except Exception:
            self._stats.misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
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
            self._client.hset(
                f"{redis_key}:meta",
                mapping={
                    k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in meta.items()
                },
            )
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
                return int(self._client.delete(*keys))
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

    def keys(self) -> list[str]:
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
        return int(self._client.ttl(self._make_key(key)))


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
        embedder: Optional[Union[Callable[[str], list[float]], SimpleEmbedder]] = None,
    ):
        self.config = config or SemanticCacheConfig()
        self._embedder = embedder or SimpleEmbedder(self.config.embedding_dimension)
        self._entries: dict[str, SemanticCacheEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._stats = SemanticCacheStats()
        self._lock = threading.RLock()
        self._similarity_sums = 0.0
        self._similarity_count = 0

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if callable(self._embedder):
            return self._embedder(text)
        return self._embedder.embed(text)

    def set(
        self,
        prompt: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
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

            for key, embedding in list(self._embeddings.items())[: self.config.max_candidates]:
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
    ) -> list[tuple[SemanticCacheEntry, float]]:
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
        embedder: Optional[Union[Callable[[str], list[float]], SimpleEmbedder]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
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
    ) -> list[tuple[SemanticCacheEntry, float]]:
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
        should_cache = use_cache and (not self._cache_temp_zero_only or temperature == 0)

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
        messages: list[dict[str, str]],
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

        should_cache = use_cache and (not self._cache_temp_zero_only or temperature == 0)

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
        **{k: v for k, v in redis_kwargs.items() if k in SemanticCacheConfig.__dataclass_fields__},
    )

    backend = "redis" if use_redis else "memory"
    return SemanticCache(config, backend=backend)


def quick_semantic_cache(
    prompt: str,
    generator: Callable[[str], Any],
    cache: Optional[SemanticCache] = None,
    threshold: float = 0.85,
) -> tuple[Any, bool, float]:
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
