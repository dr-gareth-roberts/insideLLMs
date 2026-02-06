"""Comprehensive coverage tests for semantic_cache.py and runtime/observability.py.

This file targets uncovered code paths in both modules, focusing on:
- RedisCache with mocked redis client
- VectorCache edge cases (expired entries, empty caches, embeddings disabled)
- SemanticCache with redis backend
- SemanticCacheModel metadata and chat paths
- OTelTracedModel and setup_otel_tracing with mocked OpenTelemetry
- Observability edge cases
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Observability Imports
# =============================================================================
import insideLLMs.runtime.observability as obs_module
from insideLLMs.runtime.observability import (
    CallRecord,
    OTelTracedModel,
    TelemetryCollector,
    TracedModel,
    TracingConfig,
    estimate_tokens,
    get_collector,
    instrument_model,
    set_collector,
    setup_otel_tracing,
    trace_call,
    trace_function,
)

# =============================================================================
# Semantic Cache Imports
# =============================================================================
from insideLLMs.semantic_cache import (
    RedisCache,
    SemanticCache,
    SemanticCacheConfig,
    SemanticCacheEntry,
    SemanticCacheModel,
    SemanticCacheStats,
    SemanticLookupResult,
    SimpleEmbedder,
    VectorCache,
    cosine_similarity,
    create_semantic_cache,
    quick_semantic_cache,
    wrap_model_with_semantic_cache,
)

# =============================================================================
# SEMANTIC CACHE: RedisCache with mocked redis
# =============================================================================


class TestRedisCacheMocked:
    """Tests for RedisCache using a mock Redis client."""

    def _make_mock_redis_client(self):
        """Create a mock redis client with basic behavior."""
        client = MagicMock()
        self._store = {}
        self._meta_store = {}

        def mock_get(key):
            return self._store.get(key)

        def mock_set(key, value):
            self._store[key] = value
            return True

        def mock_setex(key, ttl, value):
            self._store[key] = value
            return True

        def mock_delete(*keys):
            count = 0
            for k in keys:
                if k in self._store:
                    del self._store[k]
                    count += 1
                if k in self._meta_store:
                    del self._meta_store[k]
                    count += 1
            return count

        def mock_keys(pattern):
            prefix = pattern.rstrip("*")
            return [
                k
                for k in list(self._store.keys()) + list(self._meta_store.keys())
                if k.startswith(prefix)
            ]

        def mock_exists(key):
            return 1 if key in self._store else 0

        def mock_ttl(key):
            return -1 if key in self._store else -2

        def mock_hset(name, *args, mapping=None, **kwargs):
            self._meta_store[name] = mapping or kwargs
            return True

        def mock_hincrby(name, key=None, amount=1):
            if name not in self._meta_store:
                self._meta_store[name] = {}
            return amount

        def mock_expire(name, seconds):
            return True

        client.get = MagicMock(side_effect=mock_get)
        client.set = MagicMock(side_effect=mock_set)
        client.setex = MagicMock(side_effect=mock_setex)
        client.delete = MagicMock(side_effect=mock_delete)
        client.keys = MagicMock(side_effect=mock_keys)
        client.exists = MagicMock(side_effect=mock_exists)
        client.ttl = MagicMock(side_effect=mock_ttl)
        client.hset = MagicMock(side_effect=mock_hset)
        client.hincrby = MagicMock(side_effect=mock_hincrby)
        client.expire = MagicMock(side_effect=mock_expire)

        return client

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_init_with_client(self):
        """Test RedisCache initialization with provided client."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)
        assert cache._client is client
        assert cache.config is not None

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_init_with_config(self):
        """Test RedisCache initialization with custom config."""
        client = self._make_mock_redis_client()
        config = SemanticCacheConfig(redis_prefix="test:", ttl_seconds=300)
        cache = RedisCache(config=config, client=client)
        assert cache._prefix == "test:"

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", False)
    def test_redis_cache_init_no_redis(self):
        """Test RedisCache raises ImportError when redis not available."""
        with pytest.raises(ImportError, match="redis-py"):
            RedisCache()

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_set_and_get(self):
        """Test RedisCache set and get operations."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        # Set a value
        result = cache.set("key1", {"data": "value1"})
        assert result is True

        # Get the value
        value = cache.get("key1")
        assert value == {"data": "value1"}

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_get_miss(self):
        """Test RedisCache get returns None on miss."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        value = cache.get("nonexistent")
        assert value is None
        assert cache._stats.misses == 1

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_get_hit_stats(self):
        """Test RedisCache get updates stats on hit."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        cache.get("key1")

        assert cache._stats.hits == 1
        assert cache._stats.exact_hits == 1
        assert cache._stats.total_lookups == 1

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_set_without_ttl(self):
        """Test RedisCache set without TTL."""
        client = self._make_mock_redis_client()
        config = SemanticCacheConfig(ttl_seconds=None)
        cache = RedisCache(config=config, client=client)

        result = cache.set("key1", "value1", ttl=None)
        assert result is True
        client.set.assert_called()

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_set_with_metadata(self):
        """Test RedisCache set with metadata."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        result = cache.set("key1", "value1", metadata={"model": "gpt-4"})
        assert result is True

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_set_exception(self):
        """Test RedisCache set returns False on exception."""
        client = self._make_mock_redis_client()
        client.setex.side_effect = Exception("Connection error")
        cache = RedisCache(client=client)

        result = cache.set("key1", "value1")
        assert result is False

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_get_exception(self):
        """Test RedisCache get returns None on exception."""
        client = self._make_mock_redis_client()
        client.get.side_effect = Exception("Connection error")
        cache = RedisCache(client=client)

        value = cache.get("key1")
        assert value is None
        assert cache._stats.misses == 1

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_delete(self):
        """Test RedisCache delete."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        result = cache.delete("key1")
        assert result is True

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_delete_exception(self):
        """Test RedisCache delete returns False on exception."""
        client = self._make_mock_redis_client()
        client.delete.side_effect = Exception("Connection error")
        cache = RedisCache(client=client)

        result = cache.delete("key1")
        assert result is False

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_clear(self):
        """Test RedisCache clear."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        result = cache.clear()
        assert result >= 0

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_clear_empty(self):
        """Test RedisCache clear with no keys."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        result = cache.clear()
        assert result == 0

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_clear_exception(self):
        """Test RedisCache clear returns 0 on exception."""
        client = self._make_mock_redis_client()
        client.keys.side_effect = Exception("Connection error")
        cache = RedisCache(client=client)

        result = cache.clear()
        assert result == 0

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_stats(self):
        """Test RedisCache stats."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        stats = cache.stats()
        assert isinstance(stats, SemanticCacheStats)
        assert stats.entry_count >= 0

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_stats_exception(self):
        """Test RedisCache stats handles exception."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)
        # First call normally to set up stats
        cache.set("key1", "value1")
        # Now make keys fail
        client.keys.side_effect = Exception("Connection error")
        stats = cache.stats()
        assert isinstance(stats, SemanticCacheStats)

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_keys(self):
        """Test RedisCache keys listing."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        keys = cache.keys()
        assert isinstance(keys, list)

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_keys_exception(self):
        """Test RedisCache keys returns empty list on exception."""
        client = self._make_mock_redis_client()
        client.keys.side_effect = Exception("Connection error")
        cache = RedisCache(client=client)

        keys = cache.keys()
        assert keys == []

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_exists(self):
        """Test RedisCache exists check."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_ttl(self):
        """Test RedisCache ttl check."""
        client = self._make_mock_redis_client()
        cache = RedisCache(client=client)

        cache.set("key1", "value1")
        ttl = cache.ttl("key1")
        assert isinstance(ttl, int)

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_make_key(self):
        """Test RedisCache key prefix."""
        client = self._make_mock_redis_client()
        config = SemanticCacheConfig(redis_prefix="myapp:")
        cache = RedisCache(config=config, client=client)

        assert cache._make_key("test") == "myapp:test"


# =============================================================================
# SEMANTIC CACHE: VectorCache edge cases
# =============================================================================


class TestVectorCacheEdgeCases:
    """Tests for VectorCache edge cases not covered by existing tests."""

    def test_get_expired_entry(self):
        """Test VectorCache.get removes and misses on expired entry."""
        config = SemanticCacheConfig(ttl_seconds=None)
        cache = VectorCache(config)

        # Set an entry with a very short TTL
        cache.set("test prompt", "test value", ttl=1)
        assert cache.get("test prompt").hit is True

        # Wait for it to expire
        # Instead of sleeping, manually expire the entry
        entry_key = hashlib.sha256("test prompt".encode()).hexdigest()
        cache._entries[entry_key].expires_at = datetime.now() - timedelta(seconds=10)

        result = cache.get("test prompt")
        assert result.hit is False
        assert entry_key not in cache._entries

    def test_get_similar_with_expired_entry_in_scan(self):
        """Test get_similar skips expired entries during scan."""
        cache = VectorCache()

        # Add entries
        cache.set("What is Python?", "Python is a language")
        cache.set("What is Java?", "Java is a language")

        # Expire one entry
        for key, entry in cache._entries.items():
            if entry.prompt == "What is Java?":
                entry.expires_at = datetime.now() - timedelta(seconds=10)
                break

        # Search should skip expired entry
        result = cache.get_similar("Tell me about Java", threshold=0.1)
        # The expired entry should be skipped
        if result.hit:
            assert result.entry.prompt != "What is Java?"

    def test_find_similar_without_embeddings(self):
        """Test find_similar returns empty when embeddings disabled."""
        config = SemanticCacheConfig(use_embeddings=False)
        cache = VectorCache(config)

        cache.set("test prompt", "test value")
        results = cache.find_similar("test", limit=5, threshold=0.1)
        assert results == []

    def test_find_similar_with_expired_entries(self):
        """Test find_similar skips expired entries."""
        cache = VectorCache()

        cache.set("What is Python?", "Python is a language")
        cache.set("What is Java?", "Java is a language")

        # Expire one entry
        for key, entry in cache._entries.items():
            if entry.prompt == "What is Java?":
                entry.expires_at = datetime.now() - timedelta(seconds=10)
                break

        results = cache.find_similar("programming language", limit=5, threshold=0.01)
        # Expired entries should not appear
        for entry, _ in results:
            assert entry.prompt != "What is Java?"

    def test_remove_entry_not_found(self):
        """Test _remove_entry returns False when key doesn't exist."""
        cache = VectorCache()
        result = cache._remove_entry("nonexistent_key")
        assert result is False

    def test_remove_entry_without_embedding(self):
        """Test _remove_entry for entry that has no embedding."""
        config = SemanticCacheConfig(use_embeddings=False)
        cache = VectorCache(config)

        entry = cache.set("test prompt", "test value")
        key = entry.key
        assert key in cache._entries
        assert key not in cache._embeddings

        result = cache._remove_entry(key)
        assert result is True
        assert key not in cache._entries

    def test_evict_lru_empty(self):
        """Test _evict_lru does nothing when cache is empty."""
        cache = VectorCache()
        cache._evict_lru()  # Should not raise
        assert len(cache._entries) == 0

    def test_get_embedding_with_callable(self):
        """Test _get_embedding with a callable embedder."""

        def my_embedder(text):
            return [float(len(text))] * 10

        cache = VectorCache(
            config=SemanticCacheConfig(embedding_dimension=10),
            embedder=my_embedder,
        )
        emb = cache._get_embedding("hello")
        assert emb == [5.0] * 10

    def test_get_embedding_with_simple_embedder(self):
        """Test _get_embedding with SimpleEmbedder instance."""
        embedder = SimpleEmbedder(dimension=10)
        cache = VectorCache(
            config=SemanticCacheConfig(embedding_dimension=10),
            embedder=embedder,
        )
        emb = cache._get_embedding("hello")
        assert len(emb) == 10

    def test_set_with_explicit_ttl(self):
        """Test VectorCache.set with explicit TTL overriding config."""
        config = SemanticCacheConfig(ttl_seconds=3600)
        cache = VectorCache(config)

        entry = cache.set("test", "value", ttl=60)
        assert entry.expires_at is not None
        # TTL should be approximately 60 seconds from now
        remaining = (entry.expires_at - datetime.now()).total_seconds()
        assert remaining < 61

    def test_set_with_no_ttl_and_no_config_ttl(self):
        """Test VectorCache.set with no TTL at all."""
        config = SemanticCacheConfig(ttl_seconds=None)
        cache = VectorCache(config)

        entry = cache.set("test", "value", ttl=None)
        assert entry.expires_at is None

    def test_update_stats(self):
        """Test _update_stats method."""
        cache = VectorCache()

        # Generate some stats
        cache.set("test", "value")
        cache.get("test")
        cache.get("nonexistent")

        cache._update_stats()
        stats = cache._stats
        assert stats.total_lookups == stats.hits + stats.misses

    def test_get_similar_no_match_above_threshold(self):
        """Test get_similar returns miss when nothing above threshold."""
        cache = VectorCache()
        cache.set("apple fruit healthy", "Apples are healthy fruits")

        # Use very high threshold so nothing matches
        result = cache.get_similar("quantum physics theory", threshold=0.999)
        assert result.hit is False
        assert result.candidates_checked >= 0

    def test_stats_returns_updated_entry_count(self):
        """Test stats updates entry count correctly."""
        cache = VectorCache()
        cache.set("p1", "v1")
        cache.set("p2", "v2")

        stats = cache.stats()
        assert stats.entry_count == 2


# =============================================================================
# SEMANTIC CACHE: SemanticCache with Redis backend
# =============================================================================


class TestSemanticCacheWithRedis:
    """Tests for SemanticCache with Redis backend (mocked)."""

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_init_with_redis_backend(self):
        """Test SemanticCache initializes with redis backend."""
        mock_client = MagicMock()
        cache = SemanticCache(backend="redis", redis_client=mock_client)
        assert cache._redis is not None

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", False)
    def test_init_redis_not_available(self):
        """Test SemanticCache raises when redis not installed."""
        with pytest.raises(ImportError, match="redis-py"):
            SemanticCache(backend="redis")

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_set_with_redis(self):
        """Test SemanticCache.set stores in both vector and redis."""
        mock_client = MagicMock()
        mock_client.setex = MagicMock(return_value=True)
        mock_client.hset = MagicMock(return_value=True)
        mock_client.expire = MagicMock(return_value=True)
        cache = SemanticCache(backend="redis", redis_client=mock_client)

        entry = cache.set("test prompt", "test value")
        assert entry.prompt == "test prompt"
        # Redis set should have been called
        assert mock_client.setex.called or mock_client.set.called

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_delete_with_redis(self):
        """Test SemanticCache.delete removes from both caches."""
        mock_client = MagicMock()
        mock_client.setex = MagicMock(return_value=True)
        mock_client.hset = MagicMock(return_value=True)
        mock_client.expire = MagicMock(return_value=True)
        mock_client.delete = MagicMock(return_value=1)
        cache = SemanticCache(backend="redis", redis_client=mock_client)

        cache.set("test prompt", "test value")
        result = cache.delete("test prompt")
        assert result is True
        assert mock_client.delete.called

    @patch("insideLLMs.semantic_cache.REDIS_AVAILABLE", True)
    def test_clear_with_redis(self):
        """Test SemanticCache.clear clears both caches."""
        mock_client = MagicMock()
        mock_client.setex = MagicMock(return_value=True)
        mock_client.hset = MagicMock(return_value=True)
        mock_client.expire = MagicMock(return_value=True)
        mock_client.keys = MagicMock(return_value=[])
        cache = SemanticCache(backend="redis", redis_client=mock_client)

        cache.set("test prompt", "test value")
        cache.clear()
        assert cache.stats().entry_count == 0

    def test_get_without_semantic(self):
        """Test SemanticCache.get with use_semantic=False."""
        cache = SemanticCache()
        cache.set("exact prompt", "value")

        result = cache.get("exact prompt", use_semantic=False)
        assert result.hit is True
        assert result.value == "value"

        result = cache.get("different prompt", use_semantic=False)
        assert result.hit is False


# =============================================================================
# SEMANTIC CACHE: SemanticCacheModel extra paths
# =============================================================================


class TestSemanticCacheModelExtra:
    """Tests for SemanticCacheModel edge cases."""

    def test_generate_caches_with_model_id(self):
        """Test generate stores model_id in metadata."""
        model = MagicMock()
        model.generate.return_value = "response"
        model.model_id = "gpt-4-turbo"

        cached_model = SemanticCacheModel(model)
        cached_model.generate("test prompt", temperature=0)

        # Verify the cache entry has model metadata
        result = cached_model.cache.get("test prompt")
        assert result.hit is True
        assert result.entry.metadata.get("model") == "gpt-4-turbo"

    def test_generate_no_model_id_uses_class_name(self):
        """Test generate uses class name when model_id not available."""
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "response"

        cached_model = SemanticCacheModel(model)
        cached_model.generate("test prompt", temperature=0)

        result = cached_model.cache.get("test prompt")
        assert result.hit is True
        assert "model" in result.entry.metadata

    def test_chat_non_zero_temp_no_cache(self):
        """Test chat with non-zero temperature doesn't cache."""
        model = MagicMock()
        model.chat.return_value = "chat response"

        cached_model = SemanticCacheModel(model, cache_temperature_zero_only=True)

        cached_model.chat([{"role": "user", "content": "hi"}], temperature=0.7)
        cached_model.chat([{"role": "user", "content": "hi"}], temperature=0.7)

        assert model.chat.call_count == 2

    def test_chat_zero_temp_caches(self):
        """Test chat with zero temperature caches."""
        model = MagicMock()
        model.chat.return_value = "chat response"

        cached_model = SemanticCacheModel(model, cache_temperature_zero_only=True)

        msgs = [{"role": "user", "content": "hi"}]
        cached_model.chat(msgs, temperature=0)
        cached_model.chat(msgs, temperature=0)

        assert model.chat.call_count == 1

    def test_chat_cache_disabled(self):
        """Test chat with cache disabled."""
        model = MagicMock()
        model.chat.return_value = "response"

        cached_model = SemanticCacheModel(model)

        msgs = [{"role": "user", "content": "hi"}]
        cached_model.chat(msgs, use_cache=False)
        cached_model.chat(msgs, use_cache=False)

        assert model.chat.call_count == 2

    def test_generate_with_cache_threshold(self):
        """Test generate with custom cache threshold."""
        model = MagicMock()
        model.generate.return_value = "response"

        cached_model = SemanticCacheModel(model)
        cached_model.generate("test", temperature=0, cache_threshold=0.99)
        assert model.generate.call_count == 1


# =============================================================================
# SEMANTIC CACHE: Data class serialization
# =============================================================================


class TestDataClassSerialization:
    """Tests for to_dict methods on data classes."""

    def test_entry_to_dict_with_expires_at(self):
        """Test SemanticCacheEntry.to_dict with expires_at set."""
        expires = datetime(2025, 6, 15, 12, 0, 0)
        entry = SemanticCacheEntry(
            key="k1",
            value="v1",
            prompt="p1",
            expires_at=expires,
        )

        data = entry.to_dict()
        assert data["expires_at"] == "2025-06-15T12:00:00"

    def test_entry_to_dict_without_expires_at(self):
        """Test SemanticCacheEntry.to_dict with no expiry."""
        entry = SemanticCacheEntry(key="k1", value="v1", prompt="p1")

        data = entry.to_dict()
        assert data["expires_at"] is None

    def test_lookup_result_to_dict_with_entry(self):
        """Test SemanticLookupResult.to_dict with nested entry."""
        entry = SemanticCacheEntry(key="k1", value="v1", prompt="p1")
        result = SemanticLookupResult(
            hit=True,
            value="v1",
            key="k1",
            similarity=0.95,
            entry=entry,
        )

        data = result.to_dict()
        assert data["entry"] is not None
        assert data["entry"]["key"] == "k1"

    def test_stats_hit_rate_zero_lookups(self):
        """Test hit_rate returns 0 with no lookups."""
        stats = SemanticCacheStats()
        assert stats.hit_rate == 0.0

    def test_stats_semantic_hit_rate_zero_hits(self):
        """Test semantic_hit_rate returns 0 with no hits."""
        stats = SemanticCacheStats()
        assert stats.semantic_hit_rate == 0.0

    def test_config_to_dict_complete(self):
        """Test SemanticCacheConfig.to_dict has all expected keys."""
        config = SemanticCacheConfig()
        data = config.to_dict()
        expected_keys = {
            "max_size",
            "ttl_seconds",
            "similarity_threshold",
            "embedding_dimension",
            "use_embeddings",
            "redis_host",
            "redis_port",
            "redis_db",
            "redis_prefix",
            "max_candidates",
        }
        assert expected_keys == set(data.keys())


# =============================================================================
# SEMANTIC CACHE: Cosine similarity edge cases
# =============================================================================


class TestCosineSimilarityExtra:
    """Additional cosine similarity tests."""

    def test_cosine_similarity_both_zero_vectors(self):
        """Test cosine similarity with both zero vectors."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_cosine_similarity_one_zero_vector(self):
        """Test cosine similarity with one zero vector (reverse order)."""
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 2.0]
        assert cosine_similarity(vec1, vec2) == 0.0


# =============================================================================
# SEMANTIC CACHE: Convenience functions extra
# =============================================================================


class TestConvenienceFunctionsExtra:
    """Additional tests for convenience functions."""

    def test_create_semantic_cache_with_no_ttl(self):
        """Test create_semantic_cache with ttl_seconds=None."""
        cache = create_semantic_cache(ttl_seconds=None)
        assert cache.config.ttl_seconds is None

    def test_quick_semantic_cache_creates_cache(self):
        """Test quick_semantic_cache creates cache when none provided."""
        result, was_cached, similarity = quick_semantic_cache(
            "test",
            lambda p: f"response for {p}",
            cache=None,
        )
        assert was_cached is False
        assert "response for test" in result

    def test_wrap_model_with_cache_kwargs(self):
        """Test wrap_model_with_semantic_cache with extra kwargs."""
        model = MagicMock()
        model.generate = MagicMock(return_value="test")
        wrapped = wrap_model_with_semantic_cache(
            model,
            similarity_threshold=0.90,
            max_size=500,
            ttl_seconds=1800,
        )
        assert isinstance(wrapped, SemanticCacheModel)
        assert wrapped.cache.config.max_size == 500


# =============================================================================
# SEMANTIC CACHE: SimpleEmbedder extra
# =============================================================================


class TestSimpleEmbedderExtra:
    """Additional tests for SimpleEmbedder."""

    def test_embed_single_character_words(self):
        """Test embedding with text that has only single-char words."""
        embedder = SimpleEmbedder(dimension=10)
        # "I a" has words "I" and "a", both length 1, filtered by _tokenize
        embedding = embedder.embed("I a")
        # Character-level features should still produce non-zero embedding
        assert len(embedding) == 10


# =============================================================================
# OBSERVABILITY: OTelTracedModel and setup_otel_tracing (mocked)
# =============================================================================


class TestOTelTracedModelMocked:
    """Tests for OTelTracedModel with mocked OpenTelemetry."""

    def _setup_otel_mocks(self):
        """Set up mock OTel objects."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.set_attribute = MagicMock()
        mock_span.record_exception = MagicMock()

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span = MagicMock(return_value=mock_span)

        mock_trace = MagicMock()
        mock_trace.get_tracer = MagicMock(return_value=mock_tracer)
        mock_trace.set_tracer_provider = MagicMock()

        return mock_trace, mock_tracer, mock_span

    def test_otel_traced_model_generate(self):
        """Test OTelTracedModel.generate with mocked OTel."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.generate = MagicMock(return_value="generated text")

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            response = otel_model.generate("test prompt")
            assert response == "generated text"
            mock_span.set_attribute.assert_any_call("llm.model", "test-model")
            mock_span.set_attribute.assert_any_call("llm.success", True)

    def test_otel_traced_model_generate_with_logging(self):
        """Test OTelTracedModel.generate with prompt/response logging."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.generate = MagicMock(return_value="generated response text")

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig(log_prompts=True, log_responses=True)
            otel_model._tracer = mock_tracer

            response = otel_model.generate("test prompt")
            assert response == "generated response text"
            mock_span.set_attribute.assert_any_call("llm.prompt", "test prompt")
            mock_span.set_attribute.assert_any_call("llm.response", "generated response text")

    def test_otel_traced_model_generate_error(self):
        """Test OTelTracedModel.generate with error."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.generate = MagicMock(side_effect=ValueError("API Error"))

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            with pytest.raises(ValueError, match="API Error"):
                otel_model.generate("test prompt")

            mock_span.set_attribute.assert_any_call("llm.success", False)
            mock_span.record_exception.assert_called_once()

    def test_otel_traced_model_chat(self):
        """Test OTelTracedModel.chat with mocked OTel."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.chat = MagicMock(return_value="chat response")

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            messages = [{"role": "user", "content": "hi"}]
            response = otel_model.chat(messages)
            assert response == "chat response"
            mock_span.set_attribute.assert_any_call("llm.operation", "chat")
            mock_span.set_attribute.assert_any_call("llm.message_count", 1)

    def test_otel_traced_model_chat_not_supported(self):
        """Test OTelTracedModel.chat when model doesn't support it."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock(spec=["generate", "name"])
        model.name = "test-model"

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            with pytest.raises(NotImplementedError, match="does not support chat"):
                otel_model.chat([{"role": "user", "content": "hi"}])

    def test_otel_traced_model_chat_error(self):
        """Test OTelTracedModel.chat with error."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.chat = MagicMock(side_effect=RuntimeError("Chat failed"))

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            with pytest.raises(RuntimeError, match="Chat failed"):
                otel_model.chat([{"role": "user", "content": "hi"}])

            mock_span.set_attribute.assert_any_call("llm.success", False)
            mock_span.record_exception.assert_called_once()

    def test_otel_traced_model_info(self):
        """Test OTelTracedModel.info delegates to model."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.info = MagicMock(return_value={"name": "test-model"})

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            info = otel_model.info()
            assert info == {"name": "test-model"}

    def test_otel_traced_model_getattr(self):
        """Test OTelTracedModel.__getattr__ delegates to model."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"
        model.custom_prop = "custom_value"

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            assert otel_model.custom_prop == "custom_value"

    def test_otel_traced_model_name_property(self):
        """Test OTelTracedModel.name property."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "my-model"

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel.__new__(OTelTracedModel)
            otel_model._model = model
            otel_model._config = TracingConfig()
            otel_model._tracer = mock_tracer

            assert otel_model.name == "my-model"

    def test_otel_traced_model_init_when_available(self):
        """Test OTelTracedModel.__init__ with mocked OTel."""
        mock_trace, mock_tracer, mock_span = self._setup_otel_mocks()

        model = MagicMock()
        model.name = "test-model"

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            otel_model = OTelTracedModel(model)
            assert otel_model._model is model
            assert otel_model._config is not None
            mock_trace.get_tracer.assert_called()

    def test_otel_traced_model_init_not_available(self):
        """Test OTelTracedModel.__init__ raises when OTel not available."""
        model = MagicMock()
        model.name = "test-model"

        with patch.object(obs_module, "OTEL_AVAILABLE", False):
            with pytest.raises(ImportError, match="OpenTelemetry is required"):
                OTelTracedModel(model)


class TestSetupOTelTracingMocked:
    """Tests for setup_otel_tracing with mocked OTel."""

    def test_setup_otel_not_available(self):
        """Test setup_otel_tracing raises when OTel not available."""
        with patch.object(obs_module, "OTEL_AVAILABLE", False):
            with pytest.raises(ImportError, match="OpenTelemetry is required"):
                setup_otel_tracing(TracingConfig())

    def test_setup_otel_with_console_export(self):
        """Test setup_otel_tracing with console export."""
        mock_resource = MagicMock()
        mock_resource_cls = MagicMock(return_value=mock_resource)
        mock_resource_cls.create = MagicMock(return_value=mock_resource)

        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)

        mock_batch_processor = MagicMock()
        mock_batch_cls = MagicMock(return_value=mock_batch_processor)

        mock_console_exporter = MagicMock()
        mock_console_cls = MagicMock(return_value=mock_console_exporter)

        mock_trace = MagicMock()
        mock_resource_attrs = MagicMock()
        mock_resource_attrs.SERVICE_NAME = "service.name"

        config = TracingConfig(
            service_name="test-service",
            console_export=True,
        )

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource_cls, create=True),
            patch.object(obs_module, "TracerProvider", mock_provider_cls, create=True),
            patch.object(obs_module, "BatchSpanProcessor", mock_batch_cls, create=True),
            patch.object(obs_module, "ConsoleSpanExporter", mock_console_cls, create=True),
            patch.object(obs_module, "ResourceAttributes", mock_resource_attrs, create=True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            setup_otel_tracing(config)

            mock_resource_cls.create.assert_called_once()
            mock_provider_cls.assert_called_once()
            mock_provider.add_span_processor.assert_called()
            mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)

    def test_setup_otel_with_jaeger_endpoint(self):
        """Test setup_otel_tracing with Jaeger endpoint."""
        mock_resource_cls = MagicMock()
        mock_resource_cls.create = MagicMock(return_value=MagicMock())

        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)

        mock_trace = MagicMock()
        mock_resource_attrs = MagicMock()
        mock_resource_attrs.SERVICE_NAME = "service.name"

        config = TracingConfig(
            service_name="test-service",
            jaeger_endpoint="http://localhost:14268/api/traces",
        )

        # Mock the jaeger import to fail (ImportError path)
        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource_cls, create=True),
            patch.object(obs_module, "TracerProvider", mock_provider_cls, create=True),
            patch.object(obs_module, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(obs_module, "ConsoleSpanExporter", MagicMock(), create=True),
            patch.object(obs_module, "ResourceAttributes", mock_resource_attrs, create=True),
            patch.object(obs_module, "trace", mock_trace),
            patch.dict("sys.modules", {"opentelemetry.exporter.jaeger.thrift": None}),
        ):
            setup_otel_tracing(config)
            mock_trace.set_tracer_provider.assert_called_once()

    def test_setup_otel_with_otlp_endpoint(self):
        """Test setup_otel_tracing with OTLP endpoint."""
        mock_resource_cls = MagicMock()
        mock_resource_cls.create = MagicMock(return_value=MagicMock())

        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)

        mock_trace = MagicMock()
        mock_resource_attrs = MagicMock()
        mock_resource_attrs.SERVICE_NAME = "service.name"

        config = TracingConfig(
            service_name="test-service",
            otlp_endpoint="http://localhost:4317",
        )

        # Mock the OTLP import to fail (ImportError path)
        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource_cls, create=True),
            patch.object(obs_module, "TracerProvider", mock_provider_cls, create=True),
            patch.object(obs_module, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(obs_module, "ConsoleSpanExporter", MagicMock(), create=True),
            patch.object(obs_module, "ResourceAttributes", mock_resource_attrs, create=True),
            patch.object(obs_module, "trace", mock_trace),
            patch.dict(
                "sys.modules", {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None}
            ),
        ):
            setup_otel_tracing(config)
            mock_trace.set_tracer_provider.assert_called_once()

    def test_setup_otel_with_custom_attributes(self):
        """Test setup_otel_tracing with custom attributes."""
        mock_resource_cls = MagicMock()
        mock_resource_cls.create = MagicMock(return_value=MagicMock())

        mock_provider = MagicMock()
        mock_provider_cls = MagicMock(return_value=mock_provider)

        mock_trace = MagicMock()
        mock_resource_attrs = MagicMock()
        mock_resource_attrs.SERVICE_NAME = "service.name"

        config = TracingConfig(
            service_name="test-service",
            custom_attributes={"env": "staging", "version": "1.0"},
        )

        with (
            patch.object(obs_module, "OTEL_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource_cls, create=True),
            patch.object(obs_module, "TracerProvider", mock_provider_cls, create=True),
            patch.object(obs_module, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(obs_module, "ConsoleSpanExporter", MagicMock(), create=True),
            patch.object(obs_module, "ResourceAttributes", mock_resource_attrs, create=True),
            patch.object(obs_module, "trace", mock_trace),
        ):
            setup_otel_tracing(config)

            # Verify custom attributes were passed to Resource.create
            call_args = mock_resource_cls.create.call_args
            attrs_dict = call_args[0][0]
            assert attrs_dict.get("env") == "staging"
            assert attrs_dict.get("version") == "1.0"


# =============================================================================
# OBSERVABILITY: TelemetryCollector edge cases
# =============================================================================


class TestTelemetryCollectorCoverage:
    """Additional TelemetryCollector tests targeting coverage gaps."""

    def test_get_stats_p95_with_20_plus_records(self):
        """Test p95 latency calculation with 20+ records."""
        collector = TelemetryCollector()
        now = datetime.now()

        for i in range(25):
            collector.record(
                CallRecord(
                    model_name="test",
                    operation="generate",
                    start_time=now,
                    end_time=now,
                    latency_ms=float(i * 10),
                    success=True,
                    prompt_tokens=10,
                    completion_tokens=20,
                )
            )

        stats = collector.get_stats()
        assert stats["total_calls"] == 25
        assert "p95_latency_ms" in stats
        # p95 should be close to the 95th percentile value
        assert stats["p95_latency_ms"] > stats["p50_latency_ms"]

    def test_get_stats_all_failures(self):
        """Test stats with all failed calls."""
        collector = TelemetryCollector()
        now = datetime.now()

        for i in range(5):
            collector.record(
                CallRecord(
                    model_name="test",
                    operation="generate",
                    start_time=now,
                    end_time=now,
                    latency_ms=100.0,
                    success=False,
                    error="Failed",
                )
            )

        stats = collector.get_stats()
        assert stats["success_rate"] == 0.0
        assert stats["failures"] == 5

    def test_get_stats_combined_filters(self):
        """Test stats with multiple filters applied simultaneously."""
        collector = TelemetryCollector()
        now = datetime.now()
        old = now - timedelta(hours=2)

        # Old gpt-4 generate
        collector.record(
            CallRecord(
                model_name="gpt-4",
                operation="generate",
                start_time=old,
                end_time=old,
                latency_ms=100.0,
                success=True,
            )
        )
        # Recent gpt-4 generate
        collector.record(
            CallRecord(
                model_name="gpt-4",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=200.0,
                success=True,
            )
        )
        # Recent gpt-4 chat
        collector.record(
            CallRecord(
                model_name="gpt-4",
                operation="chat",
                start_time=now,
                end_time=now,
                latency_ms=300.0,
                success=True,
            )
        )

        since = now - timedelta(hours=1)
        stats = collector.get_stats(model_name="gpt-4", operation="generate", since=since)
        assert stats["total_calls"] == 1
        assert stats["avg_latency_ms"] == 200.0

    def test_multiple_callbacks(self):
        """Test multiple callbacks all receive records."""
        collector = TelemetryCollector()
        results1 = []
        results2 = []

        collector.add_callback(lambda r: results1.append(r.model_name))
        collector.add_callback(lambda r: results2.append(r.operation))

        now = datetime.now()
        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
        )

        assert results1 == ["test"]
        assert results2 == ["generate"]


# =============================================================================
# OBSERVABILITY: trace_call edge cases
# =============================================================================


class TestTraceCallCoverage:
    """Additional trace_call tests."""

    def test_trace_call_uses_global_collector(self):
        """Test trace_call uses global collector when none provided."""
        original = get_collector()
        test_collector = TelemetryCollector()
        set_collector(test_collector)

        try:
            with trace_call("model", "generate", "prompt") as ctx:
                ctx["response"] = "hello"

            records = test_collector.get_records()
            assert len(records) >= 1
        finally:
            set_collector(original)

    def test_trace_call_no_response_set(self):
        """Test trace_call when response is not set in context."""
        collector = TelemetryCollector()

        with trace_call("model", "op", "prompt", collector):
            pass  # Don't set response

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].completion_tokens == 0

    def test_trace_call_empty_prompt(self):
        """Test trace_call with empty prompt."""
        collector = TelemetryCollector()

        with trace_call("model", "op", "", collector) as ctx:
            ctx["response"] = "some response"

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].prompt_length == 0


# =============================================================================
# OBSERVABILITY: TracedModel edge cases
# =============================================================================


class TestTracedModelCoverage:
    """Additional TracedModel tests for coverage."""

    def test_traced_model_generate_with_kwargs_metadata(self):
        """Test generate includes kwargs in metadata."""
        from insideLLMs.models import DummyModel

        collector = TelemetryCollector()
        model = DummyModel()
        traced = TracedModel(model, collector)

        traced.generate("Hello", temperature=0.5, max_tokens=100)

        records = collector.get_records()
        assert len(records) == 1
        assert "kwargs" in records[0].metadata

    def test_traced_model_generate_with_log_prompts(self):
        """Test generate with log_prompts enabled captures prompt tokens."""
        from insideLLMs.models import DummyModel

        collector = TelemetryCollector()
        config = TracingConfig(log_prompts=True, log_responses=True)
        model = DummyModel()
        traced = TracedModel(model, collector, config)

        traced.generate("This is a test prompt")

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].prompt_tokens > 0

    def test_traced_model_chat_with_log_prompts(self):
        """Test chat with log_prompts captures message info."""
        from insideLLMs.models import DummyModel

        collector = TelemetryCollector()
        config = TracingConfig(log_prompts=True, log_responses=True)
        model = DummyModel()
        traced = TracedModel(model, collector, config)

        messages = [{"role": "user", "content": "Hello"}]
        traced.chat(messages)

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].prompt_length > 0

    def test_traced_model_generate_error_recorded(self):
        """Test generate records error when model fails."""
        collector = TelemetryCollector()
        model = MagicMock()
        model.name = "failing-model"
        model.generate = MagicMock(side_effect=ValueError("model error"))

        traced = TracedModel(model, collector)

        with pytest.raises(ValueError, match="model error"):
            traced.generate("test")

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].success is False
        assert "model error" in records[0].error


# =============================================================================
# OBSERVABILITY: trace_function edge cases
# =============================================================================


class TestTraceFunctionCoverage:
    """Additional trace_function tests."""

    def test_trace_function_preserves_return_value(self):
        """Test trace_function preserves the decorated function's return."""
        original = get_collector()
        collector = TelemetryCollector()
        set_collector(collector)

        try:

            @trace_function(operation_name="custom")
            def add(a, b):
                return a + b

            result = add(3, 4)
            assert result == 7
        finally:
            set_collector(original)

    def test_trace_function_no_args_recording(self):
        """Test trace_function without include_args."""
        original = get_collector()
        collector = TelemetryCollector()
        set_collector(collector)

        try:

            @trace_function()
            def my_func():
                return 42

            my_func()

            records = collector.get_records()
            last = records[-1]
            assert "args" not in last.metadata
        finally:
            set_collector(original)
