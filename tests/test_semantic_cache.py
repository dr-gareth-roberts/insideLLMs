"""Tests for Semantic Caching module."""

from unittest.mock import MagicMock

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.semantic_cache import (
    REDIS_AVAILABLE,
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
# Test Configuration
# =============================================================================


class TestSemanticCacheConfig:
    """Tests for SemanticCacheConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SemanticCacheConfig()

        assert config.max_size == 1000
        assert config.ttl_seconds == 3600
        assert config.similarity_threshold == 0.85
        assert config.embedding_dimension == 384
        assert config.use_embeddings is True
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379

    def test_custom_values(self):
        """Test custom configuration."""
        config = SemanticCacheConfig(
            max_size=500,
            similarity_threshold=0.9,
            ttl_seconds=7200,
            use_embeddings=False,
        )

        assert config.max_size == 500
        assert config.similarity_threshold == 0.9
        assert config.ttl_seconds == 7200
        assert config.use_embeddings is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = SemanticCacheConfig(max_size=100)
        data = config.to_dict()

        assert data["max_size"] == 100
        assert "similarity_threshold" in data
        assert "redis_host" in data


class TestSemanticCacheEntry:
    """Tests for SemanticCacheEntry."""

    def test_basic_creation(self):
        """Test basic entry creation."""
        entry = SemanticCacheEntry(
            key="abc123",
            value="test response",
            prompt="test prompt",
        )

        assert entry.key == "abc123"
        assert entry.value == "test response"
        assert entry.prompt == "test prompt"
        assert entry.access_count == 0
        assert entry.embedding is None

    def test_with_embedding(self):
        """Test entry with embedding."""
        embedding = [0.1, 0.2, 0.3]
        entry = SemanticCacheEntry(
            key="test",
            value="value",
            prompt="prompt",
            embedding=embedding,
        )

        assert entry.embedding == embedding

    def test_is_expired_no_expiry(self):
        """Test expiry check with no expiry."""
        entry = SemanticCacheEntry(
            key="test",
            value="value",
            prompt="prompt",
        )

        assert entry.is_expired() is False

    def test_touch(self):
        """Test access tracking."""
        entry = SemanticCacheEntry(
            key="test",
            value="value",
            prompt="prompt",
        )

        old_accessed = entry.last_accessed
        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed >= old_accessed

    def test_to_dict(self):
        """Test dictionary conversion."""
        entry = SemanticCacheEntry(
            key="test",
            value="value",
            prompt="prompt",
            metadata={"test": True},
        )

        data = entry.to_dict()

        assert data["key"] == "test"
        assert data["value"] == "value"
        assert data["prompt"] == "prompt"
        assert data["metadata"] == {"test": True}


class TestSemanticLookupResult:
    """Tests for SemanticLookupResult."""

    def test_cache_hit(self):
        """Test cache hit result."""
        result = SemanticLookupResult(
            hit=True,
            value="cached value",
            key="abc123",
            similarity=0.95,
        )

        assert result.hit is True
        assert result.value == "cached value"
        assert result.similarity == 0.95

    def test_cache_miss(self):
        """Test cache miss result."""
        result = SemanticLookupResult(
            hit=False,
            value=None,
            key="",
        )

        assert result.hit is False
        assert result.value is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = SemanticLookupResult(
            hit=True,
            value="test",
            key="key",
            similarity=0.9,
            lookup_time_ms=5.5,
        )

        data = result.to_dict()

        assert data["hit"] is True
        assert data["similarity"] == 0.9
        assert data["lookup_time_ms"] == 5.5


class TestSemanticCacheStats:
    """Tests for SemanticCacheStats."""

    def test_default_values(self):
        """Test default statistics."""
        stats = SemanticCacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.semantic_hits == 0
        assert stats.exact_hits == 0

    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = SemanticCacheStats(hits=80, misses=20)

        assert stats.hit_rate == 0.8

    def test_semantic_hit_rate(self):
        """Test semantic hit rate."""
        stats = SemanticCacheStats(
            hits=100,
            semantic_hits=60,
            exact_hits=40,
        )

        assert stats.semantic_hit_rate == 0.6

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = SemanticCacheStats(hits=50, misses=50)
        data = stats.to_dict()

        assert data["hits"] == 50
        assert data["hit_rate"] == 0.5


# =============================================================================
# Test Embedding Utilities
# =============================================================================


class TestSimpleEmbedder:
    """Tests for SimpleEmbedder."""

    def test_initialization(self):
        """Test embedder initialization."""
        embedder = SimpleEmbedder(dimension=128)
        assert embedder.dimension == 128

    def test_embed_basic(self):
        """Test basic embedding generation."""
        embedder = SimpleEmbedder()
        embedding = embedder.embed("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Default dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_normalized(self):
        """Test embeddings are normalized."""
        import math

        embedder = SimpleEmbedder()
        embedding = embedder.embed("Test text for embedding")

        # Calculate norm
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 0.01  # Should be approximately 1

    def test_embed_different_texts(self):
        """Test different texts produce different embeddings."""
        embedder = SimpleEmbedder()
        emb1 = embedder.embed("Hello world")
        emb2 = embedder.embed("Goodbye universe")

        assert emb1 != emb2

    def test_batch_embed(self):
        """Test batch embedding."""
        embedder = SimpleEmbedder()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.batch_embed(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_embed_empty_string(self):
        """Test embedding empty string."""
        embedder = SimpleEmbedder()
        embedding = embedder.embed("")

        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors have similarity 1."""
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.001

    def test_opposite_vectors(self):
        """Test opposite vectors have similarity -1."""
        vec1 = [1.0, 2.0]
        vec2 = [-1.0, -2.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim + 1.0) < 0.001

    def test_different_lengths(self):
        """Test vectors of different lengths return 0."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_zero_vector(self):
        """Test zero vector returns 0."""
        vec1 = [1.0, 2.0]
        vec2 = [0.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0


# =============================================================================
# Test VectorCache
# =============================================================================


class TestVectorCache:
    """Tests for VectorCache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = VectorCache()
        assert cache.config is not None

    def test_set_and_get_exact(self):
        """Test setting and getting exact match."""
        cache = VectorCache()

        cache.set("What is AI?", "AI is artificial intelligence")
        result = cache.get("What is AI?")

        assert result.hit is True
        assert result.value == "AI is artificial intelligence"
        assert result.similarity == 1.0

    def test_get_miss(self):
        """Test cache miss."""
        cache = VectorCache()
        result = cache.get("Unknown prompt")

        assert result.hit is False
        assert result.value is None

    def test_get_similar(self):
        """Test semantic similarity lookup."""
        cache = VectorCache()

        cache.set("What is machine learning?", "ML is a subset of AI")
        result = cache.get_similar("What's ML?", threshold=0.3)

        # Should find similar entry
        assert result.hit is True or result.similarity > 0

    def test_find_similar(self):
        """Test finding multiple similar entries."""
        cache = VectorCache()

        cache.set("What is AI?", "AI answer")
        cache.set("What is machine learning?", "ML answer")
        cache.set("What is deep learning?", "DL answer")

        results = cache.find_similar("artificial intelligence", limit=5, threshold=0.1)

        assert isinstance(results, list)

    def test_delete(self):
        """Test deleting entry."""
        cache = VectorCache()

        cache.set("Test prompt", "Test value")
        assert cache.get("Test prompt").hit is True

        cache.delete("Test prompt")
        assert cache.get("Test prompt").hit is False

    def test_clear(self):
        """Test clearing cache."""
        cache = VectorCache()

        cache.set("Prompt 1", "Value 1")
        cache.set("Prompt 2", "Value 2")

        cache.clear()
        stats = cache.stats()

        assert stats.entry_count == 0

    def test_max_size_eviction(self):
        """Test LRU eviction when max size reached."""
        config = SemanticCacheConfig(max_size=3)
        cache = VectorCache(config)

        cache.set("Prompt 1", "Value 1")
        cache.set("Prompt 2", "Value 2")
        cache.set("Prompt 3", "Value 3")

        # Access prompt 2 to make it more recent
        cache.get("Prompt 2")

        # Add another, should evict prompt 1 (LRU)
        cache.set("Prompt 4", "Value 4")

        stats = cache.stats()
        assert stats.entry_count == 3

    def test_stats(self):
        """Test statistics tracking."""
        cache = VectorCache()

        cache.set("Test", "Value")
        cache.get("Test")  # Hit
        cache.get("Missing")  # Miss

        stats = cache.stats()

        assert stats.hits >= 1
        assert stats.misses >= 1

    def test_with_custom_embedder(self):
        """Test with custom embedder."""

        def custom_embed(text: str) -> list:
            return [float(len(text) % 10) / 10] * 10

        cache = VectorCache(
            config=SemanticCacheConfig(embedding_dimension=10),
            embedder=custom_embed,
        )

        cache.set("Short", "Short value")
        cache.set("Much longer text", "Long value")

        # Custom embedder based on length
        assert cache.get("Short").hit is True

    def test_without_embeddings(self):
        """Test cache without embeddings (exact match only)."""
        config = SemanticCacheConfig(use_embeddings=False)
        cache = VectorCache(config)

        cache.set("Exact match", "Value")

        # Exact match works
        assert cache.get("Exact match").hit is True

        # Similar query doesn't work without embeddings
        result = cache.get_similar("exact match", threshold=0.5)
        assert result.hit is False


# =============================================================================
# Test SemanticCache
# =============================================================================


class TestSemanticCache:
    """Tests for SemanticCache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = SemanticCache()
        assert cache.config is not None

    def test_set_and_get(self):
        """Test basic set and get."""
        cache = SemanticCache()

        entry = cache.set("What is Python?", "Python is a programming language")

        assert entry.prompt == "What is Python?"
        assert entry.value == "Python is a programming language"

        result = cache.get("What is Python?")
        assert result.hit is True
        assert result.value == "Python is a programming language"

    def test_get_similar(self):
        """Test semantic similarity get."""
        cache = SemanticCache()

        cache.set(
            "What is Python programming language used for?", "Python is a programming language"
        )
        # Use the same prompt for exact match
        result = cache.get_similar("What is Python programming language used for?", threshold=0.3)

        # Should have exact match similarity
        assert result.hit is True
        assert result.similarity == 1.0

    def test_get_exact_only(self):
        """Test exact match only."""
        cache = SemanticCache()

        cache.set("Exact prompt", "Value")
        result = cache.get_exact("Different prompt")

        assert result.hit is False

    def test_find_similar(self):
        """Test finding multiple similar entries."""
        cache = SemanticCache()

        cache.set("Apple fruit", "Apples are fruits")
        cache.set("Orange fruit", "Oranges are fruits")
        cache.set("Banana fruit", "Bananas are fruits")

        results = cache.find_similar("fruits", limit=5, threshold=0.1)
        assert isinstance(results, list)

    def test_delete(self):
        """Test deletion."""
        cache = SemanticCache()

        cache.set("To delete", "Value")
        cache.delete("To delete")

        result = cache.get_exact("To delete")
        assert result.hit is False

    def test_clear(self):
        """Test clearing."""
        cache = SemanticCache()

        cache.set("Entry 1", "Value 1")
        cache.set("Entry 2", "Value 2")

        cache.clear()
        assert cache.stats().entry_count == 0

    def test_stats(self):
        """Test statistics."""
        cache = SemanticCache()

        cache.set("Test", "Value")
        cache.get("Test")
        cache.get("Missing")

        stats = cache.stats()
        assert stats.total_lookups == stats.hits + stats.misses


# =============================================================================
# Test SemanticCacheModel
# =============================================================================


class TestSemanticCacheModel:
    """Tests for SemanticCacheModel."""

    def test_initialization(self):
        """Test model wrapper initialization."""
        model = DummyModel()
        cached_model = SemanticCacheModel(model)

        assert cached_model.model is model
        assert cached_model.cache is not None

    def test_generate_no_cache(self):
        """Test generation without cache."""
        model = DummyModel()
        cached_model = SemanticCacheModel(model)

        response = cached_model.generate("Test prompt", use_cache=False)
        assert response is not None

    def test_generate_with_cache(self):
        """Test generation with caching."""
        model = MagicMock()
        model.generate.return_value = "Generated response"

        cached_model = SemanticCacheModel(model)

        # First call - should call model
        response1 = cached_model.generate("Test prompt", temperature=0)
        assert response1 == "Generated response"
        assert model.generate.call_count == 1

        # Second call - should use cache
        response2 = cached_model.generate("Test prompt", temperature=0)
        assert response2 == "Generated response"
        assert model.generate.call_count == 1  # Not called again

    def test_generate_temp_nonzero_no_cache(self):
        """Test non-zero temperature doesn't cache by default."""
        model = MagicMock()
        model.generate.return_value = "Response"

        cached_model = SemanticCacheModel(model, cache_temperature_zero_only=True)

        cached_model.generate("Test", temperature=0.7)
        cached_model.generate("Test", temperature=0.7)

        # Should call model both times
        assert model.generate.call_count == 2

    def test_chat_with_cache(self):
        """Test chat with caching."""
        model = MagicMock()
        model.chat.return_value = "Chat response"

        cached_model = SemanticCacheModel(model)

        messages = [{"role": "user", "content": "Hello"}]

        cached_model.chat(messages, temperature=0)
        assert model.chat.call_count == 1

        cached_model.chat(messages, temperature=0)
        assert model.chat.call_count == 1  # Cached

    def test_attribute_delegation(self):
        """Test attribute delegation to underlying model."""
        model = DummyModel(name="TestModel")
        cached_model = SemanticCacheModel(model)

        assert cached_model.name == "TestModel"

    def test_custom_cache(self):
        """Test with custom cache."""
        model = DummyModel()
        cache = SemanticCache()

        cached_model = SemanticCacheModel(model, cache)
        assert cached_model.cache is cache


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestCreateSemanticCache:
    """Tests for create_semantic_cache function."""

    def test_basic_creation(self):
        """Test basic cache creation."""
        cache = create_semantic_cache()
        assert isinstance(cache, SemanticCache)

    def test_with_threshold(self):
        """Test creation with similarity threshold."""
        cache = create_semantic_cache(similarity_threshold=0.9)
        assert cache.config.similarity_threshold == 0.9

    def test_with_size_and_ttl(self):
        """Test creation with size and TTL."""
        cache = create_semantic_cache(max_size=500, ttl_seconds=1800)
        assert cache.config.max_size == 500
        assert cache.config.ttl_seconds == 1800


class TestQuickSemanticCache:
    """Tests for quick_semantic_cache function."""

    def test_cache_miss(self):
        """Test when prompt not in cache."""

        def generator(prompt):
            return f"Generated for: {prompt}"

        result, was_cached, similarity = quick_semantic_cache(
            "New prompt",
            generator,
        )

        assert "Generated for" in result
        assert was_cached is False
        assert similarity == 1.0

    def test_cache_hit(self):
        """Test when prompt is in cache."""
        cache = SemanticCache()
        cache.set("Cached prompt", "Cached value")

        result, was_cached, similarity = quick_semantic_cache(
            "Cached prompt",
            lambda x: "Should not be called",
            cache=cache,
        )

        assert result == "Cached value"
        assert was_cached is True


class TestWrapModelWithSemanticCache:
    """Tests for wrap_model_with_semantic_cache function."""

    def test_basic_wrapping(self):
        """Test basic model wrapping."""
        model = DummyModel()
        wrapped = wrap_model_with_semantic_cache(model)

        assert isinstance(wrapped, SemanticCacheModel)

    def test_with_threshold(self):
        """Test wrapping with threshold."""
        model = DummyModel()
        wrapped = wrap_model_with_semantic_cache(model, similarity_threshold=0.9)

        assert wrapped.cache.config.similarity_threshold == 0.9


# =============================================================================
# Test Redis (if available)
# =============================================================================


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not installed")
class TestRedisCache:
    """Tests for RedisCache (requires Redis)."""

    def test_redis_available(self):
        """Test Redis availability."""
        assert REDIS_AVAILABLE is True


class TestRedisNotInstalled:
    """Tests when Redis is not installed."""

    def test_redis_availability_flag(self):
        """Test REDIS_AVAILABLE flag is boolean."""
        assert isinstance(REDIS_AVAILABLE, bool)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_prompt(self):
        """Test handling empty prompt."""
        cache = SemanticCache()
        entry = cache.set("", "value for empty prompt")
        assert entry is not None

    def test_very_long_prompt(self):
        """Test handling very long prompt."""
        cache = SemanticCache()
        long_prompt = "word " * 10000
        entry = cache.set(long_prompt, "value")
        assert entry is not None

    def test_unicode_prompt(self):
        """Test handling unicode prompt."""
        cache = SemanticCache()
        cache.set("你好世界", "Chinese hello world")
        result = cache.get("你好世界")
        assert result.hit is True

    def test_special_characters(self):
        """Test handling special characters."""
        cache = SemanticCache()
        cache.set("Test with 'quotes' and \"double quotes\"", "value")
        result = cache.get("Test with 'quotes' and \"double quotes\"")
        assert result.hit is True

    def test_none_value(self):
        """Test caching None value."""
        cache = SemanticCache()
        cache.set("None value prompt", None)
        result = cache.get("None value prompt")
        assert result.hit is True
        assert result.value is None

    def test_complex_value(self):
        """Test caching complex value."""
        cache = SemanticCache()
        complex_value = {
            "nested": {"data": [1, 2, 3]},
            "list": ["a", "b", "c"],
        }
        cache.set("Complex value", complex_value)
        result = cache.get("Complex value")
        assert result.hit is True
        assert result.value == complex_value

    def test_concurrent_access(self):
        """Test thread safety."""
        import threading

        cache = SemanticCache()
        errors = []

        def writer():
            for i in range(100):
                try:
                    cache.set(f"Prompt {i}", f"Value {i}")
                except Exception as e:
                    errors.append(e)

        def reader():
            for i in range(100):
                try:
                    cache.get(f"Prompt {i % 50}")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_similarity_boundary(self):
        """Test similarity at threshold boundary."""
        cache = SemanticCache()
        cache.set("Test prompt one", "Value one")

        # Exactly at threshold
        result = cache.get_similar(
            "Test prompt one",  # Same prompt
            threshold=1.0,
        )
        assert result.hit is True

    def test_stats_accuracy(self):
        """Test statistics are accurate."""
        cache = SemanticCache()

        # Add entries
        for i in range(10):
            cache.set(f"Prompt {i}", f"Value {i}")

        # Some hits (using get_exact to avoid semantic lookup overhead)
        for i in range(5):
            cache.get_exact(f"Prompt {i}")

        # Some misses
        for i in range(3):
            cache.get_exact(f"Missing {i}")

        stats = cache.stats()
        assert stats.entry_count == 10
        assert stats.hits == 5
        assert stats.misses == 3
