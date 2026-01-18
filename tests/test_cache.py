"""Tests for caching module."""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from insideLLMs.cache import (
    CachedModel,
    CacheEntry,
    CacheStats,
    DiskCache,
    InMemoryCache,
    cached,
    clear_default_cache,
    generate_cache_key,
    get_default_cache,
    set_default_cache,
)
from insideLLMs.types import ModelResponse


class TestCacheKey:
    """Tests for cache key generation."""

    def test_deterministic_key(self):
        """Test that same inputs produce same key."""
        key1 = generate_cache_key("gpt-4", "Hello world", temperature=0.0)
        key2 = generate_cache_key("gpt-4", "Hello world", temperature=0.0)
        assert key1 == key2

    def test_different_prompts(self):
        """Test that different prompts produce different keys."""
        key1 = generate_cache_key("gpt-4", "Hello world")
        key2 = generate_cache_key("gpt-4", "Goodbye world")
        assert key1 != key2

    def test_different_models(self):
        """Test that different models produce different keys."""
        key1 = generate_cache_key("gpt-4", "Hello world")
        key2 = generate_cache_key("gpt-3.5-turbo", "Hello world")
        assert key1 != key2

    def test_different_temperatures(self):
        """Test that different temperatures produce different keys."""
        key1 = generate_cache_key("gpt-4", "Hello", temperature=0.0)
        key2 = generate_cache_key("gpt-4", "Hello", temperature=0.7)
        assert key1 != key2

    def test_additional_kwargs(self):
        """Test that additional kwargs affect key."""
        key1 = generate_cache_key("gpt-4", "Hello", system="You are helpful")
        key2 = generate_cache_key("gpt-4", "Hello", system="You are evil")
        assert key1 != key2


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_not_expired_no_expiry(self):
        """Test entry without expiry is never expired."""
        entry = CacheEntry(
            key="test",
            value="{}",
            created_at=datetime.now(),
            expires_at=None,
        )
        assert not entry.is_expired()

    def test_not_expired_future_expiry(self):
        """Test entry with future expiry is not expired."""
        entry = CacheEntry(
            key="test",
            value="{}",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert not entry.is_expired()

    def test_expired_past_expiry(self):
        """Test entry with past expiry is expired."""
        entry = CacheEntry(
            key="test",
            value="{}",
            created_at=datetime.now() - timedelta(seconds=100),
            expires_at=datetime.now() - timedelta(seconds=50),
        )
        assert entry.is_expired()


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_zero(self):
        """Test hit rate with no requests."""
        stats = CacheStats(hits=0, misses=0, hit_rate=0.0)
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        # hit_rate = hits / (hits + misses)
        stats = CacheStats(hits=10, misses=0, hit_rate=1.0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        stats = CacheStats(hits=0, misses=10, hit_rate=0.0)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits/misses."""
        # hit_rate = 3 / 10 = 0.3
        stats = CacheStats(hits=3, misses=7, hit_rate=0.3)
        assert stats.hit_rate == 0.3


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    def test_set_and_get(self):
        """Test basic set and get."""
        cache = InMemoryCache()
        cache.set("key1", {"data": "value"})
        result = cache.get("key1")
        assert result == {"data": "value"}

    def test_get_missing(self):
        """Test getting missing key."""
        cache = InMemoryCache()
        assert cache.get("missing") is None

    def test_has_existing(self):
        """Test has with existing key."""
        cache = InMemoryCache()
        cache.set("key1", "value")
        assert cache.has("key1")

    def test_has_missing(self):
        """Test has with missing key."""
        cache = InMemoryCache()
        assert not cache.has("missing")

    def test_delete_existing(self):
        """Test deleting existing key."""
        cache = InMemoryCache()
        cache.set("key1", "value")
        assert cache.delete("key1")
        assert not cache.has("key1")

    def test_delete_missing(self):
        """Test deleting missing key."""
        cache = InMemoryCache()
        assert not cache.delete("missing")

    def test_clear(self):
        """Test clearing cache."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert not cache.has("key1")
        assert not cache.has("key2")

    def test_ttl_expired(self):
        """Test TTL expiration."""
        cache = InMemoryCache(default_ttl=1)
        cache.set("key1", "value")
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_ttl_not_expired(self):
        """Test TTL not expired."""
        cache = InMemoryCache(default_ttl=10)
        cache.set("key1", "value")
        assert cache.get("key1") == "value"

    def test_max_size_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = InMemoryCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")

        # key1 should be evicted (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") is not None

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = InMemoryCache()
        cache.set("key1", "value")

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.entry_count == 1

    def test_keys(self):
        """Test getting all keys."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        keys = cache.keys()
        assert set(keys) == {"key1", "key2"}


class TestDiskCache:
    """Tests for DiskCache."""

    def test_set_and_get(self):
        """Test basic set and get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(path=Path(tmpdir) / "test.db")
            cache.set("key1", {"data": "value"})
            result = cache.get("key1")
            assert result == {"data": "value"}

    def test_persistence(self):
        """Test that data persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"

            # First instance
            cache1 = DiskCache(path=path)
            cache1.set("key1", "value1")

            # Second instance
            cache2 = DiskCache(path=path)
            assert cache2.get("key1") == "value1"

    def test_get_missing(self):
        """Test getting missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(path=Path(tmpdir) / "test.db")
            assert cache.get("missing") is None

    def test_delete(self):
        """Test deleting key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(path=Path(tmpdir) / "test.db")
            cache.set("key1", "value")
            assert cache.delete("key1")
            assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(path=Path(tmpdir) / "test.db")
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_ttl_expired(self):
        """Test TTL expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(path=Path(tmpdir) / "test.db", default_ttl=1)
            cache.set("key1", "value")
            time.sleep(1.1)
            assert cache.get("key1") is None

    def test_stats(self):
        """Test statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(path=Path(tmpdir) / "test.db")
            cache.set("key1", "value")
            cache.get("key1")
            cache.get("missing")

            stats = cache.stats()
            assert stats.hits == 1
            assert stats.misses == 1
            assert stats.entry_count == 1

    def test_export_import(self):
        """Test export and import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            export_path = Path(tmpdir) / "export.json"

            # Create and populate cache
            cache1 = DiskCache(path=db_path)
            cache1.set("key1", {"value": 1})
            cache1.set("key2", {"value": 2})

            # Export
            count = cache1.export_to_file(export_path)
            assert count == 2

            # Import to new cache
            cache2 = DiskCache(path=Path(tmpdir) / "test2.db")
            imported = cache2.import_from_file(export_path)
            assert imported == 2
            assert cache2.get("key1") == {"value": 1}


class MockModel:
    """Mock model for testing CachedModel."""

    def __init__(self, response: str = "test response"):
        self.model_id = "mock-model"
        self.call_count = 0
        self._response = response

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(
            content=self._response,
            model=self.model_id,
        )


class TestCachedModel:
    """Tests for CachedModel."""

    def test_caches_deterministic_request(self):
        """Test that deterministic requests are cached."""
        model = MockModel()
        cached_model = CachedModel(model)

        # First call
        response1 = cached_model.generate("Hello", temperature=0.0)
        assert model.call_count == 1

        # Second call (should be cached)
        response2 = cached_model.generate("Hello", temperature=0.0)
        assert model.call_count == 1
        assert response1.content == response2.content

    def test_does_not_cache_nondeterministic(self):
        """Test that non-deterministic requests are not cached by default."""
        model = MockModel()
        cached_model = CachedModel(model, cache_only_deterministic=True)

        # First call
        cached_model.generate("Hello", temperature=0.7)
        assert model.call_count == 1

        # Second call (should not be cached)
        cached_model.generate("Hello", temperature=0.7)
        assert model.call_count == 2

    def test_caches_all_when_disabled(self):
        """Test caching all requests when deterministic check disabled."""
        model = MockModel()
        cached_model = CachedModel(model, cache_only_deterministic=False)

        # First call
        cached_model.generate("Hello", temperature=0.7)
        assert model.call_count == 1

        # Second call (should be cached)
        cached_model.generate("Hello", temperature=0.7)
        assert model.call_count == 1

    def test_delegates_attributes(self):
        """Test that attributes are delegated to underlying model."""
        model = MockModel()
        cached_model = CachedModel(model)
        assert cached_model.model_id == "mock-model"

    def test_access_underlying_model(self):
        """Test access to underlying model."""
        model = MockModel()
        cached_model = CachedModel(model)
        assert cached_model.model is model

    def test_access_cache(self):
        """Test access to cache."""
        model = MockModel()
        cache = InMemoryCache()
        cached_model = CachedModel(model, cache=cache)
        assert cached_model.cache is cache


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_caches_function_result(self):
        """Test that function results are cached."""
        call_count = 0

        @cached()
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    def test_different_args_not_cached(self):
        """Test that different arguments are not cached together."""
        call_count = 0

        @cached()
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2

    def test_custom_key_function(self):
        """Test custom key function."""
        call_count = 0

        def key_fn(x: int, y: int) -> str:
            return f"sum:{x + y}"

        @cached(key_fn=key_fn)
        def add(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        # Same sum, same key
        result1 = add(2, 3)
        result2 = add(1, 4)  # Same sum, should hit cache

        assert result1 == 5
        assert result2 == 5  # Returns cached result
        assert call_count == 1


class TestDefaultCache:
    """Tests for default cache management."""

    def test_get_default_cache(self):
        """Test getting default cache."""
        clear_default_cache()
        cache = get_default_cache()
        assert isinstance(cache, InMemoryCache)

    def test_set_default_cache(self):
        """Test setting default cache."""
        new_cache = InMemoryCache(max_size=50)
        set_default_cache(new_cache)
        assert get_default_cache() is new_cache

    def test_clear_default_cache(self):
        """Test clearing default cache."""
        cache = get_default_cache()
        cache.set("test", "value")
        clear_default_cache()
        assert cache.get("test") is None
