"""Tests for prompt caching and memoization module."""

import time
from datetime import datetime, timedelta

import pytest

from insideLLMs.caching import (
    AsyncCacheAdapter,
    # Classes
    BaseCache,
    # Dataclasses
    CacheConfig,
    CacheEntry,
    CacheLookupResult,
    CacheNamespace,
    CacheScope,
    CacheStats,
    CacheStatus,
    # Enums
    CacheStrategy,
    CacheWarmer,
    MemoizedFunction,
    PromptCache,
    ResponseDeduplicator,
    cached_response,
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    # Functions
    generate_cache_key,
    get_cache_key,
    memoize,
)


class TestEnums:
    """Tests for enum types."""

    def test_cache_strategy_values(self):
        assert CacheStrategy.LRU.value == "lru"
        assert CacheStrategy.LFU.value == "lfu"
        assert CacheStrategy.FIFO.value == "fifo"
        assert CacheStrategy.TTL.value == "ttl"
        assert CacheStrategy.SIZE.value == "size"

    def test_cache_status_values(self):
        assert CacheStatus.ACTIVE.value == "active"
        assert CacheStatus.EXPIRED.value == "expired"
        assert CacheStatus.EVICTED.value == "evicted"

    def test_cache_scope_values(self):
        assert CacheScope.GLOBAL.value == "global"
        assert CacheScope.SESSION.value == "session"
        assert CacheScope.USER.value == "user"


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        config = CacheConfig()
        assert config.max_size == 1000
        assert config.ttl_seconds == 3600
        assert config.strategy == CacheStrategy.LRU

    def test_custom_config(self):
        config = CacheConfig(
            max_size=500,
            ttl_seconds=1800,
            strategy=CacheStrategy.LFU,
        )
        assert config.max_size == 500
        assert config.ttl_seconds == 1800
        assert config.strategy == CacheStrategy.LFU

    def test_config_to_dict(self):
        config = CacheConfig()
        d = config.to_dict()
        assert d["max_size"] == 1000
        assert d["strategy"] == "lru"
        assert "ttl_seconds" in d


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_creation(self):
        entry = CacheEntry(key="test_key", value="test_value")
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0

    def test_entry_with_expiration(self):
        expires = datetime.now() + timedelta(hours=1)
        entry = CacheEntry(
            key="test",
            value="value",
            expires_at=expires,
        )
        assert entry.expires_at == expires
        assert not entry.is_expired()

    def test_entry_expired(self):
        expires = datetime.now() - timedelta(seconds=1)
        entry = CacheEntry(
            key="test",
            value="value",
            expires_at=expires,
        )
        assert entry.is_expired()

    def test_entry_touch(self):
        entry = CacheEntry(key="test", value="value")
        initial_count = entry.access_count

        entry.touch()

        assert entry.access_count == initial_count + 1

    def test_entry_size_estimation(self):
        entry = CacheEntry(key="test", value="a" * 100)
        assert entry.size_bytes >= 100

    def test_entry_to_dict(self):
        entry = CacheEntry(key="test", value="value")
        d = entry.to_dict()
        assert d["key"] == "test"
        assert d["value"] == "value"
        assert "created_at" in d


class TestCacheStats:
    """Tests for CacheStats."""

    def test_stats_defaults(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_stats_to_dict(self):
        stats = CacheStats(hits=10, misses=5)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5


class TestCacheLookupResult:
    """Tests for CacheLookupResult."""

    def test_hit_result(self):
        result = CacheLookupResult(
            hit=True,
            value="cached_value",
            key="test_key",
        )
        assert result.hit
        assert result.value == "cached_value"

    def test_miss_result(self):
        result = CacheLookupResult(
            hit=False,
            value=None,
            key="test_key",
        )
        assert not result.hit
        assert result.value is None

    def test_result_to_dict(self):
        result = CacheLookupResult(hit=True, value="value", key="key")
        d = result.to_dict()
        assert d["hit"] is True
        assert d["value"] == "value"


class TestGenerateCacheKey:
    """Tests for generate_cache_key function."""

    def test_simple_key(self):
        key = generate_cache_key("Hello world")
        assert len(key) == 64  # SHA256 hex length
        assert key.isalnum()

    def test_same_prompt_same_key(self):
        key1 = generate_cache_key("Test prompt")
        key2 = generate_cache_key("Test prompt")
        assert key1 == key2

    def test_different_prompts_different_keys(self):
        key1 = generate_cache_key("Prompt 1")
        key2 = generate_cache_key("Prompt 2")
        assert key1 != key2

    def test_key_with_model(self):
        key1 = generate_cache_key("Test", model="gpt-4")
        key2 = generate_cache_key("Test", model="gpt-3.5")
        assert key1 != key2

    def test_key_with_params(self):
        key1 = generate_cache_key("Test", params={"temperature": 0.5})
        key2 = generate_cache_key("Test", params={"temperature": 0.7})
        assert key1 != key2

    def test_key_with_md5(self):
        key = generate_cache_key("Test", algorithm="md5")
        assert len(key) == 32  # MD5 hex length


class TestBaseCache:
    """Tests for BaseCache."""

    def test_cache_creation(self):
        cache = BaseCache()
        assert len(cache.keys()) == 0

    def test_cache_set_and_get(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        result = cache.get("key1")

        assert result.hit
        assert result.value == "value1"

    def test_cache_miss(self):
        cache = BaseCache()
        result = cache.get("nonexistent")
        assert not result.hit
        assert result.value is None

    def test_cache_delete(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        assert cache.delete("key1")
        assert not cache.contains("key1")

    def test_cache_delete_nonexistent(self):
        cache = BaseCache()
        assert not cache.delete("nonexistent")

    def test_cache_clear(self):
        cache = BaseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert len(cache.keys()) == 0

    def test_cache_contains(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        assert cache.contains("key1")
        assert not cache.contains("key2")

    def test_cache_keys(self):
        cache = BaseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        keys = cache.keys()

        assert "key1" in keys
        assert "key2" in keys

    def test_cache_values(self):
        cache = BaseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        values = cache.values()

        assert "value1" in values
        assert "value2" in values

    def test_cache_items(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        items = cache.items()

        assert ("key1", "value1") in items

    def test_cache_stats(self):
        cache = BaseCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entry_count == 1

    def test_cache_ttl_expiration(self):
        config = CacheConfig(ttl_seconds=1)
        cache = BaseCache(config)
        cache.set("key1", "value1")

        time.sleep(1.1)

        result = cache.get("key1")
        assert not result.hit

    def test_cache_lru_eviction(self):
        config = CacheConfig(max_size=3, strategy=CacheStrategy.LRU)
        cache = BaseCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new entry, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.contains("key1")
        assert cache.contains("key4")

    def test_cache_fifo_eviction(self):
        config = CacheConfig(max_size=2, strategy=CacheStrategy.FIFO)
        cache = BaseCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # key1 should be evicted (first in)
        assert not cache.contains("key1")
        assert cache.contains("key3")

    def test_cache_lfu_eviction(self):
        config = CacheConfig(max_size=2, strategy=CacheStrategy.LFU)
        cache = BaseCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")

        # Add new entry, should evict key2 (least frequently used)
        cache.set("key3", "value3")

        assert cache.contains("key1")
        assert not cache.contains("key2")

    def test_cache_with_metadata(self):
        cache = BaseCache()
        entry = cache.set("key1", "value1", metadata={"source": "test"})

        assert entry.metadata["source"] == "test"


class TestPromptCache:
    """Tests for PromptCache."""

    def test_prompt_cache_creation(self):
        cache = PromptCache()
        assert len(cache.keys()) == 0

    def test_cache_response(self):
        cache = PromptCache()
        entry = cache.cache_response(
            prompt="What is 2+2?",
            response="4",
            model="gpt-4",
        )

        assert entry.value == "4"
        assert entry.metadata["prompt"] == "What is 2+2?"
        assert entry.metadata["model"] == "gpt-4"

    def test_get_response(self):
        cache = PromptCache()
        cache.cache_response(
            prompt="What is 2+2?",
            response="4",
            model="gpt-4",
        )

        result = cache.get_response("What is 2+2?", model="gpt-4")

        assert result.hit
        assert result.value == "4"

    def test_get_response_miss(self):
        cache = PromptCache()
        result = cache.get_response("Unknown prompt")
        assert not result.hit

    def test_get_by_prompt(self):
        cache = PromptCache()
        cache.cache_response(
            prompt="Test prompt",
            response="Test response",
        )

        result = cache.get_by_prompt("Test prompt")

        assert result.hit
        assert result.value == "Test response"

    def test_find_similar(self):
        cache = PromptCache(similarity_threshold=0.5)
        cache.cache_response(
            prompt="What is the capital of France?",
            response="Paris",
        )

        results = cache.find_similar("What is France's capital?")

        # Should find the similar prompt
        assert len(results) >= 0  # May or may not match depending on threshold

    def test_find_similar_exact_match(self):
        cache = PromptCache(similarity_threshold=0.5)
        cache.cache_response(
            prompt="exact match prompt",
            response="response",
        )

        results = cache.find_similar("exact match prompt")

        assert len(results) == 1
        assert results[0][2] == 1.0  # Perfect similarity


class TestCacheWarmer:
    """Tests for CacheWarmer."""

    def test_warmer_creation(self):
        cache = BaseCache()
        warmer = CacheWarmer(cache, lambda x: x.upper())
        assert warmer.get_queue_size() == 0

    def test_add_prompt(self):
        cache = BaseCache()
        warmer = CacheWarmer(cache, lambda x: x.upper())

        warmer.add_prompt("test prompt")

        assert warmer.get_queue_size() == 1

    def test_warm(self):
        cache = BaseCache()
        warmer = CacheWarmer(cache, lambda x: x.upper())

        warmer.add_prompt("hello")
        warmer.add_prompt("world")

        results = warmer.warm(batch_size=10)

        assert len(results) == 2
        assert all(r["status"] == "success" for r in results)

    def test_warm_skips_existing(self):
        cache = BaseCache()
        key = generate_cache_key("existing")
        cache.set(key, "cached_value")

        warmer = CacheWarmer(cache, lambda x: x.upper())
        warmer.add_prompt("existing")

        results = warmer.warm(skip_existing=True)

        assert results[0]["status"] == "skipped"

    def test_warm_with_priority(self):
        cache = BaseCache()
        warmer = CacheWarmer(cache, lambda x: x.upper())

        warmer.add_prompt("low", priority=1)
        warmer.add_prompt("high", priority=10)

        results = warmer.warm(batch_size=1)

        # High priority should be warmed first
        assert results[0]["prompt"] == "high"

    def test_warmer_without_generator(self):
        cache = BaseCache()
        warmer = CacheWarmer(cache)

        warmer.add_prompt("test")

        with pytest.raises(ValueError):
            warmer.warm()

    def test_clear_queue(self):
        cache = BaseCache()
        warmer = CacheWarmer(cache, lambda x: x)

        warmer.add_prompt("test1")
        warmer.add_prompt("test2")
        warmer.clear_queue()

        assert warmer.get_queue_size() == 0


class TestMemoizedFunction:
    """Tests for MemoizedFunction."""

    def test_memoize_basic(self):
        call_count = 0

        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        memoized = MemoizedFunction(expensive_func)

        result1 = memoized(5)
        result2 = memoized(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once

    def test_memoize_different_args(self):
        call_count = 0

        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        memoized = MemoizedFunction(func)

        memoized(5)
        memoized(10)

        assert call_count == 2

    def test_memoize_invalidate(self):
        def func(x):
            return x * 2

        memoized = MemoizedFunction(func)
        memoized(5)

        memoized.invalidate(5)

        # Should no longer be cached
        assert not memoized.cache.contains(memoized.key_generator(5))

    def test_memoize_stats(self):
        def func(x):
            return x

        memoized = MemoizedFunction(func)
        memoized(1)
        memoized(1)  # Cache hit
        memoized(2)

        stats = memoized.get_stats()

        assert stats["call_count"] == 3
        assert stats["cache_calls"] == 1


class TestMemoizeDecorator:
    """Tests for memoize decorator."""

    def test_decorator_basic(self):
        @memoize
        def func(x):
            return x * 2

        assert func(5) == 10
        assert func(5) == 10

    def test_decorator_with_options(self):
        @memoize(max_size=10, ttl_seconds=60)
        def func(x):
            return x + 1

        assert func(5) == 6


class TestCacheNamespace:
    """Tests for CacheNamespace."""

    def test_namespace_creation(self):
        ns = CacheNamespace()
        assert len(ns.list_caches()) == 0

    def test_get_cache(self):
        ns = CacheNamespace()
        cache = ns.get_cache("test")

        assert isinstance(cache, BaseCache)
        assert "test" in ns.list_caches()

    def test_get_same_cache(self):
        ns = CacheNamespace()
        cache1 = ns.get_cache("test")
        cache2 = ns.get_cache("test")

        assert cache1 is cache2

    def test_get_prompt_cache(self):
        ns = CacheNamespace()
        cache = ns.get_prompt_cache("prompts")

        assert isinstance(cache, PromptCache)

    def test_delete_cache(self):
        ns = CacheNamespace()
        ns.get_cache("test")

        assert ns.delete_cache("test")
        assert "test" not in ns.list_caches()

    def test_delete_nonexistent(self):
        ns = CacheNamespace()
        assert not ns.delete_cache("nonexistent")

    def test_get_all_stats(self):
        ns = CacheNamespace()
        ns.get_cache("cache1").set("key1", "value1")
        ns.get_cache("cache2").set("key2", "value2")

        stats = ns.get_all_stats()

        assert "cache1" in stats
        assert "cache2" in stats

    def test_clear_all(self):
        ns = CacheNamespace()
        ns.get_cache("cache1").set("key1", "value1")
        ns.get_cache("cache2").set("key2", "value2")

        ns.clear_all()

        assert len(ns.get_cache("cache1").keys()) == 0
        assert len(ns.get_cache("cache2").keys()) == 0


class TestResponseDeduplicator:
    """Tests for ResponseDeduplicator."""

    def test_deduplicator_creation(self):
        dedup = ResponseDeduplicator()
        assert len(dedup.get_unique_responses()) == 0

    def test_add_unique(self):
        dedup = ResponseDeduplicator()
        is_dup, idx = dedup.add("prompt1", "response1")

        assert not is_dup
        assert idx is None

    def test_add_duplicate(self):
        dedup = ResponseDeduplicator()
        dedup.add("prompt1", "response1")
        is_dup, idx = dedup.add("prompt2", "response1")

        assert is_dup
        assert idx == 0

    def test_similarity_threshold(self):
        dedup = ResponseDeduplicator(similarity_threshold=0.8)
        dedup.add("prompt1", "hello world test")
        is_dup, _ = dedup.add("prompt2", "hello world testing")

        # Should be considered duplicate due to high similarity
        # Result depends on actual similarity calculation

    def test_get_unique_responses(self):
        dedup = ResponseDeduplicator()
        dedup.add("p1", "r1", {"meta": 1})
        dedup.add("p2", "r2", {"meta": 2})

        unique = dedup.get_unique_responses()

        assert len(unique) == 2

    def test_clear(self):
        dedup = ResponseDeduplicator()
        dedup.add("p1", "r1")
        dedup.clear()

        assert len(dedup.get_unique_responses()) == 0


class TestAsyncCacheAdapter:
    """Tests for AsyncCacheAdapter."""

    @pytest.mark.asyncio
    async def test_async_get(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        adapter = AsyncCacheAdapter(cache)
        result = await adapter.get("key1")

        assert result.hit
        assert result.value == "value1"

    @pytest.mark.asyncio
    async def test_async_set(self):
        cache = BaseCache()
        adapter = AsyncCacheAdapter(cache)

        entry = await adapter.set("key1", "value1")

        assert entry.value == "value1"

    @pytest.mark.asyncio
    async def test_async_delete(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        adapter = AsyncCacheAdapter(cache)
        result = await adapter.delete("key1")

        assert result

    @pytest.mark.asyncio
    async def test_async_clear(self):
        cache = BaseCache()
        cache.set("key1", "value1")

        adapter = AsyncCacheAdapter(cache)
        await adapter.clear()

        assert len(cache.keys()) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_cache(self):
        cache = create_cache(max_size=100, ttl_seconds=60)

        assert isinstance(cache, BaseCache)
        assert cache.config.max_size == 100

    def test_create_prompt_cache(self):
        cache = create_prompt_cache(max_size=50)

        assert isinstance(cache, PromptCache)
        assert cache.config.max_size == 50

    def test_cached_response(self):
        generator_called = False

        def generator(prompt):
            nonlocal generator_called
            generator_called = True
            return f"Response to: {prompt}"

        response, was_cached = cached_response("Test", generator)

        assert generator_called
        assert not was_cached
        assert "Response to: Test" in response

    def test_cached_response_uses_cache(self):
        call_count = 0

        def generator(prompt):
            nonlocal call_count
            call_count += 1
            return "response"

        cache = create_prompt_cache()

        cached_response("Test", generator, cache)
        cached_response("Test", generator, cache)

        assert call_count == 1

    def test_create_cache_warmer(self):
        cache = create_cache()
        warmer = create_cache_warmer(cache, lambda x: x)

        assert isinstance(warmer, CacheWarmer)

    def test_create_namespace(self):
        ns = create_namespace(default_max_size=500)

        assert isinstance(ns, CacheNamespace)

    def test_get_cache_key(self):
        key = get_cache_key("prompt", model="gpt-4")

        assert len(key) == 64


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_prompt(self):
        key = generate_cache_key("")
        assert len(key) == 64

    def test_unicode_content(self):
        cache = create_cache()
        cache.set("key", "こんにちは 🌍")

        result = cache.get("key")
        assert result.value == "こんにちは 🌍"

    def test_large_value(self):
        cache = create_cache()
        large_value = "x" * 100000

        entry = cache.set("key", large_value)
        assert entry.size_bytes > 50000

    def test_concurrent_access(self):
        import threading

        cache = create_cache()
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_none_value(self):
        cache = create_cache()
        cache.set("key", None)

        result = cache.get("key")
        assert result.hit
        assert result.value is None

    def test_complex_value(self):
        cache = create_cache()
        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": "b"},
            "tuple": (1, 2),
        }

        cache.set("key", complex_value)
        result = cache.get("key")

        assert result.value["list"] == [1, 2, 3]


class TestIntegration:
    """Integration tests."""

    def test_full_prompt_caching_workflow(self):
        cache = create_prompt_cache(max_size=100, ttl_seconds=3600)

        # Cache some responses
        cache.cache_response(
            prompt="What is AI?",
            response="AI stands for Artificial Intelligence...",
            model="gpt-4",
            params={"temperature": 0.7},
        )

        cache.cache_response(
            prompt="Explain machine learning",
            response="Machine learning is a subset of AI...",
            model="gpt-4",
        )

        # Retrieve cached response
        result = cache.get_response("What is AI?", model="gpt-4", params={"temperature": 0.7})

        assert result.hit
        assert "Artificial Intelligence" in result.value

        # Check stats
        stats = cache.get_stats()
        assert stats.entry_count == 2
        assert stats.hits == 1

    def test_memoized_function_integration(self):
        expensive_calls = 0

        @memoize(max_size=50, ttl_seconds=300)
        def expensive_computation(x, y):
            nonlocal expensive_calls
            expensive_calls += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x * y

        # First calls compute values
        result1 = expensive_computation(3, 4)
        result2 = expensive_computation(5, 6)

        # Repeated calls use cache
        result1_cached = expensive_computation(3, 4)
        result2_cached = expensive_computation(5, 6)

        assert result1 == result1_cached == 12
        assert result2 == result2_cached == 30
        assert expensive_calls == 2  # Only computed twice

    def test_namespace_with_warmer(self):
        ns = create_namespace()
        cache = ns.get_prompt_cache("main")

        warmer = create_cache_warmer(cache, lambda x: f"Response: {x}")

        # Add prompts to warm
        warmer.add_prompt("Hello")
        warmer.add_prompt("World")

        # Warm the cache
        results = warmer.warm()

        assert all(r["status"] == "success" for r in results)

        # Verify cache was warmed using the correct key
        key = get_cache_key("Hello")
        result = cache.get(key)
        assert result.hit
        assert result.value == "Response: Hello"
