"""Coverage gap tests for insideLLMs/caching.py — Wave 7 item W7-0071.

Exercises previously uncovered classes and functions:

* CacheConfig.to_dict / CacheEntry.to_dict / CacheLookupResult.to_dict
* CacheStats.to_dict
* generate_cache_key with model, params, md5 algorithm
* generate_model_cache_key with extra kwargs
* InMemoryCache — TTL expiry, delete, stats, _evict_lru, has()
* DiskCache — full CRUD, stats, export/import, default path branch
* StrategyCache — LFU / FIFO / SIZE eviction, delete, contains, values, items
* PromptCache — cache_response, get_response, get_by_prompt, find_similar
* CachedModel — generate (cache hit/miss), model/cache properties, __getattr__
* CacheWarmer — add_prompt (priority), warm (skip/success/error), queue ops
* MemoizedFunction / memoize decorator
* CacheNamespace — get_cache reuse, get_prompt_cache, delete_cache, list,
  get_all_stats, clear_all
* ResponseDeduplicator — duplicate/unique, similarity, get_unique, clear
* AsyncCacheAdapter — async get/set/delete/clear
* Convenience helpers — create_cache, create_prompt_cache, create_cache_warmer,
  create_namespace, get_cache_key, cached_response, cached decorator
* Global default cache — get_default_cache, set_default_cache, clear_default_cache
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from insideLLMs.caching import (
    AsyncCacheAdapter,
    CacheConfig,
    CachedModel,
    CacheEntry,
    CacheLookupResult,
    CacheNamespace,
    CacheStats,
    CacheStatus,
    CacheStrategy,
    CacheWarmer,
    DiskCache,
    InMemoryCache,
    MemoizedFunction,
    PromptCache,
    ResponseDeduplicator,
    StrategyCache,
    cached,
    cached_response,
    clear_default_cache,
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    generate_cache_key,
    generate_model_cache_key,
    get_cache_key,
    get_default_cache,
    memoize,
    set_default_cache,
)
from insideLLMs.types import ModelResponse, TokenUsage

# =============================================================================
# CacheConfig
# =============================================================================


def test_cache_config_to_dict_returns_all_fields():
    config = CacheConfig(max_size=500, strategy=CacheStrategy.LFU)
    d = config.to_dict()
    assert d["max_size"] == 500
    assert d["strategy"] == "lfu"
    assert "ttl_seconds" in d
    assert "enable_stats" in d


# =============================================================================
# CacheEntry
# =============================================================================


def test_cache_entry_to_dict_has_expected_keys():
    entry = CacheEntry(key="k1", value={"a": 1})
    d = entry.to_dict()
    assert d["key"] == "k1"
    assert d["value"] == {"a": 1}
    assert "created_at" in d
    assert d["expires_at"] is None


def test_cache_entry_to_dict_with_expires_at():
    expires = datetime.now() + timedelta(hours=1)
    entry = CacheEntry(key="k", value="v", expires_at=expires)
    d = entry.to_dict()
    assert d["expires_at"] is not None


# =============================================================================
# CacheLookupResult
# =============================================================================


def test_cache_lookup_result_to_dict_hit():
    result = CacheLookupResult(hit=True, value="data", key="k1")
    d = result.to_dict()
    assert d["hit"] is True
    assert d["value"] == "data"
    assert d["key"] == "k1"
    assert d["entry"] is None


def test_cache_lookup_result_to_dict_with_entry():
    entry = CacheEntry(key="k1", value="v")
    result = CacheLookupResult(hit=True, value="v", key="k1", entry=entry)
    d = result.to_dict()
    assert d["entry"] is not None
    assert d["entry"]["key"] == "k1"


# =============================================================================
# CacheStats
# =============================================================================


def test_cache_stats_to_dict_scalar_fields():
    stats = CacheStats(hits=10, misses=5, hit_rate=0.667)
    d = stats.to_dict()
    assert d["hits"] == 10
    assert d["misses"] == 5
    assert abs(d["hit_rate"] - 0.667) < 0.001


# =============================================================================
# generate_cache_key
# =============================================================================


def test_generate_cache_key_with_model_and_params():
    key = generate_cache_key("hello", model="gpt-4", params={"temperature": 0.7})
    assert len(key) == 64  # sha256

    key_same = generate_cache_key("hello", model="gpt-4", params={"temperature": 0.7})
    assert key == key_same


def test_generate_cache_key_md5():
    key = generate_cache_key("hello", algorithm="md5")
    assert len(key) == 32


def test_generate_cache_key_kwargs():
    key1 = generate_cache_key("hello", user_id="u1")
    key2 = generate_cache_key("hello", user_id="u2")
    assert key1 != key2


# =============================================================================
# generate_model_cache_key
# =============================================================================


def test_generate_model_cache_key_with_extra_kwargs():
    key = generate_model_cache_key(
        model_id="gpt-4",
        prompt="hello",
        temperature=0.0,
        max_tokens=100,
        top_p=0.9,
    )
    assert len(key) == 64


def test_generate_model_cache_key_deterministic():
    k1 = generate_model_cache_key("m", "p", 0.0, 50)
    k2 = generate_model_cache_key("m", "p", 0.0, 50)
    assert k1 == k2


# =============================================================================
# InMemoryCache
# =============================================================================


def test_inmemory_cache_has_method():
    cache = InMemoryCache()
    cache.set("k", "v")
    assert cache.has("k") is True
    assert cache.has("missing") is False


def test_inmemory_cache_delete_existing_and_missing():
    cache = InMemoryCache()
    cache.set("a", 1)
    assert cache.delete("a") is True
    assert cache.delete("a") is False  # already gone


def test_inmemory_cache_stats_returns_hit_rate():
    cache = InMemoryCache()
    cache.set("x", 42)
    cache.get("x")  # hit
    cache.get("y")  # miss
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert abs(stats.hit_rate - 0.5) < 0.001


def test_inmemory_cache_evicts_lru_when_full():
    cache = InMemoryCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # triggers eviction of oldest
    assert cache.stats().evictions >= 1
    assert len(cache.keys()) == 2


def test_inmemory_cache_ttl_expiry():
    cache = InMemoryCache(default_ttl=1)
    cache.set("tmp", "value")
    # Manually expire by manipulating the internal entry
    cache._cache["tmp"]["expires_at"] = time.time() - 1
    result = cache.get("tmp")
    assert result is None
    assert cache.stats().misses >= 1


def test_inmemory_cache_set_with_explicit_ttl():
    cache = InMemoryCache()
    cache.set("k", "v", ttl=3600)
    assert cache.get("k") == "v"


# =============================================================================
# DiskCache
# =============================================================================


class TestDiskCache:
    """DiskCache CRUD, stats, TTL expiry, export/import."""

    def test_set_and_get(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        cache.set("key1", {"data": 42})
        assert cache.get("key1") == {"data": 42}

    def test_get_miss(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        assert cache.get("nonexistent") is None

    def test_get_expired(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        cache.set("tmp", "val", ttl=1)
        # Force expiration
        conn = cache._get_conn()
        conn.execute("UPDATE cache SET expires_at = ? WHERE key = ?", (time.time() - 1, "tmp"))
        conn.commit()
        assert cache.get("tmp") is None

    def test_delete_existing(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        cache.set("del_me", "v")
        assert cache.delete("del_me") is True
        assert cache.get("del_me") is None

    def test_delete_missing(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        assert cache.delete("nonexistent") is False

    def test_clear(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.stats().hits == 0

    def test_stats_entry_count(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        cache.set("x", 1)
        cache.set("y", 2)
        cache.get("x")  # hit
        cache.get("z")  # miss
        stats = cache.stats()
        assert stats.entry_count == 2
        assert stats.hits == 1
        assert stats.misses == 1

    def test_export_to_file(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db")
        cache.set("e1", "v1")
        cache.set("e2", "v2", metadata={"info": "test"})
        export_path = tmp_path / "export.json"
        count = cache.export_to_file(export_path)
        assert count == 2
        data = json.loads(export_path.read_text())
        assert len(data) == 2

    def test_import_from_file(self, tmp_path):
        src = DiskCache(path=tmp_path / "src.db")
        src.set("imported", "hello")
        export_path = tmp_path / "backup.json"
        src.export_to_file(export_path)

        dst = DiskCache(path=tmp_path / "dst.db")
        count = dst.import_from_file(export_path)
        assert count == 1
        assert dst.get("imported") == "hello"

    def test_default_path_uses_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        cache = DiskCache(path=None, max_size_mb=1)
        assert "insideLLMs" in str(cache._path)

    def test_set_with_default_ttl(self, tmp_path):
        cache = DiskCache(path=tmp_path / "c.db", default_ttl=3600)
        cache.set("ttl_key", "ttl_val")
        assert cache.get("ttl_key") == "ttl_val"


# =============================================================================
# StrategyCache — eviction strategies and other methods
# =============================================================================


class TestStrategyCacheEvictions:
    def test_lfu_eviction(self):
        config = CacheConfig(strategy=CacheStrategy.LFU, max_size=2, ttl_seconds=None)
        cache = StrategyCache(config)
        cache.set("a", 1)
        cache.get("a")  # access "a" more
        cache.get("a")
        cache.set("b", 2)
        # now fill up; "b" has fewer accesses so it should be evicted
        cache.set("c", 3)
        stats = cache.get_stats()
        assert stats.evictions >= 1

    def test_fifo_eviction(self):
        config = CacheConfig(strategy=CacheStrategy.FIFO, max_size=2, ttl_seconds=None)
        cache = StrategyCache(config)
        cache.set("first", 1)
        cache.set("second", 2)
        cache.set("third", 3)
        assert cache.get_stats().evictions >= 1

    def test_size_eviction(self):
        config = CacheConfig(strategy=CacheStrategy.SIZE, max_size=2, ttl_seconds=None)
        cache = StrategyCache(config)
        cache._entries["small"] = CacheEntry(key="small", value="x", size_bytes=1)
        cache._entries["big"] = CacheEntry(key="big", value="y" * 100, size_bytes=100)
        cache._evict_one()
        assert "big" not in cache._entries

    def test_delete_returns_true_and_false(self):
        cache = StrategyCache()
        cache.set("del", "v")
        assert cache.delete("del") is True
        assert cache.delete("del") is False

    def test_contains_with_expired_entry(self):
        cache = StrategyCache()
        cache._entries["exp"] = CacheEntry(
            key="exp",
            value="v",
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        assert cache.contains("exp") is False

    def test_contains_missing_key(self):
        cache = StrategyCache()
        assert cache.contains("not_there") is False

    def test_values_excludes_expired(self):
        cache = StrategyCache()
        cache.set("valid", "ok")
        cache._entries["dead"] = CacheEntry(
            key="dead",
            value="nope",
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        vals = cache.values()
        assert "ok" in vals
        assert "nope" not in vals

    def test_items_excludes_expired(self):
        cache = StrategyCache()
        cache.set("live", "val")
        cache._entries["gone"] = CacheEntry(
            key="gone",
            value="x",
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        pairs = dict(cache.items())
        assert "live" in pairs
        assert "gone" not in pairs

    def test_lru_move_to_end_on_get(self):
        config = CacheConfig(strategy=CacheStrategy.LRU, max_size=10, ttl_seconds=None)
        cache = StrategyCache(config)
        cache.set("a", 1)
        cache.set("b", 2)
        r = cache.get("a")
        assert r.hit is True

    def test_get_miss_returns_not_hit(self):
        cache = StrategyCache()
        result = cache.get("absent")
        assert result.hit is False

    def test_set_with_explicit_ttl_seconds(self):
        cache = StrategyCache()
        entry = cache.set("k", "v", ttl_seconds=3600)
        assert entry.expires_at is not None


# =============================================================================
# PromptCache
# =============================================================================


class TestPromptCache:
    def test_cache_response_and_get_response_hit(self):
        cache = PromptCache()
        cache.cache_response("What is AI?", "AI is...", model="gpt-4")
        result = cache.get_response("What is AI?", model="gpt-4")
        assert result.hit is True
        assert result.value == "AI is..."

    def test_get_response_miss(self):
        cache = PromptCache()
        result = cache.get_response("Unknown prompt")
        assert result.hit is False

    def test_get_by_prompt_hit(self):
        cache = PromptCache()
        cache.cache_response("Hello!", "Hi there!")
        result = cache.get_by_prompt("Hello!")
        assert result.hit is True

    def test_get_by_prompt_miss(self):
        cache = PromptCache()
        result = cache.get_by_prompt("not cached")
        assert result.hit is False
        assert result.key == ""

    def test_find_similar_returns_matching_prompts(self):
        cache = PromptCache(similarity_threshold=0.5)
        cache.cache_response("machine learning basics", "ML is a subset of AI")
        matches = cache.find_similar("machine learning")
        assert len(matches) >= 1
        prompt, entry, score = matches[0]
        assert "machine learning" in prompt.lower()
        assert score >= 0.5

    def test_find_similar_respects_limit(self):
        cache = PromptCache(similarity_threshold=0.0)
        for i in range(10):
            cache.cache_response(f"prompt {i}", f"response {i}")
        matches = cache.find_similar("prompt", limit=3)
        assert len(matches) <= 3

    def test_cache_response_with_params_and_metadata(self):
        cache = PromptCache()
        cache.cache_response(
            "test prompt",
            "test response",
            model="claude-3",
            params={"temperature": 0.7},
            metadata={"custom": "value"},
        )
        result = cache.get_response("test prompt", model="claude-3", params={"temperature": 0.7})
        assert result.hit is True


# =============================================================================
# CachedModel
# =============================================================================


class TestCachedModel:
    def _make_mock_model(self, response_text: str = "generated"):
        mock = MagicMock()
        mock.model_id = "test-model"
        mock.generate.return_value = ModelResponse(
            content=response_text,
            model="test-model",
            latency_ms=100.0,
        )
        return mock

    def test_generate_cache_miss_then_hit(self):
        model = self._make_mock_model("hello")
        cache = StrategyCache()
        wrapped = CachedModel(model, cache)

        r1 = wrapped.generate("p")
        assert r1.content == "hello"
        assert model.generate.call_count == 1

        r2 = wrapped.generate("p")
        assert r2.content == "hello"
        assert model.generate.call_count == 1  # cached

    def test_model_property(self):
        model = self._make_mock_model()
        wrapped = CachedModel(model)
        assert wrapped.model is model

    def test_cache_property(self):
        model = self._make_mock_model()
        cache = StrategyCache()
        wrapped = CachedModel(model, cache)
        assert wrapped.cache is cache

    def test_getattr_delegates(self):
        model = self._make_mock_model()
        model.provider = "openai"
        wrapped = CachedModel(model)
        assert wrapped.provider == "openai"

    def test_non_deterministic_skips_cache(self):
        model = self._make_mock_model("out")
        wrapped = CachedModel(model, cache_only_deterministic=True)
        wrapped.generate("p", temperature=1.0)
        wrapped.generate("p", temperature=1.0)
        # Both calls should reach the model (not cached)
        assert model.generate.call_count == 2

    def test_cached_model_with_inmemory_cache(self):
        model = self._make_mock_model("resp")
        cache = InMemoryCache()
        wrapped = CachedModel(model, cache)
        r = wrapped.generate("prompt", temperature=0.0)
        assert r.content == "resp"


# =============================================================================
# CacheWarmer
# =============================================================================


class TestCacheWarmer:
    def _make_generator(self, prefix: str = "response"):
        def gen(prompt: str) -> str:
            return f"{prefix}: {prompt}"

        return gen

    def test_add_prompt_and_warm_success(self):
        cache = StrategyCache()
        warmer = CacheWarmer(cache, self._make_generator())
        warmer.add_prompt("hello")
        warmer.add_prompt("world")
        results = warmer.warm(batch_size=10, skip_existing=False)
        assert len(results) == 2
        assert all(r["status"] == "success" for r in results)

    def test_warm_skips_existing(self):
        cache = StrategyCache()
        warmer = CacheWarmer(cache, self._make_generator())
        warmer.add_prompt("cached")
        warmer.warm(batch_size=10, skip_existing=False)  # populate
        warmer.add_prompt("cached")
        results = warmer.warm(batch_size=10, skip_existing=True)
        assert results[0]["status"] == "skipped"

    def test_warm_handles_generator_error(self):
        cache = StrategyCache()

        def broken(prompt: str) -> str:
            raise RuntimeError("fail")

        warmer = CacheWarmer(cache, broken)
        warmer.add_prompt("p")
        results = warmer.warm(skip_existing=False)
        assert results[0]["status"] == "error"
        assert "fail" in results[0]["error"]

    def test_warm_no_generator_raises(self):
        cache = StrategyCache()
        warmer = CacheWarmer(cache)
        warmer.add_prompt("p")
        with pytest.raises(ValueError, match="Generator"):
            warmer.warm()

    def test_priority_ordering(self):
        cache = StrategyCache()
        results_log = []

        def gen(prompt: str) -> str:
            results_log.append(prompt)
            return "ok"

        warmer = CacheWarmer(cache, gen)
        warmer.add_prompt("low", priority=1)
        warmer.add_prompt("high", priority=10)
        warmer.add_prompt("mid", priority=5)
        warmer.warm(batch_size=3, skip_existing=False)
        assert results_log[0] == "high"

    def test_get_queue_size_and_clear_queue(self):
        cache = StrategyCache()
        warmer = CacheWarmer(cache, self._make_generator())
        warmer.add_prompt("a")
        warmer.add_prompt("b")
        assert warmer.get_queue_size() == 2
        warmer.clear_queue()
        assert warmer.get_queue_size() == 0

    def test_get_results_after_warm(self):
        cache = StrategyCache()
        warmer = CacheWarmer(cache, self._make_generator())
        warmer.add_prompt("p")
        warmer.warm(skip_existing=False)
        results = warmer.get_results()
        assert len(results) == 1
        assert results[0]["status"] == "success"


# =============================================================================
# MemoizedFunction / memoize
# =============================================================================


class TestMemoizedFunction:
    def test_basic_memoization(self):
        calls = []

        def fn(x: int) -> int:
            calls.append(x)
            return x * 2

        memo = MemoizedFunction(fn)
        assert memo(5) == 10
        assert memo(5) == 10  # cache hit
        assert len(calls) == 1

    def test_different_args_different_cache_entry(self):
        def fn(x: int) -> int:
            return x + 1

        memo = MemoizedFunction(fn)
        assert memo(1) == 2
        assert memo(2) == 3

    def test_invalidate_forces_recompute(self):
        calls = []

        def fn(x: int) -> int:
            calls.append(x)
            return x

        memo = MemoizedFunction(fn)
        memo(7)
        memo.invalidate(7)
        memo(7)
        assert len(calls) == 2

    def test_get_stats(self):
        def fn(x: int) -> int:
            return x

        memo = MemoizedFunction(fn)
        memo(1)
        memo(1)  # hit
        stats = memo.get_stats()
        assert stats["call_count"] == 2
        assert stats["cache_calls"] == 1
        assert "cache_stats" in stats

    def test_custom_key_generator(self):
        def fn(x: int) -> int:
            return x

        def key_gen(*args, **kwargs) -> str:
            return f"custom:{args[0]}"

        memo = MemoizedFunction(fn, key_generator=key_gen)
        assert memo(3) == 3
        assert memo(3) == 3

    def test_memoize_decorator_no_parens(self):
        @memoize
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3

    def test_memoize_decorator_with_parens(self):
        @memoize(max_size=50, ttl_seconds=60)
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(3, 4) == 12
        assert multiply(3, 4) == 12


# =============================================================================
# CacheNamespace
# =============================================================================


class TestCacheNamespace:
    def test_get_cache_creates_and_reuses(self):
        ns = CacheNamespace()
        c1 = ns.get_cache("users")
        c2 = ns.get_cache("users")
        assert c1 is c2

    def test_get_prompt_cache(self):
        ns = CacheNamespace()
        pc = ns.get_prompt_cache("prompts")
        assert isinstance(pc, PromptCache)
        pc2 = ns.get_prompt_cache("prompts")
        assert pc is pc2

    def test_get_prompt_cache_replaces_non_prompt_cache(self):
        ns = CacheNamespace()
        _ = ns.get_cache("mixed")  # regular StrategyCache
        pc = ns.get_prompt_cache("mixed")  # should replace
        assert isinstance(pc, PromptCache)

    def test_delete_cache_returns_true_and_false(self):
        ns = CacheNamespace()
        ns.get_cache("temp")
        assert ns.delete_cache("temp") is True
        assert ns.delete_cache("temp") is False

    def test_list_caches(self):
        ns = CacheNamespace()
        ns.get_cache("a")
        ns.get_cache("b")
        names = ns.list_caches()
        assert "a" in names
        assert "b" in names

    def test_get_all_stats(self):
        ns = CacheNamespace()
        c = ns.get_cache("stats_test")
        c.set("k", "v")
        all_stats = ns.get_all_stats()
        assert "stats_test" in all_stats
        assert all_stats["stats_test"].entry_count >= 1

    def test_clear_all(self):
        ns = CacheNamespace()
        c = ns.get_cache("clear_me")
        c.set("item", "value")
        ns.clear_all()
        result = c.get("item")
        assert result.hit is False


# =============================================================================
# ResponseDeduplicator
# =============================================================================


class TestResponseDeduplicator:
    def test_add_unique_response(self):
        dedup = ResponseDeduplicator()
        is_dup, idx = dedup.add("p1", "Response A")
        assert is_dup is False
        assert idx is None

    def test_add_exact_duplicate(self):
        dedup = ResponseDeduplicator()
        dedup.add("p1", "Response A")
        is_dup, idx = dedup.add("p2", "Response A")
        assert is_dup is True
        assert idx == 0

    def test_similarity_based_duplicate(self):
        dedup = ResponseDeduplicator(similarity_threshold=0.6)
        dedup.add("p1", "the quick brown fox jumps")
        is_dup, _ = dedup.add("p2", "the quick brown fox leaps")
        # Both share 4/6 words = 0.667 Jaccard, above threshold 0.6
        assert is_dup is True

    def test_similarity_based_not_duplicate(self):
        dedup = ResponseDeduplicator(similarity_threshold=0.9)
        dedup.add("p1", "completely different text here")
        is_dup, _ = dedup.add("p2", "nothing in common at all")
        assert is_dup is False

    def test_empty_response_not_duplicate(self):
        dedup = ResponseDeduplicator(similarity_threshold=0.5)
        dedup.add("p1", "some text")
        is_dup, _ = dedup.add("p2", "")
        assert is_dup is False

    def test_get_unique_responses(self):
        dedup = ResponseDeduplicator()
        dedup.add("p1", "A")
        dedup.add("p2", "B")
        dedup.add("p3", "A")  # duplicate
        unique = dedup.get_unique_responses()
        assert len(unique) == 2

    def test_get_duplicate_count(self):
        dedup = ResponseDeduplicator()
        dedup.add("p1", "X")
        dedup.add("p2", "X")
        dedup.add("p3", "X")
        assert dedup.get_duplicate_count() == 2

    def test_clear_resets_state(self):
        dedup = ResponseDeduplicator()
        dedup.add("p1", "text")
        dedup.clear()
        unique = dedup.get_unique_responses()
        assert unique == []
        assert dedup.get_duplicate_count() == 0


# =============================================================================
# AsyncCacheAdapter
# =============================================================================


class TestAsyncCacheAdapter:
    def test_async_set_and_get(self):
        cache = StrategyCache()
        adapter = AsyncCacheAdapter(cache)

        async def run():
            await adapter.set("k", "v")
            result = await adapter.get("k")
            assert result.hit is True
            assert result.value == "v"

        asyncio.run(run())

    def test_async_delete(self):
        cache = StrategyCache()
        adapter = AsyncCacheAdapter(cache)

        async def run():
            await adapter.set("del_k", "v")
            deleted = await adapter.delete("del_k")
            assert deleted is True
            result = await adapter.get("del_k")
            assert result.hit is False

        asyncio.run(run())

    def test_async_clear(self):
        cache = StrategyCache()
        adapter = AsyncCacheAdapter(cache)

        async def run():
            await adapter.set("a", 1)
            await adapter.clear()
            result = await adapter.get("a")
            assert result.hit is False

        asyncio.run(run())


# =============================================================================
# Convenience functions
# =============================================================================


def test_create_cache_default():
    cache = create_cache()
    cache.set("k", "v")
    result = cache.get("k")
    assert result.hit is True


def test_create_cache_lfu():
    cache = create_cache(strategy=CacheStrategy.LFU, max_size=10)
    cache.set("x", 1)
    assert cache.get("x").hit is True


def test_create_prompt_cache():
    cache = create_prompt_cache(similarity_threshold=0.8)
    cache.cache_response("hello", "world")
    result = cache.get_response("hello")
    assert result.hit is True


def test_create_cache_warmer():
    cache = create_cache()
    warmer = create_cache_warmer(cache, lambda p: f"r:{p}")
    warmer.add_prompt("test")
    results = warmer.warm(skip_existing=False)
    assert results[0]["status"] == "success"


def test_create_namespace():
    ns = create_namespace(default_max_size=50, default_ttl_seconds=60)
    c = ns.get_cache("n1")
    c.set("k", "v")
    assert c.get("k").hit is True


def test_get_cache_key():
    key = get_cache_key("hello")
    assert len(key) == 64

    key2 = get_cache_key("hello", model="gpt-4", params={"t": 0})
    assert key != key2


def test_cached_response_miss_then_hit():
    calls = []

    def gen(prompt: str) -> str:
        calls.append(prompt)
        return f"response to {prompt}"

    cache = create_prompt_cache()
    r1, was_cached1 = cached_response("test", gen, cache=cache)
    assert was_cached1 is False
    assert r1 == "response to test"
    assert len(calls) == 1

    r2, was_cached2 = cached_response("test", gen, cache=cache)
    assert was_cached2 is True
    assert r2 == "response to test"
    assert len(calls) == 1  # no extra call


def test_cached_response_creates_cache_if_none():
    r, was_cached = cached_response("q", lambda p: "ans")
    assert r == "ans"
    assert was_cached is False


def test_cached_decorator():
    calls = []

    @cached()
    def compute(x: int) -> int:
        calls.append(x)
        return x * 3

    result1 = compute(7)
    result2 = compute(7)
    assert result1 == 21
    assert result2 == 21
    assert len(calls) == 1


def test_cached_decorator_with_key_fn():
    @cached(key_fn=lambda x: f"key:{x}")
    def fn(x: int) -> int:
        return x + 1

    assert fn(4) == 5
    assert fn(4) == 5


# =============================================================================
# Global default cache
# =============================================================================


def test_get_set_clear_default_cache():
    # Reset state from any prior test
    import insideLLMs.caching as caching_module

    caching_module._default_cache = None

    c1 = get_default_cache()
    assert c1 is not None

    c2 = get_default_cache()
    assert c1 is c2  # same singleton

    custom = InMemoryCache(max_size=5)
    set_default_cache(custom)
    assert get_default_cache() is custom

    # Set something in the custom cache, then clear
    custom.set("test", "val")
    clear_default_cache()
    assert custom.get("test") is None

    # Reset for other tests
    caching_module._default_cache = None
