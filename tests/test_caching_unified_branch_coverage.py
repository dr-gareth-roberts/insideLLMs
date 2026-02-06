"""Additional branch coverage for caching_unified."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from insideLLMs.caching_unified import (
    CacheConfig,
    CacheEntry,
    CacheLookupResult,
    CacheStatus,
    CacheStrategy,
    CacheWarmer,
    DiskCache,
    InMemoryCache,
    PromptCache,
    ResponseDeduplicator,
    StrategyCache,
    generate_cache_key,
)


def test_cache_entry_size_estimation_falls_back_for_unserializable_value():
    entry = CacheEntry(key="k", value=object())
    assert entry.size_bytes > 0


def test_cache_lookup_result_validation_paths():
    with pytest.raises(TypeError, match="requires either 'hit' or 'status'"):
        CacheLookupResult()

    with pytest.raises(ValueError, match="hit/status mismatch"):
        CacheLookupResult(hit=True, status=CacheStatus.MISS, value="x")


def test_generate_cache_key_sha1_branch():
    key = generate_cache_key("prompt", algorithm="sha1")
    assert len(key) == 40


def test_inmemory_cache_evict_lru_noop_when_empty():
    cache = InMemoryCache[str]()
    cache._evict_lru()
    assert cache.keys() == []


def test_strategy_cache_contains_removes_expired_and_empty_evict():
    cache = StrategyCache()
    cache._entries["expired"] = CacheEntry(
        key="expired",
        value="x",
        expires_at=datetime.now() - timedelta(seconds=1),
    )
    assert cache.contains("expired") is False
    assert "expired" not in cache._entries

    cache.clear()
    cache._evict_one()
    assert cache.keys() == []


def test_strategy_cache_ttl_evict_with_and_without_expiration():
    ttl_cache = StrategyCache(config=CacheConfig(strategy=CacheStrategy.TTL, max_size=10))
    ttl_cache._entries["a"] = CacheEntry(
        key="a",
        value="x",
        expires_at=datetime.now() + timedelta(seconds=10),
    )
    ttl_cache._entries["b"] = CacheEntry(
        key="b",
        value="y",
        expires_at=datetime.now() + timedelta(seconds=20),
    )
    ttl_cache._evict_one()
    assert "a" not in ttl_cache._entries

    no_expiry_cache = StrategyCache(config=CacheConfig(strategy=CacheStrategy.TTL, max_size=10))
    no_expiry_cache._entries["x"] = CacheEntry(key="x", value="1")
    no_expiry_cache._entries["y"] = CacheEntry(key="y", value="2")
    no_expiry_cache._evict_one()
    assert len(no_expiry_cache._entries) == 1


def test_strategy_cache_size_evicts_largest_entry():
    cache = StrategyCache(config=CacheConfig(strategy=CacheStrategy.SIZE, max_size=10))
    cache._entries["small"] = CacheEntry(key="small", value="1", size_bytes=1)
    cache._entries["big"] = CacheEntry(key="big", value="2", size_bytes=100)
    cache._evict_one()
    assert "big" not in cache._entries
    assert cache.get_stats().evictions >= 1


def test_prompt_cache_prompt_lookup_miss_and_skip_expired_in_similarity_search():
    cache = PromptCache(similarity_threshold=0.1)
    miss = cache.get_by_prompt("missing")
    assert not miss.hit
    assert miss.key == ""

    cache._entries["expired"] = CacheEntry(
        key="expired",
        value="cached",
        expires_at=datetime.now() - timedelta(seconds=1),
        metadata={"prompt": "hello world"},
    )
    similar = cache.find_similar("hello")
    assert similar == []


def test_cache_warmer_captures_generation_error():
    cache = StrategyCache()

    def broken_generator(prompt: str):
        raise RuntimeError(f"cannot generate for {prompt}")

    warmer = CacheWarmer(cache, broken_generator)
    warmer.add_prompt("hello")
    results = warmer.warm(skip_existing=False)

    assert results[0]["status"] == "error"
    assert "cannot generate" in results[0]["error"]


def test_response_deduplicator_non_exact_with_empty_words_is_not_duplicate():
    dedup = ResponseDeduplicator(similarity_threshold=0.5)
    assert dedup._is_duplicate("", "non-empty response") is False


def test_disk_cache_default_path_branch_uses_home_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = DiskCache(path=None, max_size_mb=1)
    # Verify default location under ~/.cache/insideLLMs.
    assert "insideLLMs" in str(cache._path)
