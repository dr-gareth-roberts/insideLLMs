"""Tests for insideLLMs.caching module.

This module provides test coverage for the unified caching infrastructure.
"""

import pytest

from insideLLMs.caching import (
    CacheConfig,
    CacheEntry,
    CacheLookupResult,
    CacheScope,
    CacheStats,
    CacheStatus,
    CacheStrategy,
    DiskCache,
    InMemoryCache,
    PromptCache,
    StrategyCache,
)


class TestCacheStrategy:
    """Tests for CacheStrategy enum."""

    def test_strategy_values_exist(self):
        assert hasattr(CacheStrategy, "LRU")
        assert hasattr(CacheStrategy, "LFU")
        assert hasattr(CacheStrategy, "TTL")

    def test_strategy_is_enum(self):
        assert isinstance(CacheStrategy.LRU, CacheStrategy)


class TestCacheStatus:
    """Tests for CacheStatus enum."""

    def test_status_values_exist(self):
        assert hasattr(CacheStatus, "HIT")
        assert hasattr(CacheStatus, "MISS")

    def test_status_is_enum(self):
        assert isinstance(CacheStatus.HIT, CacheStatus)


class TestCacheScope:
    """Tests for CacheScope enum."""

    def test_scope_values_exist(self):
        assert hasattr(CacheScope, "GLOBAL")
        assert hasattr(CacheScope, "SESSION")

    def test_scope_is_enum(self):
        assert isinstance(CacheScope.GLOBAL, CacheScope)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_instantiation(self):
        config = CacheConfig()
        assert config is not None

    def test_custom_max_size(self):
        config = CacheConfig(max_size=500)
        assert config.max_size == 500


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        entry = CacheEntry(key="test_key", value="test_value")
        assert entry.key == "test_key"
        assert entry.value == "test_value"


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_stats_creation(self):
        stats = CacheStats()
        assert stats is not None
        assert stats.hits == 0
        assert stats.misses == 0

    def test_hit_rate_zero_requests(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0


class TestCacheLookupResult:
    """Tests for CacheLookupResult dataclass."""

    def test_hit_result(self):
        result = CacheLookupResult(status=CacheStatus.HIT, value="cached_value")
        assert result.status == CacheStatus.HIT
        assert result.value == "cached_value"

    def test_miss_result(self):
        result = CacheLookupResult(status=CacheStatus.MISS, value=None)
        assert result.status == CacheStatus.MISS
        assert result.value is None


class TestInMemoryCache:
    """Tests for InMemoryCache class."""

    def test_instantiation(self):
        cache = InMemoryCache[str]()
        assert cache is not None

    def test_set_and_get(self):
        cache = InMemoryCache[str]()
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_get_miss(self):
        cache = InMemoryCache[str]()
        result = cache.get("nonexistent")
        assert result is None

    def test_clear(self):
        cache = InMemoryCache[str]()
        cache.set("key1", "value1")
        cache.clear()
        assert cache.get("key1") is None


class TestStrategyCache:
    """Tests for StrategyCache class."""

    def test_instantiation_default(self):
        cache = StrategyCache()
        assert cache is not None

    def test_instantiation_with_strategy(self):
        cache = StrategyCache(strategy=CacheStrategy.LRU, max_size=100)
        assert cache is not None


class TestPromptCache:
    """Tests for PromptCache class."""

    def test_instantiation(self):
        cache = PromptCache()
        assert cache is not None
