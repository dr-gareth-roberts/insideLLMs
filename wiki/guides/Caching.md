---
title: Caching
parent: Guides
nav_order: 1
---

# Caching

**Reduce costs. Speed up iteration.**

## Enable

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o-mini
  pipeline:
    middlewares:
      - type: cache
        args:
          cache_size: 1000
          ttl_seconds: 86400
```

## Cache Approaches

| Approach | Persistence | Use Case |
|----------|-------------|----------|
| `CacheMiddleware` in `model.pipeline` | Process memory | CLI/YAML runs |
| `InMemoryCache` + `CachedModel` | Process memory | Python scripts/tests |
| `DiskCache` + `CachedModel` | SQLite on disk | Persistent local caching |

### In-Memory Cache (YAML/CLI)

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o-mini
  pipeline:
    middlewares:
      - type: cache
        args:
          cache_size: 1000
```

Session-only; cleared when process exits.

### Programmatic In-Memory Cache

```python
from insideLLMs.caching import CachedModel, InMemoryCache

cache = InMemoryCache(max_size=1000, default_ttl=3600)
cached_model = CachedModel(base_model, cache=cache)

response = cached_model.generate("Hello")
```

### Programmatic Persistent Disk Cache

```python
from pathlib import Path

from insideLLMs.caching import CachedModel, DiskCache

cache = DiskCache(path=Path(".cache/responses.db"), max_size_mb=200, default_ttl=86400)
cached_model = CachedModel(base_model, cache=cache)
```

Persists across process restarts.

## Cache Key Generation

Cache keys include:
- Model identifier
- Prompt/messages content
- Generation parameters (temperature, max_tokens, etc.)

```
cache_key = hash(model_id + prompt + sorted(kwargs))
```

This means:
- Same prompt + same params = cache hit
- Different temperature = cache miss

## Cache Invalidation

### Clear All

```python
cache.clear()
```

### Clear Programmatically

```python
cache.clear()
```

### TTL-based Expiration

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o-mini
  pipeline:
    middlewares:
      - type: cache
        args:
          cache_size: 1000
          ttl_seconds: 86400  # 24 hours
```

## When to Disable Caching

- **Evaluating model updates**: Need fresh responses
- **Testing randomness**: Want different responses each time
- **Production benchmarks**: Measure actual latency

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o-mini
```

Or programmatically, use the uncached base model directly instead of `CachedModel`.

## Cache Statistics

```python
# CacheMiddleware stats (pipeline usage)
print(f"Hits: {cache_middleware.hits}")
print(f"Misses: {cache_middleware.misses}")
print(f"Hit rate: {cache_middleware.hit_rate:.1%}")

# InMemoryCache / DiskCache stats (programmatic usage)
stats = cache.stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
```

## Best Practices

### Do

-  Enable caching during development
-  Use SQLite for persistence
-  Set appropriate TTL for time-sensitive data
-  Clear cache when changing model behaviour

### Don't

-  Cache in production benchmarks
-  Share cache between different model versions
-  Forget to invalidate after model updates

## Determinism Note

Caching can affect determinism:
- With cache: Faster, but responses depend on cache state
- Without cache: Slower, but fresh responses each time

For CI diff-gating with `DummyModel`, caching doesn't matter (responses are fixed).

## See Also

- [Performance and Caching](../Performance-and-Caching.md) - More performance options
- [Rate Limiting](Rate-Limiting.md) - Combine with caching
