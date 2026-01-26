---
title: Caching
parent: Guides
nav_order: 1
---

# Caching

Speed up runs and reduce costs with response caching.

## Why Cache?

- **Cost reduction**: Avoid duplicate API calls
- **Faster iteration**: Skip already-computed results
- **Development**: Test without hitting rate limits
- **Reproducibility**: Get consistent responses for same inputs

## Enabling Caching

### In Config

```yaml
cache:
  enabled: true
  backend: sqlite
  path: .cache/insidellms.db
```

### Programmatically

```python
from insideLLMs.caching_unified import CacheConfig, UnifiedCache

cache = UnifiedCache(CacheConfig(
    enabled=True,
    backend="sqlite",
    path=".cache/insidellms.db"
))
```

## Cache Backends

| Backend | Persistence | Speed | Use Case |
|---------|-------------|-------|----------|
| `memory` | Session only | Fastest | Testing |
| `sqlite` | Disk | Fast | Development |
| `redis` | Network | Medium | Team sharing |

### Memory Cache

```yaml
cache:
  enabled: true
  backend: memory
```

Session-only; cleared when process exits.

### SQLite Cache

```yaml
cache:
  enabled: true
  backend: sqlite
  path: .cache/responses.db
```

Persistent; survives restarts.

### Redis Cache

```yaml
cache:
  enabled: true
  backend: redis
  url: redis://localhost:6379/0
```

Shared across machines; requires Redis server.

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

```bash
rm -rf .cache/insidellms.db
```

### Clear Programmatically

```python
cache.clear()
```

### TTL-based Expiration

```yaml
cache:
  enabled: true
  backend: sqlite
  ttl_seconds: 86400  # 24 hours
```

## When to Disable Caching

- **Evaluating model updates**: Need fresh responses
- **Testing randomness**: Want different responses each time
- **Production benchmarks**: Measure actual latency

```yaml
cache:
  enabled: false
```

Or via CLI:

```bash
insidellms run config.yaml --no-cache
```

## Cache Statistics

```python
stats = cache.stats()
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

## Best Practices

### Do

- ✅ Enable caching during development
- ✅ Use SQLite for persistence
- ✅ Set appropriate TTL for time-sensitive data
- ✅ Clear cache when changing model behavior

### Don't

- ❌ Cache in production benchmarks
- ❌ Share cache between different model versions
- ❌ Forget to invalidate after model updates

## Determinism Note

Caching can affect determinism:
- With cache: Faster, but responses depend on cache state
- Without cache: Slower, but fresh responses each time

For CI diff-gating with `DummyModel`, caching doesn't matter (responses are fixed).

## See Also

- [Performance and Caching](../Performance-and-Caching.md) - More performance options
- [Rate Limiting](Rate-Limiting.md) - Combine with caching
