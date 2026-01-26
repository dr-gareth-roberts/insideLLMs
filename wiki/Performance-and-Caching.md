---
title: Performance and Caching
nav_order: 26
---

This page collects the knobs that affect throughput, latency, cost, and determinism. It covers the
middleware pipeline, caching modules, async runners, and rate limiting utilities.

## Quick map of knobs (where to set them)

- CLI: `insidellms run --async --concurrency N` (defaults in `insideLLMs/cli.py`)
- Async runners: `RunConfig.concurrency` (default 5 in `insideLLMs/config_types.py`)
- Config YAML: `RunnerConfig.concurrency` (default 1 in `insideLLMs/config.py`)
- Pipeline middleware: `CacheMiddleware`, `RateLimitMiddleware`, `RetryMiddleware`
  (`insideLLMs/runtime/pipeline.py`)
- Adapter settings: `AdapterConfig.timeout`, `max_retries`, `retry_delay`, `rate_limit_rpm/tpm`
  (`insideLLMs/adapters.py`)
- Advanced rate limiting: `insideLLMs/rate_limiting.py` (token buckets, retry handler, circuit
  breaker, concurrency limiter)

## Caching options

### 1) Pipeline cache (in-memory, per pipeline instance)

- **Class:** `CacheMiddleware` (`insideLLMs/runtime/pipeline.py`)
- **What it caches:** `generate()` / `agenerate()` only (not chat/stream).
- **Key:** SHA-256 of JSON-serialized `{prompt, **kwargs}` (`sort_keys=True`).
- **Eviction:** oldest entries removed when `cache_size` exceeded (insertion order); optional
  `ttl_seconds` expiration (checked on access).
- **Scope:** per pipeline instance; not shared across processes; async uses `asyncio.Lock`.
- **Gotcha:** kwargs must be JSON-serializable or key generation will raise.

Example:

```python
from insideLLMs.runtime.pipeline import CacheMiddleware, ModelPipeline

pipeline = ModelPipeline(model, middlewares=[CacheMiddleware(cache_size=1000, ttl_seconds=3600)])
```

### 2) ModelWrapper cache_responses (in-memory)

- **Class:** `ModelWrapper` (`insideLLMs/models/base.py`)
- **What it caches:** `generate()` only.
- **Key:** `f"{prompt}:{sorted(kwargs.items())}"` (simple string key).
- **Scope:** per wrapper instance; no TTL.

### 3) Unified caching module (general + LLM-specific)

- **Module:** `insideLLMs.caching` (re-exports `caching_unified.py`)
- **Backends:** `InMemoryCache`, `DiskCache` (SQLite persistence), `StrategyCache`
  (LRU/LFU/FIFO/TTL/SIZE).
- **LLM helpers:** `PromptCache`, `CachedModel`, cache warming + stats.

### 4) Semantic cache (exact + similarity; Redis optional)

- **Module:** `insideLLMs/semantic_cache.py`
- **Capabilities:** semantic similarity lookup with embeddings + exact match; optional Redis backend
  for distributed caching.
- **Model wrapper:** `SemanticCacheModel` caches `generate` and `chat`.
- **Determinism guard:** `cache_temperature_zero_only=True` by default (only temp=0 responses cached).

### 5) Routing caches (for model routing)

- **Module:** `insideLLMs/routing.py`
- **Config:** `RouterConfig.cache_routes` + `cache_ttl_seconds` caches route decisions; route
  description embeddings cached in matcher.

## In-memory vs persistent/Redis

- **In-memory:** `CacheMiddleware`, `ModelWrapper`, `InMemoryCache`, `PromptCache` (fast, per process).
- **Persistent (local disk):** `DiskCache` (SQLite) from `caching_unified.py` survives restarts.
- **Distributed (Redis):** `SemanticCache` can use Redis (`redis` optional dep). Useful for
  multi-process / multi-host caching.

## Determinism and caching

- Cache keys are deterministic, but cached values reflect the first response seen for that
  prompt+params.
- For reproducible runs:
  - Set `temperature=0`.
  - Use `SemanticCacheModel` with `cache_temperature_zero_only=True` (default).
  - Prefer exact-match caching over semantic matching when determinism matters.
  - Use `concurrency=1` for strict reproducibility.
  - Treat persistent caches as part of the experiment state (clear between runs if needed).

## Async + concurrency (`--async` / `--concurrency`)

- CLI: `insidellms run --async --concurrency 5` (default 5).
- Async runner: `AsyncProbeRunner.run(..., concurrency=N)` uses an `asyncio.Semaphore` and writes
  results in input order.

Safe usage tips:

- Start low (1–5) and increase gradually; watch for 429s/timeouts.
- Combine concurrency with rate limiting to avoid bursts.
- If the base model is sync-only, async paths run in a thread pool; very high concurrency can
  exhaust threads.

## Rate limiting, retries, timeouts

### Pipeline middleware

- **RateLimitMiddleware**: token bucket in requests per minute (default 60). Sleeps when limited.
- **RetryMiddleware**: backoff + jitter; retries common transient errors (default `max_retries=3`).

Ordering matters (middleware list is request order):

- Put `CacheMiddleware` early so hits bypass rate limiting and retries.
- If you want retries to re-enter the rate limiter, place `RetryMiddleware` before
  `RateLimitMiddleware`.

### Adapter-level timeouts / retry fields

- `AdapterConfig.timeout` is passed to provider clients (e.g., OpenAI/Anthropic).
- `max_retries`, `retry_delay`, `rate_limit_rpm/tpm` are configuration fields only today (not
  enforced in adapter code); use middleware or `insideLLMs/rate_limiting.py` for actual behaviour.

## Provider rate limits (and starting values)

insideLLMs does not hard-code provider limits; use your provider's published RPM/TPM quotas.

Practical starting points (conservative):

- `--concurrency`: 1–5
- `RateLimitMiddleware(requests_per_minute=30–60, burst_size=...)`
- Increase slowly while monitoring 429s/timeouts and latency.

If your provider publishes TPM limits, add a token-based limiter and/or reduce `max_tokens`
per request.

## Cost control tips

- Turn on caching for repeated prompts (CacheMiddleware, CachedModel, SemanticCacheModel).
- Keep `temperature=0` where possible to maximise cache hits and determinism.
- Lower `max_tokens` and add stop sequences to cap output length.
- Use smaller models for development or smoke tests.
- Use `--async` to reduce wall-clock time, but keep concurrency within rate limits to avoid retries.
- Use `resume` for long runs so you don't re-pay for completed items.
