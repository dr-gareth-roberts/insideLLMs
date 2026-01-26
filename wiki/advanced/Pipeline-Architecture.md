---
title: Pipeline Architecture
parent: Advanced Features
nav_order: 1
---

# Pipeline Architecture

**Composable middleware for production-grade model interactions.**

## The Problem

You need caching. And rate limiting. And retry logic. And cost tracking. And tracing.

Traditional approach: Modify every model call site. Fragile. Error-prone.

## The Solution

Pipeline middleware. Wrap your model once. Get all cross-cutting concerns.

```python
from insideLLMs.pipeline import ModelPipeline, CacheMiddleware, RetryMiddleware, CostTrackingMiddleware

pipeline = ModelPipeline(model)
pipeline.add_middleware(CacheMiddleware())
pipeline.add_middleware(RetryMiddleware(max_attempts=3))
pipeline.add_middleware(CostTrackingMiddleware())

# Use like a normal model
response = pipeline.generate("prompt")
# Automatically cached, retried on failure, cost tracked
```

## Available Middleware

| Middleware | Purpose |
|------------|---------|
| `CacheMiddleware` | Response caching with TTL and LRU eviction |
| `RateLimitMiddleware` | Token bucket rate limiting |
| `RetryMiddleware` | Exponential backoff retry logic |
| `CostTrackingMiddleware` | Token usage and cost estimation |
| `TraceMiddleware` | Execution tracing and recording |
| `PassthroughMiddleware` | Base for observation-only middleware |

## Execution Order

Middleware executes in order for requests, reverse for responses:

```
Request:  Cache → RateLimit → Retry → Model
Response: Model → Retry → RateLimit → Cache
```

This ensures:
- Cache checked first (skip if hit)
- Rate limit enforced before retry
- Retry happens before cache write

## Complete Example

```python
from insideLLMs.models import OpenAIModel
from insideLLMs.pipeline import (
    ModelPipeline,
    CacheMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    CostTrackingMiddleware
)

# Create base model
model = OpenAIModel(model_name="gpt-4o")

# Wrap with pipeline
pipeline = ModelPipeline(model)

# Add middleware in order
pipeline.add_middleware(CacheMiddleware(
    backend="sqlite",
    path=".cache/responses.db",
    ttl_seconds=3600
))

pipeline.add_middleware(RateLimitMiddleware(
    requests_per_minute=60,
    tokens_per_minute=90000
))

pipeline.add_middleware(RetryMiddleware(
    max_attempts=3,
    initial_delay=1.0,
    exponential_base=2.0
))

pipeline.add_middleware(CostTrackingMiddleware())

# Use like normal model
response = pipeline.generate("What is quantum computing?")

# Check cost
print(f"Cost: ${pipeline.get_total_cost():.4f}")
```

## Async Support

```python
from insideLLMs.pipeline import AsyncModelPipeline

pipeline = AsyncModelPipeline(model)
pipeline.add_middleware(CacheMiddleware())
pipeline.add_middleware(RetryMiddleware())

# Concurrent execution
responses = await pipeline.abatch_generate(prompts, concurrency=10)
```

## Custom Middleware

```python
from insideLLMs.pipeline import Middleware

class LoggingMiddleware(Middleware):
    def process_request(self, prompt, **kwargs):
        print(f"Request: {prompt[:50]}...")
        return prompt, kwargs
    
    def process_response(self, response, **kwargs):
        print(f"Response: {response[:50]}...")
        return response

pipeline.add_middleware(LoggingMiddleware())
```

## Configuration

```yaml
# In harness config
pipeline:
  middleware:
    - type: cache
      args:
        backend: sqlite
        ttl_seconds: 3600
    - type: rate_limit
      args:
        requests_per_minute: 60
    - type: retry
      args:
        max_attempts: 3
```

## Why This Matters

**Without pipeline:**
```python
# Caching logic scattered everywhere
if cache.has(prompt):
    return cache.get(prompt)

# Rate limiting before every call
rate_limiter.wait_if_needed()

# Retry logic duplicated
for attempt in range(3):
    try:
        response = model.generate(prompt)
        break
    except Exception:
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)

# Cost tracking manual
tracker.record(input_tokens, output_tokens)
```

**With pipeline:**
```python
response = pipeline.generate(prompt)
# All concerns handled automatically
```

## See Also

- [Cost Management](Cost-Management.md) - Budget tracking and forecasting
- [Retry Strategies](Retry-Strategies.md) - Detailed retry configuration
- [Caching](../guides/Caching.md) - Cache configuration options
