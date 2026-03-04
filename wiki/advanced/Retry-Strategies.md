---
title: Retry Strategies
parent: Advanced Features
nav_order: 5
---

# Retry Strategies

**Handle transient failures gracefully.**

## The Problem

API calls fail. Rate limits hit. Networks timeout. Your code crashes.

## The Solution

Automatic retry with exponential backoff and circuit breakers.

```python
from insideLLMs.pipeline import ModelPipeline, RetryMiddleware

pipeline = ModelPipeline(
    model,
    middlewares=[
        RetryMiddleware(
            max_retries=3,
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
        ),
    ],
)

# Automatically retries on failure
response = pipeline.generate(prompt)
```

## Retry Configuration

```python
from insideLLMs.retry import RetryConfig

config = RetryConfig(
    max_retries=3,            # Try up to 3 times
    initial_delay=1.0,        # Start with 1 second delay
    exponential_base=2.0,     # Double delay each retry
    max_delay=60.0,           # Cap at 60 seconds
    jitter=True,              # Add randomness to prevent thundering herd
    retryable_exceptions=(TimeoutError, RateLimitError),
)
```

## Backoff Strategies

### Exponential Backoff

```python
# Delays: 1s, 2s, 4s, 8s, ...
RetryMiddleware(
    initial_delay=1.0,
    exponential_base=2.0,
)
```

### Linear Backoff

```python
# Delays: 1s, 2s, 3s, 4s, ...
RetryMiddleware(
    initial_delay=1.0,
    exponential_base=1.0,
)
```

### Constant Delay

```python
# Delays: 2s, 2s, 2s, ...
RetryMiddleware(
    initial_delay=2.0,
    exponential_base=1.0,
    max_delay=2.0,
)
```

## Circuit Breaker

Prevent cascade failures by stopping requests after repeated failures.

```python
from insideLLMs.retry import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen

config = CircuitBreakerConfig(
    failure_threshold=5,
    reset_timeout=60.0,
    half_open_max_calls=3,
)
breaker = CircuitBreaker("api_service", config)

try:
    response = breaker.execute(lambda: pipeline.generate(prompt))
except CircuitBreakerOpen:
    print("Circuit open - too many failures")
```

## Selective Retry

```python
from insideLLMs.exceptions import RateLimitError, TimeoutError
from insideLLMs.pipeline import RetryMiddleware

# RetryMiddleware retries RateLimitError, TimeoutError, and ModelError.
# For fine-grained control, use insideLLMs.retry.retry(...) on a specific call site.
retry_mw = RetryMiddleware(max_retries=3)
```

## Retry with Backpressure

```python
# Backpressure is handled via rate limiting and concurrency control.
# Use RateLimitMiddleware and AsyncModelPipeline(max_concurrency=...) for throughput tuning.
```

## Monitoring Retries

```python
# RetryMiddleware exposes counters directly.
retry_mw = RetryMiddleware(max_retries=3)
pipeline = ModelPipeline(model, middlewares=[retry_mw])

pipeline.generate(prompt)
print(f"Total retries attempted: {retry_mw.total_retries}")
```

## Configuration

```yaml
# Via model pipeline middleware in run/harness config:
model:
  type: openai
  args:
    model_name: gpt-4o-mini
  pipeline:
    middlewares:
      - type: retry
        args:
          max_retries: 3
          initial_delay: 1.0
          exponential_base: 2.0
          max_delay: 60.0
```

For advanced retry control (circuit breaker, custom exceptions, per-call retry), use the
programmatic API directly:

## Best Practices

**Do:**
- Use exponential backoff for rate limits
- Add jitter to prevent thundering herd
- Set max_delay to prevent infinite waits
- Use circuit breakers for cascading failures

**Don't:**
- Retry validation errors (they won't succeed)
- Set max_attempts too high (wastes time/money)
- Retry without backoff (hammers the API)

## Why This Matters

**Without retry:**
- Transient failures crash your pipeline
- Manual retry logic scattered everywhere
- No protection against cascade failures
- Wasted API calls on permanent errors

**With retry:**
- Transient failures handled automatically
- Centralised retry logic
- Circuit breakers prevent cascades
- Smart retry only on retriable errors

## See Also

- [Pipeline Architecture](Pipeline-Architecture.md) - Combine with other middleware
- [Rate Limiting](../guides/Rate-Limiting.md) - Prevent hitting limits
