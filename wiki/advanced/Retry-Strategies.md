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

pipeline = ModelPipeline(model)
pipeline.add_middleware(RetryMiddleware(
    max_attempts=3,
    initial_delay=1.0,
    exponential_base=2.0,
    max_delay=60.0
))

# Automatically retries on failure
response = pipeline.generate(prompt)
```

## Retry Configuration

```python
from insideLLMs.retry import RetryConfig

config = RetryConfig(
    max_attempts=3,           # Try up to 3 times
    initial_delay=1.0,        # Start with 1 second delay
    exponential_base=2.0,     # Double delay each retry
    max_delay=60.0,           # Cap at 60 seconds
    jitter=True,              # Add randomness to prevent thundering herd
    retry_on=[TimeoutError, RateLimitError]  # Which errors to retry
)
```

## Backoff Strategies

### Exponential Backoff

```python
# Delays: 1s, 2s, 4s, 8s, ...
RetryMiddleware(
    initial_delay=1.0,
    exponential_base=2.0
)
```

### Linear Backoff

```python
# Delays: 1s, 2s, 3s, 4s, ...
RetryMiddleware(
    initial_delay=1.0,
    exponential_base=1.0
)
```

### Constant Delay

```python
# Delays: 2s, 2s, 2s, ...
RetryMiddleware(
    initial_delay=2.0,
    exponential_base=1.0,
    max_delay=2.0
)
```

## Circuit Breaker

Prevent cascade failures by stopping requests after repeated failures.

```python
from insideLLMs.retry import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,    # Try again after 60s
    half_open_max_calls=3     # Test with 3 calls before fully closing
)

# Use with pipeline
pipeline.add_middleware(RetryMiddleware(circuit_breaker=breaker))

# Circuit opens after failures
try:
    response = pipeline.generate(prompt)
except CircuitBreakerOpen:
    print("Circuit open - too many failures")
```

## Selective Retry

```python
from insideLLMs.exceptions import RateLimitError, TimeoutError, ValidationError

# Retry only specific errors
RetryMiddleware(
    max_attempts=3,
    retry_on=[RateLimitError, TimeoutError],  # Retry these
    no_retry_on=[ValidationError]             # Don't retry these
)
```

## Retry with Backpressure

```python
# Reduce concurrency on retry
RetryMiddleware(
    max_attempts=3,
    reduce_concurrency_on_retry=True,
    min_concurrency=1
)
```

## Monitoring Retries

```python
from insideLLMs.retry import RetryStats

stats = pipeline.get_retry_stats()
print(f"Total retries: {stats.total_retries}")
print(f"Success after retry: {stats.retry_successes}")
print(f"Failed after retries: {stats.retry_failures}")
print(f"Average attempts: {stats.avg_attempts:.1f}")
```

## Configuration

```yaml
# In harness config
retry:
  enabled: true
  max_attempts: 3
  initial_delay: 1.0
  exponential_base: 2.0
  max_delay: 60.0
  jitter: true
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout: 60.0
```

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
