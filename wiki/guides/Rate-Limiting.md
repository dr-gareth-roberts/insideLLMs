---
title: Rate Limiting
parent: Guides
nav_order: 2
---

# Rate Limiting

Handle API rate limits gracefully.

## Why Rate Limiting?

API providers limit requests per minute/hour:

| Provider | Limit (typical) |
|----------|-----------------|
| OpenAI | 60-10,000 RPM |
| Anthropic | 60-4,000 RPM |
| Google | 60-1,000 RPM |

Exceeding limits causes errors and potential account issues.

## Enabling Rate Limiting

### In Config

```yaml
rate_limit:
  enabled: true
  requests_per_minute: 60
  requests_per_second: 1
```

### With Concurrency

```yaml
async: true
concurrency: 10

rate_limit:
  enabled: true
  requests_per_minute: 300
```

The rate limiter coordinates across concurrent requests.

## Automatic Retry

When rate limited, insideLLMs retries with exponential backoff:

```yaml
rate_limit:
  enabled: true
  requests_per_minute: 60
  retry:
    max_attempts: 3
    initial_delay: 1.0
    max_delay: 60.0
    exponential_base: 2.0
```

## Per-Provider Limits

Different limits for different providers:

```yaml
models:
  - type: openai
    args:
      model_name: gpt-4o
    rate_limit:
      requests_per_minute: 500

  - type: anthropic
    args:
      model_name: claude-3-5-sonnet
    rate_limit:
      requests_per_minute: 60
```

## Token-Based Limiting

For token limits (common with OpenAI):

```yaml
rate_limit:
  enabled: true
  tokens_per_minute: 90000
```

## Monitoring Rate Limits

### Check Current State

```python
from insideLLMs.rate_limiting import get_rate_limiter

limiter = get_rate_limiter()
print(f"Requests remaining: {limiter.remaining}")
print(f"Reset in: {limiter.reset_in} seconds")
```

### Rate Limit Headers

insideLLMs reads provider headers automatically:

```
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1234567890
```

## Strategies

### Conservative (Development)

```yaml
async: true
concurrency: 2
rate_limit:
  requests_per_minute: 30
```

### Balanced (Production)

```yaml
async: true
concurrency: 10
rate_limit:
  requests_per_minute: 300
```

### Aggressive (High Tier)

```yaml
async: true
concurrency: 50
rate_limit:
  requests_per_minute: 3000
```

## Combining with Caching

Reduce rate limit pressure:

```yaml
cache:
  enabled: true
  backend: sqlite

rate_limit:
  enabled: true
  requests_per_minute: 60
```

Cached responses don't count against rate limits.

## Error Handling

Rate limit errors are caught and handled:

```python
from insideLLMs.exceptions import RateLimitError

try:
    results = runner.run(prompt_set)
except RateLimitError as e:
    print(f"Rate limited: {e}")
    print(f"Retry after: {e.retry_after} seconds")
```

## Best Practices

### Do

-  Start conservative, increase gradually
-  Enable caching to reduce load
-  Monitor provider dashboards
-  Use appropriate tier for workload

### Don't

-  Ignore rate limit errors
-  Set limits higher than your tier
-  Run parallel jobs without coordination

## Troubleshooting

### "Rate limit exceeded"

```yaml
# Lower concurrency
concurrency: 3

# Lower rate
rate_limit:
  requests_per_minute: 30
```

### Requests still failing

Check your provider tier limits:
- OpenAI: [Rate limits](https://platform.openai.com/docs/guides/rate-limits)
- Anthropic: [Rate limits](https://docs.anthropic.com/en/api/rate-limits)

### Inconsistent limiting

Ensure single rate limiter instance:

```python
# Use singleton
from insideLLMs.rate_limiting import get_rate_limiter
limiter = get_rate_limiter()  # Same instance everywhere
```

---

## See Also

- [Caching](Caching.md) - Reduce API calls
- [Performance and Caching](../Performance-and-Caching.md) - More options
