# Rate Limiting Guide

## Overview

insideLLMs provides automatic rate limiting to prevent hitting API provider limits and manage costs.

## Quick Start

### Via RunConfig (Recommended)

```python
from insideLLMs.models import OpenAIModel
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.runner import ProbeRunner
from insideLLMs.config import RunConfig, RateLimitConfig

model = OpenAIModel()
probe = LogicProbe()

# Configure rate limiting
config = RunConfig(
    run_id="rate_limited_run",
    rate_limit=RateLimitConfig(
        requests_per_minute=60,      # 60 RPM
        tokens_per_minute=90000,     # 90K TPM
        requests_per_day=10000       # Optional daily limit
    )
)

runner = ProbeRunner(model, probe, config=config)
results = runner.run(prompts)  # Automatically rate limited
```

### Via YAML Config

```yaml
# config.yaml
model_id: gpt-3.5-turbo
probe_id: logic
dataset:
  - prompt: "What is 2+2?"
  - prompt: "What is 3+3?"

rate_limit:
  requests_per_minute: 60
  tokens_per_minute: 90000
  requests_per_day: 10000
```

```bash
insidellms harness config.yaml --run-dir ./runs/limited
```

## Provider Limits

### OpenAI

| Tier | RPM | TPM | RPD |
|------|-----|-----|-----|
| Free | 3 | 40,000 | 200 |
| Tier 1 | 500 | 60,000 | - |
| Tier 2 | 5,000 | 450,000 | - |

### Anthropic

| Tier | RPM | TPM |
|------|-----|-----|
| Free | 5 | 25,000 |
| Tier 1 | 50 | 100,000 |
| Tier 2 | 1,000 | 400,000 |

## Best Practices

### 1. Set Conservative Limits

```python
# For free tier, use conservative limits
config = RunConfig(
    run_id="free_tier",
    rate_limit=RateLimitConfig(
        requests_per_minute=2,  # Below limit
        tokens_per_minute=30000,
        requests_per_day=150
    )
)
```

### 2. Monitor Usage

```python
# Enable logging to see rate limit activity
import logging
logging.basicConfig(level=logging.INFO)

runner = ProbeRunner(model, probe, config=config)
results = runner.run(prompts)
# Logs: "RPM limit reached, sleeping 5.2s"
```

### 3. Batch Processing

```python
# Process in batches to avoid daily limits
prompts = load_large_dataset()  # 1000 prompts

batch_size = 100
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    results = runner.run(batch)
    save_results(results)
```

## Advanced Usage

### Custom Rate Limiter

```python
from insideLLMs.rate_limiting import RateLimitedModel

# Wrap model directly
base_model = OpenAIModel()
limited_model = RateLimitedModel(
    model=base_model,
    requests_per_minute=100,
    tokens_per_minute=150000,
    burst_size=10  # Allow bursts of 10 requests
)

response = limited_model.generate("Hello")
```

### Dynamic Limits

```python
# Adjust limits based on time of day
import datetime

hour = datetime.datetime.now().hour

if 9 <= hour <= 17:  # Business hours
    rpm = 30  # Conservative
else:
    rpm = 100  # Aggressive

config = RunConfig(
    run_id="dynamic",
    rate_limit=RateLimitConfig(requests_per_minute=rpm)
)
```

## Troubleshooting

### Still Hitting Rate Limits

1. **Check actual limits**: Verify your tier limits with provider
2. **Account for other usage**: Other applications may share limits
3. **Add buffer**: Set limits 10-20% below actual limits
4. **Enable logging**: See when limits are hit

```python
import logging
logging.getLogger("insideLLMs.rate_limiting").setLevel(logging.DEBUG)
```

### Slow Execution

1. **Check if rate limiting is too conservative**
2. **Increase limits** if you have capacity
3. **Use async runner** for better throughput

```python
from insideLLMs.runtime.runner import AsyncProbeRunner

runner = AsyncProbeRunner(model, probe, config=config)
results = await runner.run(prompts, concurrency=10)
```

## See Also

- [Cost Tracking](COST_TRACKING.md)
- [Caching](CACHING.md)
- [Async Execution](ASYNC.md)