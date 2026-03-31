# Rate Limiting Guide

## Overview

insideLLMs provides automatic rate limiting to prevent hitting API provider limits and manage costs.

## Quick Start

### Via ModelPipeline + RunConfig (Recommended)

```python
from insideLLMs.config_types import RunConfig
from insideLLMs.models import OpenAIModel
from insideLLMs.pipeline import ModelPipeline, RateLimitMiddleware
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.runner import ProbeRunner

# Wrap your base model with rate-limiting middleware
base_model = OpenAIModel()
model = ModelPipeline(
    base_model,
    middlewares=[RateLimitMiddleware(requests_per_minute=60, burst_size=10)],
)

probe = LogicProbe()
runner = ProbeRunner(model, probe)

# Run config controls runner behaviour (artefacts, validation, etc.)
config = RunConfig(run_root="./runs/rate_limited")

results = runner.run(prompts, config=config)
```

### Via YAML Config

```yaml
# config.yaml
model:
  type: openai
  args:
    model_name: gpt-4o-mini
  pipeline:
    middlewares:
      - type: rate_limit
        args:
          requests_per_minute: 60
          burst_size: 10

probe:
  type: logic

dataset:
  format: jsonl
  path: data/prompts.jsonl
```

```bash
insidellms run config.yaml --run-dir ./runs/limited
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
from insideLLMs.config_types import RunConfig
from insideLLMs.models import OpenAIModel
from insideLLMs.pipeline import ModelPipeline, RateLimitMiddleware
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.runner import ProbeRunner

# For free tier, use conservative limits
base_model = OpenAIModel()
model = ModelPipeline(
    base_model,
    middlewares=[RateLimitMiddleware(requests_per_minute=2, burst_size=1)],
)

probe = LogicProbe()
runner = ProbeRunner(model, probe)
results = runner.run(prompts, config=RunConfig(run_id="free_tier"))
```

### 2. Monitor Usage

```python
# Enable logging to see rate limit activity
import logging
logging.basicConfig(level=logging.INFO)

runner = ProbeRunner(model, probe)
results = runner.run(prompts, config=config)
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
from insideLLMs.models import OpenAIModel
from insideLLMs.rate_limiting import create_rate_limiter

# Standalone limiter for custom call flows
base_model = OpenAIModel()
limiter = create_rate_limiter(rate=100 / 60, capacity=10)

if limiter.acquire(tokens=1, block=True):
    response = base_model.generate("Hello")
```

### Dynamic Limits

```python
from insideLLMs.config_types import RunConfig
from insideLLMs.models import OpenAIModel
from insideLLMs.pipeline import ModelPipeline, RateLimitMiddleware
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.runner import ProbeRunner

# Adjust limits based on time of day
import datetime

hour = datetime.datetime.now().hour

if 9 <= hour <= 17:  # Business hours
    rpm = 30  # Conservative
else:
    rpm = 100  # Aggressive

base_model = OpenAIModel()
rate_middleware = RateLimitMiddleware(requests_per_minute=rpm)
model = ModelPipeline(base_model, middlewares=[rate_middleware])
probe = LogicProbe()
runner = ProbeRunner(model, probe)
results = runner.run(prompts, config=RunConfig(run_id="dynamic"))
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
from insideLLMs.config_types import RunConfig
from insideLLMs.runtime.runner import AsyncProbeRunner

runner = AsyncProbeRunner(model, probe)
results = await runner.run(prompts, config=RunConfig(concurrency=10))
```

## See Also

- [Cost Tracking](COST_TRACKING.md)
- [Caching](CACHING.md)
- [Async Execution](ASYNC.md)
