---
title: Production Shadow Capture
parent: Guides
nav_order: 6
---

# Production Shadow Capture

Capture a small, safe slice of production traffic in canonical `records.jsonl`
format, then replay and diff it through the same deterministic harness pipeline.

## Why Use Shadow Capture

- Build regression suites from real user traffic.
- Detect behavior drift earlier than synthetic-only testing.
- Reuse the same `run -> records -> report -> diff` spine.

## FastAPI Middleware Setup

```python
from fastapi import FastAPI
from insideLLMs import shadow

app = FastAPI()
app.middleware("http")(
    shadow.fastapi(
        output_path="./shadow/records.jsonl",
        sample_rate=0.01,
        model_id="prod-gpt4o",
        model_provider="openai",
        dataset_id="prod-traffic",
        include_request_headers=False,
    )
)
```

## Sampling Strategy

- Start with `sample_rate=0.01` (1%) and increase only after storage review.
- Keep sampling deterministic and stable across deploys.
- Use separate output paths per service/environment.

## Privacy and Redaction Guidance

- Default to `include_request_headers=False`.
- Redact or hash sensitive request/response fields before writing.
- Restrict access to shadow artifact directories.
- Treat shadow records as production data for retention/compliance policies.

## Replay and Diff Workflow

```bash
# 1) Generate baseline run from approved prompt/model setup
insidellms harness ci/harness.yaml --run-dir .tmp/runs/base --overwrite

# 2) Generate candidate run (new code/model/prompt)
insidellms harness ci/harness.yaml --run-dir .tmp/runs/head --overwrite

# 3) Gate behavior drift
insidellms diff .tmp/runs/base .tmp/runs/head --fail-on-changes
```

For agentic/tool workflows, add:

```bash
insidellms diff .tmp/runs/base .tmp/runs/head --fail-on-trajectory-drift
```

## Operational Tips

- Rotate `records.jsonl` outputs daily or by size.
- Attach source metadata (`service`, `env`, `version`) in `custom` fields upstream.
- Keep a reproducible baseline branch/reference for CI diffing.
