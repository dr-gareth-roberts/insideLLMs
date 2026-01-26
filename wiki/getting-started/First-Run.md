---
title: First Run
parent: Getting Started
nav_order: 2
---

# First Run

**3 minutes. No API keys.**

## Verify It Works

```bash
insidellms quicktest "What is 2 + 2?" --model dummy
```

```
Model: DummyModel
Prompt: What is 2 + 2?
Response: This is a dummy response for testing purposes.
```

DummyModel returns fixed responses. Perfect for testing without API costs.

## With a Real Model

```bash
export OPENAI_API_KEY="sk-..."
insidellms quicktest "What is 2 + 2?" --model openai
# Response: 2 + 2 equals 4.
```

## Config-Driven Run

```yaml
# my_first_run.yaml
model:
  type: dummy
probe:
  type: logic
dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
    - question: "What colour is the sky?"
```

```bash
insidellms run my_first_run.yaml --run-dir ./my_first_run
```

Creates:
- `records.jsonl` - Every input/output pair
- `manifest.json` - Run metadata
- `config.resolved.yaml` - Full config snapshot

## Test Determinism

```bash
# Run twice, diff should show 0 changes
insidellms harness ci/harness.yaml --run-dir .tmp/baseline --overwrite
insidellms harness ci/harness.yaml --run-dir .tmp/candidate --overwrite
insidellms diff .tmp/baseline .tmp/candidate
# Changes: 0 (deterministic)
```

## Next

[First Harness →](First-Harness.md) Compare multiple models.
