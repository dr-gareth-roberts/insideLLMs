---
title: First Run
parent: Getting Started
nav_order: 2
---

# First Run

Run your first insideLLMs test in 5 minutes. No API keys required.

## Quick Test

The fastest way to verify everything works:

```bash
insidellms quicktest "What is 2 + 2?" --model dummy
```

**Expected output:**

```
Model: DummyModel
Prompt: What is 2 + 2?
Response: This is a dummy response for testing purposes.
```

The DummyModel returns a fixed response, perfect for testing the framework without API costs.

## Understanding What Happened

1. **Model**: DummyModel was instantiated (no network calls)
2. **Probe**: The default probe passed your prompt to the model
3. **Output**: The response was printed to stdout

## Run with a Real Model

If you have an OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
insidellms quicktest "What is 2 + 2?" --model openai
```

**Expected output:**

```
Model: OpenAIModel (gpt-4o-mini)
Prompt: What is 2 + 2?
Response: 2 + 2 equals 4.
```

## Run from a Config File

Create a simple config file:

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
    - question: "What color is the sky?"
    - question: "Name a prime number."
```

Run it:

```bash
insidellms run my_first_run.yaml --run-dir ./my_first_run
```

**What gets created:**

```
my_first_run/
├── records.jsonl      # One JSON line per input
├── manifest.json      # Run metadata
└── config.resolved.yaml  # Full resolved config
```

## Inspect the Results

View records:

```bash
cat my_first_run/records.jsonl | head -1 | python -m json.tool
```

```json
{
  "schema_version": "1.0.0",
  "run_id": "abc123...",
  "example_id": "0",
  "input": {"question": "What is 2 + 2?"},
  "output": "This is a dummy response...",
  "status": "success"
}
```

## Golden Path (Offline CI Test)

Run the built-in determinism test:

```bash
# Run twice with identical inputs
insidellms harness ci/harness.yaml --run-dir .tmp/baseline --overwrite
insidellms harness ci/harness.yaml --run-dir .tmp/candidate --overwrite

# Diff should show no changes (deterministic)
insidellms diff .tmp/baseline .tmp/candidate
```

If everything is working, the diff will report **0 changes**.

## What's Next?

You've successfully:
- Installed insideLLMs
- Run a quick test
- Created a config-driven run
- Verified determinism

**Next steps:**

- [First Harness](First-Harness.md) - Compare multiple models
- [Understanding Outputs](Understanding-Outputs.md) - Deep dive into artifacts
- [Tutorials](../tutorials/index.md) - Step-by-step guides for real tasks
