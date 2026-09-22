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
── Response ──────────────────────────────────────────
  [DummyModel] You said: What is 2 + 2?

── Stats ─────────────────────────────────────────────
  Latency: 0.0ms
  Response length: 37 characters
```

DummyModel returns deterministic local responses. It is useful for checking the
workflow without API costs.

## With a Real Model

```bash
python3 -m pip install "insidellms[openai]"
export OPENAI_API_KEY="sk-..."
insidellms quicktest "What is 2 + 2?" \
  --model openai \
  --model-args '{"model_name":"gpt-4o"}'
# Response: 2 + 2 equals 4.
```

Keep API keys in environment variables. Do not put them in a config file: model
arguments are treated as literal values and the resolved config is saved with
the run artifacts.

## Config-Driven Run

```bash
# Creates my_first_run.yaml and data/questions.jsonl
insidellms init my_first_run.yaml --template basic

# Inspect the plan by opening the generated files, then run it
insidellms run my_first_run.yaml --run-dir ./runs/my_first_run
```

Creates:
- `records.jsonl` - Every input/output pair
- `manifest.json` - Run metadata
- `config.resolved.yaml` - Full config snapshot

## Test Determinism

```bash
# DummyModel makes this an offline deterministic example
insidellms run my_first_run.yaml --run-dir ./runs/baseline
insidellms run my_first_run.yaml --run-dir ./runs/candidate
insidellms diff ./runs/baseline ./runs/candidate
# Changes: 0 (deterministic)
```

A plain `diff` is informational and exits 0 even when it finds changes. Add a
gate such as `--fail-on-changes` when a CI job should fail. On reruns, choose new
directories or explicitly pass `--overwrite` to replace guarded run directories.

## Next

[First Harness →](First-Harness.md) Compare multiple models.
