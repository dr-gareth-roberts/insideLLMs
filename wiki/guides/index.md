---
title: Guides
nav_order: 8
has_children: true
---

# Guides

How-to guides for specific tasks and integrations.

## Available Guides

| Guide | Description |
|-------|-------------|
| [Caching](Caching.md) | Speed up runs with response caching |
| [Rate Limiting](Rate-Limiting.md) | Handle API rate limits gracefully |
| [Experiment Tracking](Experiment-Tracking.md) | Log runs to W&B, MLflow, TensorBoard |
| [Local Models](Local-Models.md) | Run Ollama, llama.cpp, vLLM |
| [Production Shadow Capture](Production-Shadow-Capture.md) | Capture sampled production traffic into canonical records |
| [IDE Extension Workflow](IDE-Extension-Workflow.md) | Run probes from VS Code/Cursor CodeLens |
| [Troubleshooting](Troubleshooting.md) | Common issues and solutions |

## Guides vs Tutorials

| Tutorials | Guides |
|-----------|--------|
| Learning-oriented | Task-oriented |
| Step-by-step walkthroughs | Focused how-tos |
| Build something complete | Solve a specific problem |
| "How do I get started?" | "How do I do X?" |

## Quick Solutions

### Speed Up Runs

```yaml
# Enable async execution
async: true
concurrency: 10
```

```bash
insidellms run config.yaml --async --concurrency 10
```

### Reduce Costs

```yaml
# Limit examples and enable caching
max_examples: 50
cache:
  enabled: true
  backend: sqlite
```

### Run Offline

```yaml
# Use DummyModel for testing
models:
  - type: dummy
```

### Debug Failures

```bash
# Check environment
insidellms doctor

# Validate artifacts
insidellms validate ./my_run

# Verbose output
insidellms run config.yaml --verbose
```

## Common Patterns

### Development Workflow

```bash
# 1. Generate and tune a local config
insidellms init dev.yaml --model dummy

# 2. Run a deterministic harness candidate
insidellms harness dev.yaml --run-dir .tmp/runs/dev --overwrite

# 3. Full run
insidellms run config.yaml --run-dir .tmp/runs/full --overwrite
```

### CI Workflow

```bash
# 1. Run candidate
insidellms harness config.yaml --run-dir ./candidate

# 2. Diff against baseline
insidellms diff ./baseline ./candidate --fail-on-changes
```
