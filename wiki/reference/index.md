---
title: Reference
nav_order: 5
has_children: true
---

# Reference

Complete reference documentation for insideLLMs.

## Reference Guides

| Guide | Description |
|-------|-------------|
| [CLI Reference](CLI.md) | All commands, flags, and options |
| [Configuration](Configuration.md) | Complete config file reference |
| [Probes Catalog](Probes-Catalog.md) | Every built-in probe with examples |
| [Models Catalog](Models-Catalog.md) | Every model provider with config |
| [Schemas](Schemas.md) | Output artifact schemas |

## Quick Links

### CLI Commands

```bash
insidellms run         # Run from config file
insidellms harness     # Multi-model comparison
insidellms quicktest   # Quick single-prompt test
insidellms diff        # Compare two runs
insidellms report      # Generate HTML report
insidellms validate    # Validate run artifacts
insidellms schema      # Schema utilities
insidellms doctor      # Check environment
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API access |
| `ANTHROPIC_API_KEY` | Anthropic API access |
| `GOOGLE_API_KEY` | Google/Gemini API access |
| `CO_API_KEY` | Cohere API access |
| `INSIDELLMS_RUN_ROOT` | Default output directory |
| `NO_COLOR` | Disable colored output |

### Python API

```python
# Models
from insideLLMs.models import OpenAIModel, AnthropicModel, DummyModel

# Probes
from insideLLMs.probes import LogicProbe, BiasProbe, AttackProbe

# Runners
from insideLLMs.runtime.runner import ProbeRunner, AsyncProbeRunner

# Registry
from insideLLMs.registry import model_registry, probe_registry

# Utilities
from insideLLMs.dataset_utils import load_jsonl_dataset, load_csv_dataset
```

## Finding What You Need

| I want to... | Go to... |
|--------------|----------|
| Look up a CLI flag | [CLI Reference](CLI.md) |
| Configure a model | [Models Catalog](Models-Catalog.md) |
| Choose a probe | [Probes Catalog](Probes-Catalog.md) |
| Write a config file | [Configuration](Configuration.md) |
| Understand output format | [Schemas](Schemas.md) |
