---
title: FAQ
nav_order: 9
---

# Frequently Asked Questions

Quick answers to common questions. For detailed troubleshooting, see [Troubleshooting](guides/Troubleshooting.md).

## Installation

### Do I need API keys to get started?

No. Use `DummyModel` for offline testing:

```bash
insidellms quicktest "Hello" --model dummy
```

API keys are only needed for hosted providers (OpenAI, Anthropic, etc.).

### What Python version do I need?

Python 3.10 or higher. Check with:

```bash
python --version
```

### How do I install optional features?

```bash
pip install -e ".[nlp]"           # NLP features
pip install -e ".[visualisation]" # Charts and reports
pip install -e ".[all]"           # Everything
```

---

## Configuration

### Why can't it find my dataset file?

Relative paths are resolved from the **config file's directory**, not your current directory.

```yaml
# If config is at /project/configs/harness.yaml
dataset:
  path: ../data/test.jsonl  # Resolves to /project/data/test.jsonl
```

### How do I use environment variables in configs?

```yaml
model:
  type: openai
  args:
    api_key: ${OPENAI_API_KEY}
```

### What's the difference between `run` and `harness`?

| Command | Models | Probes | Use Case |
|---------|--------|--------|----------|
| `run` | Single | Single | Simple tests |
| `harness` | Multiple | Multiple | Comparisons |

---

## Running

### Why does `--overwrite` refuse to overwrite?

Safety guard. insideLLMs only overwrites directories containing `.insidellms_run` marker.

**Solutions:**
1. Use `--overwrite` with a valid run directory
2. Delete the directory manually
3. Use a new directory name

### How do I resume an interrupted run?

```bash
insidellms run config.yaml --run-dir ./my_run --resume
```

This continues from where it left off using existing `records.jsonl`.

### Can I run multiple models in parallel?

Yes, with async execution:

```yaml
async: true
concurrency: 10
```

Or via CLI: `--async --concurrency 10`

---

## Models

### Can I run local models?

Yes! Supported options:

| Runner | Setup |
|--------|-------|
| Ollama | `ollama pull llama3` |
| llama.cpp | Download GGUF model |
| vLLM | `pip install vllm` |

See [Local Models Guide](guides/Local-Models.md).

### How do I compare different models?

Use a harness config:

```yaml
models:
  - type: openai
    args: {model_name: gpt-4o}
  - type: anthropic
    args: {model_name: claude-3-5-sonnet-20241022}
```

### What models are supported?

OpenAI, Anthropic, Google/Gemini, Cohere, HuggingFace, Ollama, vLLM, llama.cpp, and custom implementations. See [Models Catalog](reference/Models-Catalog.md).

---

## Cost & Performance

### How do I reduce API costs?

1. **Limit examples**: `max_examples: 50`
2. **Enable caching**: `cache: {enabled: true}`
3. **Use cheaper models**: Start with `gpt-4o-mini`
4. **Test with DummyModel**: No cost for framework testing

### How do I speed up runs?

```yaml
async: true
concurrency: 20
cache:
  enabled: true
```

### I'm hitting rate limits. What do I do?

```yaml
rate_limit:
  enabled: true
  requests_per_minute: 60

concurrency: 5  # Lower this
```

See [Rate Limiting Guide](guides/Rate-Limiting.md).

---

## Outputs

### What files does insideLLMs create?

| File | Purpose |
|------|---------|
| `records.jsonl` | Raw results |
| `manifest.json` | Run metadata |
| `summary.json` | Aggregated stats |
| `report.html` | Visual report |

### How do I keep outputs out of `~/.insidellms`?

```bash
# Per-run
insidellms run config.yaml --run-dir ./my_output

# Global default
export INSIDELLMS_RUN_ROOT=./runs
```

### How do I generate just a report?

```bash
insidellms report ./my_run
```

---

## CI Integration

### How do I detect behavioural changes in CI?

```bash
insidellms diff ./baseline ./candidate --fail-on-changes
```

Exit code 1 = changes detected, 0 = identical.

### Why do my CI runs produce different outputs?

Model responses are non-deterministic. For deterministic CI:

```yaml
models:
  - type: dummy
    args:
      response: "Fixed response"
```

### How do I update the baseline after intentional changes?

```bash
insidellms harness config.yaml --run-dir ./baseline --overwrite
git add ./baseline
git commit -m "Update baseline: [describe changes]"
```

---

## Troubleshooting

### "command not found: insidellms"

Activate your virtual environment:

```bash
source .venv/bin/activate
```

Or run as module: `python -m insideLLMs.cli`

### "Invalid API key"

1. Check key format (OpenAI: `sk-...`, Anthropic: `sk-ant-...`)
2. Verify key in provider dashboard
3. Ensure env var is set: `echo $OPENAI_API_KEY`

### How do I turn off coloured output?

```bash
export NO_COLOR=1
```

### Where can I find example datasets?

- `data/` directory in the repo
- `benchmarks/` for standard benchmarks
- `insideLLMs.benchmark_datasets` for built-in datasets
- HuggingFace datasets via config

---

## Advanced

### Can I create custom probes?

Yes! See [Custom Probe Tutorial](tutorials/Custom-Probe.md).

```python
from insideLLMs.probes.base import Probe

class MyProbe(Probe):
    def run(self, model, data, **kwargs):
        return model.generate(data["prompt"])
```

### Can I create custom models?

Yes! Implement the Model interface:

```python
from insideLLMs.models.base import Model

class MyModel(Model):
    def generate(self, prompt: str, **kwargs) -> str:
        return "response"
```

### How do I integrate with LangChain?

See [LangChain and LangGraph](LangChain-and-LangGraph.md).

---

## Getting Help

- [Troubleshooting Guide](guides/Troubleshooting.md)
- [GitHub Issues](https://github.com/dr-gareth-roberts/insideLLMs/issues)
- Run `insidellms doctor` to check your environment
