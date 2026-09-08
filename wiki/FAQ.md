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
python3 --version
```

### How do I install optional features?

```bash
python3 -m pip install "insidellms[nlp]"           # NLP features
python3 -m pip install "insidellms[visualization]" # Report dependencies
python3 -m pip install "insidellms[providers]"     # OpenAI, Anthropic, Hugging Face
```

Other provider integrations may require their SDK separately. When developing
from a repository clone, use editable forms such as
`python3 -m pip install -e ".[dev]"` from the repository root.

---

## Configuration

### Why can't it find my dataset file?

Relative paths are resolved from the **config file's directory**, not your current directory.

```yaml
# If config is at /project/configs/harness.yaml
dataset:
  path: ../data/harness_dataset.jsonl
# Resolves to /project/data/harness_dataset.jsonl
```

For the generated sample, run `insidellms init harness.yaml --template harness`
from `/project` and keep `harness.yaml` there; the initializer creates
`/project/data/harness_dataset.jsonl`.

### How do I keep API keys out of configs?

```bash
export OPENAI_API_KEY="sk-..."
insidellms run config.yaml
```

Do not write `${OPENAI_API_KEY}` in model `args`: CLI model arguments are
literal values and are not environment-expanded. Supported providers read their
usual environment variables when `api_key` is omitted. This also avoids copying
a key into `config.resolved.yaml`.

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

Yes. Use async CLI execution:

```bash
insidellms run config.yaml --async --concurrency 10
```

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
2. **Enable caching**: Add `cache` middleware in `pipeline.middlewares`
3. **Use cheaper models**: Start with `gpt-4o-mini`
4. **Test with DummyModel**: No cost for framework testing

### How do I speed up runs?

```yaml
pipeline:
  middlewares:
    - type: cache
      args:
        cache_size: 1000
```

Run with concurrency: `insidellms run config.yaml --async --concurrency 20`

### I'm hitting rate limits. What do I do?

```yaml
pipeline:
  middlewares:
    - type: rate_limit
      args:
        requests_per_minute: 60
        burst_size: 10
```

Also lower concurrency: `insidellms run config.yaml --async --concurrency 5`

See [Rate Limiting Guide](guides/Rate-Limiting.md).

---

## Outputs

### What files does insideLLMs create?

| Command | Files |
|---------|-------|
| `run` | `config.resolved.yaml`, `records.jsonl`, `manifest.json` |
| `harness` | The files above plus `summary.json`, legacy `results.jsonl`, and normally `report.html` |
| `report` | Rebuilds `summary.json` and `report.html` in an existing run directory |

`harness --skip-report` omits `report.html`. A `diff.json` file is only created
when explicitly requested with `diff --format json --output diff.json`.

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

With `--fail-on-changes`, exit code 2 means the diff found regressions, other
changes, or records present on only one side. Improvements alone do not fail
this gate, and trace/trajectory-only findings require their dedicated flags.
Without a fail flag, `diff` is informational and returns 0 even when it reports
differences. A parsed command/setup error normally returns 1; argparse usage
errors also return 2. Specialized trace and trajectory gates use codes 3–5.

### Why do my CI runs produce different outputs?

Model responses are non-deterministic. For deterministic CI:

```yaml
models:
  - type: dummy
    args:
      canned_response: "Fixed response"
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

Or run as module: `python3 -m insideLLMs.cli`

### "Invalid API key"

1. Check key format (OpenAI: `sk-...`, Anthropic: `sk-ant-...`)
2. Verify key in provider dashboard
3. Check presence without printing it: `test -n "$OPENAI_API_KEY" && echo "key is set"`

### How do I turn off coloured output?

```bash
export NO_COLOR=1
```

### Where can I find example datasets?

- `data/questions.jsonl` and `data/harness_dataset.jsonl`, created locally by `insidellms init`
- `ci/harness_dataset.jsonl` in a source checkout
- `insideLLMs.benchmark_datasets` for built-in smoke-test datasets (87 tiny handwritten examples — not real benchmarks)
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
